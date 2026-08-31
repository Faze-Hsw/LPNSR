#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-step End-to-End Noise Predictor Training
"""

import argparse
import math
import os
import ssl
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

# Disable SSL certificate verification + HF mirror (pyiqa downloads metric
# weights from HuggingFace; Windows Miniconda fails on the default CA bundle).
# Must be set BEFORE any huggingface_hub / timm import chain.
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import matplotlib
import numpy as np
import torch
import yaml
from safetensors.torch import save_file
from torch.amp import GradScaler, autocast
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend, suitable for server environments

from datapipe.train_dataloader import create_train_dataloader
from ldm.models.autoencoder import VQModelTorch
from losses.gan_loss import GANLoss, create_discriminator
from losses.lpips_loss import LPIPSLoss
from models.noise_predictor import create_noise_predictor
from models.unet import UNetModelSwin

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_named_eta_schedule(
    schedule_name,
    num_diffusion_timesteps,
    min_noise_level,
    etas_end=0.99,
    kappa=1.0,
    kwargs=None,
):
    """
    Get the eta schedule for ResShift

    This is ResShift's unique noise scheduling method, completely different from DDPM's beta schedule!

    Args:
        schedule_name: Schedule type ('exponential' or 'ldm')
        num_diffusion_timesteps: Number of diffusion steps T
        min_noise_level: Minimum noise level η_1
        etas_end: Maximum noise level η_T
        kappa: Variance control parameter κ
        kwargs: Additional parameters (e.g., power)

    Returns:
        sqrt_etas: √η_t array, shape=(T,)
    """
    if kwargs is None:
        kwargs = {}

    if schedule_name == "exponential":
        # Exponential schedule (ResShift default)
        power = kwargs.get("power", 2.0)
        etas_start = min(min_noise_level / kappa, min_noise_level)

        # Calculate growth factor
        increaser = math.exp(
            1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start)
        )
        base = (
            np.ones(
                [
                    num_diffusion_timesteps,
                ]
            )
            * increaser
        )

        # Calculate power timestep
        power_timestep = (
            np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        )
        power_timestep *= num_diffusion_timesteps - 1

        # Calculate sqrt_etas
        sqrt_etas = np.power(base, power_timestep) * etas_start

    elif schedule_name == "ldm":
        # Load from .mat file
        import scipy.io as sio

        mat_path = kwargs.get("mat_path", None)
        if mat_path is None:
            raise ValueError("ldm schedule requires mat_path")
        sqrt_etas = sio.loadmat(mat_path)["sqrt_etas"].reshape(-1)

    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name}")

    return sqrt_etas


class NoisePredictorTrainer:
    """Noise Predictor End-to-End Trainer"""

    def __init__(self, config_path):
        """
        Initialize trainer

        Args:
            config_path: Config file path
        """
        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create experiment directory
        self.exp_dir = Path(self.config["experiment"]["save_dir"])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "samples").mkdir(exist_ok=True)

        # Initialize models
        self._init_models()

        # Initialize losses
        self._init_losses()

        # Initialize optimizer
        self._init_optimizer()

        # Initialize dataloaders
        self._init_dataloaders()

        # Initialize AMP
        if self.config["training"]["use_amp"]:
            self.scaler_g = GradScaler()  # Generator-specific scaler
            self.scaler_d = (
                GradScaler() if self.discriminator is not None else None
            )  # Discriminator-specific scaler
        else:
            self.scaler_g = None
            self.scaler_d = None

        # Iteration-based training state (RFMSR / trainer.py style)
        exp = self.config["experiment"]
        self.total_iters = self.config["training"]["iterations"]
        # Logging shares the same frequency as checkpoint saving
        self.save_freq = exp.get("save_freq", 1000)
        self.keep_last_n = exp.get("keep_last_n", 0)
        # Gradient accumulation (RFMSR style): one iteration = one optimizer
        # step, accumulated over `gradient_accumulation_steps` batches
        self.accumulation_steps = self.config["training"].get(
            "gradient_accumulation_steps", 1
        )
        self.global_step = 0

        # Initialize EMA (Exponential Moving Average)
        self._init_ema()

        # Validation (RFMSR / revision trainer.py style)
        self.val_enabled = self.config.get("validation", {}).get("enabled", False)
        self.val_metrics = {}
        if self.val_enabled:
            self._init_val_metrics()

        eff_batch = self.config["training"]["batch_size"] * self.accumulation_steps
        print("Trainer initialized!")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {self.device}")
        print(
            f"Iterations: {self.total_iters} (log/save every {self.save_freq}, "
            f"accum x{self.accumulation_steps}, effective batch {eff_batch})"
        )

    def _init_ema(self):
        """Initialize EMA (Exponential Moving Average) for noise predictor"""
        ema_rate = self.config["training"].get("ema_rate", 0.999)

        if ema_rate > 0:
            self.ema_rate = ema_rate
            # Initialize EMA state from current model weights
            self.ema_state = OrderedDict(
                {
                    key: deepcopy(value.data)
                    for key, value in self.noise_predictor.state_dict().items()
                }
            )
            # Keys to ignore during EMA update (batch norm running stats, etc.)
            self.ema_ignore_keys = [
                key
                for key in self.ema_state.keys()
                if "running_" in key or "num_batches_tracked" in key
            ]
            print(f"✓ EMA initialized with rate: {self.ema_rate}")
        else:
            self.ema_rate = None
            self.ema_state = None
            self.ema_ignore_keys = None

    @torch.no_grad()
    def update_ema(self):
        """Update EMA weights after each training step"""
        if self.ema_rate is None:
            return

        source_state = self.noise_predictor.state_dict()
        for key, value in self.ema_state.items():
            if key in self.ema_ignore_keys:
                # Copy running stats directly (no EMA for these)
                self.ema_state[key] = source_state[key]
            elif not self.ema_state[key].is_floating_point():
                # Skip EMA for non-floating point tensors (e.g., int64 counters)
                self.ema_state[key] = source_state[key]
            else:
                # EMA update: ema = rate * ema + (1 - rate) * current
                self.ema_state[key].mul_(self.ema_rate).add_(
                    source_state[key].detach().data, alpha=1 - self.ema_rate
                )

    def _init_models(self):
        """Initialize models"""
        print("\n" + "=" * 70)
        print("Initializing Models")
        print("=" * 70)

        # 1. Load VQVAE (frozen)
        print("\nLoading VQVAE...")
        vae_path = self.config["resshift"]["vae_path"]

        ddconfig = {
            "double_z": False,
            "z_channels": 3,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
            "padding_mode": "zeros",
        }

        # No VQGAN LoRA fine-tuning: default (pure VQGAN) structure
        self.vae = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
        ).to(self.device)

        # Load pretrained weights
        vae_ckpt = torch.load(vae_path, map_location=self.device)
        if "state_dict" in vae_ckpt:
            state_dict = vae_ckpt["state_dict"]
        else:
            state_dict = vae_ckpt

        # Remove prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("module._orig_mod."):
                new_key = key.replace("module._orig_mod.", "")
            elif key.startswith("module."):
                new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        self.vae.load_state_dict(new_state_dict, strict=False)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print(f"✓ VQVAE loaded: {vae_path}")

        # 2. Load ResShift UNet (frozen)
        print("\nLoading ResShift UNet...")
        unet_path = self.config["resshift"]["unet_path"]

        crop_size = self.config["data"]["train"]["crop_size"]
        vae_downsample_factor = 4
        latent_size = crop_size // vae_downsample_factor

        # Architecture from config (single source, aligned with inference.yaml)
        model_config = dict(self.config["resshift_unet"])
        # Derive-and-check: image_size/lq_size must match crop_size // 4
        assert model_config["image_size"] == latent_size, (
            f"resshift_unet.image_size ({model_config['image_size']}) must equal "
            f"crop_size // {vae_downsample_factor} ({latent_size})"
        )
        assert model_config["lq_size"] == latent_size, (
            f"resshift_unet.lq_size ({model_config['lq_size']}) must equal "
            f"crop_size // {vae_downsample_factor} ({latent_size})"
        )

        self.resshift_unet = UNetModelSwin(**model_config).to(self.device)

        # Load pretrained weights
        unet_ckpt = torch.load(unet_path, map_location=self.device)
        if "state_dict" in unet_ckpt:
            state_dict = unet_ckpt["state_dict"]
        else:
            state_dict = unet_ckpt

        # Remove prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("module._orig_mod."):
                new_key = key.replace("module._orig_mod.", "")
            elif key.startswith("module."):
                new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        self.resshift_unet.load_state_dict(new_state_dict, strict=True)
        self.resshift_unet.eval()
        for param in self.resshift_unet.parameters():
            param.requires_grad = False
        print(f"✓ ResShift UNet loaded: {unet_path}")

        # 3. Initialize ResShift diffusion process parameters
        print("\nInitializing ResShift diffusion process...")
        diffusion_config = self.config["training"]["diffusion"]
        self.num_timesteps = diffusion_config["num_timesteps"]

        # ResShift-specific parameters
        self.kappa = diffusion_config["kappa"]
        self.normalize_input = diffusion_config.get("normalize_input", True)
        self.latent_flag = diffusion_config.get("latent_flag", True)
        eta_schedule = diffusion_config["eta_schedule"]
        min_noise_level = diffusion_config["min_noise_level"]
        etas_end = diffusion_config["etas_end"]
        eta_power = diffusion_config.get("eta_power", 0.3)

        # Calculate eta schedule (ResShift method)
        sqrt_etas = get_named_eta_schedule(
            schedule_name=eta_schedule,
            num_diffusion_timesteps=self.num_timesteps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=self.kappa,
            kwargs={"power": eta_power},
        )

        # Convert to torch tensor
        self.sqrt_etas = torch.from_numpy(sqrt_etas).float()
        self.etas = self.sqrt_etas**2

        # Calculate etas_prev and alpha (ResShift method)
        # alpha_t = eta_t - eta_{t-1}, this is the correct definition in ResShift!
        self.etas_prev = torch.cat([torch.tensor([0.0]), self.etas[:-1]])
        self.alpha = self.etas - self.etas_prev  # Increment

        # Calculate posterior distribution parameters (ResShift method)
        # q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t, σ̃_t²·I)
        # μ̃_t = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0
        # σ̃_t² = κ²·(η_{t-1}/η_t)·α_t
        self.posterior_mean_coef1 = self.etas_prev / self.etas  # η_{t-1}/η_t
        self.posterior_mean_coef2 = self.alpha / self.etas  # α_t/η_t
        self.posterior_variance = (
            self.kappa**2 * self.etas_prev / self.etas * self.alpha
        )

        # Handle boundary case at t=0 (avoid NaN)
        self.posterior_mean_coef1[0] = 0.0  # At t=0, eta_prev=0, so coef1=0
        self.posterior_mean_coef2[0] = 1.0  # At t=0, posterior mean is directly x_0
        self.posterior_variance[0] = self.posterior_variance[
            1
        ]  # Avoid division by zero
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        print("✓ ResShift diffusion process initialized")

        # 5. Create noise predictor (training)
        print("\nCreating noise predictor...")
        noise_config = self.config["noise_predictor"]
        # Load config if config file path is specified
        if "config_path" in noise_config:
            with open(noise_config["config_path"], "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.noise_predictor = create_noise_predictor(
                image_size=config.get("image_size", latent_size),
                latent_channels=config["latent_channels"],
                model_channels=config["model_channels"],
                out_channels=config.get("out_channels", config["latent_channels"]),
                channel_mult=tuple(config["channel_mult"]),
                num_res_blocks=config["num_res_blocks"],
                attention_resolutions=config.get(
                    "attention_resolutions", [64, 32, 16, 8]
                ),
                dropout=config.get("dropout", 0.0),
                conv_resample=config.get("conv_resample", True),
                dims=config.get("dims", 2),
                use_fp16=config.get("use_fp16", False),
                num_heads=config.get("num_heads", -1),
                num_head_channels=config.get("num_head_channels", 32),
                use_scale_shift_norm=config.get("use_scale_shift_norm", True),
                resblock_updown=config.get("resblock_updown", False),
                swin_depth=config.get("swin_depth", 2),
                swin_embed_dim=config.get("swin_embed_dim", 192),
                window_size=config.get("window_size", 8),
                mlp_ratio=config.get("mlp_ratio", 4.0),
                patch_norm=config.get("patch_norm", False),
                cond_lq=config.get("cond_lq", True),
                lq_size=config.get("lq_size", latent_size),
                use_gradient_checkpointing=self.config["training"].get(
                    "use_gradient_checkpointing", False
                ),
                double_z=config.get("double_z", True),
            ).to(self.device)
        else:
            self.noise_predictor = create_noise_predictor(
                image_size=noise_config.get("image_size", latent_size),
                latent_channels=noise_config["latent_channels"],
                model_channels=noise_config["model_channels"],
                out_channels=noise_config.get(
                    "out_channels", noise_config["latent_channels"]
                ),
                channel_mult=tuple(noise_config["channel_mult"]),
                num_res_blocks=noise_config["num_res_blocks"],
                attention_resolutions=noise_config.get(
                    "attention_resolutions", [64, 32, 16, 8]
                ),
                dropout=noise_config.get("dropout", 0.0),
                conv_resample=noise_config.get("conv_resample", True),
                dims=noise_config.get("dims", 2),
                use_fp16=noise_config.get("use_fp16", False),
                num_heads=noise_config.get("num_heads", -1),
                num_head_channels=noise_config.get("num_head_channels", 32),
                use_scale_shift_norm=noise_config.get("use_scale_shift_norm", True),
                resblock_updown=noise_config.get("resblock_updown", False),
                swin_depth=noise_config.get("swin_depth", 2),
                swin_embed_dim=noise_config.get("swin_embed_dim", 192),
                window_size=noise_config.get("window_size", 8),
                mlp_ratio=noise_config.get("mlp_ratio", 4.0),
                patch_norm=noise_config.get("patch_norm", False),
                cond_lq=noise_config.get("cond_lq", True),
                lq_size=noise_config.get("lq_size", latent_size),
                use_gradient_checkpointing=self.config["training"].get(
                    "use_gradient_checkpointing", False
                ),
                double_z=noise_config.get("double_z", True),
            ).to(self.device)

        num_params = sum(p.numel() for p in self.noise_predictor.parameters())
        print("✓ Noise predictor created")
        print(f"  - Parameters: {num_params / 1e6:.2f}M")
        print(
            f"  - Gradient checkpointing: {self.config['training']['use_gradient_checkpointing']}"
        )

        # Count trainable parameters
        total_params = sum(
            p.numel() for p in self.noise_predictor.parameters() if p.requires_grad
        )
        print(f"\nTotal trainable parameters: {total_params / 1e6:.2f}M")

    def _init_losses(self):
        """Initialize loss functions"""
        print("\n" + "=" * 70)
        print("Initializing Loss Functions")
        print("=" * 70)

        loss_config = self.config["loss"]

        # L1 loss (image space, no separate initialization needed, use F.l1_loss)
        print(f"✓ L1 loss (weight: {loss_config.get('l1_weight', 1.0)})")

        # LPIPS perceptual loss
        if loss_config.get("lpips_weight", 0) > 0:
            self.lpips_loss = LPIPSLoss(
                loss_weight=1.0, net_type=loss_config.get("lpips_net_type", "alex")
            )
            print(f"✓ LPIPS perceptual loss (weight: {loss_config['lpips_weight']})")
        else:
            self.lpips_loss = None

        # GAN loss
        if loss_config.get("gan_weight", 0) > 0:
            # Create discriminator
            self.discriminator = create_discriminator(
                disc_type=loss_config.get("disc_type", "patch"),
                input_nc=self.noise_predictor.in_channels,  # Latent space channels
                ndf=loss_config.get("disc_ndf", 64),
                n_layers=loss_config.get("disc_n_layers", 3),
                norm_type=loss_config.get("disc_norm_type", "spectral"),
            ).to(self.device)

            # Create GAN loss
            self.gan_loss = GANLoss(
                gan_type=loss_config.get("gan_type", "lsgan"), loss_weight=1.0
            )

            # Count discriminator parameters
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            print(f"✓ GAN loss (weight: {loss_config['gan_weight']})")
            print(f"  - Discriminator type: {loss_config.get('disc_type', 'patch')}")
            print(f"  - GAN type: {loss_config.get('gan_type', 'lsgan')}")
            print(f"  - Discriminator parameters: {disc_params / 1e6:.2f}M")
        else:
            self.discriminator = None
            self.gan_loss = None

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        print("\n" + "=" * 70)
        print("Initializing Optimizer")
        print("=" * 70)

        opt_config = self.config["optimizer"]

        # Optimizer
        if opt_config["type"] == "Adam":
            self.optimizer = torch.optim.Adam(
                self.noise_predictor.parameters(),
                lr=opt_config["lr"],
                betas=(opt_config["beta1"], opt_config["beta2"]),
                weight_decay=opt_config["weight_decay"],
            )
        elif opt_config["type"] == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.noise_predictor.parameters(),
                lr=opt_config["lr"],
                betas=(opt_config["beta1"], opt_config["beta2"]),
                weight_decay=opt_config["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_config['type']}")

        print(f"✓ Optimizer: {opt_config['type']}")
        print(f"  - Learning rate: {opt_config['lr']}")
        print(f"  - Weight decay: {opt_config['weight_decay']}")

        # Learning rate scheduler
        scheduler_config = self.config["scheduler"]
        if scheduler_config["type"] == "CosineAnnealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["iterations"],
                eta_min=scheduler_config["min_lr"],
            )
        elif scheduler_config["type"] == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
            )
        else:
            self.scheduler = None

        if self.scheduler:
            print(f"✓ Learning rate scheduler: {scheduler_config['type']}")

        # Discriminator optimizer (if GAN loss is enabled)
        loss_config = self.config["loss"]
        if loss_config.get("gan_weight", 0) > 0 and self.discriminator is not None:
            disc_lr = loss_config.get("disc_lr", 1.0e-4)
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=disc_lr,
                betas=(opt_config["beta1"], opt_config["beta2"]),
                weight_decay=opt_config["weight_decay"],
            )
            print("✓ Discriminator optimizer: AdamW")
            print(f"  - Learning rate: {disc_lr}")
        else:
            self.optimizer_d = None

    def _init_dataloaders(self):
        """Initialize dataloaders"""
        print("\n" + "=" * 70)
        print("Initializing Dataloaders")
        print("=" * 70 + "\n")

        data_config = self.config["data"]
        train_config = self.config["training"]

        # Training dataloader
        if self.config["degradation"]["use_degradation"]:
            # Use degradation pipeline to generate LR images
            print("Using degradation pipeline to generate LR images...")
            self.train_loader = create_train_dataloader(
                data_dir=data_config["train"]["hr_dir"],
                config_path=self.config["degradation"]["config_path"],
                batch_size=train_config["batch_size"],
                num_workers=train_config["num_workers"],
                gt_size=data_config["train"]["crop_size"],
                use_hflip=data_config["train"]["use_flip"],
                use_rot=data_config["train"]["use_rot"],
                shuffle=True,
                pin_memory=True,
            )
            print(f"✓ Training dataloader created: {len(self.train_loader)} batches")
        else:
            raise NotImplementedError(
                "Direct loading of LR-HR image pairs not yet supported"
            )

    def _extract(self, a, t, x_shape):
        """Extract values from a corresponding to t, and reshape to x_shape"""
        batch_size = t.shape[0]
        out = a.to(t.device)[t]
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _scale_input(self, inputs, t):
        """
        Normalize input (key step in ResShift!)

        This is the input normalization method consistent with the original ResShift project

        Args:
            inputs: Input tensor
            t: Timestep index

        Returns:
            Normalized input
        """
        if self.normalize_input:
            if self.latent_flag:
                # Latent space variance is approximately 1.0
                std = torch.sqrt(
                    self._extract(self.etas, t, inputs.shape) * self.kappa**2 + 1
                )
                inputs_norm = inputs / std
            else:
                inputs_max = (
                    self._extract(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                )
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Calculate ResShift posterior distribution q(x_{t-1} | x_t, x_0)

        Args:
            x_0: Predicted clean image (ResShift UNet output)
            x_t: Noisy image at current timestep
            t: Timestep

        Returns:
            mean: Posterior mean
            variance: Posterior variance
            log_variance: Posterior log variance
        """
        # ResShift: μ = coef1·x_t + coef2·x_0
        # DDPM:     μ = coef1·x_0 + coef2·x_t
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_0
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance

    def multi_step_training_loss(self, z_start, z_y, hr_image, lr_image):
        """
        Multi-step training loss calculation
        Args:
            z_start: HR image latent representation z_0 [B, C, H, W]
            z_y: LR image latent representation y [B, C, H, W]
            hr_image: Original HR image [B, 3, H, W], for image space loss calculation
            lr_image: Image-space LR image (used as UNet's lq condition)

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        batch_size = z_y.shape[0]

        # 1. Initialize x_T (using noise predictor)
        # ResShift formula: x_T = z_y + κ·√η_T·ε
        t_init = self.num_timesteps - 1
        t_init_tensor = torch.full(
            (batch_size,), t_init, device=self.device, dtype=torch.long
        )

        # Use random Gaussian noise for initialization (not using noise predictor)
        sqrt_eta_T = self._extract(self.sqrt_etas, t_init_tensor, z_y.shape)
        predicted_noise_init = torch.randn_like(z_y)
        x_t = z_y + self.kappa * sqrt_eta_T * predicted_noise_init

        # 2. Multi-step reverse sampling (consistent with inference process)
        # From num_timesteps-1 to 0
        indices = list(range(self.num_timesteps))[::-1]  # [num_timesteps-1, ..., 0]

        for i in indices:
            t_tensor = torch.full(
                (batch_size,), i, device=self.device, dtype=torch.long
            )

            # 2.1 Normalize input
            x_t_normalized = self._scale_input(x_t, t_tensor)

            # 2.2 Use UNet to predict x_0 (keep gradients to let them flow back to noise predictor)
            pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)

            # 2.3 If not the last step, perform posterior sampling
            if i > 0:
                # Calculate posterior distribution q(x_{t-1} | x_t, x_0)
                mean, variance, log_variance = self.q_posterior_mean_variance(
                    pred_x0, x_t, t_tensor
                )

                # Use noise predictor to predict noise (needs gradient).
                predicted_noise = self.noise_predictor(
                    x_t_normalized, pred_x0, lr_image, t_tensor, sample_posterior=True
                )

                # Sample x_{t-1}
                # nonzero_mask is always 1 here since i > 0
                x_t = mean + torch.exp(0.5 * log_variance) * predicted_noise

        # Final pred_x0 is our prediction result
        final_pred_x0 = pred_x0

        # 4. Calculate losses
        loss_config = self.config["loss"]
        total_loss = 0.0

        # L1 loss (latent space): regression toward the quantized HR latent
        # z_start (same target space as ResShift's latent reconstruction loss)
        l1_weight = loss_config.get("l1_weight", 1.0)
        l1_val = torch.nn.functional.l1_loss(final_pred_x0, z_start)
        loss_dict["l1"] = l1_val.item()
        total_loss += l1_weight * l1_val

        # Decode to image space (for the perceptual loss only)
        # Note: pred_image needs to keep gradients so loss can backprop to noise predictor
        # VAE is frozen, but gradients can still flow through it back to final_pred_x0
        pred_image = self.vae.decode(final_pred_x0)  # [-1, 1], keep gradients
        pred_image = pred_image * 0.5 + 0.5  # [0, 1]

        gt_image = hr_image * 0.5 + 0.5  # [0, 1]

        # LPIPS perceptual loss (image space)
        if self.lpips_loss is not None and loss_config.get("lpips_weight", 0) > 0:
            lpips_val = self.lpips_loss(pred_image, gt_image)
            loss_dict["lpips"] = lpips_val.item()
            total_loss += loss_config["lpips_weight"] * lpips_val

        # GAN generator loss (latent space)
        if self.gan_loss is not None and loss_config.get("gan_weight", 0) > 0:
            # Check if reached discriminator start iteration
            disc_start_iter = loss_config.get("disc_start_iter", 0)
            if self.global_step >= disc_start_iter:
                # Calculate generator loss in latent space: make discriminator believe generated latent is real
                fake_pred = self.discriminator(final_pred_x0)
                g_loss = self.gan_loss(fake_pred, target_is_real=True, is_disc=False)
                loss_dict["g_loss"] = g_loss.item()
                total_loss += loss_config["gan_weight"] * g_loss

        # Save latent for discriminator training (latent space GAN)
        self._pred_latent_for_disc = final_pred_x0.detach()
        self._gt_latent_for_disc = z_start.detach()

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    def train(self):
        """Iteration-based training loop with GAN alternating updates and
        gradient accumulation (RFMSR train_rfmsr_os.py style).

        - One iteration = one optimizer step (G or D), not one batch; the
          progress bar / scheduler / logging / saving all follow this counter.
        - G/D alternate at batch level (`disc_update_freq`, gated by
          `disc_start_iter`), unaffected by accumulation; each side
          accumulates gradients over `gradient_accumulation_steps` batches
          before its own optimizer step.
        - `batch_idx` is a batch-level counter only (restarts from 0 on
          resume, same as RFMSR).
        """
        self.noise_predictor.train()

        loss_config = self.config["loss"]
        disc_start_iter = loss_config.get("disc_start_iter", 0)
        disc_update_freq = loss_config.get("disc_update_freq", 1)
        gan_enabled = (
            self.discriminator is not None
            and loss_config.get("gan_weight", 0) > 0
        )
        use_amp = self.config["training"]["use_amp"]
        gradient_clip = self.config["training"]["gradient_clip"]
        accum_steps = self.accumulation_steps

        data_iter = iter(self.train_loader)
        pbar = tqdm(
            total=self.total_iters,
            initial=self.global_step,
            desc="Train",
            unit="step",
            bar_format="{desc} [{n:>6d}/{total_fmt}] {percentage:3.0f}% |{bar}| {postfix} [{rate_fmt}]",
        )

        # Window-averaged losses (reset at each save point)
        window_loss_sum = {}
        window_loss_cnt = 0

        # Independent G/D gradient accumulation counters (RFMSR style)
        g_count = 0
        d_count = 0
        # G/D alternating batch-level counter (unaffected by accumulation;
        # restarts from 0 on resume)
        batch_idx = 0

        self.optimizer.zero_grad()
        if self.optimizer_d is not None:
            self.optimizer_d.zero_grad()

        last_save_step = self.global_step

        try:
            while self.global_step < self.total_iters:
                # Cycle the dataloader (epochs no longer exist)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                hr_images = batch["gt"].to(self.device)  # [B, 3, H, W], [0, 1]
                lr_images = batch["lq"].to(self.device)  # [B, 3, H, W], [0, 1]

                # G/D alternating: batch level (unaffected by accumulation)
                train_disc = (
                    gan_enabled
                    and self.global_step >= disc_start_iter
                    and batch_idx % disc_update_freq == 0
                )
                batch_idx += 1

                # Forward + backward only (no optimizer step inside)
                loss_dict = self.train_step(
                    hr_images, lr_images, train_disc=train_disc
                )

                # Accumulate window losses
                for key, value in loss_dict.items():
                    window_loss_sum[key] = window_loss_sum.get(key, 0.0) + value
                window_loss_cnt += 1

                # ---- G/D independent gradient accumulation (RFMSR style) ----
                stepped = False

                if train_disc:
                    d_count += 1
                    if d_count >= accum_steps:
                        # Discriminator optimizer step
                        if use_amp:
                            self.scaler_d.unscale_(self.optimizer_d)
                            if gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.discriminator.parameters(), gradient_clip
                                )
                            self.scaler_d.step(self.optimizer_d)
                            self.scaler_d.update()
                        else:
                            if gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.discriminator.parameters(), gradient_clip
                                )
                            self.optimizer_d.step()
                        self.optimizer_d.zero_grad()
                        d_count = 0
                        self.global_step += 1
                        stepped = True
                else:
                    g_count += 1
                    if g_count >= accum_steps:
                        # Generator (noise predictor) optimizer step
                        if use_amp:
                            self.scaler_g.unscale_(self.optimizer)
                            if gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.noise_predictor.parameters(), gradient_clip
                                )
                            self.scaler_g.step(self.optimizer)
                            self.scaler_g.update()
                        else:
                            if gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.noise_predictor.parameters(), gradient_clip
                                )
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Update EMA after each generator optimizer step
                        self.update_ema()

                        g_count = 0
                        self.global_step += 1
                        stepped = True

                if stepped:
                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "loss": f"{loss_dict['total']:.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )
                    pbar.update(1)

                    # Per-iteration lr scheduling
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Print (window average) and save checkpoint together
                    # (`//` comparison avoids duplicate triggers when G and D
                    #  step on consecutive batches)
                    if (
                        self.global_step // self.save_freq
                        > last_save_step // self.save_freq
                    ):
                        last_save_step = self.global_step
                        print(
                            f"\n[step {self.global_step}/{self.total_iters}] "
                            f"(avg over last {window_loss_cnt} batches)"
                        )
                        for key, value in window_loss_sum.items():
                            print(f"  {key}: {value / max(window_loss_cnt, 1):.4f}")
                        print(f"  lr: {self.optimizer.param_groups[0]['lr']:.2e}")
                        window_loss_sum = {}
                        window_loss_cnt = 0

                        self.save_checkpoint(self.global_step)

                        if self.val_enabled:
                            try:
                                self.validate(self.global_step)
                            except Exception as e:
                                print(
                                    f"\n[WARN] Validation failed at step "
                                    f"{self.global_step}: {e}"
                                )

            # Final checkpoint
            self.save_checkpoint(self.global_step)
            pbar.close()
            print("Training finished!")

        except KeyboardInterrupt:
            pbar.close()
            print("\n\nTraining interrupted!")
            print("Saving current checkpoint...")
            self.save_checkpoint(self.global_step)
            print("Checkpoint saved, can resume training with --resume")

    # ------------------------------------------------------------------
    # validation (RFMSR / revision trainer.py style)
    # ------------------------------------------------------------------
    def _init_val_metrics(self):
        """Initialize validation metrics (lpips + pyiqa, fault-tolerant)."""
        try:
            import lpips
            self.val_metrics["lpips"] = lpips.LPIPS(net="alex").to(self.device)
        except ImportError:
            print("[WARN] lpips not installed, LPIPS will be skipped")
        try:
            import pyiqa

            self.val_metrics["psnr"] = pyiqa.create_metric(
                "psnr", test_y_channel=True, color_space="ycbcr", device=self.device
            )
            self.val_metrics["ssim"] = pyiqa.create_metric(
                "ssim", test_y_channel=True, color_space="ycbcr", device=self.device
            )
            self.val_metrics["dists"] = pyiqa.create_metric("dists", device=self.device)
            self.val_metrics["niqe"] = pyiqa.create_metric("niqe", device=self.device)
            self.val_metrics["musiq"] = pyiqa.create_metric("musiq", device=self.device)
            self.val_metrics["maniqa"] = pyiqa.create_metric(
                "maniqa", device=self.device
            )
            self.val_metrics["clipiqa"] = pyiqa.create_metric(
                "clipiqa", device=self.device
            )
            print(
                "Validation metrics initialized: "
                "PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA"
            )
        except ImportError:
            print("[WARN] pyiqa not installed, all FR/NR metrics will be skipped")

    @torch.no_grad()
    def reverse_sampling(self, z_y, lr_image, generator=None):
        """Full reverse sampling chain with noise-predictor correction
        (inference twin of the chain in multi_step_training_loss).

        Args:
            z_y: LR latent [B, C, H, W] (chain start)
            lr_image: image-space LR [-1, 1] (lq condition for UNet/predictor,
                size == latent size under VQVAE f=4)
            generator: optional torch.Generator for reproducible initial noise

        Returns:
            pred_x0 at t=0 [B, C, H, W]
        """
        bsz = z_y.shape[0]

        # x_T = z_y + kappa * sqrt(eta_T) * eps (same init as training)
        t_init = self.num_timesteps - 1
        t_init_tensor = torch.full(
            (bsz,), t_init, device=self.device, dtype=torch.long
        )
        sqrt_eta_T = self._extract(self.sqrt_etas, t_init_tensor, z_y.shape)
        if generator is not None:
            eps = torch.randn(
                z_y.shape,
                generator=generator,
                device=self.device,
                dtype=z_y.dtype,
            )
        else:
            eps = torch.randn_like(z_y)
        x_t = z_y + self.kappa * sqrt_eta_T * eps

        pred_x0 = None
        for i in range(self.num_timesteps - 1, -1, -1):
            t_tensor = torch.full(
                (bsz,), i, device=self.device, dtype=torch.long
            )

            # Normalize input (both the UNet and the noise predictor take the
            # scaled x_t, matching training)
            x_t_normalized = self._scale_input(x_t, t_tensor)

            # UNet predicts x_0
            pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)

            if i > 0:
                mean, _, log_variance = self.q_posterior_mean_variance(
                    pred_x0, x_t, t_tensor
                )
                predicted_noise = self.noise_predictor(
                    x_t_normalized, pred_x0, lr_image, t_tensor, sample_posterior=True
                )
                x_t = mean + torch.exp(0.5 * log_variance) * predicted_noise

        return pred_x0

    @torch.no_grad()
    def validate(self, step):
        """Validate: full sampling chain (with noise-predictor correction) on
        validation LQ/GT pairs -> metrics -> save SR images."""
        self.noise_predictor.eval()

        # Swap in EMA weights for validation
        orig_state = None
        if self.ema_state is not None:
            orig_state = OrderedDict(
                {
                    k: v.data.clone()
                    for k, v in self.noise_predictor.state_dict().items()
                }
            )
            self.noise_predictor.load_state_dict(self.ema_state)

        val_cfg = self.config.get("validation", {})
        lq_dir = Path(val_cfg.get("lq_dir", "assets/validate_lq"))
        gt_dir = Path(val_cfg.get("gt_dir", "assets/validate_gt"))
        val_scale = val_cfg.get("scale", 4.0)
        val_seed = val_cfg.get("seed", 42)
        scale = self.config["data"]["train"]["scale"]

        out_dir = self.exp_dir / "validation" / f"step_{step:08d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        lq_paths = sorted(lq_dir.glob("*.png")) + sorted(lq_dir.glob("*.jpg"))
        pairs = []
        for lp in lq_paths:
            gp = gt_dir / lp.name
            if gp.exists():
                pairs.append((lp, gp))
        total = len(pairs)
        if total == 0:
            print(f"[Val @ step {step}] No paired images found in {lq_dir}")
            if orig_state is not None:
                self.noise_predictor.load_state_dict(orig_state)
            self.noise_predictor.train()
            return

        psnr_v, ssim_v, lpips_v, dists_v = [], [], [], []
        niqe_v, musiq_v, maniqa_v, clipiqa_v = [], [], [], []

        for lq_path, gt_path in tqdm(pairs, desc=f"Val@{step}", leave=False):
            from PIL import Image

            src = Image.open(lq_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")

            # Original LR [-1, 1] (lq condition; size == latent size, as in training)
            im_np = np.array(src).astype(np.float32) / 255.0
            lr_tensor = (
                torch.from_numpy(np.moveaxis(im_np, 2, 0))
                .unsqueeze(0)
                .to(self.device)
            )
            lr_cond = lr_tensor * 2.0 - 1.0

            # Upsample LR -> HR size with the SAME interpolation as training,
            # then VQVAE encode -> z_y
            lr_up = torch.nn.functional.interpolate(
                lr_cond,
                scale_factor=scale,
                mode="bicubic",
                align_corners=False,
            )
            z_y = self.vae.encode(lr_up)
            ori_h, ori_w = lr_up.shape[-2], lr_up.shape[-1]

            # Reproducible initial noise across validation runs
            generator = torch.Generator(device=self.device).manual_seed(val_seed)
            sr_latent = self.reverse_sampling(z_y, lr_cond, generator=generator)

            # VQVAE decode -> [0, 1]
            sr_decoded = self.vae.decode(sr_latent)
            sr_decoded = torch.clamp(sr_decoded * 0.5 + 0.5, 0.0, 1.0)
            sr_decoded = sr_decoded[:, :, 0:ori_h, 0:ori_w]

            # Save SR
            sr_np = (sr_decoded[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            sr_np = np.moveaxis(sr_np, 0, 2)
            Image.fromarray(sr_np).save(out_dir / lq_path.name)

            # GT tensor [0, 1]
            gt_np = np.array(gt_img).astype(np.float32) / 255.0
            gt_tensor = (
                torch.from_numpy(np.moveaxis(gt_np, 2, 0))
                .unsqueeze(0)
                .to(self.device)
            )

            # Full-reference metrics
            if "psnr" in self.val_metrics:
                psnr_v.append(
                    self.val_metrics["psnr"](sr_decoded, gt_tensor).mean().item()
                )
            if "ssim" in self.val_metrics:
                ssim_v.append(
                    self.val_metrics["ssim"](sr_decoded, gt_tensor).mean().item()
                )
            if "dists" in self.val_metrics:
                dists_v.append(
                    self.val_metrics["dists"](sr_decoded, gt_tensor).mean().item()
                )
            if "lpips" in self.val_metrics:
                gt_norm = (gt_tensor - 0.5) / 0.5
                sr_norm = (sr_decoded - 0.5) / 0.5
                lpips_v.append(
                    self.val_metrics["lpips"](gt_norm, sr_norm).mean().item()
                )

            # No-reference metrics
            if "niqe" in self.val_metrics:
                niqe_v.append(self.val_metrics["niqe"](sr_decoded).mean().item())
            if "musiq" in self.val_metrics:
                musiq_v.append(self.val_metrics["musiq"](sr_decoded).mean().item())
            if "maniqa" in self.val_metrics:
                maniqa_v.append(self.val_metrics["maniqa"](sr_decoded).mean().item())
            if "clipiqa" in self.val_metrics:
                clipiqa_v.append(
                    self.val_metrics["clipiqa"](sr_decoded).mean().item()
                )

        print(f"\n[Val @ step {step}] images={total}")
        if psnr_v:
            print(f"  PSNR (Y):      {np.mean(psnr_v):>8.2f} dB")
        if ssim_v:
            print(f"  SSIM (Y):      {np.mean(ssim_v):>8.4f}")
        if lpips_v:
            print(f"  LPIPS (Alex):  {np.mean(lpips_v):>8.4f}")
        if dists_v:
            print(f"  DISTS:         {np.mean(dists_v):>8.4f}")
        if niqe_v:
            print(f"  NIQE:          {np.mean(niqe_v):>8.4f}")
        if musiq_v:
            print(f"  MUSIQ:         {np.mean(musiq_v):>8.4f}")
        if maniqa_v:
            print(f"  MANIQA:        {np.mean(maniqa_v):>8.4f}")
        if clipiqa_v:
            print(f"  CLIPIQA:       {np.mean(clipiqa_v):>8.4f}")
        print(f"  SR saved to: {out_dir}\n")

        if orig_state is not None:
            self.noise_predictor.load_state_dict(orig_state)
        self.noise_predictor.train()

        # Clean up old validation dirs (keep last N)
        if self.keep_last_n > 0:
            self._cleanup_old_validation()

    def _cleanup_old_validation(self):
        """Keep only the most recent `keep_last_n` validation image dirs."""
        import re
        import shutil

        val_root = self.exp_dir / "validation"
        if not val_root.exists():
            return
        pattern = re.compile(r"step_(\d+)")
        step_dirs = []
        for d in val_root.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    step_dirs.append((int(m.group(1)), d))
        step_dirs.sort(key=lambda x: x[0], reverse=True)
        for s, d in step_dirs[self.keep_last_n:]:
            shutil.rmtree(d)
            print(f"✓ Removed old validation dir: step {s}")

    def freeze_model(self, model):
        """Freeze model parameters"""
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_model(self, model):
        """Unfreeze model parameters"""
        for param in model.parameters():
            param.requires_grad = True

    def train_step(self, hr_images, lr_images, train_disc: bool):
        """
        One batch: forward the multi-step reverse chain, then backward the
        gradient scaled by 1/accumulation_steps (RFMSR train_rfmsr_os.py
        style). The optimizer step is performed by the training loop once the
        accumulation buffer is full.

        train_disc=True  → accumulate discriminator gradients
        train_disc=False → accumulate generator (noise predictor) gradients

        Args:
            hr_images: HR images [B, 3, H, W], range [0, 1]
            lr_images: LR images [B, 3, H, W], range [0, 1]
            train_disc: Whether this batch trains the discriminator

        Returns:
            loss_dict: Loss dictionary (raw loss values, without inv_accum)
        """
        self.noise_predictor.train()
        inv_accum = 1.0 / self.accumulation_steps

        # 1. Encode to latent space (frozen VAE)
        with torch.no_grad():
            # Convert to [-1, 1]
            hr_images_norm = hr_images * 2.0 - 1.0
            lr_images_norm = lr_images * 2.0 - 1.0

            # HR image direct encoding
            z_start = self.vae.encode(hr_images_norm)

            # LR image needs to be upsampled to HR size first, then encoded
            scale_factor = self.config["data"]["train"]["scale"]
            lr_images_upsampled = torch.nn.functional.interpolate(
                lr_images_norm,
                scale_factor=scale_factor,
                mode="bicubic",
                align_corners=False,
            )
            z_y = self.vae.encode(lr_images_upsampled)

        # 2. Generator forward (multi-step chain with gradients; always needed
        #    to provide fake latents for the discriminator)
        if self.config["training"]["use_amp"]:
            with autocast(device_type="cuda"):
                loss, loss_dict = self.multi_step_training_loss(
                    z_start, z_y, hr_images_norm, lr_images_norm
                )
        else:
            loss, loss_dict = self.multi_step_training_loss(
                z_start, z_y, hr_images_norm, lr_images_norm
            )

        # 3. Backward only — no optimizer step here (RFMSR style)
        if train_disc and self.discriminator is not None:
            # === Discriminator gradient accumulation ===
            self.discriminator.train()

            # Freeze generator, unfreeze discriminator
            self.freeze_model(self.noise_predictor)
            self.unfreeze_model(self.discriminator)

            # Generator-produced latents (saved by multi_step_training_loss)
            fake_latent = self._pred_latent_for_disc
            real_latent = self._gt_latent_for_disc

            if self.config["training"]["use_amp"]:
                with autocast(device_type="cuda"):
                    # Discriminate real latents
                    real_pred = self.discriminator(real_latent)
                    d_loss_real = self.gan_loss(
                        real_pred, target_is_real=True, is_disc=True
                    )

                    # Discriminate generated latents
                    fake_pred = self.discriminator(fake_latent)
                    d_loss_fake = self.gan_loss(
                        fake_pred, target_is_real=False, is_disc=True
                    )

                    d_loss = (d_loss_real + d_loss_fake) / 2

                self.scaler_d.scale(d_loss * inv_accum).backward()
            else:
                # Discriminate real latents
                real_pred = self.discriminator(real_latent)
                d_loss_real = self.gan_loss(
                    real_pred, target_is_real=True, is_disc=True
                )

                # Discriminate generated latents
                fake_pred = self.discriminator(fake_latent)
                d_loss_fake = self.gan_loss(
                    fake_pred, target_is_real=False, is_disc=True
                )

                d_loss = (d_loss_real + d_loss_fake) / 2

                (d_loss * inv_accum).backward()

            # Record losses (raw values, without inv_accum)
            loss_dict["d_loss"] = d_loss.item()
            loss_dict["d_loss_real"] = d_loss_real.item()
            loss_dict["d_loss_fake"] = d_loss_fake.item()

            # Restore generator parameter state
            self.unfreeze_model(self.noise_predictor)
        else:
            # === Generator (noise predictor) gradient accumulation ===

            # Freeze discriminator, unfreeze generator
            if self.discriminator is not None:
                self.freeze_model(self.discriminator)
            self.unfreeze_model(self.noise_predictor)

            # Accumulate generator gradients (scaled by 1/accumulation_steps)
            scaled_loss = loss * inv_accum
            if self.config["training"]["use_amp"]:
                self.scaler_g.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Restore discriminator parameter state
            if self.discriminator is not None:
                self.unfreeze_model(self.discriminator)

        return loss_dict

    def save_checkpoint(self, step):
        """Save EMA weights (safetensors) + full training state (pth) at `step`."""
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. EMA weights only (for inference)
        model_weights = (
            self.ema_state
            if self.ema_state is not None
            else self.noise_predictor.state_dict()
        )
        # safetensors only accepts plain, contiguous, non-shared tensors
        safeweight = {
            k: v.detach().cpu().clone().contiguous() for k, v in model_weights.items()
        }
        weights_path = ckpt_dir / f"noise_predictor_step{step}.safetensors"
        save_file(safeweight, str(weights_path))
        print(f"✓ EMA weights saved: {weights_path}")

        # 2. Full training state (for resuming training)
        training_ckpt = {
            "step": step,
            "global_step": self.global_step,
            "noise_predictor": self.noise_predictor.state_dict(),  # Current weights (not EMA)
            "ema_state": self.ema_state,  # EMA state
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_g": self.scaler_g.state_dict() if self.scaler_g else None,
            "scaler_d": self.scaler_d.state_dict() if self.scaler_d else None,
            "config": self.config,
        }

        # Save EMA rate for reference
        if self.ema_rate is not None:
            training_ckpt["ema_rate"] = self.ema_rate

        # Save discriminator state (if GAN loss is enabled)
        if self.discriminator is not None:
            training_ckpt["discriminator"] = self.discriminator.state_dict()
        if self.optimizer_d is not None:
            training_ckpt["optimizer_d"] = self.optimizer_d.state_dict()

        training_ckpt_path = ckpt_dir / f"training_state_step{step}.pth"
        torch.save(training_ckpt, training_ckpt_path)
        print(f"✓ Training state saved: {training_ckpt_path}")

        # 3. Clean up old checkpoints (keep last N steps)
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(ckpt_dir)

    def _cleanup_old_checkpoints(self, ckpt_dir):
        """Keep only the most recent `keep_last_n` checkpoint steps."""
        import re

        pattern = re.compile(r"(noise_predictor_step|training_state_step)(\d+)")
        ckpt_steps = {}
        for f in ckpt_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                s = int(m.group(2))
                ckpt_steps.setdefault(s, []).append(f)
        sorted_steps = sorted(ckpt_steps.keys(), reverse=True)
        for s in sorted_steps[self.keep_last_n:]:
            for f in ckpt_steps[s]:
                f.unlink()
            print(f"✓ Removed old checkpoint: step {s}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint (training state for resuming)"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.noise_predictor.load_state_dict(checkpoint["noise_predictor"])

        # Load EMA state if exists
        if "ema_state" in checkpoint and checkpoint["ema_state"] is not None:
            self.ema_state = checkpoint["ema_state"]
            print("  - EMA state loaded")
        elif self.ema_rate is not None:
            # Initialize EMA state from loaded weights
            self.ema_state = OrderedDict(
                {
                    key: deepcopy(value.data)
                    for key, value in checkpoint["noise_predictor"].items()
                }
            )
            print("  - EMA state initialized from checkpoint weights")

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # Restore AMP scaler states if they were saved
        if self.scaler_g is not None and checkpoint.get("scaler_g") is not None:
            self.scaler_g.load_state_dict(checkpoint["scaler_g"])
        if self.scaler_d is not None and checkpoint.get("scaler_d") is not None:
            self.scaler_d.load_state_dict(checkpoint["scaler_d"])

        self.global_step = checkpoint.get("step", checkpoint.get("global_step", 0))

        # Load discriminator state (if exists)
        if self.discriminator is not None and "discriminator" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            print("  - Discriminator weights loaded")
        if self.optimizer_d is not None and "optimizer_d" in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            print("  - Discriminator optimizer state loaded")

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  - Step: {self.global_step}")

        return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Train noise predictor (multi-step, iteration-based)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trainer.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Training state checkpoint path (training_state_step*.pth)",
    )
    args = parser.parse_args()

    # Create trainer
    trainer = NoisePredictorTrainer(args.config)

    # Resume training
    if args.resume:
        trainer.load_checkpoint(args.resume)

    print("\n" + "=" * 70)
    print(f"Starting training for {trainer.total_iters} iterations!")
    print("=" * 70 + "\n")

    # Iteration-based training loop (logging + checkpointing handled inside)
    trainer.train()


if __name__ == "__main__":
    main()

