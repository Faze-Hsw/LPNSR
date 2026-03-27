#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-step End-to-End Noise Predictor Training
"""

import argparse
import math
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
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
            if self.discriminator is not None:
                self.scaler_d = GradScaler()  # Discriminator-specific scaler
        else:
            self.scaler_g = None
            self.scaler_d = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Initialize EMA (Exponential Moving Average)
        self._init_ema()

        print("Trainer initialized!")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {self.device}")

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

        lora_rank = 8
        lora_alpha = 1.0
        lora_tune_decoder = False

        self.vae = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
            rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_tune_decoder=lora_tune_decoder,
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

        model_structure = {
            "image_size": latent_size,
            "in_channels": 3,
            "model_channels": 160,
            "out_channels": 3,
            "attention_resolutions": [64, 32, 16, 8],
            "channel_mult": [1, 2, 2, 4],
            "num_res_blocks": [2, 2, 2, 2],
            "num_head_channels": 32,
            "use_scale_shift_norm": True,
            "resblock_updown": False,
            "swin_depth": 2,
            "swin_embed_dim": 192,
            "window_size": 8,
            "mlp_ratio": 4,
            "cond_lq": True,
            "lq_size": latent_size,
        }

        model_config = {
            **model_structure,
            "dropout": 0.0,
            "use_fp16": False,
            "conv_resample": True,
            "dims": 2,
            "patch_norm": False,
        }

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
        self.sampling_steps = self.config["training"]["sampling_steps"]

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
                input_nc=3,
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
                T_max=self.config["training"]["num_epochs"],
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

                # Use noise predictor to predict noise (needs gradient)
                predicted_noise = self.noise_predictor(
                    x_t, pred_x0, lr_image, t_tensor, sample_posterior=True
                )

                # Sample x_{t-1}
                # nonzero_mask is always 1 here since i > 0
                x_t = mean + torch.exp(0.5 * log_variance) * predicted_noise

        # Final pred_x0 is our prediction result
        final_pred_x0 = pred_x0

        # 4. Calculate losses
        loss_config = self.config["loss"]
        total_loss = 0.0

        # Decode to image space
        # Note: pred_image needs to keep gradients so loss can backprop to noise predictor
        # VAE is frozen, but gradients can still flow through it back to final_pred_x0
        pred_image = self.vae.decode(final_pred_x0)  # [-1, 1], keep gradients
        pred_image = torch.clamp(pred_image, -1, 1)
        pred_image = pred_image * 0.5 + 0.5  # [0, 1]

        gt_image = hr_image * 0.5 + 0.5  # [0, 1]

        # L1 loss (image space)
        l1_weight = loss_config.get("l1_weight", 1.0)
        l1_val = torch.nn.functional.l1_loss(pred_image, gt_image)
        loss_dict["l1"] = l1_val.item()
        total_loss += l1_weight * l1_val

        # LPIPS perceptual loss (image space)
        if self.lpips_loss is not None and loss_config.get("lpips_weight", 0) > 0:
            lpips_val = self.lpips_loss(pred_image, gt_image)
            loss_dict["lpips"] = lpips_val.item()
            total_loss += loss_config["lpips_weight"] * lpips_val

        # GAN generator loss (image space)
        if self.gan_loss is not None and loss_config.get("gan_weight", 0) > 0:
            # Check if reached discriminator start epoch
            disc_start_epoch = loss_config.get("disc_start_epoch", 0)
            if self.current_epoch >= disc_start_epoch:
                # Calculate generator loss: make discriminator believe generated image is real
                fake_pred = self.discriminator(pred_image)
                g_loss = self.gan_loss(fake_pred, target_is_real=True, is_disc=False)
                loss_dict["g_loss"] = g_loss.item()
                total_loss += loss_config["gan_weight"] * g_loss

        # Save decoded image for discriminator training
        self._pred_image_for_disc = pred_image.detach()
        self._gt_image_for_disc = gt_image

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    def train_epoch(self, epoch):
        """
        Train one epoch

        Args:
            epoch: Current epoch number

        Returns:
            avg_loss_dict: Average loss dictionary
        """
        self.noise_predictor.train()

        # Record current epoch and step within epoch for debug output
        self.current_epoch = epoch
        self.epoch_step = 0

        total_loss_dict = {}

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}",
        )

        for step, batch in enumerate(pbar):
            self.epoch_step = step
            # Get data
            hr_images = batch["gt"].to(self.device)  # [B, 3, H, W], [0, 1]
            lr_images = batch["lq"].to(self.device)  # [B, 3, H, W], [0, 1]

            # Train one step
            loss_dict = self.train_step(hr_images, lr_images)

            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0.0
                total_loss_dict[key] += value

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            # Print log
            if (step + 1) % self.config["experiment"]["log_interval"] == 0:
                print(
                    f"\nEpoch [{epoch}/{self.config['training']['num_epochs']}] "
                    f"Step [{step + 1}/{len(self.train_loader)}]"
                )
                for key, value in loss_dict.items():
                    print(f"  {key}: {value:.4f}")
                print(f"  lr: {self.optimizer.param_groups[0]['lr']:.2e}")

            self.global_step += 1

        # Calculate average losses
        num_batches = len(self.train_loader)
        avg_loss_dict = {
            key: value / num_batches for key, value in total_loss_dict.items()
        }

        return avg_loss_dict

    def freeze_model(self, model):
        """Freeze model parameters"""
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_model(self, model):
        """Unfreeze model parameters"""
        for param in model.parameters():
            param.requires_grad = True

    def train_step(self, hr_images, lr_images):
        """
        Single training step (similar to ResShift and InvSR), supports gradient accumulation

        Args:
            hr_images: HR images [B, 3, H, W], range [0, 1]
            lr_images: LR images [B, 3, H, W], range [0, 1]

        Returns:
            loss_dict: Loss dictionary
        """
        self.noise_predictor.train()
        batch_size = hr_images.shape[0]

        loss_config = self.config["loss"]

        # Get discriminator update frequency (for true alternating training)
        disc_update_freq = loss_config.get("disc_update_freq", 1)
        should_update_disc = self.global_step % disc_update_freq == 0

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

        # 2. First calculate generator loss (always needed, to provide fake images for discriminator)
        if self.config["training"]["use_amp"]:
            with autocast(device_type="cuda"):
                loss, loss_dict = self.multi_step_training_loss(
                    z_start, z_y, hr_images_norm, lr_images_norm
                )
        else:
            loss, loss_dict = self.multi_step_training_loss(
                z_start, z_y, hr_images_norm, lr_images_norm
            )

        # 3. True alternating training logic
        if (
            should_update_disc
            and self.discriminator is not None
            and loss_config.get("gan_weight", 0) > 0
        ):
            # === This step only updates discriminator ===
            disc_start_epoch = loss_config.get("disc_start_epoch", 0)
            if self.current_epoch >= disc_start_epoch:
                # Get generator produced images (already saved in multi_step_training_loss)
                if hasattr(self, "_pred_image_for_disc") and hasattr(
                    self, "_gt_image_for_disc"
                ):
                    fake_image = self._pred_image_for_disc
                    real_image = self._gt_image_for_disc

                    self.discriminator.train()

                    # Freeze generator, unfreeze discriminator
                    self.freeze_model(self.noise_predictor)
                    self.unfreeze_model(self.discriminator)

                    # Calculate discriminator loss
                    if self.config["training"]["use_amp"]:
                        with autocast(device_type="cuda"):
                            # Discriminate real images
                            real_pred = self.discriminator(real_image)
                            d_loss_real = self.gan_loss(
                                real_pred, target_is_real=True, is_disc=True
                            )

                            # Discriminate generated images
                            fake_pred = self.discriminator(fake_image)
                            d_loss_fake = self.gan_loss(
                                fake_pred, target_is_real=False, is_disc=True
                            )

                            d_loss = (d_loss_real + d_loss_fake) / 2

                        self.scaler_d.scale(d_loss).backward()

                        # Gradient clipping
                        if self.config["training"]["gradient_clip"] > 0:
                            self.scaler_d.unscale_(self.optimizer_d)
                            torch.nn.utils.clip_grad_norm_(
                                self.discriminator.parameters(),
                                self.config["training"]["gradient_clip"],
                            )
                        self.scaler_d.step(self.optimizer_d)
                        self.scaler_d.update()
                        self.optimizer_d.zero_grad()
                    else:
                        # Discriminate real images
                        real_pred = self.discriminator(real_image)
                        d_loss_real = self.gan_loss(
                            real_pred, target_is_real=True, is_disc=True
                        )

                        # Discriminate generated images
                        fake_pred = self.discriminator(fake_image)
                        d_loss_fake = self.gan_loss(
                            fake_pred, target_is_real=False, is_disc=True
                        )

                        d_loss = (d_loss_real + d_loss_fake) / 2

                        d_loss.backward()

                        # Gradient clipping
                        if self.config["training"]["gradient_clip"] > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.discriminator.parameters(),
                                self.config["training"]["gradient_clip"],
                            )
                        self.optimizer_d.step()
                        self.optimizer_d.zero_grad()

                    # Record losses
                    loss_dict["d_loss"] = d_loss.item()
                    loss_dict["d_loss_real"] = d_loss_real.item()
                    loss_dict["d_loss_fake"] = d_loss_fake.item()

                # Restore generator parameter state
                self.unfreeze_model(self.noise_predictor)
        else:
            # === This step only updates generator ===

            # Freeze discriminator, unfreeze generator
            if self.discriminator is not None:
                self.freeze_model(self.discriminator)
            self.unfreeze_model(self.noise_predictor)

            # Backpropagate generator loss
            if self.config["training"]["use_amp"]:
                self.scaler_g.scale(loss).backward()

                # Gradient clipping
                if self.config["training"]["gradient_clip"] > 0:
                    self.scaler_g.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.noise_predictor.parameters(),
                        self.config["training"]["gradient_clip"],
                    )

                self.scaler_g.step(self.optimizer)
                self.scaler_g.update()
                self.optimizer.zero_grad()
            else:
                loss.backward()

                # Gradient clipping
                if self.config["training"]["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.noise_predictor.parameters(),
                        self.config["training"]["gradient_clip"],
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update EMA after generator update
            self.update_ema()

            # Restore discriminator parameter state
            if self.discriminator is not None:
                self.unfreeze_model(self.discriminator)

        return loss_dict

    def save_checkpoint(self, epoch):
        """Save EMA weights only (no best loss tracking)"""
        # Use EMA weights if available, otherwise use regular weights
        model_weights = (
            self.ema_state
            if self.ema_state is not None
            else self.noise_predictor.state_dict()
        )

        # Save EMA weights directly (for inference)
        ckpt_path = self.exp_dir / "checkpoints" / f"noise_predictor_epoch{epoch}.pth"
        torch.save(model_weights, ckpt_path)
        print(f"✓ EMA weights saved: {ckpt_path}")

        # Also save training state checkpoint (for resuming training)
        training_ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "noise_predictor": self.noise_predictor.state_dict(),  # Current weights (not EMA)
            "ema_state": self.ema_state,  # EMA state
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
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

        training_ckpt_path = (
            self.exp_dir / "checkpoints" / "checkpoint" / f"checkpoint_epoch{epoch}.pth"
        )
        training_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(training_ckpt, training_ckpt_path)
        print(f"✓ Training checkpoint saved: {training_ckpt_path}")

        # Delete old checkpoints (keep most recent N)
        keep_recent = self.config["experiment"].get("keep_recent_checkpoints", 5)
        # Sort by epoch number
        ema_ckpts = sorted(
            (self.exp_dir / "checkpoints").glob("noise_predictor_epoch*.pth"),
            key=lambda x: int(x.stem.replace("noise_predictor_epoch", "")),
        )
        training_ckpts = sorted(
            (self.exp_dir / "checkpoints" / "checkpoint").glob("checkpoint_epoch*.pth"),
            key=lambda x: int(x.stem.replace("checkpoint_epoch", "")),
        )

        # Delete old training checkpoints
        if len(training_ckpts) > keep_recent:
            for old_ckpt in training_ckpts[:-keep_recent]:
                old_ckpt.unlink()
                print(f"✓ Deleted old checkpoint: {old_ckpt.name}")

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

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        # Load discriminator state (if exists)
        if self.discriminator is not None and "discriminator" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            print("  - Discriminator weights loaded")
        if self.optimizer_d is not None and "optimizer_d" in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            print("  - Discriminator optimizer state loaded")

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  - Epoch: {self.current_epoch}")
        print(f"  - Global step: {self.global_step}")

        return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Train noise predictor (single-step training)"
    )
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume training"
    )
    args = parser.parse_args()

    # Create trainer
    trainer = NoisePredictorTrainer(args.config)

    # Resume training
    start_epoch = 1
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        if checkpoint is not None:
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"\nResuming training from epoch {start_epoch}")
        else:
            print("\nWarning: Could not load checkpoint, starting from scratch")

    print("\n" + "=" * 70)
    print("Starting training!")
    print("=" * 70 + "\n")

    # Training loop
    num_epochs = trainer.config["training"]["num_epochs"]

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 70}\n")

            # Train one epoch
            avg_loss_dict = trainer.train_epoch(epoch)

            # Print average losses
            print(f"\nEpoch {epoch} average loss:")
            for key, value in avg_loss_dict.items():
                print(f"  {key}: {value:.4f}")

            # Update learning rate
            if trainer.scheduler is not None:
                trainer.scheduler.step()
                print(
                    f"\nCurrent learning rate: {trainer.optimizer.param_groups[0]['lr']:.2e}"
                )

            # Save checkpoint
            if epoch % trainer.config["experiment"]["save_interval"] == 0:
                trainer.save_checkpoint(epoch)

        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        print("Saving current checkpoint...")
        trainer.save_checkpoint(epoch)
        print("Checkpoint saved, can resume training with --resume")


if __name__ == "__main__":
    main()
