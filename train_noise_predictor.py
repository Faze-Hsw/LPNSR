#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-step Residual Regression Training for the Noise Predictor

The predictor learns the denoiser's prediction error r* = x_0 - x'_0
(the KL-optimal mean correction, in the correction direction).
The per-step schedule weight w_t = sqrt(alpha_t) / (kappa * sqrt(eta_t * eta_{t-1}))
is applied at inference time (see infer.py), keeping the regression target
scale-invariant across timesteps.
"""

import argparse
import logging
import math
import os
import ssl
import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

# Disable SSL certificate verification (Windows Miniconda env fails on the
# default CA bundle). Must be set BEFORE any huggingface_hub / requests import.
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Silence third-party deprecation noise (clip / timm / torchvision / lpips)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")

# Silence the "triton not found" noise from torch.utils.flop_counter and
# xformers (harmless; flop counting and some fused kernels are disabled).
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("xformers").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import matplotlib
import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend, suitable for server environments

from datapipe.train_dataloader import create_train_dataloader
from diffusers import AutoencoderKL
from models.unet import UNetModelSwin
from safetensors.torch import load_file, save_file

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
    """Train the noise predictor with single-step residual regression (iteration-based)."""

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

        # Training schedule (iteration-based, same convention as trainer.py)
        self.total_iters = int(self.config["training"]["iterations"])
        self.save_freq = int(self.config["experiment"]["save_freq"])
        self.keep_last_n = int(self.config["experiment"].get("keep_last_n", 3))
        self.accumulation_steps = int(
            self.config["training"].get("gradient_accumulation_steps", 1)
        )

        # Initialize models
        self._init_models()

        # Initialize optimizer
        self._init_optimizer()

        # Initialize dataloaders
        self._init_dataloaders()

        # Initialize AMP
        if self.config["training"]["use_amp"]:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Training state
        self.global_step = 0

        # Initialize EMA (Exponential Moving Average)
        self._init_ema()

        # Validation (same convention as trainer.py)
        self.val_enabled = self.config.get("validation", {}).get("enabled", False)
        self.val_metrics = {}
        if self.val_enabled:
            self._init_val_metrics()

        print("Trainer initialized!")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {self.device}")
        print(f"Iterations: {self.total_iters}")

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

        # 1. Load SD2.1 VAE (frozen)
        print("\nLoading SD2.1 VAE...")
        vae_path = self.config["resshift"]["vae_path"]
        self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae_scale = self.vae.config.scaling_factor
        print(f"✓ VAE loaded: {vae_path} (scaling_factor={self.vae_scale})")

        # 2. Load frozen denoiser (structure from model_params, same as trainer.yaml)
        print("\nLoading denoiser...")
        denoiser_path = self.config["resshift"]["denoiser_path"]

        crop_size = self.config["data"]["train"]["crop_size"]
        latent_size = crop_size // 8  # SD2.1 VAE downsample factor

        unet_cfg = dict(self.config["model_params"])
        assert unet_cfg["image_size"] == latent_size, (
            f"model_params.image_size ({unet_cfg['image_size']}) must equal "
            f"crop_size / 8 = {latent_size}"
        )

        self.denoiser = UNetModelSwin(**unet_cfg).to(self.device)

        # Load pretrained weights (safetensors preferred; .pth kept for legacy)
        if str(denoiser_path).endswith(".safetensors"):
            denoiser_ckpt = load_file(str(denoiser_path), device="cpu")
        else:
            denoiser_ckpt = torch.load(denoiser_path, map_location=self.device)
        if "state_dict" in denoiser_ckpt:
            state_dict = denoiser_ckpt["state_dict"]
        else:
            state_dict = denoiser_ckpt

        # Remove prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("module._orig_mod."):
                new_key = key.replace("module._orig_mod.", "")
            elif key.startswith("module."):
                new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        self.denoiser.load_state_dict(new_state_dict, strict=True)
        self.denoiser.eval()
        for param in self.denoiser.parameters():
            param.requires_grad = False
        print(f"✓ Denoiser loaded: {denoiser_path}")

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

        # Posterior q(x_{t-1} | x_t, x_0) parameters (validation reverse sampling)
        self.etas_prev = torch.cat([torch.tensor([0.0]), self.etas[:-1]])
        self.alpha = self.etas - self.etas_prev
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        self.posterior_variance = self.kappa**2 * self.etas_prev / self.etas * self.alpha
        # Boundary handling at t=0
        self.posterior_mean_coef1[0] = 0.0
        self.posterior_mean_coef2[0] = 1.0
        self.posterior_variance[0] = self.posterior_variance[1]
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        # Per-step weight for the noise-predictor mean correction (KL-optimal):
        # w_t = sqrt(alpha_t) / (kappa * sqrt(eta_t * eta_{t-1}))
        # t=0 has zero posterior variance and is masked; overwrite to avoid 0/0.
        self.noise_step_weight = torch.sqrt(self.alpha) / (
            self.kappa * torch.sqrt(self.etas * self.etas_prev)
        )
        self.noise_step_weight[0] = 0.0

        print("✓ ResShift diffusion process initialized")

        # 5. Create noise predictor (training)
        # Same architecture as the denoiser UNet; only the input differs:
        # in_channels = 2 * latent_channels (z_t + predicted x_0), with the LR
        # latent condition via cond_lq (first-conv concat, Identity extractor).
        # Output = the residual r = x_0 - x'_0 (deterministic, no dist head).
        print("\nCreating noise predictor...")
        np_cfg = dict(self.config["noise_predictor_params"])
        # Structural consistency with the denoiser: the predictor consumes
        # cat(scale(x_t), pred_x0) and predicts the residual in latent space.
        assert np_cfg["image_size"] == unet_cfg["image_size"], (
            "noise_predictor_params.image_size must match model_params.image_size"
        )
        assert np_cfg["in_channels"] == unet_cfg["in_channels"] * 2, (
            "noise_predictor_params.in_channels must equal 2 * model_params.in_channels "
            "(concat of scale(x_t) and pred_x0)"
        )
        assert np_cfg["out_channels"] == unet_cfg["out_channels"], (
            "noise_predictor_params.out_channels must equal model_params.out_channels "
            "(both are latent-space quantities: x_0 prediction vs residual)"
        )
        assert np_cfg.get("lq_channels") == unet_cfg["in_channels"], (
            "noise_predictor_params.lq_channels must equal model_params.in_channels "
            "(the lq condition is the LR latent)"
        )
        np_cfg["use_checkpoint"] = bool(
            self.config["training"].get("use_gradient_checkpointing", False)
        )
        self.noise_predictor = UNetModelSwin(**np_cfg).to(self.device)

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
                T_max=self.total_iters,
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

    # ------------------------------------------------------------------
    # validation (same convention as trainer.py)
    # ------------------------------------------------------------------
    def _init_val_metrics(self):
        """Initialize validation metrics (identical to RFMSR cal_metrics logic)."""
        try:
            import lpips
            self.val_metrics["lpips"] = lpips.LPIPS(net="alex").to(self.device)
        except ImportError:
            print("[WARN] lpips not installed, LPIPS will be skipped")
        try:
            import pyiqa
            self.val_metrics["psnr"] = pyiqa.create_metric(
                "psnr", test_y_channel=True, color_space="ycbcr", device=self.device)
            self.val_metrics["ssim"] = pyiqa.create_metric(
                "ssim", test_y_channel=True, color_space="ycbcr", device=self.device)
            self.val_metrics["dists"] = pyiqa.create_metric("dists", device=self.device)
            self.val_metrics["niqe"] = pyiqa.create_metric("niqe", device=self.device)
            self.val_metrics["musiq"] = pyiqa.create_metric("musiq", device=self.device)
            self.val_metrics["maniqa"] = pyiqa.create_metric("maniqa", device=self.device)
            self.val_metrics["clipiqa"] = pyiqa.create_metric("clipiqa", device=self.device)
            print("Validation metrics initialized: PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA")
        except ImportError:
            print("[WARN] pyiqa not installed, all FR/NR metrics will be skipped")

    def _q_posterior_mean_variance(self, x0_pred, x_t, t):
        """Posterior q(x_{t-1} | x_t, x_0) mean/variance (validation sampling)."""
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x0_pred
        )
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)
        return mean, variance, log_variance

    @torch.no_grad()
    def reverse_sampling(self, lr_latent):
        """ResShift reverse sampling with the noise-predictor mean correction.

        x_{t-1} = mu_theta_t + sqrt(Sigma_t) * (w_t * r_pred + eps),  t > 0
        x_0     = mu_theta_0                                          (t = 0)
        """
        bsz = lr_latent.shape[0]
        t_T = torch.full(
            (bsz,), self.num_timesteps - 1, device=self.device, dtype=torch.long
        )
        x_t = lr_latent + self.kappa * self._extract(
            self.sqrt_etas, t_T, lr_latent.shape
        ) * torch.randn_like(lr_latent)

        for i in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((bsz,), i, device=self.device, dtype=torch.long)
            x_t_norm = self._scale_input(x_t, t)
            pred_x0 = self.denoiser(x_t_norm, t, lq=lr_latent)
            mean, _, log_variance = self._q_posterior_mean_variance(pred_x0, x_t, t)
            if i > 0:
                r_pred = self.noise_predictor(
                    torch.cat([x_t_norm, pred_x0], dim=1), t, lq=lr_latent
                )
                w_t = self._extract(self.noise_step_weight, t, x_t.shape)
                noise = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_variance) * (w_t * r_pred + noise)
            else:
                x_t = mean
        return x_t

    @torch.no_grad()
    def validate(self, step):
        """Validate: full sampling chain (denoiser + noise predictor) on
        validation LQ/GT pairs -> metrics -> save SR images."""
        self.noise_predictor.eval()

        # Swap in EMA weights for validation
        orig_state = None
        if self.ema_state is not None:
            orig_state = OrderedDict(
                {k: v.data.clone() for k, v in self.noise_predictor.state_dict().items()}
            )
            self.noise_predictor.load_state_dict(self.ema_state)

        val_cfg = self.config.get("validation", {})
        lq_dir = Path(val_cfg.get("lq_dir", "assets/validate_lq"))
        gt_dir = Path(val_cfg.get("gt_dir", "assets/validate_gt"))
        val_scale = val_cfg.get("scale", 4.0)

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
            print(f"[Val @ step {step}] No paired images found")
            if orig_state is not None:
                self.noise_predictor.load_state_dict(orig_state)
            self.noise_predictor.train()
            return

        from PIL import Image

        psnr_v, ssim_v, lpips_v, dists_v = [], [], [], []
        niqe_v, musiq_v, maniqa_v, clipiqa_v = [], [], [], []

        for lq_path, gt_path in tqdm(pairs, desc=f"Val@{step}", leave=False):
            src = Image.open(lq_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            exact_w = int(src.size[0] * val_scale)
            exact_h = int(src.size[1] * val_scale)
            target = src.resize((exact_w, exact_h), Image.BICUBIC)
            ori_h, ori_w = target.size[1], target.size[0]

            # Upsampled LR -> VAE encode to latent
            im_np = np.array(target).astype(np.float32) / 255.0
            im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0).to(self.device)
            z_lr = self.vae.encode(im_cond * 2.0 - 1.0).latent_dist.sample() * self.vae_scale

            # Full sampling chain (denoiser + noise predictor correction)
            sr_latent = self.reverse_sampling(z_lr)

            # VAE decode
            sr_decoded = self.vae.decode(sr_latent / self.vae_scale).sample
            sr_decoded = torch.clamp((sr_decoded + 1.0) / 2.0, 0.0, 1.0)
            sr_decoded = sr_decoded[:, :, 0:ori_h, 0:ori_w]

            # Save SR
            sr_np = (sr_decoded[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            sr_np = np.moveaxis(sr_np, 0, 2)
            Image.fromarray(sr_np).save(out_dir / lq_path.name)

            # GT tensor
            gt_np = np.array(gt_img).astype(np.float32) / 255.0
            gt_tensor = torch.from_numpy(np.moveaxis(gt_np, 2, 0)).unsqueeze(0).to(self.device)

            # Full-reference
            if "psnr" in self.val_metrics:
                psnr_v.append(self.val_metrics["psnr"](sr_decoded, gt_tensor).mean().item())
            if "ssim" in self.val_metrics:
                ssim_v.append(self.val_metrics["ssim"](sr_decoded, gt_tensor).mean().item())
            if "dists" in self.val_metrics:
                dists_v.append(self.val_metrics["dists"](sr_decoded, gt_tensor).mean().item())
            if "lpips" in self.val_metrics:
                gt_norm = (gt_tensor - 0.5) / 0.5
                sr_norm = (sr_decoded - 0.5) / 0.5
                lpips_v.append(self.val_metrics["lpips"](gt_norm, sr_norm).mean().item())

            # No-reference
            if "niqe" in self.val_metrics:
                niqe_v.append(self.val_metrics["niqe"](sr_decoded).mean().item())
            if "musiq" in self.val_metrics:
                musiq_v.append(self.val_metrics["musiq"](sr_decoded).mean().item())
            if "maniqa" in self.val_metrics:
                maniqa_v.append(self.val_metrics["maniqa"](sr_decoded).mean().item())
            if "clipiqa" in self.val_metrics:
                clipiqa_v.append(self.val_metrics["clipiqa"](sr_decoded).mean().item())

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
            print(f"  Removed old validation dir: step {s}")

    def single_step_training_loss(self, z_start, z_y):
        """
        Single-step residual regression loss (KL-optimal mean correction).

        Forward:  x_t = x_0 + eta_t * (y - x_0) + kappa * sqrt(eta_t) * eps
        Target:   r* = x_0 - x'_0, the frozen denoiser's prediction error
                  (correction direction toward the true clean latent).
        Sampling: x_{t-1} = mu_theta_t + sqrt(Sigma_t) * (w_t * r_pred + eps),
                  with w_t = sqrt(alpha_t) / (kappa * sqrt(eta_t * eta_{t-1}))
                  applied at inference time (see infer.py), NOT here.

        Args:
            z_start: HR latent x_0 [B, C, H, W]
            z_y: LR latent y [B, C, H, W] (also used as the lq condition)

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        batch_size = z_y.shape[0]
        T = self.num_timesteps

        # Sample t in [1, T-1]: t=0 has zero posterior variance (no noise term)
        t = torch.randint(1, T, (batch_size,), device=self.device, dtype=torch.long)

        # Forward construction of x_t (same as trainer.py)
        eta_t = self._extract(self.etas, t, z_start.shape)
        sqrt_eta_t = self._extract(self.sqrt_etas, t, z_start.shape)
        x_t = (
            z_start
            + eta_t * (z_y - z_start)
            + self.kappa * sqrt_eta_t * torch.randn_like(z_start)
        )

        # Frozen denoiser prediction (no_grad: constant for both input and target)
        with torch.no_grad():
            x_t_normalized = self._scale_input(x_t, t)
            pred_x0 = self.denoiser(x_t_normalized, t, lq=z_y)

        # Regression target: correction direction r* = x_0 - x'_0
        target = z_start - pred_x0

        # Predicted residual (deterministic output of the UNet).
        # Input x_t uses the same _scale_input normalization as the denoiser,
        # so all three input branches are ~unit variance across timesteps.
        r_pred = self.noise_predictor(
            torch.cat([x_t_normalized, pred_x0], dim=1), t, lq=z_y
        )

        loss = torch.nn.functional.mse_loss(r_pred, target)
        loss_dict["residual_mse"] = loss.item()
        loss_dict["total"] = loss.item()

        return loss, loss_dict

    def train(self):
        """Iteration-based training loop with gradient accumulation (same style as trainer.py)."""
        self.noise_predictor.train()
        grad_clip = self.config["training"].get("gradient_clip", 0)
        accum_steps = self.accumulation_steps

        data_iter = iter(self.train_loader)
        pbar = tqdm(
            total=self.total_iters,
            initial=self.global_step,
            desc="Train",
            unit="step",
            bar_format="{desc} [{n:>6d}/{total_fmt}] {percentage:3.0f}% |{bar}| {postfix} [{rate_fmt}]",
        )

        ema_loss = 0.0
        ema_cnt = 0
        accum_loss = 0.0
        accum_count = 0

        try:
            self.optimizer.zero_grad()

            while self.global_step < self.total_iters:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                hr_images = batch["gt"].to(self.device)  # [B, 3, H, W], [0, 1]
                lr_images = batch["lq"].to(self.device)  # [B, 3, H, W], [0, 1]

                ld = self.train_step(hr_images, lr_images)

                ema_loss += ld["residual_mse"]
                ema_cnt += 1
                accum_loss += ld["residual_mse"]
                accum_count += 1

                if accum_count >= accum_steps:
                    # Gradient clipping
                    if grad_clip > 0:
                        if self.config["training"]["use_amp"]:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.noise_predictor.parameters(), grad_clip
                        )

                    # Optimizer step
                    if self.config["training"]["use_amp"]:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.update_ema()

                    self.global_step += 1

                    pbar.set_postfix(
                        {
                            "res_mse": f"{accum_loss / accum_count:.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )
                    pbar.update(1)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    if self.global_step % self.save_freq == 0:
                        avg_loss = ema_loss / max(ema_cnt, 1)
                        print(
                            f"\n[step {self.global_step}/{self.total_iters}]  "
                            f"residual_mse={avg_loss:.6f}  "
                            f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                        )
                        ema_loss = 0.0
                        ema_cnt = 0
                        self.save_checkpoint(self.global_step)
                        if self.val_enabled:
                            try:
                                self.validate(self.global_step)
                            except Exception as e:
                                print(
                                    f"\n[WARN] Validation failed at "
                                    f"step {self.global_step}: {e}"
                                )

                    accum_loss = 0.0
                    accum_count = 0

            self.save_checkpoint(self.global_step)
            pbar.close()
            print("Training finished!")
        except KeyboardInterrupt:
            pbar.close()
            print("\nInterrupted, saving checkpoint ...")
            self.save_checkpoint(self.global_step)

    def train_step(self, hr_images, lr_images):
        """
        Forward + backward. Loss is scaled by 1/accumulation_steps;
        the optimizer step is performed by the training loop.

        Args:
            hr_images: HR images [B, 3, H, W], range [0, 1]
            lr_images: LR images [B, 3, H, W], range [0, 1]

        Returns:
            loss_dict: Loss dictionary
        """
        self.noise_predictor.train()

        # 1. Encode to latent space (frozen VAE)
        with torch.no_grad():
            # Convert to [-1, 1]
            hr_images_norm = hr_images * 2.0 - 1.0
            lr_images_norm = lr_images * 2.0 - 1.0

            # HR image direct encoding
            z_start = self.vae.encode(hr_images_norm.float()).latent_dist.sample() * self.vae_scale

            # LR image needs to be upsampled to HR size first, then encoded
            scale_factor = self.config["data"]["train"]["scale"]
            lr_images_upsampled = torch.nn.functional.interpolate(
                lr_images_norm,
                scale_factor=scale_factor,
                mode="bicubic",
                align_corners=False,
            )
            z_y = self.vae.encode(lr_images_upsampled.float()).latent_dist.sample() * self.vae_scale

        # 2. Single-step residual regression loss
        if self.config["training"]["use_amp"]:
            with autocast(device_type="cuda"):
                loss, loss_dict = self.single_step_training_loss(z_start, z_y)
        else:
            loss, loss_dict = self.single_step_training_loss(z_start, z_y)

        # 3. Backward pass (only the noise predictor is trained);
        # loss is scaled by 1/accumulation_steps, the optimizer step is
        # performed by the training loop after accumulation.
        scaled_loss = loss * (1.0 / self.accumulation_steps)
        if self.config["training"]["use_amp"]:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss_dict

    def save_checkpoint(self, step):
        """Save EMA weights (safetensors) + full training state for resume."""
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # EMA weights for inference (safetensors, same convention as the denoiser)
        model_weights = (
            self.ema_state
            if self.ema_state is not None
            else self.noise_predictor.state_dict()
        )
        weight_path = ckpt_dir / f"noise_predictor_step{step}.safetensors"
        save_file(dict(model_weights), str(weight_path))
        print(f"✓ Noise predictor weights saved: {weight_path.name}")

        # Full training state (resume)
        training_ckpt = {
            "global_step": self.global_step,
            "noise_predictor": self.noise_predictor.state_dict(),
            "ema_state": self.ema_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
        }
        if self.ema_rate is not None:
            training_ckpt["ema_rate"] = self.ema_rate
        state_path = ckpt_dir / f"training_state_step{step}.pth"
        torch.save(training_ckpt, state_path)
        print(f"✓ Training state saved: {state_path.name}")

        # Keep only the most recent N checkpoints
        if self.keep_last_n > 0:
            for pattern in ("noise_predictor_step*.safetensors", "training_state_step*.pth"):
                files = sorted(
                    ckpt_dir.glob(pattern),
                    key=lambda x: int(x.stem.split("step")[-1]),
                )
                for old in files[:-self.keep_last_n]:
                    old.unlink()
                    print(f"✓ Deleted old checkpoint: {old.name}")

    def load_checkpoint(self, checkpoint_path):
        """Load training state checkpoint (for resuming)"""
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

        self.global_step = checkpoint["global_step"]

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  - Global step: {self.global_step}")

        return checkpoint


    def print_training_config(self):
        """Print the full training configuration (same style as trainer.py)."""
        cfg = self.config
        tc = cfg["training"]
        dc = cfg["data"]["train"]
        diff = tc["diffusion"]
        opt = cfg["optimizer"]
        sched = cfg.get("scheduler", {})
        val = cfg.get("validation", {})
        latent_size = dc["crop_size"] // 8  # SD2.1 f=8

        line = "-" * 70
        print("\n" + "=" * 70)
        print("Training Configuration")
        print("=" * 70)

        print("Experiment:")
        print(f"  exp_dir             : {self.exp_dir}")
        print(f"  total_iters         : {self.total_iters}")
        print(f"  save_freq           : {self.save_freq}")
        print(f"  keep_last_n         : {self.keep_last_n}")

        print(line)
        print("Data:")
        print(f"  batch_size          : {tc['batch_size']}")
        print(f"  accum_steps         : {self.accumulation_steps}")
        print(f"  effective_batch     : {tc['batch_size'] * self.accumulation_steps}")
        print(f"  crop_size           : {dc['crop_size']} (latent {latent_size})")
        print(f"  scale               : {dc['scale']}")
        print(f"  num_workers         : {tc['num_workers']}")
        print(f"  hr_dir              : {dc['hr_dir']}")

        print(line)
        print("Diffusion:")
        print(f"  num_timesteps       : {diff['num_timesteps']}")
        print(f"  kappa               : {diff['kappa']}")
        print(f"  min_noise_level     : {diff['min_noise_level']}")
        print(f"  etas_end            : {diff['etas_end']}")
        print(f"  eta_power           : {diff.get('eta_power', 0.3)}")
        print(f"  normalize_input     : {diff.get('normalize_input', True)}")
        print(f"  latent_flag         : {diff.get('latent_flag', True)}")

        print(line)
        print("Optimizer:")
        print(f"  type                : {opt['type']}")
        print(f"  lr                  : {opt['lr']}")
        print(f"  scheduler           : {sched.get('type', 'none')}")
        print(f"  gradient_clip       : {tc.get('gradient_clip', 0)}")
        print(f"  grad_checkpointing  : {tc.get('use_gradient_checkpointing', False)}")
        print(f"  ema_rate            : {tc.get('ema_rate', 0.999)}")
        print(f"  use_amp             : {tc.get('use_amp', False)}")

        print(line)
        print("Model:")
        print(f"  vae                 : {cfg['resshift']['vae_path']}")
        print(f"  vae_scaling_factor  : {self.vae_scale}")
        print(f"  denoiser (frozen)   : {cfg['resshift']['denoiser_path']}")
        den_params = sum(p.numel() for p in self.denoiser.parameters())
        print(f"  denoiser_params     : {den_params / 1e6:.2f}M")
        np_params = sum(p.numel() for p in self.noise_predictor.parameters())
        print(f"  predictor_params    : {np_params / 1e6:.2f}M")

        print(line)
        print("Validation:")
        print(f"  enabled             : {self.val_enabled}")
        if self.val_enabled:
            print(f"  lq_dir              : {val.get('lq_dir')}")
            print(f"  gt_dir              : {val.get('gt_dir')}")

        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train the noise predictor (single-step residual regression)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_noise_predictor.yaml",
        help="Config file path (default: configs/train_noise_predictor.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Training state path (training_state_step*.pth) to resume from",
    )
    args = parser.parse_args()

    # Create trainer
    trainer = NoisePredictorTrainer(args.config)

    # Resume training
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"\nResuming training from step {trainer.global_step}")

    print("\n" + "=" * 70)
    print(
        f"Starting noise predictor training for {trainer.total_iters} iterations"
    )
    print("=" * 70 + "\n")

    trainer.print_training_config()

    trainer.train()


if __name__ == "__main__":
    main()
