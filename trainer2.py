#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Residual Corrector Training (trainer2)

Train the noise predictor as a DETERMINISTIC residual corrector stacked on the
frozen first-stage ResShift denoiser:

    1. forward:     x_t = x_0 + eta_t * (z_y - x_0) + kappa * sqrt(eta_t) * eps
    2. first stage: x0' = denoiser(scale(x_t), t, lq=LR)     (frozen weights)
    3. corrector:   r   = predictor(scale(x_t), x0', LR, t)  (deterministic,
                    double_z=False -- no probabilistic head, no sampling)
    4. objective:   loss = MSE(r, x0 - x0') + LPIPS(decode(x0' + r), GT)

i.e. the corrector regresses the residual between the ground-truth HR latent
(x0) and the first denoiser's prediction (x0'). At inference the corrected
estimate is x0'' = x0' + r.
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
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file
from torch.amp import GradScaler, autocast
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend, suitable for server environments

from datapipe.train_dataloader import create_train_dataloader
from ldm.models.autoencoder import VQModelTorch
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

    Args:
        schedule_name: Schedule type ('exponential')
        num_diffusion_timesteps: Number of diffusion steps T
        min_noise_level: Minimum noise level eta_1
        etas_end: Maximum noise level eta_T
        kappa: Variance control parameter kappa
        kwargs: Additional parameters (e.g., power)

    Returns:
        sqrt_etas: sqrt(eta_t) array, shape=(T,)
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
    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name}")

    return sqrt_etas


class ResidualCorrectorTrainer:
    """Train the noise predictor as a deterministic residual corrector."""

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

        # Iteration-based training state (RFMSR style)
        exp = self.config["experiment"]
        self.total_iters = self.config["training"]["iterations"]
        self.save_freq = exp.get("save_freq", 1000)
        self.keep_last_n = exp.get("keep_last_n", 0)
        self.accumulation_steps = self.config["training"].get(
            "gradient_accumulation_steps", 1
        )
        self.global_step = 0

        # Initialize everything
        self._init_models()
        self._init_losses()
        self._init_optimizer()
        self._init_dataloaders()
        self._init_ema()

        # Validation (same as trainer.py)
        self.val_enabled = self.config.get("validation", {}).get("enabled", False)
        self.val_metrics = {}
        if self.val_enabled:
            self._init_val_metrics()

        # AMP
        if self.config["training"]["use_amp"]:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        print("Trainer initialized!")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {self.device}")
        eff_batch = self.config["training"]["batch_size"] * self.accumulation_steps
        print(
            f"Iterations: {self.total_iters} (log/save every {self.save_freq}, "
            f"accum x{self.accumulation_steps}, effective batch {eff_batch})"
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

        # 2. Load ResShift UNet (frozen first-stage denoiser)
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

        # 3. ResShift diffusion parameters (forward only: eta schedule)
        print("\nInitializing ResShift diffusion parameters...")
        diffusion_config = self.config["diffusion"]
        self.num_timesteps = diffusion_config["num_timesteps"]
        self.kappa = diffusion_config["kappa"]
        self.normalize_input = diffusion_config.get("normalize_input", True)
        self.latent_flag = diffusion_config.get("latent_flag", True)

        sqrt_etas = get_named_eta_schedule(
            schedule_name=diffusion_config["eta_schedule"],
            num_diffusion_timesteps=self.num_timesteps,
            min_noise_level=diffusion_config["min_noise_level"],
            etas_end=diffusion_config["etas_end"],
            kappa=self.kappa,
            kwargs={"power": diffusion_config.get("eta_power", 0.3)},
        )
        self.sqrt_etas = torch.from_numpy(sqrt_etas).float()
        self.etas = self.sqrt_etas**2

        # Posterior q(x_{t-1} | x_t, x0) parameters -- only needed by the
        # validation sampling chain; the single-step training objective does
        # not use them.
        self.etas_prev = torch.cat([torch.tensor([0.0]), self.etas[:-1]])
        alpha = self.etas - self.etas_prev
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = alpha / self.etas
        self.posterior_variance = self.kappa**2 * self.etas_prev / self.etas * alpha
        # Boundary handling at t=0 (avoid division by zero / NaN)
        self.posterior_mean_coef1[0] = 0.0
        self.posterior_mean_coef2[0] = 1.0
        self.posterior_variance[0] = self.posterior_variance[1]
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        print("✓ ResShift diffusion parameters initialized")

        # 4. Residual corrector (noise predictor backbone, DETERMINISTIC output)
        print("\nCreating residual corrector (noise predictor backbone)...")
        noise_config = self.config["noise_predictor"]
        with open(noise_config["config_path"], "r", encoding="utf-8") as f:
            np_cfg = yaml.safe_load(f)

        # double_z is FORCED to False: the corrector outputs a deterministic
        # residual, the probabilistic (mean+logvar) head is not used here.
        self.noise_predictor = create_noise_predictor(
            image_size=np_cfg.get("image_size", latent_size),
            latent_channels=np_cfg["latent_channels"],
            model_channels=np_cfg["model_channels"],
            out_channels=np_cfg.get("out_channels", np_cfg["latent_channels"]),
            channel_mult=tuple(np_cfg["channel_mult"]),
            num_res_blocks=np_cfg["num_res_blocks"],
            attention_resolutions=np_cfg.get(
                "attention_resolutions", [64, 32, 16, 8]
            ),
            dropout=np_cfg.get("dropout", 0.0),
            conv_resample=np_cfg.get("conv_resample", True),
            dims=np_cfg.get("dims", 2),
            use_fp16=np_cfg.get("use_fp16", False),
            num_heads=np_cfg.get("num_heads", -1),
            num_head_channels=np_cfg.get("num_head_channels", 32),
            use_scale_shift_norm=np_cfg.get("use_scale_shift_norm", True),
            resblock_updown=np_cfg.get("resblock_updown", False),
            swin_depth=np_cfg.get("swin_depth", 2),
            swin_embed_dim=np_cfg.get("swin_embed_dim", 192),
            window_size=np_cfg.get("window_size", 8),
            mlp_ratio=np_cfg.get("mlp_ratio", 4.0),
            patch_norm=np_cfg.get("patch_norm", False),
            cond_lq=np_cfg.get("cond_lq", True),
            lq_size=np_cfg.get("lq_size", latent_size),
            use_gradient_checkpointing=self.config["training"].get(
                "use_gradient_checkpointing", False
            ),
            double_z=False,  # deterministic residual output
        ).to(self.device)

        num_params = sum(p.numel() for p in self.noise_predictor.parameters())
        print("✓ Residual corrector created (noise predictor backbone)")
        print(f"  - Parameters: {num_params / 1e6:.2f}M")
        print("  - Output: deterministic residual (double_z=False)")

    def _init_losses(self):
        """Initialize the LPIPS perceptual loss (the residual MSE is inline)."""
        loss_config = self.config["loss"]
        self.l2_weight = loss_config.get("l2_weight", 1.0)
        self.lpips_weight = loss_config.get("lpips_weight", 1.0)

        self.lpips_loss = None
        if self.lpips_weight > 0:
            try:
                from losses.lpips_loss import LPIPSLoss

                self.lpips_loss = LPIPSLoss(
                    loss_weight=1.0,
                    net_type=loss_config.get("lpips_net_type", "vgg"),
                )
            except ImportError:
                print("[WARN] lpips not installed, LPIPS loss will be skipped")
                self.lpips_loss = None

        print(
            f"✓ Losses: L2(w={self.l2_weight}) + "
            f"LPIPS(w={self.lpips_weight}, "
            f"net={loss_config.get('lpips_net_type', 'vgg')})"
        )

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        print("\n" + "=" * 70)
        print("Initializing Optimizer")
        print("=" * 70)

        opt_config = self.config["optimizer"]

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

    def _init_dataloaders(self):
        """Initialize dataloaders"""
        print("\n" + "=" * 70)
        print("Initializing Dataloaders")
        print("=" * 70 + "\n")

        data_config = self.config["data"]
        train_config = self.config["training"]

        if self.config["degradation"]["use_degradation"]:
            print("Using degradation pipeline to generate LR images...")
            self.train_loader = create_train_dataloader(
                data_dir=data_config["train"]["hr_dir"],
                config_path=self.config["degradation"]["config_path"],
                batch_size=train_config["batch_size"],
                num_workers=train_config["num_workers"],
                gt_size=data_config["train"]["crop_size"],
                use_hflip=data_config["train"].get("use_flip", True),
                use_rot=data_config["train"].get("use_rot", True),
                shuffle=True,
                pin_memory=True,
            )
            print(f"✓ Training dataloader created: {len(self.train_loader)} batches")
        else:
            raise NotImplementedError(
                "Direct loading of LR-HR image pairs not yet supported"
            )

    def _init_ema(self):
        """Initialize EMA (Exponential Moving Average) for the corrector"""
        ema_rate = self.config["training"].get("ema_rate", 0.999)

        if ema_rate > 0:
            self.ema_rate = ema_rate
            self.ema_state = OrderedDict(
                {
                    key: deepcopy(value.data)
                    for key, value in self.noise_predictor.state_dict().items()
                }
            )
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
        """Update EMA weights after each optimizer step"""
        if self.ema_state is None:
            return

        source_state = self.noise_predictor.state_dict()
        for key, value in self.ema_state.items():
            if key in self.ema_ignore_keys:
                self.ema_state[key] = source_state[key]
            elif not self.ema_state[key].is_floating_point():
                self.ema_state[key] = source_state[key]
            else:
                self.ema_state[key].mul_(self.ema_rate).add_(
                    source_state[key].detach().data, alpha=1 - self.ema_rate
                )

    def _extract(self, a, t, x_shape):
        """Extract values from a corresponding to t, and reshape to x_shape"""
        batch_size = t.shape[0]
        out = a.to(t.device)[t]
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _scale_input(self, inputs, t):
        """Normalize input (key step in ResShift)"""
        if self.normalize_input:
            if self.latent_flag:
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

    def training_loss(self, z_start, z_y, lr_image, hr_images_norm):
        """Single-step residual correction loss (the whole training objective).

        1. random t, forward: x_t = x0 + eta_t*(z_y - x0) + kappa*sqrt(eta_t)*eps
        2. frozen first denoiser: x0' = UNet(scale(x_t), t, lq=LR)
        3. deterministic corrector: r = P(scale(x_t), x0', LR, t)
        4. loss = MSE(r, x0 - x0') + LPIPS(decode(x0' + r), GT)
        """
        bsz = z_y.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (bsz,), device=self.device, dtype=torch.long
        )

        # 1. Forward diffusion (ResShift)
        eta_t = self._extract(self.etas, t, z_start.shape)
        sqrt_eta_t = self._extract(self.sqrt_etas, t, z_start.shape)
        e_0 = z_y - z_start
        eps = torch.randn_like(z_start)
        x_t = z_start + eta_t * e_0 + self.kappa * sqrt_eta_t * eps

        # 2. First denoiser (frozen weights, gradients flow through)
        x_t_normalized = self._scale_input(x_t, t)
        x0_pred = self.resshift_unet(x_t_normalized, t, lq=lr_image)

        # 3. Deterministic residual correction
        r = self.noise_predictor(
            x_t_normalized, x0_pred, lr_image, t, sample_posterior=False
        )

        # 4. Residual supervision (latent space): r should match (x0 - x0')
        res_target = z_start - x0_pred
        mse_loss = torch.nn.functional.mse_loss(r, res_target)
        loss = self.l2_weight * mse_loss
        loss_dict = {"res_mse": mse_loss.item()}

        # 5. LPIPS perceptual loss (image space) on the corrected estimate
        #    x0_corr = x0' + r; gradients flow through the frozen VQVAE decoder
        if self.lpips_loss is not None:
            x0_corr = x0_pred + r
            pred_image = self.vae.decode(x0_corr)  # [-1, 1], keep gradients
            pred_image = pred_image * 0.5 + 0.5  # [0, 1]
            gt_image = hr_images_norm * 0.5 + 0.5  # [0, 1]
            lpips_loss = self.lpips_loss(pred_image, gt_image)
            loss_dict["lpips"] = lpips_loss.item()
            loss = loss + self.lpips_weight * lpips_loss

        return loss, loss_dict

    def train_step(self, hr_images, lr_images):
        """Forward + backward (loss scaled by 1/accum); step performed in train()."""
        self.noise_predictor.train()
        inv_accum = 1.0 / self.accumulation_steps

        # 1. Encode to latent space (frozen VQVAE)
        with torch.no_grad():
            hr_images_norm = hr_images * 2.0 - 1.0
            lr_images_norm = lr_images * 2.0 - 1.0

            z_start = self.vae.encode(hr_images_norm)

            scale_factor = self.config["data"]["train"]["scale"]
            lr_images_upsampled = torch.nn.functional.interpolate(
                lr_images_norm,
                scale_factor=scale_factor,
                mode="bicubic",
                align_corners=False,
            )
            z_y = self.vae.encode(lr_images_upsampled)

        # 2. Single-step residual correction loss
        if self.config["training"]["use_amp"]:
            with autocast(device_type="cuda"):
                loss, loss_dict = self.training_loss(
                    z_start, z_y, lr_images_norm, hr_images_norm
                )
        else:
            loss, loss_dict = self.training_loss(
                z_start, z_y, lr_images_norm, hr_images_norm
            )

        # 3. Backward only -- no optimizer step here
        scaled_loss = loss * inv_accum
        if self.config["training"]["use_amp"]:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss_dict

    def train(self):
        """Iteration-based loop with gradient accumulation (RFMSR style)."""
        self.noise_predictor.train()

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

        g_count = 0
        self.optimizer.zero_grad()
        last_save_step = self.global_step
        window_loss_sum = {}
        window_loss_cnt = 0

        try:
            while self.global_step < self.total_iters:
                # Cycle the dataloader
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                hr_images = batch["gt"].to(self.device)  # [B, 3, H, W], [0, 1]
                lr_images = batch["lq"].to(self.device)  # [B, 3, H, W], [0, 1]

                # Forward + backward only (no optimizer step inside)
                loss_dict = self.train_step(hr_images, lr_images)

                # Accumulate window losses (averaged at each save-point print)
                for key, value in loss_dict.items():
                    window_loss_sum[key] = window_loss_sum.get(key, 0.0) + value
                window_loss_cnt += 1

                stepped = False
                g_count += 1
                if g_count >= accum_steps:
                    # Optimizer step
                    if use_amp:
                        self.scaler.unscale_(self.optimizer)
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.noise_predictor.parameters(), gradient_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.noise_predictor.parameters(), gradient_clip
                            )
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Update EMA after each optimizer step
                    self.update_ema()

                    g_count = 0
                    self.global_step += 1
                    stepped = True

                if stepped:
                    postfix = {"res_mse": f"{loss_dict['res_mse']:.4f}"}
                    if "lpips" in loss_dict:
                        postfix["lpips"] = f"{loss_dict['lpips']:.4f}"
                    postfix["lr"] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                    # Per-iteration lr scheduling
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Print + save together
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
    # validation (same as trainer.py, with the residual corrector applied
    # at every reverse step: x0_corr = x0' + r)
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
        """Full reverse sampling chain with the residual corrector applied at
        every step (inference twin of the training objective).

        Each reverse step:
            x0'    = UNet(scale(x_t), t, lq=LR)      (frozen first stage)
            x0_corr = x0' + r                         (deterministic corrector)
            x_{t-1} = posterior_mean(x0_corr, x_t) + sigma * eps   (i > 0)

        Returns:
            x0_corr at t=0 [B, C, H, W]
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

        x0_corr = None
        for i in range(self.num_timesteps - 1, -1, -1):
            t_tensor = torch.full(
                (bsz,), i, device=self.device, dtype=torch.long
            )

            x_t_normalized = self._scale_input(x_t, t_tensor)
            pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)
            r = self.noise_predictor(
                x_t_normalized, pred_x0, lr_image, t_tensor, sample_posterior=False
            )
            x0_corr = pred_x0 + r

            if i > 0:
                mean = (
                    self._extract(self.posterior_mean_coef1, t_tensor, x_t.shape)
                    * x_t
                    + self._extract(self.posterior_mean_coef2, t_tensor, x_t.shape)
                    * x0_corr
                )
                log_variance = self._extract(
                    self.posterior_log_variance_clipped, t_tensor, x_t.shape
                )
                if generator is not None:
                    noise = torch.randn(
                        x_t.shape,
                        generator=generator,
                        device=self.device,
                        dtype=x_t.dtype,
                    )
                else:
                    noise = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_variance) * noise

        return x0_corr

    @torch.no_grad()
    def validate(self, step):
        """Validate: corrected full sampling chain on validation LQ/GT pairs
        -> metrics -> save SR images."""
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

            # Original LR [-1, 1] (lq condition; size == latent size)
            im_np = np.array(src).astype(np.float32) / 255.0
            lr_tensor = (
                torch.from_numpy(np.moveaxis(im_np, 2, 0))
                .unsqueeze(0)
                .to(self.device)
            )
            lr_cond = lr_tensor * 2.0 - 1.0

            # Upsample LR -> HR size with the SAME interpolation as training,
            # then VQVAE encode -> z_y
            lr_up = F.interpolate(
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
        weights_path = ckpt_dir / f"noise_predictor_residual_step{step}.safetensors"
        save_file(safeweight, str(weights_path))
        print(f"✓ EMA weights saved: {weights_path}")

        # 2. Full training state (for resuming training)
        training_ckpt = {
            "step": step,
            "global_step": self.global_step,
            "noise_predictor": self.noise_predictor.state_dict(),
            "ema_state": self.ema_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "config": self.config,
        }

        if self.ema_rate is not None:
            training_ckpt["ema_rate"] = self.ema_rate

        training_ckpt_path = ckpt_dir / f"training_state_step{step}.pth"
        torch.save(training_ckpt, training_ckpt_path)
        print(f"✓ Training state saved: {training_ckpt_path}")

        # 3. Clean up old checkpoints (keep last N steps)
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(ckpt_dir)

    def _cleanup_old_checkpoints(self, ckpt_dir):
        """Keep only the most recent `keep_last_n` checkpoint steps."""
        import re

        pattern = re.compile(
            r"(noise_predictor_residual_step|training_state_step)(\d+)"
        )
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
        """Load training state checkpoint (for resuming)"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.noise_predictor.load_state_dict(checkpoint["noise_predictor"])

        if "ema_state" in checkpoint and checkpoint["ema_state"] is not None:
            self.ema_state = checkpoint["ema_state"]
            print("  - EMA state loaded")
        elif self.ema_rate is not None:
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

        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.global_step = checkpoint.get("step", checkpoint.get("global_step", 0))

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  - Step: {self.global_step}")

        return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Train the noise predictor as a deterministic residual "
        "corrector for the frozen ResShift denoiser"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trainer2.yaml",
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
    trainer = ResidualCorrectorTrainer(args.config)

    # Resume training
    if args.resume:
        trainer.load_checkpoint(args.resume)

    print("\n" + "=" * 70)
    print(f"Starting residual corrector training for {trainer.total_iters} iterations!")
    print("=" * 70 + "\n")

    # Iteration-based training loop (logging + checkpointing handled inside)
    trainer.train()


if __name__ == "__main__":
    main()
