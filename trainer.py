#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the ResShift denoising UNet (x0-prediction) using the standard
single-step diffusion objective, following the original ResShift training
recipe (see ResShift/trainer.py + models/gaussian_diffusion.py).

Key differences from `train_noise_predictor.py`:
  - The ResShift UNet (self.denoiser) is TRAINED (not frozen).
  - The VQ-VAE is frozen and only used to encode/decode latents.
  - There is NO noise predictor / GAN / LPIPS stack; the objective is a pure
    MSE regression in latent space over the predicted clean latent x0.
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

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datapipe.train_dataloader import create_train_dataloader  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from models.unet import UNetModelSwin  # noqa: E402


def get_named_eta_schedule(
    schedule_name,
    num_diffusion_timesteps,
    min_noise_level,
    etas_end=0.99,
    kappa=1.0,
    kwargs=None,
):
    """ResShift's exponential eta schedule (returns sqrt_etas, shape [T])."""
    if kwargs is None:
        kwargs = {}

    if schedule_name == "exponential":
        power = kwargs.get("power", 2.0)
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(
            1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start)
        )
        base = np.ones([num_diffusion_timesteps]) * increaser
        power_timestep = (
            np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        )
        power_timestep *= num_diffusion_timesteps - 1
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == "ldm":
        import scipy.io as sio

        mat_path = kwargs.get("mat_path", None)
        if mat_path is None:
            raise ValueError("ldm schedule requires mat_path")
        sqrt_etas = sio.loadmat(mat_path)["sqrt_etas"].reshape(-1)
    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name}")

    return sqrt_etas


class ResShiftTrainer:
    """Train the ResShift denoising UNet with the x0-prediction objective."""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Experiment dir
        self.exp_dir = Path(self.config["experiment"]["save_dir"])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)

        # Init everything
        self._init_models()
        self._init_optimizer()
        self._init_dataloaders()
        self._init_ema()

        # AMP
        self.use_amp = bool(self.config["training"].get("use_amp", False))
        self.scaler = GradScaler() if self.use_amp else None

        # Iteration-based training (RFMSR style)
        exp = self.config["experiment"]
        self.total_iters = self.config["training"]["iterations"]
        # Logging shares the same frequency as checkpoint saving.
        self.save_freq = exp.get("save_freq", 1000)
        self.keep_last_n = exp.get("keep_last_n", 0)

        # Gradient accumulation (RFMSR style)
        self.accumulation_steps = self.config["training"].get(
            "gradient_accumulation_steps", 1
        )

        # Validation (RFMSR style)
        self.val_enabled = self.config.get("validation", {}).get("enabled", False)
        self.val_metrics = {}
        if self.val_enabled:
            self._init_val_metrics()

        # State
        self.global_step = 0

        print("Trainer initialized!")
        print(f"  Exp dir   : {self.exp_dir}")
        print(f"  Device    : {self.device}")
        print(f"  Iterations: {self.total_iters}")

    # ------------------------------------------------------------------
    # model / diffusion init
    # ------------------------------------------------------------------
    def _init_models(self):
        print("\n" + "=" * 70)
        print("Initializing Models")
        print("=" * 70)

        # ---- 1) SD2.1 VAE (AutoencoderKL, frozen) ----
        print("\nLoading SD2.1 VAE...")
        vae_path = self.config["resshift"]["vae_path"]
        self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae_scale = self.vae.config.scaling_factor
        print(f"  VAE loaded: {vae_path}")
        print(f"  scaling_factor: {self.vae_scale}")

        # ---- 2) Denoiser UNet (trained from scratch) ----
        print("\nCreating denoiser UNet (from scratch)...")
        unet_cfg = dict(self.config["model_params"])
        self.denoiser = UNetModelSwin(**unet_cfg).to(self.device)

        n_params = sum(p.numel() for p in self.denoiser.parameters())
        print(f"  Denoiser parameters: {n_params / 1e6:.2f}M")

        # ---- 3) ResShift diffusion schedule ----
        print("\nInitializing ResShift diffusion...")
        diff_cfg = self.config["training"]["diffusion"]
        self.num_timesteps = diff_cfg["num_timesteps"]
        self.kappa = float(diff_cfg["kappa"])
        self.normalize_input = bool(diff_cfg.get("normalize_input", True))
        self.latent_flag = bool(diff_cfg.get("latent_flag", True))

        sqrt_etas = get_named_eta_schedule(
            schedule_name=diff_cfg["eta_schedule"],
            num_diffusion_timesteps=self.num_timesteps,
            min_noise_level=diff_cfg["min_noise_level"],
            etas_end=diff_cfg["etas_end"],
            kappa=self.kappa,
            kwargs={"power": diff_cfg.get("eta_power", 2.0)},
        )
        self.sqrt_etas = torch.from_numpy(sqrt_etas).float()
        self.etas = self.sqrt_etas ** 2
        # ResShift alpha: alpha_t = eta_t - eta_{t-1}
        self.etas_prev = torch.cat([torch.tensor([0.0]), self.etas[:-1]])
        self.alpha = self.etas - self.etas_prev

        # Posterior q(x_{t-1} | x_t, x_0) parameters (for reverse sampling)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        self.posterior_variance = self.kappa ** 2 * self.etas_prev / self.etas * self.alpha
        # Boundary handling at t=0
        self.posterior_mean_coef1[0] = 0.0
        self.posterior_mean_coef2[0] = 1.0
        self.posterior_variance[0] = self.posterior_variance[1]
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        print(f"  num_timesteps = {self.num_timesteps}")
        print(f"  kappa         = {self.kappa}")

    # ------------------------------------------------------------------
    # optimizer / data / ema
    # ------------------------------------------------------------------
    def _init_optimizer(self):
        print("\n" + "=" * 70)
        print("Initializing Optimizer")
        print("=" * 70)
        opt_cfg = self.config["optimizer"]
        if opt_cfg["type"] == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.denoiser.parameters(),
                lr=opt_cfg["lr"],
                betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                weight_decay=opt_cfg.get("weight_decay", 0.0),
            )
        elif opt_cfg["type"] == "Adam":
            self.optimizer = torch.optim.Adam(
                self.denoiser.parameters(),
                lr=opt_cfg["lr"],
                betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                weight_decay=opt_cfg.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg['type']}")

        sched_cfg = self.config["scheduler"]
        if sched_cfg["type"] == "CosineAnnealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["iterations"],
                eta_min=sched_cfg["min_lr"],
            )
        elif sched_cfg["type"] == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg["step_size"],
                gamma=sched_cfg["gamma"],
            )
        else:
            self.scheduler = None

        print(f"  Optimizer: {opt_cfg['type']}  lr={opt_cfg['lr']}")

    def _init_dataloaders(self):
        print("\nInitializing Dataloaders...")
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]
        if not self.config["degradation"]["use_degradation"]:
            raise NotImplementedError("Only on-the-fly degradation is supported.")
        self.train_loader = create_train_dataloader(
            data_dir=data_cfg["train"]["hr_dir"],
            config_path=self.config["degradation"]["config_path"],
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg["num_workers"],
            gt_size=data_cfg["train"]["crop_size"],
            use_hflip=data_cfg["train"].get("use_flip", True),
            use_rot=data_cfg["train"].get("use_rot", True),
            shuffle=True,
            pin_memory=True,
        )
        print(f"  {len(self.train_loader)} batches/epoch")

    def _init_ema(self):
        ema_rate = self.config["training"].get("ema_rate", 0.999)
        if ema_rate > 0:
            self.ema_rate = ema_rate
            self.ema_state = OrderedDict(
                {k: deepcopy(v.data) for k, v in self.denoiser.state_dict().items()}
            )
            self.ema_ignore_keys = [
                k for k in self.ema_state if "running_" in k or "num_batches_tracked" in k
            ]
            print(f"EMA initialized (rate={ema_rate})")
        else:
            self.ema_rate = None
            self.ema_state = None
            self.ema_ignore_keys = None

    @torch.no_grad()
    def update_ema(self):
        if self.ema_state is None:
            return
        src = self.denoiser.state_dict()
        for k, v in self.ema_state.items():
            if k in self.ema_ignore_keys or not v.is_floating_point():
                self.ema_state[k] = src[k].clone()
            else:
                self.ema_state[k].mul_(self.ema_rate).add_(
                    src[k].detach().data, alpha=1 - self.ema_rate
                )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.to(t.device)[t].float()
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def _scale_input(self, x, t):
        """ResShift input normalization (latent branch)."""
        if not self.normalize_input:
            return x
        if self.latent_flag:
            std = torch.sqrt(self._extract(self.etas, t, x.shape) * self.kappa ** 2 + 1)
            return x / std
        inputs_max = self._extract(self.sqrt_etas, t, x.shape) * self.kappa * 3 + 1
        return x / inputs_max

    # ------------------------------------------------------------------
    # SD2.1 VAE encode/decode (official: sample + scaling_factor)
    # ------------------------------------------------------------------
    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image [-1,1] -> latent, scaled by scaling_factor."""
        return self.vae.encode(img.float()).latent_dist.sample() * self.vae_scale

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent -> image [-1,1]."""
        return self.vae.decode(latent / self.vae_scale).sample

    # ------------------------------------------------------------------
    # core loss (single-step diffusion training, x0 prediction)
    # ------------------------------------------------------------------
    def training_loss(self, z_start, z_y):
        """
        Standard single-step ResShift diffusion loss.

        Forward:  x_t = x_0 + eta_t * (y_0 - x_0) + kappa * sqrt(eta_t) * eps
        Target:   the UNet predicts the clean latent x_0 (START_X).
        Loss:     mean((x0_pred - x_0)^2), optionally weighted per timestep.
        Note:     the UNet's lq condition is the LR latent (z_y), since SD2.1
                  is f=8 and the LR image does NOT match the latent size.
        """
        bsz = z_start.shape[0]
        T = self.num_timesteps

        # Sample a random timestep per batch element: t in [0, T-1]
        t = torch.randint(0, T, (bsz,), device=self.device, dtype=torch.long)

        # Forward diffusion (residual shifting)
        eta_t = self._extract(self.etas, t, z_start.shape)
        sqrt_eta_t = self._extract(self.sqrt_etas, t, z_start.shape)
        e_0 = z_y - z_start  # residual LR - HR
        eps = torch.randn_like(z_start)
        x_t = z_start + eta_t * e_0 + self.kappa * sqrt_eta_t * eps

        # Normalize input and predict x0 with the (trainable) UNet.
        # lq condition = LR latent (z_y), matching x_t's latent size.
        x_t_norm = self._scale_input(x_t, t)
        x0_pred = self.denoiser(x_t_norm, t, lq=z_y)

        # MSE regression toward the clean latent
        loss = torch.nn.functional.mse_loss(x0_pred, z_start)

        loss_dict = {"mse": loss.item()}
        return loss, loss_dict

    # ------------------------------------------------------------------
    # reverse sampling (for validation inference)
    # ------------------------------------------------------------------
    def _q_posterior_mean_variance(self, x0_pred, x_t, t):
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x0_pred
        )
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)
        return mean, variance, log_variance

    @torch.no_grad()
    def reverse_sampling(self, lr_latent):
        """ResShift reverse sampling (x0-prediction).

        lq condition = lr_latent itself (the LR latent), matching x_t size.
        """
        bsz = lr_latent.shape[0]
        # prior: x_T = y + kappa * sqrt(eta_T) * eps
        t_T = torch.full((bsz,), self.num_timesteps - 1, device=self.device, dtype=torch.long)
        x_t = lr_latent + self.kappa * self._extract(self.sqrt_etas, t_T, lr_latent.shape) * torch.randn_like(lr_latent)

        for i in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((bsz,), i, device=self.device, dtype=torch.long)
            x_t_norm = self._scale_input(x_t, t)
            pred_x0 = self.denoiser(x_t_norm, t, lq=lr_latent)
            mean, _, log_variance = self._q_posterior_mean_variance(pred_x0, x_t, t)
            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_variance) * noise
            else:
                x_t = mean
        return x_t

    # ------------------------------------------------------------------
    # validation (RFMSR style)
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

    @torch.no_grad()
    def validate(self, step):
        """Validate: inference on validation LQ/GT pairs -> metrics -> save SR."""
        self.denoiser.eval()

        # Swap in EMA weights for validation
        orig_state = None
        if self.ema_state is not None:
            orig_state = OrderedDict(
                {k: v.data.clone() for k, v in self.denoiser.state_dict().items()}
            )
            self.denoiser.load_state_dict(self.ema_state)

        val_cfg = self.config.get("validation", {})
        lq_dir = Path(val_cfg.get("lq_dir", "assets/validate_lq"))
        gt_dir = Path(val_cfg.get("gt_dir", "assets/validate_gt"))
        val_scale = val_cfg.get("scale", 4.0)
        max_images = val_cfg.get("max_images", 0)

        out_dir = self.exp_dir / "validation" / f"step_{step:08d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        lq_paths = sorted(lq_dir.glob("*.png")) + sorted(lq_dir.glob("*.jpg"))
        pairs = []
        for lp in lq_paths:
            gp = gt_dir / lp.name
            if gp.exists():
                pairs.append((lp, gp))
        if max_images > 0:
            pairs = pairs[:max_images]
        total = len(pairs)
        if total == 0:
            print(f"[Val @ step {step}] No paired images found")
            if orig_state is not None:
                self.denoiser.load_state_dict(orig_state)
            self.denoiser.train()
            return

        psnr_v, ssim_v, lpips_v, dists_v = [], [], [], []
        niqe_v, musiq_v, maniqa_v, clipiqa_v = [], [], [], []

        for lq_path, gt_path in tqdm(pairs, desc=f"Val@{step}", leave=False):
            from PIL import Image

            src = Image.open(lq_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            exact_w = int(src.size[0] * val_scale)
            exact_h = int(src.size[1] * val_scale)
            target = src.resize((exact_w, exact_h), Image.BICUBIC)
            ori_h, ori_w = target.size[1], target.size[0]

            # Upsampled LR -> VAE encode to latent
            im_np = np.array(target).astype(np.float32) / 255.0
            im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0).to(self.device)
            z_lr = self.vae_encode(im_cond * 2.0 - 1.0)

            # LR condition for UNet = the LR latent (z_lr), matching latent size.
            # (SD2.1 is f=8, so the LR image [128] != latent [64]; use z_lr directly.)
            sr_latent = self.reverse_sampling(z_lr)

            # VAE decode
            sr_decoded = self.vae_decode(sr_latent)
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
            self.denoiser.load_state_dict(orig_state)
        self.denoiser.train()

        # Clean up old validation dirs (keep last N)
        if self.keep_last_n > 0:
            self._cleanup_old_validation()

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def train_step(self, hr_images, lr_images):
        """Forward + backward. Loss is scaled by 1/accumulation_steps;
        the optimizer step is performed by the training loop."""
        self.denoiser.train()
        with torch.no_grad():
            hr_norm = hr_images * 2.0 - 1.0
            lr_norm = lr_images * 2.0 - 1.0
            z_start = self.vae_encode(hr_norm)

            scale = self.config["data"]["train"]["scale"]
            lr_up = torch.nn.functional.interpolate(
                lr_norm, scale_factor=scale, mode="bicubic", align_corners=False
            )
            z_y = self.vae_encode(lr_up)

        ctx = autocast(device_type="cuda") if self.use_amp else torch.amp.autocast("cuda", enabled=False)
        with ctx:
            loss, loss_dict = self.training_loss(z_start, z_y)

        # Scale loss by 1/accumulation_steps (RFMSR style)
        loss_scale = 1.0 / self.accumulation_steps
        scaled_loss = loss * loss_scale

        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}

    def train(self):
        """Iteration-based training loop with gradient accumulation (RFMSR style)."""
        self.denoiser.train()
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

        ema_mse = 0.0
        ema_cnt = 0
        accum_mse = 0.0
        accum_count = 0

        try:
            self.optimizer.zero_grad()

            while self.global_step < self.total_iters:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                hr = batch["gt"].to(self.device)
                lr = batch["lq"].to(self.device)
                ld = self.train_step(hr, lr)

                ema_mse += ld["mse"]
                ema_cnt += 1
                accum_mse += ld["mse"]
                accum_count += 1

                if accum_count >= accum_steps:
                    # Gradient clipping
                    if grad_clip > 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.denoiser.parameters(), grad_clip
                        )

                    # Optimizer step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.update_ema()

                    self.global_step += 1
                    step_avg = accum_mse / accum_count
                    pbar.set_postfix({"mse": f"{step_avg:.4f}",
                                      "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})
                    pbar.update(1)

                    accum_mse = 0.0
                    accum_count = 0

                    if self.scheduler is not None:
                        self.scheduler.step()

                    if self.global_step % self.save_freq == 0:
                        avg_mse = ema_mse / max(ema_cnt, 1)
                        print(f"\n[step {self.global_step}/{self.total_iters}]  mse={avg_mse:.6f}  "
                              f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                        ema_mse = 0.0
                        ema_cnt = 0
                        self.save_checkpoint(self.global_step)
                        if self.val_enabled:
                            try:
                                self.validate(self.global_step)
                            except Exception as e:
                                print(f"\n[WARN] Validation failed at step {self.global_step}: {e}")

            self.save_checkpoint(self.global_step)
            pbar.close()
            print("Training finished!")
        except KeyboardInterrupt:
            pbar.close()
            print("\nInterrupted, saving checkpoint ...")
            self.save_checkpoint(self.global_step)

    def save_checkpoint(self, step):
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        weights = (
            self.ema_state if self.ema_state is not None else self.denoiser.state_dict()
        )
        out_path = ckpt_dir / f"unet_step{step}.pth"
        torch.save(weights, out_path)

        full = {
            "step": step,
            "global_step": self.global_step,
            "unet": self.denoiser.state_dict(),
            "ema_state": self.ema_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "config": self.config,
        }
        full_path = ckpt_dir / f"training_state_step{step}.pth"
        torch.save(full, full_path)
        print(f"  Checkpoint saved: step {step}")

        # Clean up old checkpoints (keep last N)
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(ckpt_dir)

    def _cleanup_old_checkpoints(self, ckpt_dir):
        import re

        pattern = re.compile(r"(unet_step|training_state_step)(\d+)")
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
            print(f"  Removed old checkpoint: step {s}")

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

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.denoiser.load_state_dict(ck["unet"])
        if ck.get("ema_state") is not None:
            self.ema_state = ck["ema_state"]
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler and ck.get("scheduler"):
            self.scheduler.load_state_dict(ck["scheduler"])
        # Restore AMP scaler state if it was saved (newer checkpoints).
        if self.scaler is not None and ck.get("scaler") is not None:
            self.scaler.load_state_dict(ck["scaler"])
        self.global_step = ck["step"]
        print(f"Resumed from {path}  step={self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train the ResShift denoising UNet (x0 prediction)")
    parser.add_argument("--config", type=str, default="configs/trainer.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    trainer = ResShiftTrainer(args.config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print("\n" + "=" * 70)
    print(f"Starting ResShift UNet training for {trainer.total_iters} iterations")
    print("=" * 70 + "\n")

    trainer.train()


if __name__ == "__main__":
    main()
