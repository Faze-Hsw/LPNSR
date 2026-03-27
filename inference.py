#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Predictor Inference Script
"""

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from ldm.models.autoencoder import VQModelTorch
from models.noise_predictor import create_noise_predictor
from models.swinir_sr import SwinIRWrapper, create_swinir
from models.unet import UNetModelSwin


def get_named_eta_schedule(
    schedule_name,
    num_diffusion_timesteps,
    min_noise_level,
    etas_end=0.99,
    kappa=1.0,
    power=2.0,
):
    """
    Get the eta schedule for ResShift

    Args:
        schedule_name: Schedule type ('exponential')
        num_diffusion_timesteps: Number of diffusion steps T
        min_noise_level: Minimum noise level η_1
        etas_end: Maximum noise level η_T
        kappa: Variance control parameter κ
        power: Power for exponential schedule

    Returns:
        sqrt_etas: √η_t array, shape=(T,)
    """
    if schedule_name == "exponential":
        # Exponential schedule (ResShift default)
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


def space_timesteps(num_timesteps, sample_timesteps):
    """
    Create a list of timesteps for sampling (uniformly selected from the original diffusion process)

    Args:
        num_timesteps: Total number of steps in the original diffusion process
        sample_timesteps: Number of steps to use during sampling

    Returns:
        use_timesteps: Set of selected timesteps
    """
    all_steps = [
        int((num_timesteps / sample_timesteps) * x) for x in range(sample_timesteps)
    ]
    return set(all_steps)


class ImageSpliterTh:
    """Image patch splitting class (using Gaussian weighted aggregation)"""

    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1):
        """
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
            extra_bs: aggregate pchs to processing
        """
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs

        self.dtype = torch.float64

        bs, chn, height, width = im.shape
        self.true_bs = bs

        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = self.__len__()
        self.count_pchs = 0

        self.im_ori = im
        self.device = im.device
        # Use float64 precision for accumulation
        self.im_res = torch.zeros(
            [bs, chn, height * sf, width * sf], dtype=self.dtype, device="cpu"
        )
        self.pixel_count = torch.zeros(
            [bs, chn, height * sf, width * sf], dtype=self.dtype, device="cpu"
        )

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [
                0,
            ]
        else:
            starts = list(range(0, length, self.stride))
            for ii in range(len(starts)):
                if starts[ii] + self.pch_size > length:
                    starts[ii] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[
                self.count_pchs : self.count_pchs + self.extra_bs
            ]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_size
                h_end = h_start + self.pch_size
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch = current_pch
                else:
                    pch = torch.cat([pch, current_pch], dim=0)

                h_start *= self.sf
                h_end *= self.sf
                w_start *= self.sf
                w_end *= self.sf
                index_infos.append([h_start, h_end, w_start, w_end])

            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()

        return pch, index_infos

    @staticmethod
    def generate_kernel_1d(ksize):
        """Generate 1D Gaussian kernel"""
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # opencv default setting
        if ksize % 2 == 0:
            kernel = cv2.getGaussianKernel(
                ksize=ksize + 1, sigma=sigma, ktype=cv2.CV_64F
            )
            kernel = kernel[1:,]
        else:
            kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_64F)
        return kernel

    def get_weight(self, height, width):
        """Generate 2D Gaussian weight matrix"""
        kernel_h = self.generate_kernel_1d(height).reshape(-1, 1)
        kernel_w = self.generate_kernel_1d(width).reshape(1, -1)
        kernel = np.matmul(kernel_h, kernel_w)
        kernel = (
            torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        )  # 1 x 1 x height x width
        return kernel.to(dtype=self.dtype, device=self.im_res.device)

    def update(self, pch_res, index_infos):
        """
        Aggregate patch results using Gaussian weighting

        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
        """
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)

        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii]
            # Get current patch device
            current_device = current_pch.device
            # Generate Gaussian weight (on the same device as the patch)
            current_weight = self.get_weight(
                current_pch.shape[-2], current_pch.shape[-1]
            ).to(current_device)
            # Convert to float64 and perform weighted accumulation
            weighted_pch = (current_pch * current_weight).type(self.dtype).cpu()
            weighted_weight = current_weight.type(self.dtype).cpu()
            # Accumulate to result
            self.im_res[:, :, h_start:h_end, w_start:w_end] += weighted_pch
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += weighted_weight

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        result = self.im_res.div(self.pixel_count)
        return result.to(self.device)


class NoisePredictorInference:
    """Noise Predictor Inference Class"""

    def __init__(self, config_path, device="cuda"):
        """
        Initialize the inference engine

        Args:
            config_path: Path to the config file
            device: Device ('cuda' or 'cpu')
        """
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.project_root = Path(__file__).resolve().parent

        # Load config
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        requested_device = device or self.config.get("inference", {}).get(
            "device", "cuda"
        )
        self.device = self._resolve_device(requested_device)

        # Set random seed
        seed = self.config["inference"].get("seed", 12345)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Inference config (must be set before model initialization)
        self.num_steps = self.config["inference"]["num_steps"]
        self.scale_factor = self.config["inference"]["scale_factor"]
        self.chop_size = self.config["inference"]["chop_size"]
        self.chop_stride = self.config["inference"]["chop_stride"]
        self.chop_bs = self.config["inference"]["chop_bs"]
        self.use_amp = self.config["inference"]["use_amp"] and self.device == "cuda"
        self.use_noise_predictor = self.config["inference"].get(
            "use_noise_predictor", True
        )
        self.use_swinir = self.config["inference"].get("use_swinir", True)

        # Initialize models
        self._init_models()

        # Initialize diffusion parameters
        self._init_diffusion()

        # Color correction config (to resolve color shift after super-resolution)
        self.color_correction = self.config["inference"].get("color_correction", True)

        print("✓ Inference engine initialized")
        print(f"  - Sampling steps: {self.num_steps}")
        print(f"  - Scale factor: {self.scale_factor}x")
        print(f"  - Chop size: {self.chop_size}x{self.chop_size}")
        print(f"  - Chop stride: {self.chop_stride}")
        print(f"  - Device: {self.device}")
        print(f"  - AMP: {'enabled' if self.use_amp else 'disabled'}")
        print(
            f"  - Color correction: {'enabled' if self.color_correction else 'disabled'}"
        )
        print(
            f"  - Noise predictor: {'enabled' if self.use_noise_predictor else 'disabled'}"
        )
        print(f"  - SwinIR SR: {'enabled' if self.swinir else 'disabled'}")

    def _resolve_device(self, requested_device):
        """Resolve runtime device with safe CUDA fallback."""
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU")
            return "cpu"
        return requested_device

    def _init_models(self):
        """Initialize models"""
        print("Loading models...")

        # 1. Load VAE
        print("  Loading VAE...")
        vae_config = self.config["vae"]

        # VAE model architecture parameters (must match pretrained weights)
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

        # Get LoRA parameters from config
        lora_config = vae_config.get("lora", {})

        self.vae = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
            rank=lora_config.get("rank", 8),
            lora_alpha=lora_config.get("alpha", 1.0),
            lora_tune_decoder=lora_config.get("tune_decoder", False),
        )

        # Load pretrained weights
        vae_path = self.project_root / self.config["model"]["vae_path"]
        vae_ckpt = torch.load(vae_path, map_location="cpu")

        # Process state_dict format
        if "state_dict" in vae_ckpt:
            state_dict = vae_ckpt["state_dict"]
        else:
            state_dict = vae_ckpt

        # Smart prefix handling: detect prefix format in checkpoint
        first_key = list(state_dict.keys())[0]
        has_module_prefix = first_key.startswith("module.")
        has_orig_mod_prefix = "_orig_mod." in first_key

        # Remove or add prefixes as needed
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if has_orig_mod_prefix:
                new_key = new_key.replace("_orig_mod.", "")
            if has_module_prefix:
                new_key = new_key.replace("module.", "")
            new_state_dict[new_key] = value

        # Use strict=True to ensure all weights are loaded correctly
        missing_keys, unexpected_keys = self.vae.load_state_dict(
            new_state_dict, strict=False
        )
        if missing_keys:
            print(f"  ⚠️ VAE missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  ⚠️ VAE unexpected keys: {unexpected_keys[:5]}...")  # Show first 5

        self.vae = self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("  ✓ VAE loaded")

        # 2. Load ResShift UNet
        print("  Loading ResShift UNet...")
        unet_config = self.config["resshift_unet"]
        self.resshift_unet = UNetModelSwin(**unet_config)

        # Load pretrained weights
        resshift_path = self.project_root / self.config["model"]["resshift_path"]
        resshift_ckpt = torch.load(resshift_path, map_location="cpu")

        # Process state_dict format
        if "state_dict" in resshift_ckpt:
            state_dict = resshift_ckpt["state_dict"]
        else:
            state_dict = resshift_ckpt

        # Remove possible prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("module._orig_mod."):
                new_key = key.replace("module._orig_mod.", "")
            elif key.startswith("module."):
                new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        self.resshift_unet.load_state_dict(new_state_dict, strict=True)
        self.resshift_unet = self.resshift_unet.to(self.device)
        self.resshift_unet.eval()
        for param in self.resshift_unet.parameters():
            param.requires_grad = False
        print("  ✓ ResShift UNet loaded")

        # 3. Load noise predictor
        print("  Loading noise predictor...")
        noise_predictor_config = self.config["noise_predictor"]

        # Load config if config file path is specified
        if "config_path" in noise_predictor_config:
            noise_predictor_config_path = (
                self.project_root / noise_predictor_config["config_path"]
            )
            with open(noise_predictor_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.noise_predictor = create_noise_predictor(
                image_size=config.get("image_size", 64),
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
                lq_size=config.get("lq_size", 64),
            )
        else:
            self.noise_predictor = create_noise_predictor(
                image_size=noise_predictor_config.get("image_size", 64),
                latent_channels=noise_predictor_config["latent_channels"],
                model_channels=noise_predictor_config["model_channels"],
                out_channels=noise_predictor_config.get(
                    "out_channels", noise_predictor_config["latent_channels"]
                ),
                channel_mult=tuple(noise_predictor_config["channel_mult"]),
                num_res_blocks=noise_predictor_config["num_res_blocks"],
                attention_resolutions=noise_predictor_config.get(
                    "attention_resolutions", [64, 32, 16, 8]
                ),
                dropout=noise_predictor_config.get("dropout", 0.0),
                conv_resample=noise_predictor_config.get("conv_resample", True),
                dims=noise_predictor_config.get("dims", 2),
                use_fp16=noise_predictor_config.get("use_fp16", False),
                num_heads=noise_predictor_config.get("num_heads", -1),
                num_head_channels=noise_predictor_config.get("num_head_channels", 32),
                use_scale_shift_norm=noise_predictor_config.get(
                    "use_scale_shift_norm", True
                ),
                resblock_updown=noise_predictor_config.get("resblock_updown", False),
                swin_depth=noise_predictor_config.get("swin_depth", 2),
                swin_embed_dim=noise_predictor_config.get("swin_embed_dim", 192),
                window_size=noise_predictor_config.get("window_size", 8),
                mlp_ratio=noise_predictor_config.get("mlp_ratio", 4.0),
                patch_norm=noise_predictor_config.get("patch_norm", False),
                cond_lq=noise_predictor_config.get("cond_lq", True),
                lq_size=noise_predictor_config.get("lq_size", 64),
            )

        # Load weights
        noise_predictor_path = (
            self.project_root / self.config["model"]["noise_predictor_path"]
        )
        noise_ckpt = torch.load(noise_predictor_path, map_location="cpu")
        state_dict = noise_ckpt
        print(f" Loading from {noise_predictor_path.name} (weights only)")

        self.noise_predictor.load_state_dict(state_dict, strict=True)
        self.noise_predictor = self.noise_predictor.to(self.device)
        self.noise_predictor.eval()
        for param in self.noise_predictor.parameters():
            param.requires_grad = False
        print("  ✓ Noise predictor loaded")

        # 4. Load SwinIR SR model (optional)
        self.swinir = None
        if self.config["inference"].get("use_swinir", False):
            print("  Loading SwinIR SR model...")
            swinir_config = self.config["inference"].get("swinir", {})

            # SwinIR model path
            swinir_path_str = swinir_config.get(
                "model_path",
                self.config["model"].get(
                    "swinir_path",
                    "pretrained/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
                ),
            )
            swinir_model_path = self.project_root / swinir_path_str

            # Create SwinIR model
            swinir_model = create_swinir(
                upscale=self.scale_factor,
                img_size=swinir_config.get("img_size", 64),
                window_size=swinir_config.get("window_size", 8),
                img_range=1.0,
                depths=swinir_config.get("depths", [6, 6, 6, 6, 6, 6]),
                embed_dim=swinir_config.get("embed_dim", 180),
                num_heads=swinir_config.get("num_heads", [6, 6, 6, 6, 6, 6]),
                mlp_ratio=swinir_config.get("mlp_ratio", 2),
                upsampler=swinir_config.get("upsampler", "nearest+conv"),
                resi_connection=swinir_config.get("resi_connection", "1conv"),
                model_path=str(swinir_model_path),
                device=self.device,
            )

            # Use wrapper to handle data range conversion
            self.swinir = SwinIRWrapper(swinir_model)
            print("  ✓ SwinIR SR model loaded")
        else:
            print("  SwinIR SR model not enabled")

    def _init_diffusion(self):
        """
        Initialize diffusion parameters

        ResShift v3 is trained directly with steps=4, timestep_respacing=None
        So no timestep remapping is needed, directly use num_steps diffusion parameters
        """
        diffusion_config = self.config["diffusion"]

        # Diffusion steps (ResShift v3 trained directly with num_steps)
        self.diffusion_num_timesteps = diffusion_config[
            "num_timesteps"
        ]  # Should equal num_steps
        self.kappa = diffusion_config["kappa"]
        self.normalize_input = diffusion_config.get("normalize_input", True)
        self.latent_flag = diffusion_config.get("latent_flag", True)

        # Calculate eta schedule (directly with num_timesteps steps)
        sqrt_etas = get_named_eta_schedule(
            schedule_name=diffusion_config["eta_schedule"],
            num_diffusion_timesteps=self.diffusion_num_timesteps,
            min_noise_level=diffusion_config["min_noise_level"],
            etas_end=diffusion_config["etas_end"],
            kappa=self.kappa,
            power=diffusion_config["eta_power"],
        )

        # Use sqrt_etas directly (length=num_timesteps)
        self.sqrt_etas = sqrt_etas.astype(np.float64)
        self.etas = self.sqrt_etas**2

        # Calculate alpha (ResShift definition: alpha_t = eta_t - eta_{t-1})
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev  # This is the alpha in ResShift!

        # Calculate posterior distribution parameters
        self.posterior_variance = (
            self.kappa**2 * self.etas_prev / self.etas * self.alpha
        )
        # Handle variance at t=0 (avoid division by zero and NaN)
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)

        # Posterior mean coefficients
        self.posterior_mean_coef1 = self.etas_prev / self.etas  # η_{t-1}/η_t
        self.posterior_mean_coef2 = self.alpha / self.etas  # α_t/η_t
        # Handle division by zero at t=0
        self.posterior_mean_coef1[0] = 0.0
        self.posterior_mean_coef2[0] = 1.0  # When t=0, posterior mean is directly x_0

        # Convert to tensors
        self.sqrt_etas = torch.from_numpy(self.sqrt_etas).float()
        self.etas = torch.from_numpy(self.etas).float()
        self.etas_prev = torch.from_numpy(self.etas_prev).float()
        self.alpha = torch.from_numpy(self.alpha).float()
        self.posterior_variance = torch.from_numpy(self.posterior_variance).float()
        self.posterior_variance_clipped = torch.from_numpy(
            self.posterior_variance_clipped
        ).float()
        self.posterior_log_variance_clipped = torch.from_numpy(
            self.posterior_log_variance_clipped
        ).float()
        self.posterior_mean_coef1 = torch.from_numpy(self.posterior_mean_coef1).float()
        self.posterior_mean_coef2 = torch.from_numpy(self.posterior_mean_coef2).float()

        print("  ✓ Diffusion parameters initialized")

    def _wavelet_blur(self, image: torch.Tensor, radius: int):
        """
        Apply wavelet blur to input tensor

        Args:
            image: Input image tensor (B, C, H, W)
            radius: Blur radius

        Returns:
            Blurred tensor
        """
        # Convolution kernel - Gaussian blur kernel
        kernel_vals = [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ]
        kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
        # Add channel dimension, becomes 4D tensor
        kernel = kernel[None, None]
        # Repeat across all input channels
        kernel = kernel.repeat(3, 1, 1, 1)
        # Use replicate mode for padding
        image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
        # Apply grouped convolution
        output = F.conv2d(image, kernel, groups=3, dilation=radius)
        return output

    def _wavelet_decomposition(self, image: torch.Tensor, levels: int = 5):
        """
        Perform wavelet decomposition on input tensor

        Args:
            image: Input image tensor (B, C, H, W)
            levels: Number of decomposition levels

        Returns:
            high_freq: High frequency component (detail information)
            low_freq: Low frequency component (color/brightness information)
        """
        high_freq = torch.zeros_like(image)
        for i in range(levels):
            radius = 2**i
            low_freq = self._wavelet_blur(image, radius)
            high_freq += image - low_freq
            image = low_freq

        return high_freq, low_freq

    def _color_correction(self, sr_tensor, lr_tensor):
        """
        Color correction: Use wavelet reconstruction to correct SR image color shift

        Principle:
        - Perform wavelet decomposition on SR image to extract high frequency component (texture, edges, etc.)
        - Perform wavelet decomposition on LR image to extract low frequency component (overall color, brightness)
        - Reconstruct using SR high frequency + LR low frequency to preserve SR details while correcting colors

        Args:
            sr_tensor: SR image tensor (B, C, H, W), [-1, 1]
            lr_tensor: LR image tensor (B, C, H, W), [-1, 1]

        Returns:
            Color corrected SR image tensor
        """
        # Convert range from [-1, 1] to [0, 1] for wavelet processing
        sr_01 = (sr_tensor + 1.0) / 2.0
        lr_01 = (lr_tensor + 1.0) / 2.0

        # Wavelet decomposition on SR image, extract high frequency component (detail information)
        sr_high_freq, _ = self._wavelet_decomposition(sr_01)

        # Wavelet decomposition on LR image, extract low frequency component (color information)
        _, lr_low_freq = self._wavelet_decomposition(lr_01)

        # Reconstruction: SR high frequency (details) + LR low frequency (color)
        corrected_01 = sr_high_freq + lr_low_freq

        # Clamp to [0, 1] range
        corrected_01 = torch.clamp(corrected_01, 0.0, 1.0)

        # Convert back to [-1, 1] range
        corrected = corrected_01 * 2.0 - 1.0

        return corrected

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from array and broadcast to target shape

        Args:
            arr: 1D tensor array
            timesteps: Timestep indices
            broadcast_shape: Target shape

        Returns:
            Broadcasted tensor
        """
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _scale_input(self, inputs, t):
        """
        Normalize input (key step in ResShift!)

        Args:
            inputs: Input tensor
            t: Timestep index (in remapped space)
        """
        if self.normalize_input:
            if self.latent_flag:
                # Latent space variance is approximately 1.0
                std = torch.sqrt(
                    self._extract_into_tensor(self.etas, t, inputs.shape)
                    * self.kappa**2
                    + 1
                )
                inputs_norm = inputs / std
            else:
                inputs_max = (
                    self._extract_into_tensor(self.sqrt_etas, t, inputs.shape)
                    * self.kappa
                    * 3
                    + 1
                )
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Calculate ResShift posterior distribution q(x_{t-1}|x_t, x_0)

        Posterior mean: μ = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0
        Posterior variance: σ² = κ²·η_{t-1}·α_t/η_t

        Where α_t = η_t - η_{t-1} (ResShift definition)

        Args:
            x_0: Predicted x_0 (pred_xstart)
            x_t: Current x_t
            t: Timestep index (in remapped space, 0 to num_steps-1)

        Returns:
            mean: Posterior mean
            variance: Posterior variance
            log_variance: Log of posterior variance (clipped)
        """
        # Use precomputed coefficients
        mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_0
        )

        variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return mean, variance, log_variance

    @torch.no_grad()
    def sample_func(self, lr_tensor):
        """
        Complete super-resolution pipeline (corresponding to ResShift's sample_func)

        Args:
            lr_tensor: LR image (B, C, H, W), [-1, 1], RGB

        Returns:
            sr_tensor: SR image (B, C, H*sf, W*sf), [-1, 1], RGB
        """

        # 1. Upsample LR image
        if self.swinir is not None and self.use_swinir:
            # Use SwinIR for super-resolution
            lr_upsampled = self.swinir(lr_tensor)
        else:
            # Use bicubic interpolation
            lr_upsampled = F.interpolate(
                lr_tensor,
                scale_factor=self.scale_factor,
                mode="bicubic",
                align_corners=False,
            )

        # 2. Encode to latent space
        with torch.no_grad():
            lr_latent = self.vae.encode(lr_upsampled)

        # 3. Reverse sampling
        # Note: UNet's lq condition requires image-space LR, not latent-space
        sr_latent = self.reverse_sampling(lr_latent, lr_tensor)

        # 4. Decode to image space
        with torch.no_grad():
            sr_tensor = self.vae.decode(sr_latent)

        # 5. Clamp to valid range to prevent color overflow
        sr_tensor = torch.clamp(sr_tensor, -1.0, 1.0)

        return sr_tensor

    def prior_sample(self, y, noise=None):
        """
        Sample from prior distribution, i.e., q(x_T|y) ~= N(x_T|y, κ²η_T)

        Args:
            y: Degraded image latent representation (lr_latent)
            noise: Optional noise

        Returns:
            x_T: Initial sample
        """
        # Use last timestep (i.e., num_steps-1, corresponding to original max timestep)
        t = torch.tensor([self.num_steps - 1] * y.shape[0], device=self.device).long()

        # Use random Gaussian noise (original ResShift)
        noise = torch.randn_like(y)

        return (
            y
            + self._extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise
        )

    def reverse_sampling(self, lr_latent, lr_image):
        """
        ResShift reverse sampling process (used during inference)

        Fully follows the p_sample_loop_progressive implementation from the original ResShift project

        Note: ResShift v3 is trained directly with 4 steps (timestep_respacing=None),
        so no timestep remapping is needed, directly use indices 0-3 as timesteps

        Args:
            lr_latent: LR image latent representation y (already VAE encoded)
            lr_image: Image-space LR image (used as UNet's lq condition)

        Returns:
            x_0: Final SR latent representation
        """
        # Use prior_sample to initialize x_T = y + κ·√η_T·ε
        x_t = self.prior_sample(lr_latent)

        # Reverse sampling: from num_steps-1 to 0
        indices = list(range(self.num_steps))[
            ::-1
        ]  # [num_steps-1, num_steps-2, ..., 0]

        for i in indices:
            # Timestep index (0 to num_steps-1)
            t_tensor = torch.tensor([i] * lr_latent.shape[0], device=self.device).long()

            # 1. Normalize input
            x_t_normalized = self._scale_input(x_t, t_tensor)

            # 2. Use ResShift's UNet to predict x_0
            # ResShift v3 is trained directly with 4 steps, so timestep is directly passed as i
            pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)

            # 3. Calculate ResShift posterior distribution
            mean, variance, log_variance = self.q_posterior_mean_variance(
                pred_x0, x_t, t_tensor
            )

            # 4. Generate noise
            # Choose noise source for intermediate sampling based on use_noise_predictor
            if self.use_noise_predictor:
                # Use noise predictor to predict noise
                noise = self.noise_predictor(
                    x_t, pred_x0, lr_image, t_tensor, sample_posterior=True
                )
            else:
                # Use random Gaussian noise
                noise = torch.randn_like(x_t)

            # 5. Sample x_{t-1}: add noise when t>0, use mean directly when t=0
            nonzero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
            x_t = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

        return x_t

    def pad_image(self, img, multiple=64):
        """
        Pad image to a multiple of the specified value

        Args:
            img: Input image (H, W, C) numpy array
            multiple: Multiple value

        Returns:
            padded_img: Padded image
            (pad_h, pad_w): Padding size
        """
        h, w = img.shape[:2]

        # Calculate required padding size
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            # Use reflect padding
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        return img, (pad_h, pad_w)

    def process_single_image(self, lr_image):
        """
        Process a single image (fully follows ResShift pipeline)

        Args:
            lr_image: LR image (H, W, C) numpy array, [0, 1], RGB

        Returns:
            sr_image: SR image (H*4, W*4, C) numpy array, [0, 1], RGB
        """
        # 1. Pad image
        padding_offset = self.config["inference"].get("padding_offset", 64)

        # First pad to multiple of 64 (consistent with ResShift)
        lr_padded, (pad_h, pad_w) = self.pad_image(lr_image, multiple=padding_offset)

        # Ensure power of 2 (for FFT operations)
        h, w = lr_padded.shape[:2]
        # Find smallest power of 2 >= h and w
        next_pow2_h = 1 if h == 0 else 2 ** ((h - 1).bit_length())
        next_pow2_w = 1 if w == 0 else 2 ** ((w - 1).bit_length())

        if h != next_pow2_h or w != next_pow2_w:
            # Need extra padding to power of 2
            extra_pad_h = next_pow2_h - h
            extra_pad_w = next_pow2_w - w
            lr_padded = np.pad(
                lr_padded, ((0, extra_pad_h), (0, extra_pad_w), (0, 0)), mode="reflect"
            )
            pad_h += extra_pad_h
            pad_w += extra_pad_w

        # 2. Convert to tensor
        lr_tensor = (
            torch.from_numpy(lr_padded).permute(2, 0, 1).unsqueeze(0).float()
        )  # 1 x C x H x W
        lr_tensor = lr_tensor.to(self.device)

        # 3. Normalize to [-1, 1]
        lr_tensor = lr_tensor * 2.0 - 1.0

        # 4. Determine if chop is needed
        context = lambda: torch.amp.autocast("cuda") if self.use_amp else nullcontext()

        if lr_tensor.shape[2] > self.chop_size or lr_tensor.shape[3] > self.chop_size:
            # Use chop for large images
            print(
                f"  Using chop processing (image space size: {lr_tensor.shape[3]}x{lr_tensor.shape[2]})"
            )

            im_spliter = ImageSpliterTh(
                lr_tensor,
                self.chop_size,
                stride=self.chop_stride,
                sf=self.scale_factor,  # Scale factor
                extra_bs=self.chop_bs,
            )

            for lr_pch, index_infos in im_spliter:
                with context():
                    # Complete SR pipeline for each patch
                    sr_pch = self.sample_func(lr_pch)
                im_spliter.update(sr_pch, index_infos)

            sr_tensor = im_spliter.gather()
        else:
            # Direct processing
            print(
                f"  Direct processing (image space size: {lr_tensor.shape[3]}x{lr_tensor.shape[2]})"
            )
            with context():
                sr_tensor = self.sample_func(lr_tensor)

        # 5. Denormalize to [0, 1] (sr_tensor is already clamped to [-1,1] in sample_func)
        sr_tensor = sr_tensor * 0.5 + 0.5

        # Apply color correction
        if self.color_correction:
            lr_upsampled_full = F.interpolate(
                lr_tensor,
                scale_factor=self.scale_factor,
                mode="bicubic",
                align_corners=False,
            )
            lr_upsampled_full = lr_upsampled_full * 0.5 + 0.5
            sr_tensor = self._color_correction(sr_tensor, lr_upsampled_full)
            print("  ✓ Applied color correction to image")

        # Extra clamp to ensure [0, 1] range
        sr_tensor = torch.clamp(sr_tensor, 0, 1)

        # 6. Convert to numpy
        sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 7. Remove padding
        if pad_h > 0 or pad_w > 0:
            h_end = sr_image.shape[0] - pad_h * self.scale_factor
            w_end = sr_image.shape[1] - pad_w * self.scale_factor
            sr_image = sr_image[:h_end, :w_end]

        return sr_image

    def inference(self, input_path, output_path):
        """
        Inference entry point

        Args:
            input_path: Input path (image or folder)
            output_path: Output path
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Get image list
        if input_path.is_file():
            image_paths = [input_path]
        else:
            image_paths = []
            for ext in [
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.bmp",
                "*.PNG",
                "*.JPG",
                "*.JPEG",
            ]:
                image_paths.extend(input_path.glob(ext))
            image_paths = sorted(image_paths)
        image_paths = list(set(image_paths))
        print(f"\nFound {len(image_paths)} images")

        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            # Read image
            lr_image = cv2.imread(str(img_path))
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
            lr_image = lr_image.astype(np.float32) / 255.0

            print(
                f"\nProcessing: {img_path.name} (size: {lr_image.shape[1]}x{lr_image.shape[0]})"
            )

            # Super-resolution
            sr_image = self.process_single_image(lr_image)

            # Save result
            sr_image = (sr_image * 255.0).astype(np.uint8)
            if self.config["inference"]["rgb2bgr"]:
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)

            output_file = output_path / f"{img_path.stem}_sr.png"
            cv2.imwrite(str(output_file), sr_image)

            print(f"  ✓ Saved to: {output_file}")

        print(f"\n✓ All done! Results saved in: {output_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="Noise Predictor Inference Script")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input path (image or folder)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./results", help="Output path"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of sampling steps (overrides config file)",
    )
    parser.add_argument(
        "--disable_noise_predictor",
        action="store_true",
        help="Disable noise predictor, use random noise (original ResShift method)",
    )
    parser.add_argument(
        "--disable_swinir", action="store_true", help="Disable SwinIR SR model"
    )

    return parser


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 60)
    print("Noise Predictor Inference Script")
    print("=" * 60)

    # Initialize inference engine
    inferencer = NoisePredictorInference(args.config, device=args.device)

    # Override sampling steps
    if args.num_steps is not None:
        inferencer.num_steps = args.num_steps
        print(f"Sampling steps overridden to: {inferencer.num_steps}")

    # Override noise mode
    if args.disable_noise_predictor:
        inferencer.use_noise_predictor = False

    if args.disable_swinir:
        inferencer.use_swinir = False

    # Print final inference strategy
    print("\nInference strategy:")
    print(
        f"  - Intermediate sampling: {'Noise predictor' if inferencer.use_noise_predictor else 'Random Gaussian noise'}"
    )
    print(
        f"  - Upsampling: {'SwinIR' if inferencer.use_swinir else 'Bicubic interpolation'}"
    )

    # Execute inference
    inferencer.inference(args.input, args.output)


if __name__ == "__main__":
    main()
