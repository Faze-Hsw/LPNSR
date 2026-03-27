"""
SR Project Image Quality Assessment Metrics Module

Includes the following metrics:
|- PSNR: Peak Signal-to-Noise Ratio (full-reference metric)
|- SSIM: Structural Similarity Index (full-reference metric)
|- LPIPS: Learned Perceptual Image Patch Similarity (full-reference metric)
|- NIQE: Natural Image Quality Evaluator (no-reference metric)
|- PI: Perceptual Index (no-reference metric)
|- CLIPIQA: CLIP-based Image Quality Assessment (no-reference metric)
|- MUSIQ: Multi-Scale Image Quality Transformer (no-reference metric)
"""

from .clipiqa import CLIPIQA, calculate_clipiqa
from .lpips import LPIPS, calculate_lpips
from .metric_utils import (
    bgr2ycbcr,
    img2tensor,
    reorder_image,
    rgb2ycbcr,
    tensor2img,
    to_y_channel,
)
from .musiq import MUSIQ, calculate_musiq
from .niqe import NIQE, calculate_niqe
from .pi import PI, calculate_pi
from .psnr import PSNR, calculate_psnr
from .ssim import SSIM, calculate_ssim

__all__ = [
    # PSNR
    "calculate_psnr",
    "PSNR",
    # SSIM
    "calculate_ssim",
    "SSIM",
    # LPIPS
    "calculate_lpips",
    "LPIPS",
    # NIQE
    "calculate_niqe",
    "NIQE",
    # PI
    "calculate_pi",
    "PI",
    # CLIPIQA
    "calculate_clipiqa",
    "CLIPIQA",
    # MUSIQ
    "calculate_musiq",
    "MUSIQ",
    # Utility functions
    "img2tensor",
    "tensor2img",
    "rgb2ycbcr",
    "bgr2ycbcr",
    "to_y_channel",
    "reorder_image",
]
