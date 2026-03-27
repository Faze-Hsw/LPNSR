"""
SSIM (Structural Similarity Index) Structural Similarity Index

Full-reference image quality assessment metric, comprehensively considering
image luminance, contrast, and structural information.
SSIM value range is [-1, 1], higher values indicate more similar images.
"""

from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from .metric_utils import reorder_image, to_y_channel


def _ssim_single_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
    data_range: float = 255.0,
) -> float:
    """Calculate SSIM value for single-channel image using Gaussian weighting"""
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Create Gaussian kernel with sigma=1.5 (standard implementation)
    kernel = cv2.getGaussianKernel(win_size, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[
        win_size // 2 : -win_size // 2, win_size // 2 : -win_size // 2
    ]
    mu2 = cv2.filter2D(img2, -1, window)[
        win_size // 2 : -win_size // 2, win_size // 2 : -win_size // 2
    ]

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        cv2.filter2D(img1**2, -1, window)[
            win_size // 2 : -win_size // 2, win_size // 2 : -win_size // 2
        ]
        - mu1_sq
    )
    sigma2_sq = (
        cv2.filter2D(img2**2, -1, window)[
            win_size // 2 : -win_size // 2, win_size // 2 : -win_size // 2
        ]
        - mu2_sq
    )
    sigma12 = (
        cv2.filter2D(img1 * img2, -1, window)[
            win_size // 2 : -win_size // 2, win_size // 2 : -win_size // 2
        ]
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return float(ssim_map.mean())


def calculate_ssim(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    crop_border: int = 0,
    input_order: str = "HWC",
    test_y_channel: bool = False,
    data_range: float = 255.0,
    win_size: int = 11,
) -> float:
    """
    Calculate SSIM value between two images

    Args:
        img1: First image
        img2: Second image
        crop_border: Number of border pixels to crop before calculation
        input_order: Dimension order of input images
        test_y_channel: Whether to test only on Y channel
        data_range: Image data range
        win_size: Sliding window size

    Returns:
        SSIM value
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    if img1.ndim == 2:
        ssim_val = _ssim_single_channel(
            img1, img2, win_size=win_size, data_range=data_range
        )
    elif img1.ndim == 3:
        ssim_vals = []
        for i in range(img1.shape[2]):
            ssim_val = _ssim_single_channel(
                img1[..., i], img2[..., i], win_size=win_size, data_range=data_range
            )
            ssim_vals.append(ssim_val)
        ssim_val = np.mean(ssim_vals)
    else:
        raise ValueError(f"Wrong image dimension: {img1.ndim}")

    return float(ssim_val)


class SSIM(nn.Module):
    """SSIM calculation class"""

    def __init__(
        self,
        crop_border: int = 0,
        input_order: str = "HWC",
        test_y_channel: bool = False,
        data_range: float = 255.0,
        window_size: int = 11,
    ):
        super().__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel
        self.data_range = data_range
        self.window_size = window_size

    def forward(
        self,
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor],
    ) -> float:
        return calculate_ssim(
            img1,
            img2,
            crop_border=self.crop_border,
            input_order=self.input_order,
            test_y_channel=self.test_y_channel,
            data_range=self.data_range,
            win_size=self.window_size,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"crop_border={self.crop_border}, "
            f"test_y_channel={self.test_y_channel})"
        )
