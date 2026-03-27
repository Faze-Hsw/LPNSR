"""
PSNR (Peak Signal-to-Noise Ratio) Peak Signal-to-Noise Ratio

Full-reference image quality assessment metric, used to measure image reconstruction quality.
Higher PSNR values indicate better image quality, usually expressed in dB.
"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .metric_utils import reorder_image, to_y_channel


def calculate_psnr(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    crop_border: int = 0,
    input_order: str = "HWC",
    test_y_channel: bool = False,
    data_range: float = 255.0,
) -> float:
    """
    Calculate PSNR value between two images

    Args:
        img1: First image (usually reconstructed/SR image)
        img2: Second image (usually reference/GT image)
        crop_border: Number of border pixels to crop before calculation
        input_order: Dimension order of input images, 'HWC' or 'CHW'
        test_y_channel: Whether to test only on Y channel (for color images)
        data_range: Image data range, default 255

    Returns:
        PSNR value (dB)
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

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 10.0 * np.log10((data_range**2) / mse)

    return float(psnr)


class PSNR(nn.Module):
    """PSNR calculation class"""

    def __init__(
        self,
        crop_border: int = 0,
        input_order: str = "HWC",
        test_y_channel: bool = False,
        data_range: float = 255.0,
    ):
        super().__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel
        self.data_range = data_range

    def forward(
        self,
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor],
    ) -> float:
        return calculate_psnr(
            img1,
            img2,
            crop_border=self.crop_border,
            input_order=self.input_order,
            test_y_channel=self.test_y_channel,
            data_range=self.data_range,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"crop_border={self.crop_border}, "
            f"test_y_channel={self.test_y_channel})"
        )
