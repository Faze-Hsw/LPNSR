"""
NIQE (Natural Image Quality Evaluator) Natural Image Quality Evaluator

No-reference image quality assessment metric, based on natural scene statistical properties.
Lower NIQE values indicate better image quality.
"""

import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn


def calculate_niqe(
    img: Union[np.ndarray, torch.Tensor],
    crop_border: int = 0,
    input_order: str = "HWC",
    convert_to_gray: bool = True,
    **kwargs,
) -> float:
    """
    Calculate NIQE score of image

    Args:
        img: Input image
        crop_border: Number of border pixels to crop
        input_order: Dimension order of input image
        convert_to_gray: Whether to convert to grayscale

    Returns:
        NIQE score (lower is better)
    """
    # Prefer using pyiqa library
    try:
        import pyiqa

        device = "cuda" if torch.cuda.is_available() else "cpu"
        niqe_metric = pyiqa.create_metric("niqe", device=device)

        if isinstance(img, np.ndarray):
            if img.ndim == 3 and input_order == "HWC":
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            if img.max() > 5.0:
                img = img / 255.0

        if img.ndim == 3:
            img = img.unsqueeze(0)

        img = img.to(device)

        with torch.no_grad():
            score = niqe_metric(img)

        return float(score.item())

    except ImportError:
        warnings.warn("pyiqa包未安装，NIQE需要pyiqa支持。请安装: pip install pyiqa")
        return 0.0


class NIQE(nn.Module):
    """NIQE计算类"""

    def __init__(
        self,
        crop_border: int = 0,
        input_order: str = "HWC",
        convert_to_gray: bool = True,
        device: str = None,
    ):
        super().__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to_gray = convert_to_gray
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.pyiqa_model = None
        try:
            import pyiqa

            self.pyiqa_model = pyiqa.create_metric("niqe", device=self.device)
        except ImportError:
            warnings.warn(
                "pyiqa package is not installed, please install: pip install pyiqa"
            )

    def forward(self, img: Union[np.ndarray, torch.Tensor]) -> float:
        if self.pyiqa_model is not None:
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and self.input_order == "HWC":
                    img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).float()
                if img.max() > 5.0:
                    img = img / 255.0

            if img.ndim == 3:
                img = img.unsqueeze(0)

            img = img.to(self.device)

            with torch.no_grad():
                score = self.pyiqa_model(img)

            return float(score.item())
        else:
            return calculate_niqe(
                img,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to_gray=self.convert_to_gray,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
