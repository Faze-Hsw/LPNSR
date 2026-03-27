"""
CLIP-IQA (CLIP-based Image Quality Assessment) CLIP-based Image Quality Assessment

No-reference image quality assessment metric, using CLIP model's
vision-language alignment capability to assess image quality.
Higher scores indicate better image quality.
"""

import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn


def calculate_clipiqa(
    img: Union[np.ndarray, torch.Tensor],
    input_order: str = "HWC",
    device: str = None,
    **kwargs,
) -> float:
    """
    Calculate CLIP-IQA score of image

    Args:
        img: Input image
        input_order: Dimension order of input image
        device: Computation device

    Returns:
        CLIP-IQA score (higher is better, range [0, 1])
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import pyiqa

        clipiqa_metric = pyiqa.create_metric("clipiqa", device=device)

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
            score = clipiqa_metric(img)

        return float(score.item())

    except ImportError:
        warnings.warn("pyiqa包未安装，CLIP-IQA需要pyiqa支持。请安装: pip install pyiqa")
        return 0.0
    except Exception as e:
        warnings.warn(f"CLIP-IQA计算失败: {e}")
        return 0.0


class CLIPIQA(nn.Module):
    """CLIP-IQA计算类"""

    def __init__(self, input_order: str = "HWC", device: str = None):
        super().__init__()
        self.input_order = input_order
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.pyiqa_model = None
        try:
            import pyiqa

            self.pyiqa_model = pyiqa.create_metric("clipiqa", device=self.device)
        except ImportError:
            warnings.warn(
                "pyiqa package is not installed, please install: pip install pyiqa"
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize CLIP-IQA: {e}")

    def forward(self, img: Union[np.ndarray, torch.Tensor]) -> float:
        if self.pyiqa_model is None:
            warnings.warn("CLIP-IQA model is not initialized")
            return 0.0

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
