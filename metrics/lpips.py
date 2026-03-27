"""
LPIPS (Learned Perceptual Image Patch Similarity) Learned Perceptual Image Patch Similarity

Full-reference image quality assessment metric, using pre-trained deep networks
to extract features for measuring perceptual similarity.
Lower LPIPS values indicate more similar perceptual quality.
"""

import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class LPIPS(nn.Module):
    """LPIPS calculation class"""

    def __init__(self, net: str = "alex", use_gpu: bool = True, spatial: bool = False):
        super().__init__()
        self.net_type = net
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.spatial = spatial
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize LPIPS model"""
        try:
            import lpips

            self.model = lpips.LPIPS(net=self.net_type, spatial=self.spatial)
            if self.use_gpu:
                self.model = self.model.cuda()
            self.model.eval()
        except ImportError:
            warnings.warn(
                "lpips package is not installed, please use 'pip install lpips' to install"
            )
            self.model = None

    def forward(
        self,
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> float:
        if self.model is None:
            warnings.warn("LPIPS model is not initialized")
            return 0.0

        # Convert to tensor
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float()
            if img1.ndim == 3:
                img1 = img1.permute(2, 0, 1).unsqueeze(0)
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float()
            if img2.ndim == 3:
                img2 = img2.permute(2, 0, 1).unsqueeze(0)

        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)

        # Normalize to [0, 1]
        if img1.max() > 5.0:
            img1 = img1 / 255.0
        if img2.max() > 5.0:
            img2 = img2 / 255.0

        # Normalize to [-1, 1]
        if normalize:
            img1 = img1 * 2.0 - 1.0
            img2 = img2 * 2.0 - 1.0

        if self.use_gpu:
            img1 = img1.cuda()
            img2 = img2.cuda()

        with torch.no_grad():
            lpips_val = self.model(img1, img2)

        return float(lpips_val.mean().item())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(net={self.net_type})"


def calculate_lpips(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    net: str = "alex",
    use_gpu: bool = True,
) -> float:
    """Calculate LPIPS value between two images"""
    lpips_model = LPIPS(net=net, use_gpu=use_gpu)
    return lpips_model(img1, img2)
