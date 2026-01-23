"""
LPIPS (Learned Perceptual Image Patch Similarity) 学习感知图像块相似度

全参考图像质量评估指标，使用预训练深度网络提取特征来衡量感知相似度。
LPIPS值越低表示感知质量越相似。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union
import warnings


class LPIPS(nn.Module):
    """LPIPS计算类"""
    
    def __init__(
        self,
        net: str = 'alex',
        use_gpu: bool = True,
        spatial: bool = False
    ):
        super().__init__()
        self.net_type = net
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.spatial = spatial
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """初始化LPIPS模型"""
        try:
            import lpips
            self.model = lpips.LPIPS(net=self.net_type, spatial=self.spatial)
            if self.use_gpu:
                self.model = self.model.cuda()
            self.model.eval()
        except ImportError:
            warnings.warn("lpips包未安装，请使用 'pip install lpips' 安装")
            self.model = None
    
    def forward(
        self,
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> float:
        if self.model is None:
            warnings.warn("LPIPS模型未初始化")
            return 0.0
        
        # 转换为张量
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
        
        # 归一化到[0, 1]
        if img1.max() > 5.0:
            img1 = img1 / 255.0
        if img2.max() > 5.0:
            img2 = img2 / 255.0
        
        # 归一化到[-1, 1]
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
        return f'{self.__class__.__name__}(net={self.net_type})'


def calculate_lpips(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    net: str = 'alex',
    use_gpu: bool = True
) -> float:
    """计算两张图像之间的LPIPS值"""
    lpips_model = LPIPS(net=net, use_gpu=use_gpu)
    return lpips_model(img1, img2)
