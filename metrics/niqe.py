"""
NIQE (Natural Image Quality Evaluator) 自然图像质量评估器

无参考图像质量评估指标，基于自然场景统计特性。
NIQE值越低表示图像质量越好。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union
import warnings


def calculate_niqe(
    img: Union[np.ndarray, torch.Tensor],
    crop_border: int = 0,
    input_order: str = 'HWC',
    convert_to_gray: bool = True,
    **kwargs
) -> float:
    """
    计算图像的NIQE分数
    
    Args:
        img: 输入图像
        crop_border: 裁剪边界像素数
        input_order: 输入图像的维度顺序
        convert_to_gray: 是否转换为灰度图
        
    Returns:
        NIQE分数 (越低越好)
    """
    # 优先使用pyiqa库
    try:
        import pyiqa
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        niqe_metric = pyiqa.create_metric('niqe', device=device)
        
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and input_order == 'HWC':
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
        input_order: str = 'HWC',
        convert_to_gray: bool = True,
        device: str = None
    ):
        super().__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to_gray = convert_to_gray
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pyiqa_model = None
        try:
            import pyiqa
            self.pyiqa_model = pyiqa.create_metric('niqe', device=self.device)
        except ImportError:
            warnings.warn("pyiqa包未安装，请安装: pip install pyiqa")
    
    def forward(self, img: Union[np.ndarray, torch.Tensor]) -> float:
        if self.pyiqa_model is not None:
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and self.input_order == 'HWC':
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
                convert_to_gray=self.convert_to_gray
            )
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
