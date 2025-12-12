"""
超分辨率训练损失函数模块

包含以下损失函数：
1. L2Loss: 标准的L2损失（MSE）
2. FocalFrequencyLoss: 频域感知损失（ICCV 2021）
3. LPIPSLoss: LPIPS感知损失（CVPR 2018）
"""

from .basic_loss import L2Loss
from .frequency_loss import FocalFrequencyLoss
from .lpips_loss import LPIPSLoss

__all__ = [
    'L2Loss',
    'FocalFrequencyLoss',
    'LPIPSLoss',
]
