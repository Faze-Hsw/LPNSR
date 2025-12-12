"""
LPIPS感知损失（Learned Perceptual Image Patch Similarity）

设计思想：
LPIPS是一种基于深度学习的感知相似度度量，通过预训练网络提取的特征来衡量图像之间的感知差异。
相比于传统的像素级损失（如L1、L2），LPIPS能够更好地捕捉人类视觉系统对图像质量的感知。

核心优势：
1. 感知一致性：与人类感知高度相关
2. 特征级比较：捕捉高层语义信息
3. 预训练优势：利用大规模数据集学习到的表示
4. 多尺度特征：综合不同层级的特征信息

参考论文：
The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (CVPR 2018)
https://arxiv.org/abs/1801.03924
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class LPIPSLoss(nn.Module):
    """
    LPIPS感知损失
    
    该损失函数使用预训练的深度网络提取特征，计算图像之间的感知相似度。
    LPIPS已被证明与人类感知高度相关，广泛应用于图像生成、超分辨率等任务。
    
    Args:
        loss_weight: 损失权重，用于多损失加权
        net_type: 预训练网络类型，可选 'alex'(AlexNet), 'vgg'(VGG), 'squeeze'(SqueezeNet)
                  默认'alex'，计算效率最高且效果良好
        use_gpu: 是否使用GPU加速，默认True
        spatial: 是否返回空间维度的损失图，默认False
        normalize: 是否将输入归一化到[-1, 1]范围，默认True
    
    Examples:
        >>> criterion = LPIPSLoss(loss_weight=0.5, net_type='alex')
        >>> pred = torch.randn(4, 3, 256, 256)  # 需要3通道RGB图像
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    
    Note:
        - 输入图像应该是RGB格式，3通道
        - 输入范围应该是[-1, 1]或[0, 1]（如果normalize=True会自动处理）
        - LPIPS模型参数是冻结的，不参与训练
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        net_type: str = 'alex',
        use_gpu: bool = True,
        spatial: bool = False,
        normalize: bool = True
    ):
        super().__init__()
        
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips库未安装。请运行 'pip install lpips' 安装。"
            )
        
        self.loss_weight = loss_weight
        self.net_type = net_type
        self.spatial = spatial
        self.normalize = normalize
        
        # 创建LPIPS模型
        # net_type可选: 'alex', 'vgg', 'squeeze'
        # alex: AlexNet，计算最快，效果良好
        # vgg: VGG，计算较慢，但某些场景效果更好
        # squeeze: SqueezeNet，介于两者之间
        self.lpips_model = lpips.LPIPS(
            net=net_type,
            spatial=spatial,
            verbose=False
        )
        
        # 冻结LPIPS模型参数
        for param in self.lpips_model.parameters():
            param.requires_grad = False
        
        # 设置为评估模式
        self.lpips_model.eval()
    
    def _convert_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入转换为3通道RGB格式
        
        如果输入是4通道（如latent space），取前3通道
        如果输入是1通道（灰度图），复制到3通道
        
        Args:
            x: 输入张量 [B, C, H, W]
        
        Returns:
            rgb: RGB图像 [B, 3, H, W]
        """
        c = x.shape[1]
        
        if c == 3:
            return x
        elif c == 4:
            # 取前3通道
            return x[:, :3, :, :]
        elif c == 1:
            # 复制到3通道
            return x.repeat(1, 3, 1, 1)
        else:
            # 其他情况，取前3通道或复制
            if c > 3:
                return x[:, :3, :, :]
            else:
                # 补齐到3通道
                pad = torch.zeros(x.shape[0], 3 - c, x.shape[2], x.shape[3], 
                                  device=x.device, dtype=x.dtype)
                return torch.cat([x, pad], dim=1)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算LPIPS感知损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            weight: 可选的像素权重（仅在spatial=True时有效）
        
        Returns:
            loss: LPIPS损失值
        """
        assert pred.shape == target.shape, \
            f"pred和target形状必须相同，得到{pred.shape}和{target.shape}"
        
        # 转换为RGB格式
        pred_rgb = self._convert_to_rgb(pred)
        target_rgb = self._convert_to_rgb(target)
        
        # 确保LPIPS模型在正确的设备上
        if pred.device != next(self.lpips_model.parameters()).device:
            self.lpips_model = self.lpips_model.to(pred.device)
        
        # 计算LPIPS损失
        # LPIPS模型期望输入范围为[-1, 1]
        # normalize=True时会自动处理[0,1]范围的输入
        loss = self.lpips_model(
            pred_rgb, 
            target_rgb, 
            normalize=self.normalize
        )
        
        # 如果返回空间维度损失图，应用权重
        if self.spatial and weight is not None:
            loss = loss * weight
        
        # 取平均
        loss = loss.mean()
        
        return self.loss_weight * loss
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"loss_weight={self.loss_weight}, "
                f"net_type='{self.net_type}', "
                f"spatial={self.spatial}, "
                f"normalize={self.normalize})")
