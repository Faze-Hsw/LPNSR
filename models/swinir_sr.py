"""
SwinIR super-resolution model wrapper for LPNSR
"""

import os

import torch
import torch.nn as nn

from .network_swinir import SwinIR as SwinIRNet


def create_swinir(
    upscale=4,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="nearest+conv",
    resi_connection="1conv",
    model_path=None,
    device="cuda",
):
    """
    创建SwinIR超分模型并加载预训练权重

    Args:
        upscale: 超分倍数
        img_size: 训练时的图像尺寸
        window_size: 窗口大小
        img_range: 图像范围
        depths: 每层深度
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        mlp_ratio: MLP比率
        upsampler: 上采样器类型
        resi_connection: 残差连接方式
        model_path: 预训练权重路径
        device: 设备

    Returns:
        SwinIR模型
    """
    model = SwinIRNet(
        upscale=upscale,
        in_chans=3,
        img_size=img_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )

    if model_path is not None and os.path.exists(model_path):
        print(f"Loading SwinIR weights from {model_path}")
        pretrained_model = torch.load(model_path, map_location="cpu")

        # 尝试加载不同的key名称
        if "params_ema" in pretrained_model:
            model.load_state_dict(pretrained_model["params_ema"], strict=True)
        elif "params" in pretrained_model:
            model.load_state_dict(pretrained_model["params"], strict=True)
        else:
            model.load_state_dict(pretrained_model, strict=True)
        print("✓ SwinIR weights loaded successfully!")
    else:
        print(
            f"Warning: SwinIR model path {model_path} not found, using random initialization"
        )

    model = model.to(device)
    model.eval()
    return model


class SwinIRWrapper(nn.Module):
    """
    SwinIR包装器,处理数据范围转换和padding策略
    输入: [-1, 1]范围
    输出: [-1, 1]范围
    """

    def __init__(self, swinir_model):
        super().__init__()
        self.swinir_model = swinir_model
        self.window_size = swinir_model.window_size
        self.upscale = swinir_model.upscale

    def forward(self, x):
        """
        Args:
            x: LR图像, shape [B, C, H, W], range [-1, 1]

        Returns:
            SR图像, shape [B, C, H*scale, W*scale], range [-1, 1]
        """
        # 将[-1, 1]转换到[0, 1]
        x_01 = (x + 1.0) / 2.0
        # SwinIR推理
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                sr_01 = self.swinir_model(x_01)
        # 将[0, 1]转换回[-1, 1]
        sr = sr_01 * 2.0 - 1.0

        return sr
