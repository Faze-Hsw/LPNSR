"""
基于EDSR-Unet架构的噪声预测器
输入为中间状态、初始LR和当前时间步，完全重构网络架构
保留原有的输出分布预测和核心功能接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union
from dataclasses import dataclass

from SR.ldm.modules.diffusionmodules.openaimodel import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
)

# 尝试导入xformers
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except ImportError:
    XFORMERS_IS_AVAILABLE = False
    print("xformers not available, using standard attention implementation")


@dataclass
class NoisePredictorOutput:
    noise: torch.Tensor
    latent_dist: Optional['DiagonalGaussianDistribution'] = None


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        eps = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: 'DiagonalGaussianDistribution' = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0]).to(self.parameters.device)

        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )


class EDSRResidualBlock(nn.Module):
    """EDSR风格的残差块，包含密集连接特征融合"""
    def __init__(self, channels: int, growth_rate: int = 32, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale

        # 密集连接卷积层
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, channels, 3, padding=1)

        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 密集连接
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.activation(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))

        # 残差缩放
        return residual + self.res_scale * x4


class EDSRTimeModulatedBlock(nn.Module):
    """带有时间步调制的EDSR残差块"""
    def __init__(self, channels: int, emb_channels: int, growth_rate: int = 32, res_scale: float = 0.1):
        super().__init__()
        self.res_block = EDSRResidualBlock(channels, growth_rate, res_scale)
        self.time_mlp = nn.Sequential(
            nn.ReLU(inplace=False),
            linear(emb_channels, channels),
            nn.Unflatten(1, (-1, 1, 1))  # 调整为空间维度
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # 时间步调制
        t_mod = self.time_mlp(t_emb)
        x = x + t_mod  # 时间信息注入

        # EDSR残差块
        return self.res_block(x)


class Downsample(nn.Module):
    """EDSR风格下采样模块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class Upsample(nn.Module):
    """EDSR风格上采样模块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先上采样再卷积
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.activation(self.conv(x))


class EDSRUnetNoisePredictor(nn.Module):
    """
    基于EDSR-Unet架构的噪声预测器
    输入: 中间状态(z_t)、初始LR(lr_latent)、时间步(timesteps)
    """
    def __init__(
        self,
        latent_channels: int = 3,
        model_channels: int = 192,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        growth_rate: int = 32,
        res_scale: float = 0.1,
        double_z: bool = True
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_levels = len(channel_mult)
        self.double_z = double_z

        # 时间步嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.ReLU(inplace=False),
            linear(time_embed_dim, time_embed_dim)
        )

        # 输入处理: 融合中间状态和初始LR
        self.z_proj = nn.Conv2d(latent_channels, model_channels, 3, padding=1)
        self.lr_proj = nn.Conv2d(latent_channels, model_channels, 3, padding=1)
        self.input_fusion = nn.Conv2d(model_channels * 2, model_channels, 3, padding=1)

        # 编码器
        self.encoder = nn.ModuleList()
        current_channels = model_channels

        for level in range(self.num_levels):
            block = nn.ModuleList()
            # 添加多个残差块
            for _ in range(num_res_blocks):
                block.append(
                    EDSRTimeModulatedBlock(
                        current_channels,
                        time_embed_dim,
                        growth_rate,
                        res_scale
                    )
                )

            # 添加下采样（最后一层不下采样）
            if level < self.num_levels - 1:
                block.append(Downsample(current_channels))
                current_channels *= 2  # 下采样后通道翻倍

            self.encoder.append(block)

        # 瓶颈层
        self.bottleneck = nn.ModuleList([
            EDSRTimeModulatedBlock(
                current_channels,
                time_embed_dim,
                growth_rate,
                res_scale
            ),
            EDSRTimeModulatedBlock(
                current_channels,
                time_embed_dim,
                growth_rate,
                res_scale
            )
        ])

        # 解码器
        self.decoder = nn.ModuleList()
        
        # 存储每个解码器层的输入通道数（用于跳跃连接）
        decoder_channel_sizes = []
        
        for level in reversed(range(self.num_levels)):
            decoder_channel_sizes.append(current_channels)
            
            block = nn.ModuleList()
            # 添加多个残差块
            for _ in range(num_res_blocks):
                block.append(
                    EDSRTimeModulatedBlock(
                        current_channels,
                        time_embed_dim,
                        growth_rate,
                        res_scale
                    )
                )

            # 添加上采样（最后一层不上采样）
            if level > 0:
                block.append(Upsample(current_channels))
                current_channels = current_channels // 2  # 上采样后通道减半

            self.decoder.append(block)
        
        # 创建跳跃连接的通道匹配卷积
        self.skip_convs = nn.ModuleList()
        
        # 编码器保存的通道数列表（下采样前的）
        encoder_channels_list = []
        current_channels = model_channels
        for level in range(self.num_levels):
            encoder_channels_list.append(current_channels)
            if level < self.num_levels - 1:
                # 下采样后通道翻倍
                current_channels *= 2
        # 添加最后一个（最深层）
        encoder_channels_list.append(current_channels)
        
        # 为每个解码器层创建对应的跳跃连接卷积
        for i in range(self.num_levels):
            # 从后往前取编码器保存的通道数（跳过最后一个）
            encoder_channels = encoder_channels_list[-(i+2)]
            # 对应解码器层的输入通道数
            decoder_channels = decoder_channel_sizes[i]
            
            # 创建通道匹配卷积
            self.skip_convs.append(
                nn.Conv2d(encoder_channels, decoder_channels, kernel_size=1)
            )

        # 输出层
        output_channels = latent_channels * 2 if double_z else latent_channels
        # 使用解码器最后一层的输出通道数
        # 解码器最后一层是decoder_channel_sizes的最后一个元素
        final_decoder_channels = decoder_channel_sizes[-1]
        self.output = nn.Sequential(
            nn.Conv2d(final_decoder_channels, model_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            zero_module(nn.Conv2d(model_channels, output_channels, 3, padding=1))
        )

    def forward(
        self,
        z_t: torch.Tensor,          # 中间状态
        lr_latent: torch.Tensor,    # 初始LR
        timesteps: torch.Tensor,    # 当前时间步
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> Union[torch.Tensor, DiagonalGaussianDistribution, NoisePredictorOutput]:
        # 时间步嵌入
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # 输入融合: 中间状态 + 初始LR
        z_feat = self.z_proj(z_t)
        lr_feat = self.lr_proj(lr_latent)
        x = self.input_fusion(torch.cat([z_feat, lr_feat], dim=1))

        # 编码器前向
        skip_connections = []
        for level in range(self.num_levels):
            for i, block in enumerate(self.encoder[level]):
                if isinstance(block, EDSRTimeModulatedBlock):
                    x = block(x, t_emb)
                else:  # 下采样
                    # 在下采样前保存特征
                    skip_connections.append(x.clone())
                    x = block(x)
            # 如果是最后一层，也需要保存特征
            if level == self.num_levels - 1:
                skip_connections.append(x.clone())

        # 瓶颈层
        for block in self.bottleneck:
            x = block(x, t_emb)

        # 解码器前向
        # 注意：现在skip_connections包含了所有的跳跃连接，包括最后一个
        # 从后往前使用：第一个使用的是最后一个下采样前的特征，以此类推
        skip_idx = len(skip_connections) - 1
        
        for level in range(self.num_levels):
            # 获取对应的跳跃连接
            skip = skip_connections[skip_idx]
            skip_idx -= 1
            
            # 跳跃连接通道匹配
            skip = self.skip_convs[level](skip)
            
            # 确保空间尺寸匹配（可能需要裁剪或填充）
            if skip.shape[2:] != x.shape[2:]:
                # 使用中心裁剪或填充来匹配尺寸
                skip_h, skip_w = skip.shape[2], skip.shape[3]
                x_h, x_w = x.shape[2], x.shape[3]
                
                if skip_h > x_h or skip_w > x_w:
                    # 裁剪
                    skip = skip[:, :, :x_h, :x_w]
                else:
                    # 填充
                    pad_h = x_h - skip_h
                    pad_w = x_w - skip_w
                    skip = F.pad(skip, (0, pad_w, 0, pad_h))
            
            # 跳跃连接融合
            x = x + skip

            for i, block in enumerate(self.decoder[level]):
                if isinstance(block, EDSRTimeModulatedBlock):
                    x = block(x, t_emb)
                else:  # 上采样
                    x = block(x)

        # 输出处理
        h_out = self.output(x)

        if self.double_z:
            posterior = DiagonalGaussianDistribution(h_out)
            if sample_posterior:
                return posterior.sample(generator=generator)
            else:
                return posterior
        else:
            return h_out


def create_noise_predictor(
    latent_channels: int = 3, **kwargs
) -> nn.Module:
    return EDSRUnetNoisePredictor(latent_channels=latent_channels, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = create_noise_predictor(
        latent_channels=3,
        model_channels=192,
        channel_mult=(1, 2, 4,8),
        num_res_blocks=2,
        growth_rate=32,
        res_scale=0.1
    ).to(device)

    # 测试输入: 中间状态、初始LR、时间步
    batch_size = 2
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    z_t = torch.randn(batch_size, 3, 64, 64).to(device)  # 中间状态
    lr_latent = torch.randn(batch_size, 3, 64, 64).to(device)  # 初始LR

    # 测试前向传播
    print("=" * 50)
    print("EDSR-Unet噪声预测器测试")
    print("=" * 50)

    predicted_noise = model(z_t, lr_latent, timesteps, sample_posterior=True)
    print(f"中间状态形状: {z_t.shape}")
    print(f"初始LR形状: {lr_latent.shape}")
    print(f"预测噪声形状: {predicted_noise.shape}")

    posterior = model(z_t, lr_latent, timesteps, sample_posterior=False)
    print(f"分布均值形状: {posterior.mean.shape}")
    print(f"分布方差形状: {posterior.var.shape}")

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")