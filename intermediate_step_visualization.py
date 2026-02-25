#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中间步骤可视化脚本

功能：
1. 可视化SWINIR超分后的图像（作为初始上采样）
2. 可视化最终超分结果
3. 可视化NoisePredictor预测的噪声图（每一步的噪声预测）
4. 帮助分析噪声预测器的工作原理

注意：
- SWINIR只在开始时调用一次，用于初始上采样（替代bicubic插值）
- 噪声预测器在每一步都被调用，用于生成精确的噪声
"""

import os
import sys
import warnings

# 在导入其他模块之前设置警告过滤器
warnings.filterwarnings("ignore", message=".*A matching Triton is not available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")

import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from contextlib import nullcontext

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# LPNSR模块导入
from LPNSR.models.noise_predictor import create_noise_predictor
from LPNSR.models.unet import UNetModelSwin
from LPNSR.ldm.models.autoencoder import VQModelTorch


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        power=2.0):
    """
    获取ResShift的eta调度

    Args:
        schedule_name: 调度类型（'exponential'）
        num_diffusion_timesteps: 扩散步数T
        min_noise_level: 最小噪声水平η_1
        etas_end: 最大噪声水平η_T
        kappa: 方差控制参数κ
        power: 指数调度的幂次

    Returns:
        sqrt_etas: √η_t数组，shape=(T,)
    """
    if schedule_name == 'exponential':
        # 指数调度（ResShift默认）
        etas_start = min(min_noise_level / kappa, min_noise_level)

        # 计算增长因子
        increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser

        # 计算幂次时间步
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        power_timestep *= (num_diffusion_timesteps - 1)

        # 计算sqrt_etas
        sqrt_etas = np.power(base, power_timestep) * etas_start
    else:
        raise ValueError(f"未知的schedule_name: {schedule_name}")

    return sqrt_etas


class ModelOutputsVisualizer:
    """超分中间步骤可视化类"""

    def __init__(self, config_path, device='cuda'):
        """
        初始化可视化器

        Args:
            config_path: 配置文件路径
            device: 设备（'cuda' 或 'cpu'）
        """
        self.device = device

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置随机种子
        seed = self.config['inference'].get('seed', 12345)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 推理配置（需要在初始化模型之前设置）
        self.num_steps = self.config['inference']['num_steps']
        self.scale_factor = self.config['inference']['scale_factor']
        self.use_amp = self.config['inference']['use_amp']

        # 初始化模型
        self._init_models()

        # 初始化扩散参数
        self._init_diffusion()

        # 颜色校正配置
        self.color_correction = self.config['inference'].get('color_correction', True)

        # 上采样方式配置
        self.use_swinir = self.config['inference'].get('use_swinir', False)

        print(f"✓ 可视化器初始化完成")
        print(f"  - 采样步数: {self.num_steps}")
        print(f"  - 超分倍数: {self.scale_factor}x")
        print(f"  - 上采样方式: {'SWINIR' if self.swinir is not None else 'Bicubic'}")

    def _init_models(self):
        """初始化模型"""
        print("正在加载模型...")

        # 1. 加载VAE
        print("  加载VAE...")
        vae_config = self.config['vae']

        # VAE模型结构参数（必须与预训练权重一致）
        ddconfig = {
            'double_z': False,
            'z_channels': 3,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
            'padding_mode': 'zeros',
        }

        # 从配置文件获取LoRA参数
        lora_config = vae_config.get('lora', {})

        self.vae = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
            rank=lora_config.get('rank', 8),
            lora_alpha=lora_config.get('alpha', 1.0),
            lora_tune_decoder=lora_config.get('tune_decoder', False),
        )

        # 加载预训练权重
        vae_ckpt = torch.load(self.config['model']['vae_path'], map_location='cpu')

        # 处理state_dict格式
        if 'state_dict' in vae_ckpt:
            state_dict = vae_ckpt['state_dict']
        else:
            state_dict = vae_ckpt

        # 智能处理前缀
        first_key = list(state_dict.keys())[0]
        has_module_prefix = first_key.startswith('module.')
        has_orig_mod_prefix = '_orig_mod.' in first_key

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if has_orig_mod_prefix:
                new_key = new_key.replace('_orig_mod.', '')
            if has_module_prefix:
                new_key = new_key.replace('module.', '')
            new_state_dict[new_key] = value

        self.vae.load_state_dict(new_state_dict, strict=False)
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("  ✓ VAE加载完成")

        # 2. 加载ResShift UNet
        print("  加载ResShift UNet...")
        unet_config = self.config['resshift_unet']
        self.resshift_unet = UNetModelSwin(**unet_config)

        resshift_ckpt = torch.load(self.config['model']['resshift_path'], map_location='cpu')

        if 'state_dict' in resshift_ckpt:
            state_dict = resshift_ckpt['state_dict']
        else:
            state_dict = resshift_ckpt

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module._orig_mod.'):
                new_key = key.replace('module._orig_mod.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        self.resshift_unet.load_state_dict(new_state_dict, strict=True)
        self.resshift_unet = self.resshift_unet.to(self.device)
        self.resshift_unet.eval()
        for param in self.resshift_unet.parameters():
            param.requires_grad = False
        print("  ✓ ResShift UNet加载完成")


        # 3. 加载噪声预测器
        print("  加载噪声预测器...")
        noise_predictor_config = self.config['noise_predictor']

        if 'config_path' in noise_predictor_config:
            with open(noise_predictor_config['config_path'], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.noise_predictor = create_noise_predictor(
                image_size=config.get('image_size', 64),
                latent_channels=config['latent_channels'],
                model_channels=config['model_channels'],
                out_channels=config.get('out_channels', config['latent_channels']),
                channel_mult=tuple(config['channel_mult']),
                num_res_blocks=config['num_res_blocks'],
                attention_resolutions=config.get('attention_resolutions', [64, 32, 16, 8]),
                dropout=config.get('dropout', 0.0),
                conv_resample=config.get('conv_resample', True),
                dims=config.get('dims', 2),
                use_fp16=config.get('use_fp16', False),
                num_heads=config.get('num_heads', -1),
                num_head_channels=config.get('num_head_channels', 32),
                use_scale_shift_norm=config.get('use_scale_shift_norm', True),
                resblock_updown=config.get('resblock_updown', False),
                swin_depth=config.get('swin_depth', 2),
                swin_embed_dim=config.get('swin_embed_dim', 192),
                window_size=config.get('window_size', 8),
                mlp_ratio=config.get('mlp_ratio', 4.0),
                patch_norm=config.get('patch_norm', False),
                cond_lq=config.get('cond_lq', True),
                lq_size=config.get('lq_size', 64),
            )
        else:
            self.noise_predictor = create_noise_predictor(
                image_size=noise_predictor_config.get('image_size', 64),
                latent_channels=noise_predictor_config['latent_channels'],
                model_channels=noise_predictor_config['model_channels'],
                out_channels=noise_predictor_config.get('out_channels', noise_predictor_config['latent_channels']),
                channel_mult=tuple(noise_predictor_config['channel_mult']),
                num_res_blocks=noise_predictor_config['num_res_blocks'],
                attention_resolutions=noise_predictor_config.get('attention_resolutions', [64, 32, 16, 8]),
                dropout=noise_predictor_config.get('dropout', 0.0),
                conv_resample=noise_predictor_config.get('conv_resample', True),
                dims=noise_predictor_config.get('dims', 2),
                use_fp16=noise_predictor_config.get('use_fp16', False),
                num_heads=noise_predictor_config.get('num_heads', -1),
                num_head_channels=noise_predictor_config.get('num_head_channels', 32),
                use_scale_shift_norm=noise_predictor_config.get('use_scale_shift_norm', True),
                resblock_updown=noise_predictor_config.get('resblock_updown', False),
                swin_depth=noise_predictor_config.get('swin_depth', 2),
                swin_embed_dim=noise_predictor_config.get('swin_embed_dim', 192),
                window_size=noise_predictor_config.get('window_size', 8),
                mlp_ratio=noise_predictor_config.get('mlp_ratio', 4.0),
                patch_norm=noise_predictor_config.get('patch_norm', False),
                cond_lq=noise_predictor_config.get('cond_lq', True),
                lq_size=noise_predictor_config.get('lq_size', 64),
            )

        noise_ckpt = torch.load(self.config['model']['noise_predictor_path'], map_location='cpu')

        if isinstance(noise_ckpt, dict):
            if 'noise_predictor' in noise_ckpt:
                state_dict = noise_ckpt['noise_predictor']
            elif 'model_state_dict' in noise_ckpt:
                state_dict = noise_ckpt['model_state_dict']
            else:
                state_dict = noise_ckpt
        else:
            raise ValueError(f"不支持的checkpoint格式: {type(noise_ckpt)}")

        self.noise_predictor.load_state_dict(state_dict, strict=True)
        self.noise_predictor = self.noise_predictor.to(self.device)
        self.noise_predictor.eval()
        for param in self.noise_predictor.parameters():
            param.requires_grad = False
        print("  ✓ 噪声预测器加载完成")


        # 5. 加载SWINIR超分模型（可选）
        self.swinir = None
        if self.config['inference'].get('use_swinir', False):
            print("  加载SWINIR超分模型...")
            from LPNSR.models.swinir_sr import create_swinir, SwinIRWrapper

            swinir_config = self.config['inference'].get('swinir', {})

            # SWINIR模型路径
            swinir_model_path = 'LPNSR/pretrained/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'

            # 创建SWINIR模型
            swinir_model = create_swinir(
                upscale=self.scale_factor,
                img_size=swinir_config.get('img_size', 64),
                window_size=swinir_config.get('window_size', 8),
                img_range=swinir_config.get('img_range', 1.0),
                depths=swinir_config.get('depths', [6, 6, 6, 6, 6, 6]),
                embed_dim=swinir_config.get('embed_dim', 180),
                num_heads=swinir_config.get('num_heads', [6, 6, 6, 6, 6, 6]),
                mlp_ratio=swinir_config.get('mlp_ratio', 2),
                upsampler=swinir_config.get('upsampler', 'pixelshuffle'),
                resi_connection=swinir_config.get('resi_connection', '1conv'),
                model_path=swinir_model_path,
                device=self.device
            )
            # 使用包装器处理数据范围转换
            self.swinir = SwinIRWrapper(swinir_model)
            print("  ✓ SWINIR加载完成")
        else:
            print("  ! SWINIR未配置或未启用，将使用bicubic插值")

    def _init_diffusion(self):
        """初始化扩散参数"""
        diffusion_config = self.config['diffusion']

        self.diffusion_num_timesteps = diffusion_config['num_timesteps']
        self.kappa = diffusion_config['kappa']
        self.normalize_input = diffusion_config.get('normalize_input', True)
        self.latent_flag = diffusion_config.get('latent_flag', True)

        # 计算eta调度
        sqrt_etas = get_named_eta_schedule(
            schedule_name=diffusion_config['eta_schedule'],
            num_diffusion_timesteps=self.diffusion_num_timesteps,
            min_noise_level=diffusion_config['min_noise_level'],
            etas_end=diffusion_config['etas_end'],
            kappa=self.kappa,
            power=diffusion_config['eta_power']
        )

        self.sqrt_etas = sqrt_etas.astype(np.float64)
        self.etas = self.sqrt_etas ** 2

        # 计算alpha
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # 计算后验分布参数
        self.posterior_variance = self.kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)

        # 后验均值系数
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        self.posterior_mean_coef1[0] = 0.0
        self.posterior_mean_coef2[0] = 1.0

        # 转换为tensor
        self.sqrt_etas = torch.from_numpy(self.sqrt_etas).float()
        self.etas = torch.from_numpy(self.etas).float()
        self.etas_prev = torch.from_numpy(self.etas_prev).float()
        self.alpha = torch.from_numpy(self.alpha).float()
        self.posterior_variance = torch.from_numpy(self.posterior_variance).float()
        self.posterior_variance_clipped = torch.from_numpy(self.posterior_variance_clipped).float()
        self.posterior_log_variance_clipped = torch.from_numpy(self.posterior_log_variance_clipped).float()
        self.posterior_mean_coef1 = torch.from_numpy(self.posterior_mean_coef1).float()
        self.posterior_mean_coef2 = torch.from_numpy(self.posterior_mean_coef2).float()

        print(f"  ✓ 扩散参数初始化完成")
        print(f"    - 扩散步数: {self.diffusion_num_timesteps}")
        print(f"    - κ: {self.kappa}")

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """从数组中提取值并广播到目标形状"""
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _scale_input(self, inputs, t):
        """对输入进行归一化"""
        if self.normalize_input:
            if self.latent_flag:
                std = torch.sqrt(self._extract_into_tensor(self.etas, t, inputs.shape) * self.kappa ** 2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = self._extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """计算ResShift后验分布"""
        mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_0
        )

        variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, variance, log_variance

    def prior_sample_random(self, y):
        """
        使用随机噪声从先验分布采样
        q(x_T|y) ~= N(x_T|y, κ²η_T)
        """
        t = torch.tensor([self.num_steps - 1] * y.shape[0], device=y.device).long()
        noise = torch.randn_like(y)
        return y + self._extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def prior_sample_initializer(self, y):
        """
        使用Initializer从先验分布采样
        q(x_T|y) ~= N(x_T|y, κ²η_T)
        """
        t = torch.tensor([self.num_steps - 1] * y.shape[0], device=y.device).long()
        # 使用Initializer预测噪声
        noise = self.initializer(y, t, sample_posterior=True)
        return y + self._extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def prior_sample(self, y, t=None):
        """
        从先验分布采样，即 q(x_T|y) ~= N(x_T|y, κ²η_T)

        Args:
            y: 退化图像的潜在表示（lr_latent）
            t: 可选的时间步，默认使用最后一个时间步

        Returns:
            x_T: 初始采样
        """
        if t is None:
            t = torch.tensor([self.num_steps - 1] * y.shape[0], device=y.device).long()

        # 使用随机高斯噪声
        noise = torch.randn_like(y)

        return y + self._extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def _wavelet_blur(self, image: torch.Tensor, radius: int):
        """对输入tensor应用小波模糊"""
        kernel_vals = [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ]
        kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
        kernel = kernel[None, None]
        kernel = kernel.repeat(3, 1, 1, 1)
        image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
        output = F.conv2d(image, kernel, groups=3, dilation=radius)
        return output

    def _wavelet_decomposition(self, image: torch.Tensor, levels: int = 5):
        """对输入tensor进行小波分解"""
        high_freq = torch.zeros_like(image)
        for i in range(levels):
            radius = 2 ** i
            low_freq = self._wavelet_blur(image, radius)
            high_freq += (image - low_freq)
            image = low_freq
        return high_freq, low_freq

    def _color_correction(self, sr_tensor, lr_tensor):
        """颜色校正"""
        sr_01 = (sr_tensor + 1.0) / 2.0
        lr_01 = (lr_tensor + 1.0) / 2.0
        sr_high_freq, _ = self._wavelet_decomposition(sr_01)
        _, lr_low_freq = self._wavelet_decomposition(lr_01)
        corrected_01 = sr_high_freq + lr_low_freq
        corrected_01 = torch.clamp(corrected_01, 0.0, 1.0)
        corrected = corrected_01 * 2.0 - 1.0
        return corrected

    def latent_to_image(self, latent):
        """将潜在表示转换为图像"""
        with torch.no_grad():
            img_tensor = self.vae.decode(latent)
        img_tensor = torch.clamp(img_tensor, -1.0, 1.0)
        # 转换到[0, 1]
        img_tensor = img_tensor * 0.5 + 0.5
        # 转换为numpy
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return img

    @torch.no_grad()
    def reverse_sampling_with_outputs(self, lr_latent, lr_image, lr_upsampled):
        """
        反向采样过程，收集NoisePredictor的输出

        Args:
            lr_latent: LR图像的潜在表示
            lr_image: 图像空间的LR图像（用作UNet的lq条件）
            lr_upsampled: SWINIR或bicubic上采样的HR图像

        Returns:
            noise_predictor_outputs: NoisePredictor在每一步的输出列表
            intermediate_latents: 每一步的中间潜在表示列表
            final_sr_image: 最终的超分图像
        """
        noise_predictor_outputs = []
        intermediate_latents = []

        # 初始化x_T (使用先验采样)
        t_T = torch.tensor([self.num_steps - 1] * lr_latent.shape[0], device=self.device).long()
        x_t = self.prior_sample(lr_latent, t_T)

        # 保存初始潜在状态
        intermediate_latents.append(x_t.clone())

        # 反向采样：从num_steps-1到0
        indices = list(range(self.num_steps))[::-1]

        for i in indices:
            t_tensor = torch.tensor([i] * lr_latent.shape[0], device=self.device).long()

            # 1. 对输入进行归一化
            x_t_normalized = self._scale_input(x_t, t_tensor)

            # 2. 使用ResShift的UNet预测x_0
            pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)

            # 3. 计算ResShift后验分布
            mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t_tensor)

            # 4. 使用NoisePredictor生成噪声
            # 注意：noise_predictor现在接受原始图像空间的lr_image[-1,1]，而不是潜空间的lr_latent
            # noise_predictor 内部会处理分布采样
            noise = self.noise_predictor(x_t, lr_image, t_tensor, sample_posterior=True)
            noise_predictor_outputs.append(noise.clone())

            # 5. 采样x_{t-1}
            nonzero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
            x_t = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

            # 保存中间潜在状态
            intermediate_latents.append(x_t.clone())

        # 6. 解码最终的x_0得到最终图像
        final_sr_image = self.latent_to_image(x_t)

        return noise_predictor_outputs, intermediate_latents, final_sr_image

    def pad_image(self, img, multiple=64):
        """Padding图像到multiple的倍数"""
        h, w = img.shape[:2]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        return img, (pad_h, pad_w)

    def visualize_intermediate_steps(self, lr_image_path, output_path):
        """
        可视化SWINIR超分结果和NoisePredictor每步的噪声预测

        Args:
            lr_image_path: LR图像路径
            output_path: 输出路径
        """
        # 读取图像
        lr_image = cv2.imread(str(lr_image_path))
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_image = lr_image.astype(np.float32) / 255.0

        print(f"\n处理: {lr_image_path}")
        print(f"  原始尺寸: {lr_image.shape[1]}x{lr_image.shape[0]}")

        # Padding图像
        padding_offset = self.config['inference'].get('padding_offset', 64)
        lr_padded, (pad_h, pad_w) = self.pad_image(lr_image, multiple=padding_offset)

        # 转换为tensor
        lr_tensor = torch.from_numpy(lr_padded).permute(2, 0, 1).unsqueeze(0).float()
        lr_tensor = lr_tensor.to(self.device)
        lr_tensor = lr_tensor * 2.0 - 1.0  # 归一化到[-1, 1]

        # 上采样LR图像（使用SWINIR或bicubic）
        use_swinir = self.config['inference'].get('use_swinir', False)
        if use_swinir and hasattr(self, 'swinir') and self.swinir is not None:
            # 使用SWINIR进行超分
            lr_upsampled = self.swinir(lr_tensor)
            print(f"  上采样方法: SWINIR")
        else:
            # 使用双三次插值
            lr_upsampled = F.interpolate(
                lr_tensor,
                scale_factor=self.scale_factor,
                mode='bicubic',
                align_corners=False
            )
            print(f"  上采样方法: Bicubic")

        # 编码到潜在空间
        with torch.no_grad():
            lr_latent = self.vae.encode(lr_upsampled)

        print("  开始采样...")

        # 运行反向采样，收集NoisePredictor的输出
        noise_predictor_outputs, intermediate_latents, final_sr_image = self.reverse_sampling_with_outputs(
            lr_latent, lr_tensor, lr_upsampled
        )

        # 创建输出目录
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取图像名称
        img_name = Path(lr_image_path).stem

        # 准备LR和上采样图像用于显示
        lr_upsampled_display = lr_upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()
        lr_upsampled_display = (lr_upsampled_display + 1.0) / 2.0
        lr_upsampled_display = np.clip(lr_upsampled_display, 0, 1)

        # 去除padding
        if pad_h > 0 or pad_w > 0:
            h_end = lr_upsampled_display.shape[0] - pad_h * self.scale_factor
            w_end = lr_upsampled_display.shape[1] - pad_w * self.scale_factor
            lr_upsampled_display = lr_upsampled_display[:h_end, :w_end]
            final_sr_image = final_sr_image[:h_end, :w_end]

        # 创建SWINIR超分结果可视化图
        self._create_results_figure(
            lr_image, lr_upsampled_display, final_sr_image,
            output_path / f"{img_name}_results.png"
        )

        # 创建噪声预测器输出可视化图
        self._create_noise_predictor_figure(
            noise_predictor_outputs,
            output_path / f"{img_name}_noise_predictor.png",
            h_end if pad_h > 0 or pad_w > 0 else None,
            w_end if pad_h > 0 or pad_w > 0 else None
        )

        # 保存噪声预测器的输出
        self._save_noise_images(
            noise_predictor_outputs,
            output_path / "noise_outputs", img_name,
            h_end if pad_h > 0 or pad_w > 0 else None,
            w_end if pad_h > 0 or pad_w > 0 else None
        )

        print(f"  ✓ 可视化结果保存到: {output_path}")

    def _create_results_figure(self, lr_image, lr_upsampled, final_sr_image, output_file):
        """
        创建结果对比可视化图

        Args:
            lr_image: 原始LR图像
            lr_upsampled: 上采样的LR图像（SWINIR或bicubic）
            final_sr_image: 最终的超分结果
            output_file: 输出文件路径
        """
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.flatten()

        # 显示LR图像
        axes[0].imshow(lr_image)
        axes[0].set_title('LR Input', fontsize=10)
        axes[0].axis('off')

        # 显示上采样图像
        axes[1].imshow(lr_upsampled)
        axes[1].set_title('Upsampled (Initial)', fontsize=10)
        axes[1].axis('off')

        # 显示最终SR图像
        axes[2].imshow(final_sr_image)
        axes[2].set_title('Final Super-Resolution', fontsize=10)
        axes[2].axis('off')

        plt.suptitle('Super-Resolution Results Comparison',
                     fontsize=12, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    保存结果对比图: {output_file}")

    def _latent_to_image(self, latent):
        """将潜在表示转换为图像（用于可视化模型输出）"""
        with torch.no_grad():
            img_tensor = self.vae.decode(latent)
        img_tensor = torch.clamp(img_tensor, -1.0, 1.0)
        img_tensor = img_tensor * 0.5 + 0.5
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(img, 0, 1)

    def _create_noise_predictor_figure(self, noise_predictor_outputs, output_file,
                                       h_end=None, w_end=None):
        """
        创建噪声预测器输出可视化图

        Args:
            noise_predictor_outputs: NoisePredictor在每一步的输出列表
            output_file: 输出文件路径
            h_end: 图像高度结束位置（去除padding后）
            w_end: 图像宽度结束位置（去除padding后）
        """
        num_steps = len(noise_predictor_outputs)

        # 计算网格行列数
        num_cols = min(8, num_steps)
        num_rows = (num_steps + num_cols - 1) // num_cols

        # 创建图表
        fig_width = 2.5 * num_cols
        fig_height = 2.5 * num_rows
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, wspace=0.05, hspace=0.1)

        # 显示NoisePredictor在每一步的输出
        for i, noise_output in enumerate(noise_predictor_outputs):
            row = i // num_cols
            col = i % num_cols
            noise_img = self._latent_to_image(noise_output)
            if h_end is not None and w_end is not None:
                noise_img = noise_img[:h_end, :w_end]

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(noise_img)
            step_num = num_steps - i
            ax.set_title(f'Step {step_num}\nNoise Pred', fontsize=8)
            ax.axis('off')

        plt.suptitle('NoisePredictor Outputs (Predicted Noise at Each Step)',
                     fontsize=12, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    保存噪声预测器可视化图: {output_file}")

    def _save_noise_images(self, noise_predictor_outputs, output_dir, img_name,
                        h_end=None, w_end=None):
        """
        保存噪声预测器的输出

        Args:
            noise_predictor_outputs: NoisePredictor在每一步的输出列表
            output_dir: 输出目录
            img_name: 图像名称
            h_end: 图像高度结束位置（去除padding后）
            w_end: 图像宽度结束位置（去除padding后）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存NoisePredictor在每一步的输出
        num_steps = len(noise_predictor_outputs)
        for i, noise_output in enumerate(noise_predictor_outputs):
            noise_img = self._latent_to_image(noise_output)
            if h_end is not None and w_end is not None:
                noise_img = noise_img[:h_end, :w_end]

            step_num = num_steps - i
            img_uint8 = (np.clip(noise_img, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{img_name}_step{step_num:03d}_noise.png"), img_bgr)


        print(f"    保存噪声预测图像到: {output_dir}")

    def run(self, input_path, output_path):
        """
        运行可视化

        Args:
            input_path: 输入路径（图像或文件夹）
            output_path: 输出路径
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 获取图像列表
        if input_path.is_file():
            image_paths = [input_path]
        else:
            image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG']:
                image_paths.extend(input_path.glob(ext))
            image_paths = sorted(image_paths)

        print(f"\n找到 {len(image_paths)} 张图像")

        # 处理每张图像
        for img_path in tqdm(image_paths, desc="可视化模型输出"):
            self.visualize_intermediate_steps(img_path, output_path)

        print(f"\n✓ 全部完成！结果保存在: {output_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="中间步骤可视化脚本")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="输入路径（图像或文件夹）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./visualization_results",
        help="输出路径"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="LPNSR/configs/inference.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备"
    )

    return parser


def main():
    # 解析参数
    parser = get_parser()
    args = parser.parse_args()

    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("超分中间步骤可视化脚本")
    print("=" * 60)
    print("\n可视化内容：")
    print("  1. 超分结果对比：LR图像、初始上采样（SWINIR/Bicubic）、最终结果")
    print("  2. 噪声预测器输出：每一步预测的噪声图")
    print("\n输出文件：")
    print("  - {img_name}_results.png: 超分结果对比图（LR vs 初始上采样 vs 最终结果）")
    print("  - {img_name}_noise_predictor.png: 噪声预测器输出汇总图")
    print("  - noise_outputs/{img_name}_stepXXX_noise.png: 噪声预测器的单独输出")
    print("\n注意：")
    print("  - SWINIR只在开始时调用一次，用于初始上采样")
    print("  - 噪声预测器在每一步都被调用，用于生成精确的噪声")
    print("=" * 60)

    # 初始化可视化器
    visualizer = ModelOutputsVisualizer(args.config, device=args.device)

    # 执行可视化
    visualizer.run(args.input, args.output)


if __name__ == '__main__':
    main()
