#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中间状态可视化脚本

功能：
1. 可视化反向扩散过程中每一步的中间状态图
2. 对比两种不同的噪声策略：
   - 模式A：随机噪声初始化 + 随机噪声采样（原始ResShift）
   - 模式B：随机噪声初始化 + 噪声预测器采样
3. 生成对比图表，方便分析不同策略的差异
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


class IntermediateStepVisualizer:
    """中间状态可视化类"""

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

        # 推理配置（需要在初始化扩散参数之前设置）
        self.num_steps = self.config['inference']['num_steps']

        # 初始化模型
        self._init_models()

        # 初始化扩散参数
        self._init_diffusion()
        self.scale_factor = self.config['inference']['scale_factor']
        self.use_amp = self.config['inference']['use_amp']

        # 颜色校正配置
        self.color_correction = self.config['inference'].get('color_correction', True)

        print(f"✓ 可视化器初始化完成")
        print(f"  - 采样步数: {self.num_steps}")
        print(f"  - 超分倍数: {self.scale_factor}x")

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
                latent_channels=config['latent_channels'],
                model_channels=config['model_channels'],
                channel_mult=tuple(config['channel_mult']),
                num_res_blocks=config['num_res_blocks'],
                growth_rate=config['growth_rate'],
                res_scale=config['res_scale'],
                double_z=config['double_z']
            )
        else:
            self.noise_predictor = create_noise_predictor(
                latent_channels=noise_predictor_config['latent_channels'],
                model_channels=noise_predictor_config['model_channels'],
                channel_mult=tuple(noise_predictor_config['channel_mult']),
                num_res_blocks=noise_predictor_config['num_res_blocks'],
                growth_rate=noise_predictor_config.get('growth_rate', 32),
                res_scale=noise_predictor_config.get('res_scale', 0.1),
                double_z=noise_predictor_config.get('double_z', True)
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

    def prior_sample_predicted(self, y, lr_latent):
        """
        使用噪声预测器从先验分布采样
        q(x_T|y) ~= N(x_T|y, κ²η_T)
        """
        t = torch.tensor([self.num_steps - 1] * y.shape[0], device=y.device).long()
        # 使用噪声预测器预测噪声
        noise = self.noise_predictor(y, lr_latent, t, sample_posterior=True)
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
    def reverse_sampling_with_intermediate(self, lr_latent, lr_image, use_random_noise=True,
                                           use_random_init=True):
        """
        反向采样过程，返回每一步的中间状态

        Args:
            lr_latent: LR图像的潜在表示
            lr_image: 图像空间的LR图像（用作UNet的lq条件）
            use_random_noise: 是否在中间步骤使用随机噪声
            use_random_init: 是否使用随机噪声初始化（否则使用噪声预测器）

        Returns:
            intermediate_states: 每一步的中间状态列表（潜在空间）
            intermediate_images: 每一步的中间状态图像列表
        """
        # 存储中间状态
        intermediate_latents = []
        intermediate_images = []

        # 初始化x_T
        if use_random_init:
            x_t = self.prior_sample_random(lr_latent)
        else:
            x_t = self.prior_sample_predicted(lr_latent, lr_latent)

        # 保存初始状态 (x_T)
        intermediate_latents.append(x_t.clone())
        intermediate_images.append(self.latent_to_image(x_t))

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

            # 4. 生成噪声
            if use_random_noise:
                noise = torch.randn_like(x_t)
            else:
                noise = self.noise_predictor(x_t, lr_latent, t_tensor, sample_posterior=True)

            # 5. 采样x_{t-1}
            nonzero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
            x_t = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

            # 保存中间状态
            intermediate_latents.append(x_t.clone())
            intermediate_images.append(self.latent_to_image(x_t))

        return intermediate_latents, intermediate_images

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
        可视化中间步骤

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

        # 上采样LR图像
        lr_upsampled = F.interpolate(
            lr_tensor,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # 编码到潜在空间
        with torch.no_grad():
            lr_latent = self.vae.encode(lr_upsampled)

        print("  开始采样...")

        # 模式A：随机噪声初始化 + 随机噪声采样
        print("  模式A: 随机噪声初始化 + 随机噪声采样")
        _, images_mode_a = self.reverse_sampling_with_intermediate(
            lr_latent, lr_tensor, use_random_noise=True, use_random_init=True
        )

        # 模式B：随机噪声初始化 + 噪声预测器采样
        print("  模式B: 随机噪声初始化 + 噪声预测器采样")
        _, images_mode_b = self.reverse_sampling_with_intermediate(
            lr_latent, lr_tensor, use_random_noise=False, use_random_init=True
        )

        # 创建输出目录
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取图像名称
        img_name = Path(lr_image_path).stem

        # 准备LR上采样图像用于显示
        lr_upsampled_display = lr_upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()
        lr_upsampled_display = (lr_upsampled_display + 1.0) / 2.0
        lr_upsampled_display = np.clip(lr_upsampled_display, 0, 1)

        # 去除padding
        if pad_h > 0 or pad_w > 0:
            h_end = lr_upsampled_display.shape[0] - pad_h * self.scale_factor
            w_end = lr_upsampled_display.shape[1] - pad_w * self.scale_factor
            lr_upsampled_display = lr_upsampled_display[:h_end, :w_end]

            # 对所有中间状态图像去除padding
            for i in range(len(images_mode_a)):
                images_mode_a[i] = images_mode_a[i][:h_end, :w_end]
                images_mode_b[i] = images_mode_b[i][:h_end, :w_end]

        # 创建对比图
        self._create_comparison_figure(
            lr_image, lr_upsampled_display,
            images_mode_a, images_mode_b,
            output_path / f"{img_name}_comparison.png"
        )

        # 保存单独的中间状态图像
        self._save_individual_images(
            images_mode_a, images_mode_b,
            output_path, img_name
        )

        print(f"  ✓ 可视化结果保存到: {output_path}")

    def _create_comparison_figure(self, lr_image, lr_upsampled,
                                  images_a, images_b, output_file):
        """
        创建对比图表

        Args:
            lr_image: 原始LR图像
            lr_upsampled: 上采样的LR图像
            images_a: 模式A的中间状态图像列表
            images_b: 模式B的中间状态图像列表
            output_file: 输出文件路径
        """
        num_steps = len(images_a)  # 包括初始状态，所以是num_steps + 1

        # 创建图表
        # 行：LR | 模式A | 模式B
        # 列：LR/初始状态 | 步骤1 | 步骤2 | ... | 最终结果
        fig = plt.figure(figsize=(4 * (num_steps + 1), 4 * 3))
        gs = gridspec.GridSpec(3, num_steps + 1, figure=fig, wspace=0.05, hspace=0.1)

        # 第一行：LR图像和上采样图像
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(lr_image)
        ax.set_title('LR Input', fontsize=10)
        ax.axis('off')

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(lr_upsampled)
        ax.set_title('Bicubic Upsampled', fontsize=10)
        ax.axis('off')

        # 填充第一行剩余位置
        for j in range(2, num_steps + 1):
            ax = fig.add_subplot(gs[0, j])
            ax.axis('off')

        # 行标签
        row_labels = [
            'Input',
            'Mode A\n(Random Init +\nRandom Noise)',
            'Mode B\n(Random Init +\nPredictor)'
        ]

        # 列标签
        col_labels = ['Initial (x_T)'] + [f'Step {i + 1}' for i in range(num_steps - 1)] + ['Final (x_0)']

        # 模式A
        for j, img in enumerate(images_a):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(np.clip(img, 0, 1))
            if j == 0:
                ax.set_ylabel(row_labels[1], fontsize=9)
            ax.set_title(col_labels[j], fontsize=9)
            ax.axis('off')

        # 模式B
        for j, img in enumerate(images_b):
            ax = fig.add_subplot(gs[2, j])
            ax.imshow(np.clip(img, 0, 1))
            if j == 0:
                ax.set_ylabel(row_labels[2], fontsize=9)
            ax.axis('off')

        plt.suptitle('Intermediate Steps Comparison\n(Reverse Diffusion Process)',
                     fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    保存对比图: {output_file}")

    def _save_individual_images(self, images_a, images_b, output_path, img_name):
        """
        保存单独的中间状态图像

        Args:
            images_a: 模式A的中间状态图像列表
            images_b: 模式B的中间状态图像列表
            output_path: 输出目录
            img_name: 图像名称
        """
        # 创建子目录
        mode_a_dir = output_path / "mode_a_random_random"
        mode_b_dir = output_path / "mode_b_random_predictor"

        mode_a_dir.mkdir(parents=True, exist_ok=True)
        mode_b_dir.mkdir(parents=True, exist_ok=True)

        # 保存模式A的图像
        for i, img in enumerate(images_a):
            if i == 0:
                step_name = "initial_xT"
            elif i == len(images_a) - 1:
                step_name = "final_x0"
            else:
                step_name = f"step_{i}"

            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(mode_a_dir / f"{img_name}_{step_name}.png"), img_bgr)

        # 保存模式B的图像
        for i, img in enumerate(images_b):
            if i == 0:
                step_name = "initial_xT"
            elif i == len(images_b) - 1:
                step_name = "final_x0"
            else:
                step_name = f"step_{i}"

            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(mode_b_dir / f"{img_name}_{step_name}.png"), img_bgr)

        print(f"    保存单独图像到: {mode_a_dir}, {mode_b_dir}")

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
        for img_path in tqdm(image_paths, desc="可视化中间状态"):
            self.visualize_intermediate_steps(img_path, output_path)

        print(f"\n✓ 全部完成！结果保存在: {output_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="中间状态可视化脚本")
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
    print("中间状态可视化脚本")
    print("=" * 60)
    print("\n两种模式说明：")
    print("  模式A: 随机噪声初始化 + 随机噪声采样（原始ResShift）")
    print("  模式B: 随机噪声初始化 + 噪声预测器采样")
    print("=" * 60)

    # 初始化可视化器
    visualizer = IntermediateStepVisualizer(args.config, device=args.device)

    # 执行可视化
    visualizer.run(args.input, args.output)


if __name__ == '__main__':
    main()
