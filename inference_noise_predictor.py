#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
噪声预测器推理脚本

功能：
1. 加载训练好的噪声预测器
2. 对LR图像进行超分辨率重建
3. 支持chop分块处理大图像
4. 自动padding不足尺寸的图像
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
from contextlib import nullcontext

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# SR模块导入
from SR.models.noise_predictor import create_noise_predictor
from SR.models.unet import UNetModelSwin
from SR.ldm.models.autoencoder import VQModelTorch


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


def space_timesteps(num_timesteps, sample_timesteps):
    """
    创建用于采样的时间步列表（从原始扩散过程中均匀选取）

    Args:
        num_timesteps: 原始扩散过程的总步数
        sample_timesteps: 采样时使用的步数

    Returns:
        use_timesteps: 选中的时间步集合
    """
    all_steps = [int((num_timesteps / sample_timesteps) * x) for x in range(sample_timesteps)]
    return set(all_steps)


class ImageSpliterTh:
    """图像分块处理类（用于处理大图像）"""

    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
            extra_bs: aggregate pchs to processing
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs

        bs, chn, height, width = im.shape
        self.true_bs = bs

        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = self.__len__()
        self.count_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height * sf, width * sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height * sf, width * sf], dtype=im.dtype, device=im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0, ]
        else:
            starts = list(range(0, length, self.stride))
            for ii in range(len(starts)):
                if starts[ii] + self.pch_size > length:
                    starts[ii] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[self.count_pchs:self.count_pchs + self.extra_bs]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_size
                h_end = h_start + self.pch_size
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch = current_pch
                else:
                    pch = torch.cat([pch, current_pch], dim=0)

                h_start *= self.sf
                h_end *= self.sf
                w_start *= self.sf
                w_end *= self.sf
                index_infos.append([h_start, h_end, w_start, w_end])

            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()

        return pch, index_infos

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
        '''
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii]
            self.im_res[:, :, h_start:h_end, w_start:w_end] += current_pch
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)


class NoisePredictorInference:
    """噪声预测器推理类"""

    def __init__(self, config_path, device='cuda'):
        """
        初始化推理器

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
        self.chop_size = self.config['inference']['chop_size']
        self.chop_stride = self.config['inference']['chop_stride']
        self.chop_bs = self.config['inference']['chop_bs']
        self.use_amp = self.config['inference']['use_amp']
        self.use_random_noise = self.config['inference'].get('use_random_noise', False)

        # 颜色校正配置（解决超分后图像颜色偏移问题）
        self.color_correction = self.config['inference'].get('color_correction', True)

        print(f"✓ 推理器初始化完成")
        print(f"  - 采样步数: {self.num_steps}")
        print(f"  - 超分倍数: {self.scale_factor}x")
        print(f"  - Chop尺寸: {self.chop_size}x{self.chop_size}")
        print(f"  - Chop步长: {self.chop_stride}")
        print(f"  - 颜色校正: {'启用' if self.color_correction else '禁用'}")

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

        # 智能处理前缀：检测checkpoint中的前缀格式
        first_key = list(state_dict.keys())[0]
        has_module_prefix = first_key.startswith('module.')
        has_orig_mod_prefix = '_orig_mod.' in first_key

        # 根据需要去除或添加前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if has_orig_mod_prefix:
                new_key = new_key.replace('_orig_mod.', '')
            if has_module_prefix:
                new_key = new_key.replace('module.', '')
            new_state_dict[new_key] = value

        # 使用strict=True确保所有权重都正确加载
        missing_keys, unexpected_keys = self.vae.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"  ⚠️ VAE缺少的权重: {missing_keys}")
        if unexpected_keys:
            print(f"  ⚠️ VAE多余的权重: {unexpected_keys[:5]}...")  # 只显示前5个

        # 检查关键的quantize embedding权重是否正确加载
        if hasattr(self.vae, 'quantize') and hasattr(self.vae.quantize, 'embedding'):
            embed_weight = self.vae.quantize.embedding.weight
            print(
                f"  VQ embedding shape: {embed_weight.shape}, mean: {embed_weight.mean().item():.4f}, std: {embed_weight.std().item():.4f}")

        self.vae = self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("  ✓ VAE加载完成")

        # 2. 加载ResShift UNet
        print("  加载ResShift UNet...")
        unet_config = self.config['resshift_unet']
        self.resshift_unet = UNetModelSwin(**unet_config)

        # 加载预训练权重
        resshift_ckpt = torch.load(self.config['model']['resshift_path'], map_location='cpu')

        # 处理state_dict格式
        if 'state_dict' in resshift_ckpt:
            state_dict = resshift_ckpt['state_dict']
        else:
            state_dict = resshift_ckpt

        # 去除可能的前缀
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
        
        # 如果指定了配置文件路径，则加载配置
        if 'config_path' in noise_predictor_config:
            with open(noise_predictor_config['config_path'], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # 使用配置文件中的参数
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
            # 使用直接指定的参数
            self.noise_predictor = create_noise_predictor(
                latent_channels=noise_predictor_config['latent_channels'],
                model_channels=noise_predictor_config['model_channels'],
                channel_mult=tuple(noise_predictor_config['channel_mult']),
                num_res_blocks=noise_predictor_config['num_res_blocks'],
                growth_rate=noise_predictor_config.get('growth_rate', 32),
                res_scale=noise_predictor_config.get('res_scale', 0.1),
                double_z=noise_predictor_config.get('double_z', True)
            )

        # 加载权重
        noise_ckpt = torch.load(self.config['model']['noise_predictor_path'], map_location='cpu')

        # 智能处理不同的checkpoint格式
        if isinstance(noise_ckpt, dict):
            if 'noise_predictor' in noise_ckpt:
                # 完整训练checkpoint格式（checkpoint_epoch_*.pth）
                state_dict = noise_ckpt['noise_predictor']
                print(f"  从训练checkpoint加载 (epoch {noise_ckpt.get('epoch', 'unknown')})")
            elif 'model_state_dict' in noise_ckpt:
                # 标准格式
                state_dict = noise_ckpt['model_state_dict']
            else:
                # 直接是state_dict（best_model.pth的新格式）
                state_dict = noise_ckpt
                print(f"  从best_model.pth加载（仅权重）")
        else:
            raise ValueError(f"不支持的checkpoint格式: {type(noise_ckpt)}")

        self.noise_predictor.load_state_dict(state_dict, strict=True)
        self.noise_predictor = self.noise_predictor.to(self.device)
        self.noise_predictor.eval()
        for param in self.noise_predictor.parameters():
            param.requires_grad = False
        print("  ✓ 噪声预测器加载完成")

    def _init_diffusion(self):
        """
        初始化扩散参数

        ResShift v3直接用steps=4训练，timestep_respacing=None
        因此不需要时间步重映射，直接使用num_steps步的扩散参数
        """
        diffusion_config = self.config['diffusion']

        # 扩散步数（ResShift v3直接用num_steps训练）
        self.diffusion_num_timesteps = diffusion_config['num_timesteps']  # 应该等于num_steps
        self.kappa = diffusion_config['kappa']
        self.normalize_input = diffusion_config.get('normalize_input', True)
        self.latent_flag = diffusion_config.get('latent_flag', True)
        self.scale_factor = diffusion_config.get('scale_factor', 1.0)

        # 计算eta调度（直接用num_timesteps步）
        sqrt_etas = get_named_eta_schedule(
            schedule_name=diffusion_config['eta_schedule'],
            num_diffusion_timesteps=self.diffusion_num_timesteps,
            min_noise_level=diffusion_config['min_noise_level'],
            etas_end=diffusion_config['etas_end'],
            kappa=self.kappa,
            power=diffusion_config['eta_power']
        )

        # 直接使用sqrt_etas（长度=num_timesteps）
        self.sqrt_etas = sqrt_etas.astype(np.float64)
        self.etas = self.sqrt_etas ** 2

        # 计算alpha（ResShift定义：alpha_t = eta_t - eta_{t-1}）
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev  # 这才是ResShift中的alpha！

        # 计算后验分布参数
        self.posterior_variance = self.kappa ** 2 * self.etas_prev / self.etas * self.alpha
        # 处理t=0时的方差（避免除以0和NaN）
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)

        # 后验均值系数
        self.posterior_mean_coef1 = self.etas_prev / self.etas  # η_{t-1}/η_t
        self.posterior_mean_coef2 = self.alpha / self.etas  # α_t/η_t
        # 处理t=0时的除以0问题
        self.posterior_mean_coef1[0] = 0.0
        self.posterior_mean_coef2[0] = 1.0  # 当t=0时，后验均值直接是x_0

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
        print(f"    - 采样步数: {self.num_steps}")
        print(f"    - κ: {self.kappa}")
        print(f"    - sqrt_etas: {self.sqrt_etas.numpy()}")
        print(f"    - posterior_mean_coef1: {self.posterior_mean_coef1.numpy()}")
        print(f"    - posterior_mean_coef2: {self.posterior_mean_coef2.numpy()}")

    def _wavelet_blur(self, image: torch.Tensor, radius: int):
        """
        对输入tensor应用小波模糊

        Args:
            image: 输入图像 tensor (B, C, H, W)
            radius: 模糊半径

        Returns:
            模糊后的tensor
        """
        # 卷积核 - 高斯模糊核
        kernel_vals = [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ]
        kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
        # 添加通道维度，变成4D tensor
        kernel = kernel[None, None]
        # 在所有输入通道上重复
        kernel = kernel.repeat(3, 1, 1, 1)
        # 使用replicate模式进行padding
        image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
        # 应用分组卷积
        output = F.conv2d(image, kernel, groups=3, dilation=radius)
        return output

    def _wavelet_decomposition(self, image: torch.Tensor, levels: int = 5):
        """
        对输入tensor进行小波分解

        Args:
            image: 输入图像 tensor (B, C, H, W)
            levels: 分解层数

        Returns:
            high_freq: 高频分量（细节信息）
            low_freq: 低频分量（颜色/亮度信息）
        """
        high_freq = torch.zeros_like(image)
        for i in range(levels):
            radius = 2 ** i
            low_freq = self._wavelet_blur(image, radius)
            high_freq += (image - low_freq)
            image = low_freq

        return high_freq, low_freq

    def _color_correction(self, sr_tensor, lr_tensor):
        """
        颜色校正：使用小波重建方法来校正SR图像的颜色偏移

        原理：
        - 对SR图像进行小波分解，提取高频分量（纹理、边缘等细节）
        - 对LR图像进行小波分解，提取低频分量（整体颜色、亮度）
        - 将SR的高频 + LR的低频进行重建，保留SR的细节同时修正颜色

        Args:
            sr_tensor: SR图像 tensor (B, C, H, W), [-1, 1]
            lr_tensor: LR图像 tensor (B, C, H, W), [-1, 1]

        Returns:
            颜色校正后的SR图像 tensor
        """
        # 将范围从 [-1, 1] 转换到 [0, 1] 以进行小波处理
        sr_01 = (sr_tensor + 1.0) / 2.0
        lr_01 = (lr_tensor + 1.0) / 2.0

        # 对SR图像进行小波分解，提取高频分量（细节信息）
        sr_high_freq, _ = self._wavelet_decomposition(sr_01)

        # 对LR图像进行小波分解，提取低频分量（颜色信息）
        _, lr_low_freq = self._wavelet_decomposition(lr_01)

        # 重建：SR的高频（细节）+ LR的低频（颜色）
        corrected_01 = sr_high_freq + lr_low_freq

        # Clamp到 [0, 1] 范围
        corrected_01 = torch.clamp(corrected_01, 0.0, 1.0)

        # 转换回 [-1, 1] 范围
        corrected = corrected_01 * 2.0 - 1.0

        return corrected

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        从数组中提取值并广播到目标形状

        Args:
            arr: 一维tensor数组
            timesteps: 时间步索引
            broadcast_shape: 目标形状

        Returns:
            广播后的tensor
        """
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _scale_input(self, inputs, t):
        """
        对输入进行归一化（ResShift的关键步骤！）

        Args:
            inputs: 输入tensor
            t: 时间步索引（在重映射后的空间中）
        """
        if self.normalize_input:
            if self.latent_flag:
                # 潜在空间的方差约为1.0
                std = torch.sqrt(self._extract_into_tensor(self.etas, t, inputs.shape) * self.kappa ** 2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = self._extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        计算ResShift后验分布 q(x_{t-1}|x_t, x_0)

        后验均值：μ = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0
        后验方差：σ² = κ²·η_{t-1}·α_t/η_t

        其中 α_t = η_t - η_{t-1}（ResShift定义）

        Args:
            x_0: 预测的x_0 (pred_xstart)
            x_t: 当前x_t
            t: 时间步索引（在重映射后的空间中，0到num_steps-1）

        Returns:
            mean: 后验均值
            variance: 后验方差
            log_variance: 后验方差的对数（clipped）
        """
        # 使用预计算的系数
        mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_0
        )

        variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, variance, log_variance

    @torch.no_grad()
    def sample_func(self, lr_tensor):
        """
        完整的超分流程（对应ResShift的sample_func）

        Args:
            lr_tensor: LR图像 (B, C, H, W), [-1, 1], RGB

        Returns:
            sr_tensor: SR图像 (B, C, H*sf, W*sf), [-1, 1], RGB
        """
        # 调试：检查输入范围
        # print(f"  [Debug] lr_tensor: min={lr_tensor.min().item():.3f}, max={lr_tensor.max().item():.3f}")

        # 1. 上采样LR图像
        lr_upsampled = F.interpolate(
            lr_tensor,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # 2. 编码到潜在空间
        with torch.no_grad():
            lr_latent = self.vae.encode(lr_upsampled)
        # print(f"  [Debug] lr_latent: min={lr_latent.min().item():.3f}, max={lr_latent.max().item():.3f}")

        # 3. 反向采样
        # 注意：UNet的lq条件需要图像空间的LR，不是潜在空间的
        sr_latent = self.reverse_sampling(lr_latent, lr_tensor)
        # print(f"  [Debug] sr_latent: min={sr_latent.min().item():.3f}, max={sr_latent.max().item():.3f}")

        # 4. 解码到图像空间
        with torch.no_grad():
            sr_tensor = self.vae.decode(sr_latent)
        # print(f"  [Debug] sr_tensor (after VAE decode): min={sr_tensor.min().item():.3f}, max={sr_tensor.max().item():.3f}")

        # 5. clamp到有效范围，防止颜色溢出
        sr_tensor = torch.clamp(sr_tensor, -1.0, 1.0)

        # 6. 【颜色校正】使用小波重建校正颜色偏移
        if self.color_correction:
            sr_tensor = self._color_correction(sr_tensor, lr_upsampled)

        return sr_tensor

    def prior_sample(self, y, noise=None):
        """
        从先验分布采样，即 q(x_T|y) ~= N(x_T|y, κ²η_T)

        【重要】初始化和中间步骤保持一致：
        - use_random_noise=True: 都用随机噪声
        - use_random_noise=False: 都用噪声预测器

        Args:
            y: 退化图像的潜在表示（lr_latent）
            noise: 可选的噪声

        Returns:
            x_T: 初始采样
        """
        # 使用最后一个时间步（即num_steps-1，对应原始的最大时间步）
        t = torch.tensor([self.num_steps - 1] * y.shape[0], device=y.device).long()

        if noise is None:
            # 初始化时总是使用随机高斯噪声（不使用噪声预测器）
            noise = torch.randn_like(y)

        return y + self._extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def reverse_sampling(self, lr_latent, lr_image):
        """
        ResShift反向采样过程（推理时使用）

        完全按照ResShift原项目的p_sample_loop_progressive实现

        注意：ResShift v3直接用4步训练（timestep_respacing=None），
        所以不需要时间步重映射，直接使用索引0-3作为时间步

        Args:
            lr_latent: LR图像的潜在表示 y（已通过VAE编码）
            lr_image: 图像空间的LR图像（用作UNet的lq条件）

        Returns:
            x_0: 最终的SR潜在表示
        """
        # 【关键】使用prior_sample初始化x_T = y + κ·√η_T·ε
        x_t = self.prior_sample(lr_latent)

        # 反向采样：从num_steps-1到0
        indices = list(range(self.num_steps))[::-1]  # [num_steps-1, num_steps-2, ..., 0]

        for i in indices:
            # 时间步索引（0到num_steps-1）
            t_tensor = torch.tensor([i] * lr_latent.shape[0], device=self.device).long()

            # 1. 对输入进行归一化
            x_t_normalized = self._scale_input(x_t, t_tensor)

            # 2. 使用ResShift的UNet预测x_0
            # ResShift v3直接用4步训练，所以时间步直接传入i
            # 注意：lq应该是图像空间的LR图像，不是潜在空间的lr_latent！
            pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)

            # 3. 计算ResShift后验分布
            mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t_tensor)

            # 4. 生成噪声
            # 根据use_random_noise选择中间采样的噪声来源
            if self.use_random_noise:
                # 使用随机高斯噪声
                noise = torch.randn_like(x_t)
            else:
                # 使用噪声预测器预测噪声
                # EDSR-Unet噪声预测器需要(z_t, lr_latent, timestep)作为输入
                noise = self.noise_predictor(x_t, lr_latent, t_tensor, sample_posterior=True)

            # 5. 采样x_{t-1}：当t>0时添加噪声，t=0时直接使用均值
            nonzero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
            x_t = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

        return x_t

    def pad_image(self, img, multiple=64):
        """
        Padding图像到multiple的倍数

        Args:
            img: 输入图像 (H, W, C) numpy array
            multiple: 倍数

        Returns:
            padded_img: padding后的图像
            (pad_h, pad_w): padding的尺寸
        """
        h, w = img.shape[:2]

        # 计算需要padding的尺寸
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            # 使用反射padding
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        return img, (pad_h, pad_w)

    def process_single_image(self, lr_image):
        """
        处理单张图像（完全按照ResShift流程）

        Args:
            lr_image: LR图像 (H, W, C) numpy array, [0, 1], RGB

        Returns:
            sr_image: SR图像 (H*4, W*4, C) numpy array, [0, 1], RGB
        """
        # 1. Padding图像
        padding_offset = self.config['inference'].get('padding_offset', 64)

        # 先padding到64的倍数（与ResShift保持一致）
        lr_padded, (pad_h, pad_w) = self.pad_image(lr_image, multiple=padding_offset)

        # 确保是2的幂次（用于FFT操作）
        h, w = lr_padded.shape[:2]
        # 找到大于等于h和w的最小2的幂次
        next_pow2_h = 1 if h == 0 else 2 ** ((h - 1).bit_length())
        next_pow2_w = 1 if w == 0 else 2 ** ((w - 1).bit_length())

        if h != next_pow2_h or w != next_pow2_w:
            # 需要额外的padding到2的幂次
            extra_pad_h = next_pow2_h - h
            extra_pad_w = next_pow2_w - w
            lr_padded = np.pad(lr_padded, ((0, extra_pad_h), (0, extra_pad_w), (0, 0)), mode='reflect')
            pad_h += extra_pad_h
            pad_w += extra_pad_w

        # 2. 转换为tensor
        lr_tensor = torch.from_numpy(lr_padded).permute(2, 0, 1).unsqueeze(0).float()  # 1 x C x H x W
        lr_tensor = lr_tensor.to(self.device)

        # 3. 归一化到[-1, 1]
        lr_tensor = lr_tensor * 2.0 - 1.0

        # 4. 判断是否需要chop（在图像空间判断！）
        context = lambda: torch.amp.autocast('cuda') if self.use_amp else nullcontext

        if lr_tensor.shape[2] > self.chop_size or lr_tensor.shape[3] > self.chop_size:
            # 使用chop处理大图像（在图像空间！）
            print(f"  使用chop处理 (图像空间尺寸: {lr_tensor.shape[3]}x{lr_tensor.shape[2]})")

            im_spliter = ImageSpliterTh(
                lr_tensor,
                self.chop_size,
                stride=self.chop_stride,
                sf=self.scale_factor,  # 超分倍数
                extra_bs=self.chop_bs,
            )

            for lr_pch, index_infos in im_spliter:
                with context():
                    # 对每个patch进行完整的超分流程
                    sr_pch = self.sample_func(lr_pch)
                im_spliter.update(sr_pch, index_infos)

            sr_tensor = im_spliter.gather()
        else:
            # 直接处理
            print(f"  直接处理 (图像空间尺寸: {lr_tensor.shape[3]}x{lr_tensor.shape[2]})")
            with context():
                sr_tensor = self.sample_func(lr_tensor)

        # 5. 反归一化到[0, 1]（sr_tensor已在sample_func中clamp到[-1,1]）
        sr_tensor = sr_tensor * 0.5 + 0.5
        # 额外clamp确保在[0, 1]范围内
        sr_tensor = torch.clamp(sr_tensor, 0, 1)

        # 6. 转换为numpy
        sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 7. 去除padding
        if pad_h > 0 or pad_w > 0:
            h_end = sr_image.shape[0] - pad_h * self.scale_factor
            w_end = sr_image.shape[1] - pad_w * self.scale_factor
            sr_image = sr_image[:h_end, :w_end]

        return sr_image

    def inference(self, input_path, output_path):
        """
        推理入口

        Args:
            input_path: 输入路径（图像或文件夹）
            output_path: 输出路径
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

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
        for img_path in tqdm(image_paths, desc="处理图像"):
            # 读取图像
            lr_image = cv2.imread(str(img_path))
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
            lr_image = lr_image.astype(np.float32) / 255.0

            print(f"\n处理: {img_path.name} (尺寸: {lr_image.shape[1]}x{lr_image.shape[0]})")

            # 超分辨率
            sr_image = self.process_single_image(lr_image)

            # 保存结果
            sr_image = (sr_image * 255.0).astype(np.uint8)
            if self.config['inference']['rgb2bgr']:
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)

            output_file = output_path / f"{img_path.stem}_sr.png"
            cv2.imwrite(str(output_file), sr_image)

            print(f"  ✓ 保存到: {output_file}")

        print(f"\n✓ 全部完成！结果保存在: {output_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="噪声预测器推理脚本")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="输入路径（图像或文件夹）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./results",
        help="输出路径"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="SR/configs/inference_noise_predictor.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="采样步数（覆盖配置文件）"
    )
    parser.add_argument(
        "--use_random_noise",
        action="store_true",
        help="使用随机噪声（原始ResShift方式）而非噪声预测器"
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
    print("噪声预测器推理脚本")
    print("=" * 60)

    # 初始化推理器
    inferencer = NoisePredictorInference(args.config, device=args.device)

    # 覆盖采样步数
    if args.num_steps is not None:
        inferencer.num_steps = args.num_steps
        print(f"采样步数已覆盖为: {args.num_steps}")

    # 覆盖噪声模式
    if args.use_random_noise:
        inferencer.use_random_noise = True

    # 打印最终的噪声策略
    if inferencer.use_random_noise:
        print(f"噪声策略: 初始化(x_T)使用随机高斯噪声 + 中间采样使用随机高斯噪声")
    else:
        print(f"噪声策略: 初始化(x_T)使用随机高斯噪声 + 中间采样使用噪声预测器")

    # 执行推理
    inferencer.inference(args.input, args.output)


if __name__ == '__main__':
    main()
