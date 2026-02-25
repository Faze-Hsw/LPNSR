#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多步端到端训练噪声预测器（与推理流程完全一致）

训练思路：
1. 输入HR图像和LR图像
2. 使用冻结的VAE编码到潜在空间
3. 使用噪声预测器初始化 x_T = z_y + κ·√η_T·ε
4. 多步反向采样（与推理完全一致）：
   - UNet预测x_0
   - 噪声预测器预测采样噪声
   - 后验采样得到x_{t-1}
5. 计算最终x_0与真实z_start的损失
6. 反向传播，只更新噪声预测器

关键：多步训练与推理流程完全一致，避免训练-推理不一致问题！
- 使用eta调度而非beta调度
- 残差偏移扩散：x_t = (1-η_t)·x_0 + η_t·y + √η_t·κ·ε
- 后验分布：μ = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0
"""

import os
import sys
import warnings

# 在导入其他模块之前设置警告过滤器
warnings.filterwarnings("ignore", message=".*A matching Triton is not available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境

# LPNSR模块导入
from LPNSR.models.noise_predictor import create_noise_predictor
from LPNSR.models.unet import UNetModelSwin
from LPNSR.ldm.models.autoencoder import VQModelTorch
from LPNSR.losses.lpips_loss import LPIPSLoss
from LPNSR.losses.gan_loss import GANLoss, create_discriminator
from LPNSR.datapipe.train_dataloader import create_train_dataloader


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    获取ResShift的eta调度

    这是ResShift特有的噪声调度方式，与DDPM的beta调度完全不同！

    Args:
        schedule_name: 调度类型（'exponential' 或 'ldm'）
        num_diffusion_timesteps: 扩散步数T
        min_noise_level: 最小噪声水平η_1
        etas_end: 最大噪声水平η_T
        kappa: 方差控制参数κ
        kwargs: 额外参数（如power）

    Returns:
        sqrt_etas: √η_t数组，shape=(T,)
    """
    if kwargs is None:
        kwargs = {}

    if schedule_name == 'exponential':
        # 指数调度（ResShift默认）
        power = kwargs.get('power', 2.0)
        etas_start = min(min_noise_level / kappa, min_noise_level)

        # 计算增长因子
        increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser

        # 计算幂次时间步
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        power_timestep *= (num_diffusion_timesteps - 1)

        # 计算sqrt_etas
        sqrt_etas = np.power(base, power_timestep) * etas_start

    elif schedule_name == 'ldm':
        # 从.mat文件加载
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        if mat_path is None:
            raise ValueError("ldm schedule需要提供mat_path")
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)

    else:
        raise ValueError(f"未知的schedule_name: {schedule_name}")

    return sqrt_etas


class NoisePredictorTrainer:
    """噪声预测器端到端训练器"""

    def __init__(self, config_path):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建实验目录
        self.exp_dir = Path(self.config['experiment']['save_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'samples').mkdir(exist_ok=True)

        # 初始化模型
        self._init_models()

        # 初始化损失函数
        self._init_losses()

        # 初始化优化器
        self._init_optimizer()

        # 初始化数据加载器
        self._init_dataloaders()

        # 初始化AMP
        if self.config['training']['use_amp']:
            self.scaler_g = GradScaler()  # 生成器专用scaler
            if self.discriminator is not None:
                self.scaler_d = GradScaler()  # 判别器专用scaler
        else:
            self.scaler_g = None
            self.scaler_d = None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        print(f"训练器初始化完成！")
        print(f"实验目录: {self.exp_dir}")
        print(f"设备: {self.device}")

    def _init_models(self):
        """初始化模型"""
        print("\n" + "=" * 70)
        print("初始化模型")
        print("=" * 70)

        # 1. 加载VQVAE（冻结）
        print("\n加载VQVAE...")
        vae_path = self.config['resshift']['vae_path']

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

        lora_rank = 8
        lora_alpha = 1.0
        lora_tune_decoder = False

        self.vae = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
            rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_tune_decoder=lora_tune_decoder,
        ).to(self.device)

        # 加载预训练权重
        vae_ckpt = torch.load(vae_path, map_location=self.device)
        if 'state_dict' in vae_ckpt:
            state_dict = vae_ckpt['state_dict']
        else:
            state_dict = vae_ckpt

        # 去除前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module._orig_mod.'):
                new_key = key.replace('module._orig_mod.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        self.vae.load_state_dict(new_state_dict, strict=False)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print(f"✓ VQVAE加载完成: {vae_path}")

        # 2. 加载ResShift UNet（冻结）
        print("\n加载ResShift UNet...")
        unet_path = self.config['resshift']['unet_path']

        crop_size = self.config['data']['train']['crop_size']
        vae_downsample_factor = 4
        latent_size = crop_size // vae_downsample_factor

        model_structure = {
            'image_size': latent_size,
            'in_channels': 3,
            'model_channels': 160,
            'out_channels': 3,
            'attention_resolutions': [64, 32, 16, 8],
            'channel_mult': [1, 2, 2, 4],
            'num_res_blocks': [2, 2, 2, 2],
            'num_head_channels': 32,
            'use_scale_shift_norm': True,
            'resblock_updown': False,
            'swin_depth': 2,
            'swin_embed_dim': 192,
            'window_size': 8,
            'mlp_ratio': 4,
            'cond_lq': True,
            'lq_size': latent_size,
        }

        model_config = {
            **model_structure,
            'dropout': 0.0,
            'use_fp16': False,
            'conv_resample': True,
            'dims': 2,
            'patch_norm': False,
        }

        self.resshift_unet = UNetModelSwin(**model_config).to(self.device)

        # 加载预训练权重
        unet_ckpt = torch.load(unet_path, map_location=self.device)
        if 'state_dict' in unet_ckpt:
            state_dict = unet_ckpt['state_dict']
        else:
            state_dict = unet_ckpt

        # 去除前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module._orig_mod.'):
                new_key = key.replace('module._orig_mod.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        self.resshift_unet.load_state_dict(new_state_dict, strict=True)
        self.resshift_unet.eval()
        for param in self.resshift_unet.parameters():
            param.requires_grad = False
        print(f"✓ ResShift UNet加载完成: {unet_path}")

        # 3. 初始化ResShift扩散过程参数
        print("\n初始化ResShift扩散过程...")
        diffusion_config = self.config['training']['diffusion']
        self.num_timesteps = diffusion_config['num_timesteps']
        self.sampling_steps = self.config['training']['sampling_steps']

        # ResShift特有参数
        self.kappa = diffusion_config['kappa']
        self.normalize_input = diffusion_config.get('normalize_input', True)
        self.latent_flag = diffusion_config.get('latent_flag', True)
        eta_schedule = diffusion_config['eta_schedule']
        min_noise_level = diffusion_config['min_noise_level']
        etas_end = diffusion_config['etas_end']
        eta_power = diffusion_config.get('eta_power', 0.3)

        # 计算eta调度（ResShift方式）
        sqrt_etas = get_named_eta_schedule(
            schedule_name=eta_schedule,
            num_diffusion_timesteps=self.num_timesteps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=self.kappa,
            kwargs={'power': eta_power}
        )

        # 转换为torch tensor
        self.sqrt_etas = torch.from_numpy(sqrt_etas).float()
        self.etas = self.sqrt_etas ** 2

        # 计算etas_prev和alpha（ResShift方式）
        # alpha_t = eta_t - eta_{t-1}，这是ResShift的正确定义！
        self.etas_prev = torch.cat([torch.tensor([0.0]), self.etas[:-1]])
        self.alpha = self.etas - self.etas_prev  # 增量

        # 计算后验分布参数（ResShift方式）
        # q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t, σ̃_t²·I)
        # μ̃_t = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0
        # σ̃_t² = κ²·(η_{t-1}/η_t)·α_t
        self.posterior_mean_coef1 = self.etas_prev / self.etas  # η_{t-1}/η_t
        self.posterior_mean_coef2 = self.alpha / self.etas  # α_t/η_t
        self.posterior_variance = self.kappa ** 2 * self.etas_prev / self.etas * self.alpha

        # 处理t=0的边界情况（避免NaN）
        self.posterior_mean_coef1[0] = 0.0  # t=0时，eta_prev=0，所以coef1=0
        self.posterior_mean_coef2[0] = 1.0  # t=0时，后验均值直接是x_0
        self.posterior_variance[0] = self.posterior_variance[1]  # 避免除零
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        print(f"✓ ResShift扩散过程初始化完成")
        print(f"  - 噪声调度类型: {eta_schedule}（ResShift特有）")
        print(f"  - 总时间步数T: {self.num_timesteps}")
        print(f"  - κ (kappa): {self.kappa}")
        print(f"  - η范围: [{self.etas[0]:.4f}, {self.etas[-1]:.4f}]")
        print(f"  - √η范围: [{self.sqrt_etas[0]:.4f}, {self.sqrt_etas[-1]:.4f}]")
        print(f"  - 采样步数S: {self.sampling_steps}")
        print(f"  - 输入归一化: {self.normalize_input}")
        print(f"  - 初始化分布: N(y, κ²·η_T·I) = N(y, {self.kappa ** 2 * self.etas[-1]:.4f}·I)")

        # 5. 创建噪声预测器（训练）
        print("\n创建噪声预测器...")
        noise_config = self.config['noise_predictor']
        # 如果指定了配置文件路径，则加载配置
        if 'config_path' in noise_config:
            with open(noise_config['config_path'], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.noise_predictor = create_noise_predictor(
                image_size=config.get('image_size', latent_size),
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
                lq_size=config.get('lq_size', latent_size),
                use_gradient_checkpointing=self.config['training'].get('use_gradient_checkpointing', False),
                double_z=config.get('double_z', True),
            ).to(self.device)
        else:
            self.noise_predictor = create_noise_predictor(
                image_size=noise_config.get('image_size', latent_size),
                latent_channels=noise_config['latent_channels'],
                model_channels=noise_config['model_channels'],
                out_channels=noise_config.get('out_channels', noise_config['latent_channels']),
                channel_mult=tuple(noise_config['channel_mult']),
                num_res_blocks=noise_config['num_res_blocks'],
                attention_resolutions=noise_config.get('attention_resolutions', [64, 32, 16, 8]),
                dropout=noise_config.get('dropout', 0.0),
                conv_resample=noise_config.get('conv_resample', True),
                dims=noise_config.get('dims', 2),
                use_fp16=noise_config.get('use_fp16', False),
                num_heads=noise_config.get('num_heads', -1),
                num_head_channels=noise_config.get('num_head_channels', 32),
                use_scale_shift_norm=noise_config.get('use_scale_shift_norm', True),
                resblock_updown=noise_config.get('resblock_updown', False),
                swin_depth=noise_config.get('swin_depth', 2),
                swin_embed_dim=noise_config.get('swin_embed_dim', 192),
                window_size=noise_config.get('window_size', 8),
                mlp_ratio=noise_config.get('mlp_ratio', 4.0),
                patch_norm=noise_config.get('patch_norm', False),
                cond_lq=noise_config.get('cond_lq', True),
                lq_size=noise_config.get('lq_size', latent_size),
                use_gradient_checkpointing=self.config['training'].get('use_gradient_checkpointing', False),
                double_z=noise_config.get('double_z', True),
            ).to(self.device)

        num_params = sum(p.numel() for p in self.noise_predictor.parameters())
        print(f"✓ 噪声预测器创建完成")
        print(f"  - 参数量: {num_params / 1e6:.2f}M")
        print(f"  - 梯度检查点: {self.config['training']['use_gradient_checkpointing']}")

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.noise_predictor.parameters() if p.requires_grad)
        print(f"\n总可训练参数: {total_params / 1e6:.2f}M")

    def _init_losses(self):
        """初始化损失函数"""
        print("\n" + "=" * 70)
        print("初始化损失函数")
        print("=" * 70)

        loss_config = self.config['loss']

        # L1损失（图像空间，不需要单独初始化，使用 F.l1_loss）
        print(f"✓ L1损失 (权重: {loss_config.get('l1_weight', 1.0)})")

        # LPIPS感知损失
        if loss_config.get('lpips_weight', 0) > 0:
            self.lpips_loss = LPIPSLoss(
                loss_weight=1.0,
                net_type=loss_config.get('lpips_net_type', 'alex')
            )
            print(f"✓ LPIPS感知损失 (权重: {loss_config['lpips_weight']})")
        else:
            self.lpips_loss = None

        # GAN损失
        if loss_config.get('gan_weight', 0) > 0:
            # 创建判别器
            self.discriminator = create_discriminator(
                disc_type=loss_config.get('disc_type', 'patch'),
                input_nc=3,
                ndf=loss_config.get('disc_ndf', 64),
                n_layers=loss_config.get('disc_n_layers', 3),
                norm_type=loss_config.get('disc_norm_type', 'spectral')
            ).to(self.device)

            # 创建GAN损失
            self.gan_loss = GANLoss(
                gan_type=loss_config.get('gan_type', 'lsgan'),
                loss_weight=1.0
            )

            # 统计判别器参数量
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            print(f"✓ GAN损失 (权重: {loss_config['gan_weight']})")
            print(f"  - 判别器类型: {loss_config.get('disc_type', 'patch')}")
            print(f"  - GAN类型: {loss_config.get('gan_type', 'lsgan')}")
            print(f"  - 判别器参数量: {disc_params / 1e6:.2f}M")
        else:
            self.discriminator = None
            self.gan_loss = None

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        print("\n" + "=" * 70)
        print("初始化优化器")
        print("=" * 70)

        opt_config = self.config['optimizer']

        # 优化器
        if opt_config['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.noise_predictor.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.noise_predictor.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {opt_config['type']}")

        print(f"✓ 优化器: {opt_config['type']}")
        print(f"  - 学习率: {opt_config['lr']}")
        print(f"  - Weight decay: {opt_config['weight_decay']}")

        # 学习率调度器
        scheduler_config = self.config['scheduler']
        if scheduler_config['type'] == 'CosineAnnealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=scheduler_config['min_lr']
            )
        elif scheduler_config['type'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        else:
            self.scheduler = None

        if self.scheduler:
            print(f"✓ 学习率调度器: {scheduler_config['type']}")

        # 判别器优化器（如果启用GAN损失）
        loss_config = self.config['loss']
        if loss_config.get('gan_weight', 0) > 0 and self.discriminator is not None:
            disc_lr = loss_config.get('disc_lr', 1.0e-4)
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=disc_lr,
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
            print(f"✓ 判别器优化器: AdamW")
            print(f"  - 学习率: {disc_lr}")
        else:
            self.optimizer_d = None

    def _init_dataloaders(self):
        """初始化数据加载器"""
        print("\n" + "=" * 70)
        print("初始化数据加载器")
        print("=" * 70 + "\n")

        data_config = self.config['data']
        train_config = self.config['training']

        # 训练数据加载器
        if self.config['degradation']['use_degradation']:
            # 使用退化管道生成LR图像
            print("使用退化管道生成LR图像...")
            self.train_loader = create_train_dataloader(
                data_dir=data_config['train']['hr_dir'],
                config_path=self.config['degradation']['config_path'],
                batch_size=train_config['batch_size'],
                num_workers=train_config['num_workers'],
                gt_size=data_config['train']['crop_size'],
                use_hflip=data_config['train']['use_flip'],
                use_rot=data_config['train']['use_rot'],
                shuffle=True,
                pin_memory=True
            )
            print(f"✓ 训练数据加载器创建成功：{len(self.train_loader)} batches")
        else:
            raise NotImplementedError("暂不支持直接加载LR-HR图像对")


    def _extract(self, a, t, x_shape):
        """从a中提取t对应的值，并reshape到x_shape"""
        batch_size = t.shape[0]
        out = a.to(t.device)[t]
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _scale_input(self, inputs, t):
        """
        对输入进行归一化（ResShift的关键步骤！）

        这是与ResShift原项目保持一致的输入归一化方法

        Args:
            inputs: 输入tensor
            t: 时间步索引

        Returns:
            归一化后的输入
        """
        if self.normalize_input:
            if self.latent_flag:
                # 潜在空间的方差约为1.0
                std = torch.sqrt(self._extract(self.etas, t, inputs.shape) * self.kappa ** 2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = self._extract(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        计算ResShift的后验分布 q(x_{t-1} | x_t, x_0)

        ResShift后验分布公式：
        q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t, σ̃_t²·I)

        其中：
        μ̃_t = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0
        σ̃_t² = κ²·(η_{t-1}/η_t)·α_t
        α_t = η_t - η_{t-1}

        Args:
            x_0: 预测的干净图像（ResShift UNet的输出）
            x_t: 当前时间步的含噪图像
            t: 时间步

        Returns:
            mean: 后验均值
            variance: 后验方差
            log_variance: 后验对数方差
        """
        # ResShift: μ = coef1·x_t + coef2·x_0
        # DDPM:     μ = coef1·x_0 + coef2·x_t
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
                self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_0
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def multi_step_training_loss(self, z_start, z_y, hr_image,lr_image):
        """
        多步训练损失计算 - 与推理流程完全一致

        多步训练流程：
        1. 使用噪声预测器初始化 x_T = z_y + κ·√η_T·noise
        2. 多步反向采样：for t in [T-1, T-2, ..., 0]:
           - UNet 预测 x_0
           - 噪声预测器预测噪声
           - 后验采样得到 x_{t-1}
        3. 计算最终的 x_0 与真实 z_start 的损失

        这样训练与推理完全一致，避免训练-推理不一致的问题。

        Args:
            z_start: HR图像的潜在表示 z_0 [B, C, H, W]
            z_y: LR图像的潜在表示 y [B, C, H, W]
            hr_image: 原始HR图像 [B, 3, H, W],用于图像空间损失计算
            lr_image: 图像空间的LR图像（用作UNet的lq条件）

        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        batch_size = z_y.shape[0]

        # 1. 初始化 x_T（使用噪声预测器）
        # ResShift 公式：x_T = z_y + κ·√η_T·ε
        t_init = self.num_timesteps - 1
        t_init_tensor = torch.full((batch_size,), t_init, device=self.device, dtype=torch.long)

        # 初始化时使用随机高斯噪声（不使用噪声预测器）
        sqrt_eta_T = self._extract(self.sqrt_etas, t_init_tensor, z_y.shape)
        predicted_noise_init = torch.randn_like(z_y)
        x_t = z_y + self.kappa * sqrt_eta_T * predicted_noise_init

        # 2. 多步反向采样（与推理流程一致）
        # 从 num_timesteps-1 到 0
        indices = list(range(self.num_timesteps))[::-1]  # [num_timesteps-1, ..., 0]

        for i in indices:
            t_tensor = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            # 2.1 对输入进行归一化
            x_t_normalized = self._scale_input(x_t, t_tensor)

            # 2.2 使用 UNet 预测 x_0（保留梯度，让梯度流回噪声预测器）
            pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)

            # 2.3 如果不是最后一步，进行后验采样
            if i > 0:
                # 计算后验分布 q(x_{t-1} | x_t, x_0)
                mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t_tensor)

                # 使用噪声预测器预测噪声（需要梯度）
                predicted_noise = self.noise_predictor(x_t, pred_x0, lr_image, t_tensor, sample_posterior=True)

                # 采样 x_{t-1}
                # nonzero_mask 在这里总是 1，因为 i > 0
                x_t = mean + torch.exp(0.5 * log_variance) * predicted_noise

        # 最终的 pred_x0 就是我们的预测结果
        final_pred_x0 = pred_x0

        # 4. 计算损失
        loss_config = self.config['loss']
        total_loss = 0.0

        # 解码到图像空间
        # 注意：pred_image 需要保留梯度以便损失能够反向传播到噪声预测器
        # VAE虽然是冻结的，但梯度仍然可以通过它传回到 final_pred_x0
        pred_image = self.vae.decode(final_pred_x0)  # [-1, 1]，保留梯度
        pred_image = torch.clamp(pred_image, -1, 1)
        pred_image = pred_image * 0.5 + 0.5  # [0, 1]

        gt_image = hr_image * 0.5 + 0.5  # [0, 1]

        # L1 损失（图像空间）
        l1_weight = loss_config.get('l1_weight', 1.0)
        l1_val = torch.nn.functional.l1_loss(pred_image, gt_image)
        loss_dict['l1'] = l1_val.item()
        total_loss += l1_weight * l1_val

        # LPIPS 感知损失（图像空间）
        if self.lpips_loss is not None and loss_config.get('lpips_weight', 0) > 0:
            lpips_val = self.lpips_loss(pred_image, gt_image)
            loss_dict['lpips'] = lpips_val.item()
            total_loss += loss_config['lpips_weight'] * lpips_val

        # GAN 生成器损失（图像空间）
        if self.gan_loss is not None and loss_config.get('gan_weight', 0) > 0:
            # 检查是否达到判别器开始训练的epoch
            disc_start_epoch = loss_config.get('disc_start_epoch', 0)
            if self.current_epoch >= disc_start_epoch:
                # 计算生成器损失：让判别器认为生成图像是真的
                fake_pred = self.discriminator(pred_image)
                g_loss = self.gan_loss(fake_pred, target_is_real=True, is_disc=False)
                loss_dict['g_loss'] = g_loss.item()
                total_loss += loss_config['gan_weight'] * g_loss

        # 保存解码后的图像供判别器训练使用
        self._pred_image_for_disc = pred_image.detach()
        self._gt_image_for_disc = gt_image

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def train_epoch(self, epoch):
        """
        训练一个epoch

        Args:
            epoch: 当前epoch数

        Returns:
            avg_loss_dict: 平均损失字典
        """
        self.noise_predictor.train()

        # 记录当前epoch和epoch内的step，用于调试输出
        self.current_epoch = epoch
        self.epoch_step = 0

        total_loss_dict = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}")

        for step, batch in enumerate(pbar):
            self.epoch_step = step
            # 获取数据
            hr_images = batch['gt'].to(self.device)  # [B, 3, H, W], [0, 1]
            lr_images = batch['lq'].to(self.device)  # [B, 3, H, W], [0, 1]

            # 训练一步
            loss_dict = self.train_step(hr_images, lr_images)

            # 累积损失
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0.0
                total_loss_dict[key] += value

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # 打印日志
            if (step + 1) % self.config['experiment']['log_interval'] == 0:
                print(f"\nEpoch [{epoch}/{self.config['training']['num_epochs']}] "
                      f"Step [{step + 1}/{len(self.train_loader)}]")
                for key, value in loss_dict.items():
                    print(f"  {key}: {value:.4f}")
                print(f"  lr: {self.optimizer.param_groups[0]['lr']:.2e}")

            self.global_step += 1

        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss_dict = {key: value / num_batches for key, value in total_loss_dict.items()}

        return avg_loss_dict

    def freeze_model(self, model):
        """冻结模型参数"""
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_model(self, model):
        """解冻模型参数"""
        for param in model.parameters():
            param.requires_grad = True

    def train_step(self, hr_images, lr_images):
        """
        单步训练（类似 ResShift 和 InvSR），支持梯度累计

        Args:
            hr_images: HR图像 [B, 3, H, W], 范围[0, 1]
            lr_images: LR图像 [B, 3, H, W], 范围[0, 1]

        Returns:
            loss_dict: 损失字典
        """
        self.noise_predictor.train()
        batch_size = hr_images.shape[0]

        loss_config = self.config['loss']

        # 获取判别器更新频率（实现真正的交替训练）
        disc_update_freq = loss_config.get('disc_update_freq', 1)
        should_update_disc = (self.global_step % disc_update_freq == 0)

        # 1. 编码到潜在空间（冻结的VAE）
        with torch.no_grad():
            # 转换到[-1, 1]
            hr_images_norm = hr_images * 2.0 - 1.0
            lr_images_norm = lr_images * 2.0 - 1.0

            # HR图像直接编码
            z_start = self.vae.encode(hr_images_norm)

            # LR图像需要先上采样到与HR相同尺寸，再编码
            scale_factor = self.config['data']['train']['scale']
            lr_images_upsampled = torch.nn.functional.interpolate(
                lr_images_norm, scale_factor=scale_factor, mode='bicubic', align_corners=False
            )
            z_y = self.vae.encode(lr_images_upsampled)

        # 2. 先计算生成器损失（总是需要计算，为判别器提供假图像）
        if self.config['training']['use_amp']:
            with autocast(device_type='cuda'):
                loss, loss_dict = self.multi_step_training_loss(
                    z_start, z_y, hr_images_norm,lr_images_norm
                )
        else:
            loss, loss_dict = self.multi_step_training_loss(
                z_start, z_y, hr_images_norm,lr_images_norm
            )

        # 3. 真正的交替训练逻辑
        if should_update_disc and self.discriminator is not None and loss_config.get('gan_weight', 0) > 0:
            # === 这个step只更新判别器 ===
            disc_start_epoch = loss_config.get('disc_start_epoch', 0)
            if self.current_epoch >= disc_start_epoch:
                # 获取生成器产生的图像（已在multi_step_training_loss中保存）
                if hasattr(self, '_pred_image_for_disc') and hasattr(self, '_gt_image_for_disc'):
                    fake_image = self._pred_image_for_disc
                    real_image = self._gt_image_for_disc

                    self.discriminator.train()

                    # 冻结生成器，解冻判别器
                    self.freeze_model(self.noise_predictor)
                    self.unfreeze_model(self.discriminator)

                    # 计算判别器损失
                    if self.config['training']['use_amp']:
                        with autocast(device_type='cuda'):
                            # 判别真实图像
                            real_pred = self.discriminator(real_image)
                            d_loss_real = self.gan_loss(real_pred, target_is_real=True, is_disc=True)

                            # 判别生成图像
                            fake_pred = self.discriminator(fake_image)
                            d_loss_fake = self.gan_loss(fake_pred, target_is_real=False, is_disc=True)

                            d_loss = (d_loss_real + d_loss_fake) / 2

                        self.scaler_d.scale(d_loss).backward()

                        # 梯度裁剪
                        if self.config['training']['gradient_clip'] > 0:
                            self.scaler_d.unscale_(self.optimizer_d)
                            torch.nn.utils.clip_grad_norm_(
                                self.discriminator.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        self.scaler_d.step(self.optimizer_d)
                        self.scaler_d.update()
                        self.optimizer_d.zero_grad()
                    else:
                        # 判别真实图像
                        real_pred = self.discriminator(real_image)
                        d_loss_real = self.gan_loss(real_pred, target_is_real=True, is_disc=True)

                        # 判别生成图像
                        fake_pred = self.discriminator(fake_image)
                        d_loss_fake = self.gan_loss(fake_pred, target_is_real=False, is_disc=True)

                        d_loss = (d_loss_real + d_loss_fake) / 2

                        d_loss.backward()

                        # 梯度裁剪
                        if self.config['training']['gradient_clip'] > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.discriminator.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        self.optimizer_d.step()
                        self.optimizer_d.zero_grad()

                    # 记录损失
                    loss_dict['d_loss'] = d_loss.item()
                    loss_dict['d_loss_real'] = d_loss_real.item()
                    loss_dict['d_loss_fake'] = d_loss_fake.item()

                # 恢复生成器参数状态
                self.unfreeze_model(self.noise_predictor)
        else:
            # === 这个step只更新生成器 ===

            # 冻结判别器，解冻生成器
            if self.discriminator is not None:
                self.freeze_model(self.discriminator)
            self.unfreeze_model(self.noise_predictor)

            # 反向传播生成器损失
            if self.config['training']['use_amp']:
                self.scaler_g.scale(loss).backward()

                # 梯度裁剪
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler_g.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.noise_predictor.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.scaler_g.step(self.optimizer)
                self.scaler_g.update()
                self.optimizer.zero_grad()
            else:
                loss.backward()

                # 梯度裁剪
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.noise_predictor.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # 恢复判别器参数状态
            if self.discriminator is not None:
                self.unfreeze_model(self.discriminator)

        return loss_dict

    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'noise_predictor': self.noise_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config,
        }

        # 保存判别器状态（如果启用GAN损失）
        if self.discriminator is not None:
            checkpoint['discriminator'] = self.discriminator.state_dict()
        if self.optimizer_d is not None:
            checkpoint['optimizer_d'] = self.optimizer_d.state_dict()

        # 保存最新的checkpoint（包含完整训练状态）
        ckpt_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, ckpt_path)
        print(f"✓ Checkpoint已保存: {ckpt_path}")

        # 保存最佳模型（只保存噪声预测器权重）
        if is_best:
            best_path = self.exp_dir / 'checkpoints' / 'noise_predictor.pth'
            # 只保存噪声预测器的state_dict，方便推理时直接加载
            torch.save(self.noise_predictor.state_dict(), best_path)
            print(f"✓ 最佳模型已保存（仅噪声预测器权重）: {best_path}")

        # 删除旧的checkpoint（保留最近N个）
        keep_recent = self.config['experiment'].get('keep_recent_checkpoints', 5)
        # 按epoch数字排序，而不是按字母顺序
        checkpoints = sorted(
            (self.exp_dir / 'checkpoints').glob('checkpoint_epoch_*.pth'),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        if len(checkpoints) > keep_recent:
            for old_ckpt in checkpoints[:-keep_recent]:
                old_ckpt.unlink()
                print(f"✓ 删除旧checkpoint: {old_ckpt.name}")

    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.noise_predictor.load_state_dict(checkpoint['noise_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        # 加载判别器状态（如果存在）
        if self.discriminator is not None and 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            print(f"  - 已加载判别器权重")
        if self.optimizer_d is not None and 'optimizer_d' in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            print(f"  - 已加载判别器优化器状态")

        print(f"✓ Checkpoint已加载: {checkpoint_path}")
        print(f"  - Epoch: {self.current_epoch}")
        print(f"  - Global step: {self.global_step}")
        print(f"  - Best loss: {self.best_loss:.6f}")

        return checkpoint

def main():
    parser = argparse.ArgumentParser(description='训练噪声预测器（单步训练）')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    args = parser.parse_args()

    # 创建训练器
    trainer = NoisePredictorTrainer(args.config)

    # 恢复训练
    start_epoch = 1
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        if checkpoint is not None:
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"\n从epoch {start_epoch}恢复训练")
        else:
            print("\n警告: 无法加载checkpoint，从头开始训练")

    print("\n" + "=" * 70)
    print("开始训练！")
    print("=" * 70 + "\n")

    # 训练循环
    num_epochs = trainer.config['training']['num_epochs']

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 70}\n")

            # 训练一个epoch
            avg_loss_dict = trainer.train_epoch(epoch)

            # 打印平均损失
            print(f"\nEpoch {epoch} 平均损失:")
            for key, value in avg_loss_dict.items():
                print(f"  {key}: {value:.4f}")

            # 更新学习率
            if trainer.scheduler is not None:
                trainer.scheduler.step()
                print(f"\n当前学习率: {trainer.optimizer.param_groups[0]['lr']:.2e}")

            # 保存checkpoint
            if epoch % trainer.config['experiment']['save_interval'] == 0:
                is_best = avg_loss_dict['total'] < trainer.best_loss
                if is_best:
                    trainer.best_loss = avg_loss_dict['total']
                trainer.save_checkpoint(epoch, is_best=is_best)

        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n训练被中断！")
        print("保存当前checkpoint...")
        trainer.save_checkpoint(epoch, is_best=False)
        print("Checkpoint已保存，可以使用--resume恢复训练")


if __name__ == '__main__':
    main()