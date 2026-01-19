#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
初始化器训练脚本

训练思路：
1. 从T=4,3,2,1随机采样一个时间步
2. 使用initializer生成中间状态 x_t = z_y + κ·√η_t·predicted_noise
3. 使用预训练的UNet直接预测干净HR x_0
4. 计算预测的x_0与真实z_start的损失
5. 只更新initializer的参数

关键：
- UNet和VAE都是冻结的
- 不使用EMA更新
- 不使用判别器
"""

import os
import sys
import warnings

# 将项目根目录添加到Python路径

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 在导入其他模块之前设置警告过滤器
warnings.filterwarnings("ignore", message=".*A matching Triton is not available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.")

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
import matplotlib
matplotlib.use('Agg')

# LPNSR模块导入
from LPNSR.models.initializer import create_initializer
from LPNSR.models.unet import UNetModelSwin
from LPNSR.ldm.models.autoencoder import VQModelTorch
from LPNSR.losses.basic_loss import L2Loss
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


class InitializerTrainer:
    """初始化器训练器"""

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

        # 不使用EMA
        self.ema = None

        # 初始化AMP（为生成器和判别器使用独立的scaler）
        if self.config['training']['use_amp']:
            self.scaler_g = GradScaler()  # 生成器专用scaler
            self.scaler_d = GradScaler()  # 判别器专用scaler
        else:
            self.scaler_g = None
            self.scaler_d = None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # 损失记录列表
        self.loss_history = {
            'epoch': [],
            'train_l2': [],
            'train_lpips': [],
            'train_g_loss': [],
            'train_d_loss': [],
        }

        print(f"\n训练器初始化完成！")
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

        vae_ckpt = torch.load(vae_path, map_location=self.device)
        if 'state_dict' in vae_ckpt:
            state_dict = vae_ckpt['state_dict']
        else:
            state_dict = vae_ckpt

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

        unet_ckpt = torch.load(unet_path, map_location=self.device)
        if 'state_dict' in unet_ckpt:
            state_dict = unet_ckpt['state_dict']
        else:
            state_dict = unet_ckpt

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

        # 3. 加载初始化器（训练）
        print("\n加载初始化器...")
        initializer_config_path = self.config['initializer']['config_path']

        with open(initializer_config_path, 'r', encoding='utf-8') as f:
            initializer_config = yaml.safe_load(f)

        self.initializer = create_initializer(
            latent_channels=initializer_config['latent_channels'],
            model_channels=initializer_config['model_channels'],
            channel_mult=tuple(initializer_config['channel_mult']),
            num_res_blocks=initializer_config['num_res_blocks'],
            growth_rate=initializer_config['growth_rate'],
            res_scale=initializer_config['res_scale'],
            double_z=initializer_config['double_z']
        ).to(self.device)
        self.initializer.train()
        print(f"✓ 初始化器加载完成")

        # 统计参数量
        total_params = sum(p.numel() for p in self.initializer.parameters() if p.requires_grad)
        print(f"\n总可训练参数: {total_params / 1e6:.2f}M")

        # 4. 初始化ResShift扩散参数
        diffusion_config = self.config['training']['diffusion']
        self.num_timesteps = diffusion_config['num_timesteps']
        self.kappa = diffusion_config['kappa']

        # 获取eta调度
        self.sqrt_etas = get_named_eta_schedule(
            schedule_name=diffusion_config['eta_schedule'],
            num_diffusion_timesteps=self.num_timesteps,
            min_noise_level=diffusion_config['min_noise_level'],
            etas_end=diffusion_config['etas_end'],
            kappa=self.kappa,
            kwargs={'power': diffusion_config.get('eta_power', 2.0)}
        )

        # 转换为tensor并移动到设备
        self.sqrt_etas = torch.tensor(self.sqrt_etas, dtype=torch.float32).to(self.device)
        self.etas = self.sqrt_etas ** 2

        print(f"\n✓ ResShift扩散参数初始化完成")
        print(f"  - 扩散步数: {self.num_timesteps}")
        print(f"  - kappa: {self.kappa}")
        print(f"  - eta范围: [{self.etas[0]:.4f}, {self.etas[-1]:.4f}]")

    def _extract(self, a, t, x_shape):
        """
        从数组a中提取t时间步的值，并扩展到x_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _scale_input(self, x_t, t):
        """
        对输入进行归一化（ResShift的关键步骤）

        """
        std = torch.sqrt(self._extract(self.etas, t, x_t.shape) * self.kappa ** 2 + 1)
        return x_t / std

    def _init_losses(self):
        """初始化损失函数"""
        print("\n" + "=" * 70)
        print("初始化损失函数")
        print("=" * 70)

        loss_config = self.config['loss']

        # L2损失
        self.l2_loss = L2Loss()
        print(f"✓ L2损失 (权重: {loss_config['l2_weight']})")

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
        loss_config = self.config['loss']

        # 初始化器优化器
        if opt_config['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.initializer.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.initializer.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {opt_config['type']}")

        print(f"✓ 优化器: {opt_config['type']}")
        print(f"  - 学习率: {opt_config['lr']}")
        print(f"  - Weight decay: {opt_config['weight_decay']}")

        # 判别器优化器（如果启用GAN损失）
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

    def _init_dataloaders(self):
        """初始化数据加载器"""
        print("\n" + "=" * 70)
        print("初始化数据加载器")
        print("=" * 70 + "\n")

        data_config = self.config['data']
        train_config = self.config['training']

        # 训练数据加载器
        if self.config['degradation']['use_degradation']:
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

    def train_loss(self, z_start, z_y, lr_image):
        """
        计算初始化器训练损失

        训练流程：
        1. 从T=4,3,2,1随机采样一个时间步
        2. 使用initializer生成噪声
        3. 使用ResShift初始化公式生成x_t: x_t = z_y + κ·√η_t·predicted_noise
        4. 使用UNet预测x_0
        5. 计算x_0与真实z_start的损失

        Args:
            z_start: HR图像的潜在表示 [B, C, H, W]
            z_y: LR图像的潜在表示 [B, C, H, W]
            lr_image: 图像空间的LR图像（用作UNet的lq条件）

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        batch_size = z_y.shape[0]

        # 1. 随机采样时间步：从[4, 3, 2, 1]中随机选择（注意：ResShift的时间步是0-based）
        # ResShift的时间步：0, 1, 2, 3 (共4步)
        # 我们从T=4,3,2,1随机采样，对应时间步索引为[3, 2, 1, 0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # 2. 使用initializer生成噪声
        predicted_noise = self.initializer(z_y, t, sample_posterior=True)

        # 3. 使用ResShift初始化公式生成x_t
        # x_t = z_y + κ·√η_t·predicted_noise
        sqrt_eta_t = self._extract(self.sqrt_etas, t, z_y.shape)
        x_t = z_y + self.kappa * sqrt_eta_t * predicted_noise

        # 4. 使用UNet预测x_0
        # 首先归一化输入
        x_t_normalized = self._scale_input(x_t, t)
        pred_x0 = self.resshift_unet(x_t_normalized, t, lq=lr_image)

        # 5. 计算损失
        loss_config = self.config['loss']
        total_loss = 0.0

        # L2损失（在潜空间计算）
        l2 = self.l2_loss(pred_x0, z_start)
        loss_dict['l2'] = l2.item()
        total_loss += loss_config['l2_weight'] * l2

        # LPIPS感知损失和GAN损失（在图像空间计算）
        need_image_space = (
                (self.lpips_loss is not None and loss_config.get('lpips_weight', 0) > 0) or
                (self.gan_loss is not None and loss_config.get('gan_weight', 0) > 0)
        )

        if need_image_space:
            # 解码到图像空间
            # 注意：pred_image 需要保留梯度以便感知损失能够反向传播到initializer
            # VAE虽然是冻结的，但梯度仍然可以通过它传回到 pred_x0
            pred_image = self.vae.decode(pred_x0)
            pred_image = pred_image * 0.5 + 0.5

            # gt_image 不需要梯度
            with torch.no_grad():
                gt_image = self.vae.decode(z_start)
                gt_image = gt_image * 0.5 + 0.5

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
        self.initializer.train()
        self.current_epoch = epoch

        total_loss_dict = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}")

        for step, batch in enumerate(pbar):
            # 获取数据
            hr_images = batch['gt'].to(self.device)
            lr_images = batch['lq'].to(self.device)

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
        单步训练

        Args:
            hr_images: HR图像 [B, 3, H, W], 范围[0, 1]
            lr_images: LR图像 [B, 3, H, W], 范围[0, 1]

        Returns:
            loss_dict: 损失字典
        """
        self.initializer.train()
        batch_size = hr_images.shape[0]

        loss_config = self.config['loss']

        # 获取判别器更新频率（实现真正的交替训练）
        disc_update_freq = loss_config.get('disc_update_freq', 1)
        should_update_disc = (self.global_step % disc_update_freq == 0)

        # 1. 编码到潜在空间（冻结的VAE）
        with torch.no_grad():
            hr_images_norm = hr_images * 2.0 - 1.0
            lr_images_norm = lr_images * 2.0 - 1.0

            z_start = self.vae.encode(hr_images_norm)

            scale_factor = self.config['data']['train']['scale']
            lr_images_upsampled = torch.nn.functional.interpolate(
                lr_images_norm, scale_factor=scale_factor, mode='bicubic', align_corners=False
            )
            z_y = self.vae.encode(lr_images_upsampled)

        # 2. 计算生成器损失（initializer训练）
        if self.config['training']['use_amp']:
            with autocast(device_type='cuda'):
                loss, loss_dict = self.train_loss(z_start, z_y, lr_images_norm)
        else:
            loss, loss_dict = self.train_loss(z_start, z_y, lr_images_norm)

        # 3. 真正的交替训练逻辑
        if should_update_disc and self.discriminator is not None and loss_config.get('gan_weight', 0) > 0:
            # === 这个step只更新判别器 ===
            disc_start_epoch = loss_config.get('disc_start_epoch', 0)
            if self.current_epoch >= disc_start_epoch:
                # 获取生成器产生的图像（已在train_loss中保存）
                if hasattr(self, '_pred_image_for_disc') and hasattr(self, '_gt_image_for_disc'):
                    fake_image = self._pred_image_for_disc
                    real_image = self._gt_image_for_disc

                    self.discriminator.train()

                    # 冻结生成器，解冻判别器
                    self.freeze_model(self.initializer)
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
                self.unfreeze_model(self.initializer)
        else:
            # === 这个step只更新生成器 ===

            # 冻结判别器，解冻生成器
            if self.discriminator is not None:
                self.freeze_model(self.discriminator)
            self.unfreeze_model(self.initializer)

            # 反向传播生成器损失
            if self.config['training']['use_amp']:
                self.scaler_g.scale(loss).backward()

                # 梯度裁剪
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler_g.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.initializer.parameters(),
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
                        self.initializer.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # 恢复判别器参数状态
            if self.discriminator is not None:
                self.unfreeze_model(self.discriminator)

        return loss_dict

    def save_checkpoint(self, epoch, is_best=False):
        """
        保存checkpoint

        Args:
            epoch: 当前epoch数
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'initializer_state_dict': self.initializer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'loss_history': self.loss_history
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存判别器状态（如果存在）
        if self.discriminator is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
        if self.optimizer_d is not None:
            checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()

        # 保存epoch checkpoint（完整的训练状态，用于恢复训练）
        ckpt_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, ckpt_path)
        print(f"✓ 保存checkpoint: {ckpt_path}")

        # 保存最佳模型（只保存initializer权重，方便推理）
        if is_best:
            best_path = self.exp_dir / 'checkpoints' / 'initializer.pth'
            torch.save(self.initializer.state_dict(), best_path)
            print(f"✓ 最佳模型已保存（仅initializer权重）: {best_path}")

        # 清理旧checkpoint
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """清理旧的checkpoint，只保留最近的N个"""
        keep_recent = self.config['experiment'].get('keep_recent_checkpoints', 5)
        ckpt_dir = self.exp_dir / 'checkpoints'

        # 获取所有checkpoint_epoch_*.pth文件
        epoch_ckpts = sorted(
            ckpt_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda x: int(x.stem.split('_')[-1])
        )

        # 删除多余的checkpoint
        while len(epoch_ckpts) > keep_recent:
            old_ckpt = epoch_ckpts.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                print(f"  删除旧checkpoint: {old_ckpt.name}")

    def load_checkpoint(self, ckpt_path):
        """
        加载checkpoint

        Args:
            ckpt_path: checkpoint路径
        """
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.initializer.load_state_dict(checkpoint['initializer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.loss_history = checkpoint.get('loss_history', self.loss_history)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载判别器状态（如果存在）
        if self.discriminator is not None and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if self.optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

        print(f"✓ 加载checkpoint: {ckpt_path}")
        print(f"  - Epoch: {self.current_epoch}")
        print(f"  - Best loss: {self.best_loss:.4f}")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("开始训练")
        print("=" * 70)

        num_epochs = self.config['training']['num_epochs']

        for epoch in range(self.current_epoch, num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")

            # 训练一个epoch
            avg_loss_dict = self.train_epoch(epoch + 1)

            # 记录训练损失
            self.loss_history['epoch'].append(epoch + 1)
            self.loss_history['train_l2'].append(avg_loss_dict.get('l2', 0))
            self.loss_history['train_lpips'].append(avg_loss_dict.get('lpips', 0))
            self.loss_history['train_g_loss'].append(avg_loss_dict.get('g_loss', 0))
            self.loss_history['train_d_loss'].append(avg_loss_dict.get('d_loss', 0))

            # 打印epoch总结
            print(f"\nEpoch {epoch + 1} 训练总结:")
            for key, value in avg_loss_dict.items():
                print(f"  {key}: {value:.4f}")

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 保存checkpoint
            is_best = avg_loss_dict['total'] < self.best_loss
            if is_best:
                self.best_loss = avg_loss_dict['total']
                print(f"\n✓ 新的最佳模型: {self.best_loss:.4f}")

            if (epoch + 1) % self.config['experiment']['save_interval'] == 0:
                self.save_checkpoint(epoch + 1, is_best)

        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        print(f"最佳损失: {self.best_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='训练初始化器')
    parser.add_argument('--config', type=str, default='LPNSR/configs/train_initializer.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建训练器
    trainer = InitializerTrainer(args.config)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
