#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端训练噪声预测器

训练思路：
1. 输入HR图像和LR图像
2. 使用冻结的VAE编码到潜在空间
3. 从x_T开始进行N步反向采样（ResShift方式）
4. 每步使用冻结的ResShift UNet预测x_0，计算后验分布
5. 使用噪声预测器生成噪声（替代随机噪声）
6. 得到最终的SR图像
7. 计算SR与HR的损失
8. 反向传播，只更新噪声预测器

关键：完全按照ResShift论文实现扩散过程！
- 使用eta调度而非beta调度
- 残差偏移扩散：x_t = (1-η_t)·x_0 + η_t·y + √η_t·κ·ε
- 初始化：x_T = y + κ·√η_T·ε
- 后验分布：μ = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0
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
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

# SR模块导入
from SR.models.noise_predictor import create_noise_predictor
from SR.models.unet import UNetModelSwin
from SR.ldm.models.autoencoder import VQModelTorch
from SR.losses.basic_loss import L2Loss
from SR.losses.frequency_loss import FocalFrequencyLoss
from SR.losses.lpips_loss import LPIPSLoss
from SR.datapipe.train_dataloader import create_train_dataloader


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
        increaser = math.exp(1/(num_diffusion_timesteps-1) * math.log(etas_end/etas_start))
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


class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 注册参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


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
        (self.exp_dir / 'tensorboard').mkdir(exist_ok=True)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / 'tensorboard'))
        
        # 初始化模型
        self._init_models()
        
        # 初始化损失函数
        self._init_losses()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化EMA
        if self.config['training']['use_ema']:
            self.ema = EMA(self.noise_predictor, decay=self.config['training']['ema_decay'])
        else:
            self.ema = None
        
        # 初始化AMP
        if self.config['training']['use_amp']:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print(f"训练器初始化完成！")
        print(f"实验目录: {self.exp_dir}")
        print(f"设备: {self.device}")
    
    def _init_models(self):
        """初始化模型"""
        print("\n" + "="*70)
        print("初始化模型")
        print("="*70)
        
        # 1. 加载VQVAE（冻结）
        print("\n加载VQVAE...")
        vae_config_path = self.config['resshift']['vae_config_path']
        vae_path = self.config['resshift']['vae_path']
        
        # 加载VAE配置
        with open(vae_config_path, 'r', encoding='utf-8') as f:
            vae_config = yaml.safe_load(f)
        
        # VQVAE模型结构参数（与预训练权重一致）
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
        
        lora_config = vae_config.get('lora', {})
        self.vae = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
            rank=lora_config.get('rank', 8),
            lora_alpha=lora_config.get('alpha', 1.0),
            lora_tune_decoder=lora_config.get('tune_decoder', False),
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
        unet_config_path = self.config['resshift']['unet_config_path']
        unet_path = self.config['resshift']['unet_path']
        
        # 加载UNet配置
        with open(unet_config_path, 'r', encoding='utf-8') as f:
            unet_config = yaml.safe_load(f)
        
        # 从配置文件计算潜在空间尺寸
        # crop_size / VAE下采样倍数 = 潜在空间尺寸
        # 注意：VAE下采样倍数由VAE架构决定（ch_mult: [1,2,4] → 2^2=4倍）
        # 这个值必须与预训练VAE权重一致，不能随意修改
        crop_size = self.config['data']['train']['crop_size']
        vae_downsample_factor = 4  # 由VAE的ch_mult长度决定：2^(len(ch_mult)-1) = 2^2 = 4
        latent_size = crop_size // vae_downsample_factor
        
        # UNet模型结构参数（与预训练权重一致）
        model_structure = {
            'image_size': latent_size,  # 动态计算，而非硬编码
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
            'lq_size': latent_size,  # 也需要动态计算
        }
        
        # 合并配置
        model_config = {
            **model_structure,
            'dropout': unet_config.get('dropout', 0.0),
            'use_fp16': unet_config.get('use_fp16', False),
            'conv_resample': unet_config.get('conv_resample', True),
            'dims': unet_config.get('dims', 2),
            'patch_norm': unet_config.get('patch_norm', False),
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
        self.posterior_mean_coef2 = self.alpha / self.etas      # α_t/η_t
        self.posterior_variance = self.kappa**2 * self.etas_prev / self.etas * self.alpha
        
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
        print(f"  - 初始化分布: N(y, κ²·η_T·I) = N(y, {self.kappa**2 * self.etas[-1]:.4f}·I)")
        print(f"\n  ResShift扩散公式：")
        print(f"  - 前向: x_t = (1-η_t)·x_0 + η_t·y + √η_t·κ·ε")
        print(f"  - 后验: μ = (η_{{t-1}}/η_t)·x_t + (α_t/η_t)·x_0")
        print(f"  - 方差: σ² = κ²·(η_{{t-1}}/η_t)·α_t")
        print(f"  - posterior_mean_coef1: {self.posterior_mean_coef1.numpy()}")
        print(f"  - posterior_mean_coef2: {self.posterior_mean_coef2.numpy()}")
        
        # 5. 创建噪声预测器（训练）
        print("\n创建噪声预测器...")
        noise_config = self.config['noise_predictor']
        self.noise_predictor = create_noise_predictor(
            latent_channels=noise_config['latent_channels'],
            model_channels=noise_config['model_channels'],
            channel_mult=tuple(noise_config['channel_mult']),
            num_res_blocks=noise_config['num_res_blocks'],
            attention_levels=noise_config['attention_levels'],
            num_heads=noise_config['num_heads'],
            use_cross_attention=noise_config['use_cross_attention'],
            use_frequency_aware=noise_config['use_frequency_aware'],
            use_xformers=noise_config.get('use_xformers', True),
            use_checkpoint=self.config['training']['use_gradient_checkpointing']
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
        print("\n" + "="*70)
        print("初始化损失函数")
        print("="*70)
        
        loss_config = self.config['loss']
        
        # L2损失
        self.l2_loss = L2Loss()
        print(f"✓ L2损失 (权重: {loss_config['l2_weight']})")
        
        # 频域损失
        if loss_config['freq_weight'] > 0:
            self.freq_loss = FocalFrequencyLoss(
                loss_weight=1.0,
                alpha=loss_config.get('freq_alpha', 1.0)
            )
            print(f"✓ 频域损失 (权重: {loss_config['freq_weight']})")
        else:
            self.freq_loss = None
        
        # LPIPS感知损失
        if loss_config.get('lpips_weight', 0) > 0:
            self.lpips_loss = LPIPSLoss(
                loss_weight=1.0,
                net_type=loss_config.get('lpips_net_type', 'alex')
            )
            print(f"✓ LPIPS感知损失 (权重: {loss_config['lpips_weight']})")
        else:
            self.lpips_loss = None
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        print("\n" + "="*70)
        print("初始化优化器")
        print("="*70)
        
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
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        print("\n" + "="*70)
        print("初始化数据加载器")
        print("="*70 + "\n")
        
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
        
        # 验证数据加载器
        if data_config['val']['hr_dir'] is not None:
            print("\n创建验证数据加载器...")
            self.val_loader = create_train_dataloader(
                data_dir=data_config['val']['hr_dir'],
                config_path=self.config['degradation']['config_path'],
                batch_size=train_config['batch_size'],
                num_workers=train_config['num_workers'],
                gt_size=data_config['val']['crop_size'],
                use_hflip=False,  # 验证时不使用数据增强
                use_rot=False,
                shuffle=False,
                pin_memory=True
            )
            print(f"✓ 验证数据加载器创建成功：{len(self.val_loader)} batches")
        else:
            self.val_loader = None
            print("\n未配置验证数据")
        
        print()
    
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
                std = torch.sqrt(self._extract(self.etas, t, inputs.shape) * self.kappa**2 + 1)
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
        # 注意：ResShift的系数顺序与DDPM不同！
        # ResShift: μ = coef1·x_t + coef2·x_0
        # DDPM:     μ = coef1·x_0 + coef2·x_t
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_0
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample_with_noise_predictor(self, x_t, y, lr_image, t_tensor):
        """
        使用噪声预测器的采样步骤
        
        Args:
            x_t: 当前时间步的含噪潜在表示 [B, C, H, W]
            y: LR图像的潜在表示 [B, C, H, W]
            lr_image: 图像空间的LR图像（用作UNet的lq条件）
            t_tensor: 时间步张量 [B]
        
        Returns:
            x_{t-1}: 下一时间步的潜在表示
        """
        # 1. 对输入进行归一化（ResShift的关键步骤！）
        x_t_normalized = self._scale_input(x_t, t_tensor)
        
        # 2. 使用ResShift的UNet预测x_0
        # 注意：lq应该是图像空间的LR图像，不是潜在空间的y！
        pred_x0 = self.resshift_unet(x_t_normalized, t_tensor, lq=lr_image)
        
        # clip预测结果到[-1, 1]
        pred_x0 = pred_x0.clamp(-1, 1)
        
        # 3. 计算后验分布 q(x_{t-1} | x_t, x_0)
        mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t_tensor)
        
        # 4. 使用噪声预测器生成噪声（替代随机噪声）
        # 注意：训练时需要梯度，推理时不需要
        if self.training:
            predicted_noise, _ = self.noise_predictor(x_t, t_tensor, y)
        else:
            with torch.no_grad():
                predicted_noise, _ = self.noise_predictor(x_t, t_tensor, y)
        
        # 5. 采样x_{t-1}
        nonzero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
        sample = mean + nonzero_mask * torch.exp(0.5 * log_variance) * predicted_noise
        
        return sample
    
    @torch.no_grad()
    def reverse_sampling(self, hr_latent, lr_latent, lr_image, num_steps):
        """
        完整的ResShift反向采样过程（推理时使用）
        
        ResShift v3直接用4步训练，所以时间步直接从0到num_timesteps-1
        
        Args:
            hr_latent: HR图像的潜在表示（未使用，保留接口兼容性）
            lr_latent: LR图像的潜在表示 y
            lr_image: 图像空间的LR图像（用作UNet的lq条件）
            num_steps: 采样步数S（应该等于num_timesteps）
        
        Returns:
            x_0: 最终的潜在表示
        """
        # ResShift初始化：x_T = y + κ·√η_T·ε
        t_init = self.num_timesteps - 1
        sqrt_eta_T = self.sqrt_etas[t_init].to(lr_latent.device)
        x_t = lr_latent + self.kappa * sqrt_eta_T * torch.randn_like(lr_latent)
        
        # 反向采样：\u4ecnum_timesteps-1到0
        indices = list(range(self.num_timesteps))[::-1]  # [num_timesteps-1, ..., 0]
        
        for i in indices:
            t_tensor = torch.full((lr_latent.shape[0],), i, device=self.device, dtype=torch.long)
            x_t = self.p_sample_with_noise_predictor(x_t, lr_latent, lr_image, t_tensor)
        
        return x_t
    
    def reverse_sampling_train(self, hr_latent, lr_latent, lr_image, num_steps):
        """
        完整的ResShift反向采样过程（训练时使用，需要梯度）
        
        ResShift v3直接用4步训练，所以时间步直接从0到num_timesteps-1
        
        Args:
            hr_latent: HR图像的潜在表示（未使用，保留接口兼容性）
            lr_latent: LR图像的潜在表示 y
            lr_image: 图像空间的LR图像（用作UNet的lq条件）
            num_steps: 采样步数S（应该等于num_timesteps）
        
        Returns:
            x_0: 最终的潜在表示
        """
        # ResShift初始化：x_T = y + κ·√η_T·ε
        t_init = self.num_timesteps - 1
        sqrt_eta_T = self.sqrt_etas[t_init].to(lr_latent.device)
        x_t = lr_latent + self.kappa * sqrt_eta_T * torch.randn_like(lr_latent)
        
        # 反向采样：从num_timesteps-1到0
        indices = list(range(self.num_timesteps))[::-1]  # [num_timesteps-1, ..., 0]
        
        for i in indices:
            t_tensor = torch.full((lr_latent.shape[0],), i, device=self.device, dtype=torch.long)
            
            # 1. 对输入进行归一化（ResShift的关键步骤！）
            x_t_normalized = self._scale_input(x_t, t_tensor)
            
            # 2. 使用ResShift的UNet预测x_0（冻结，不需要梯度）
            # 注意：lq应该是图像空间的LR图像，不是潜在空间的lr_latent！
            with torch.no_grad():
                pred_x0 = self.resshift_unet(x_t_normalized.detach(), t_tensor, lq=lr_image)
                
                # 计算ResShift后验分布
                mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t.detach(), t_tensor)
            
            # 3. 使用噪声预测器生成噪声（需要梯度）
            predicted_noise, _ = self.noise_predictor(x_t, t_tensor, lr_latent)
            
            # 4. 采样x_{t-1}（保留梯度）
            nonzero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
            x_t = mean + nonzero_mask * torch.exp(0.5 * log_variance) * predicted_noise
        
        return x_t
    
    def compute_loss(self, sr_latent, hr_latent):
        """
        计算损失
        
        Args:
            sr_latent: SR图像的潜在表示
            hr_latent: HR图像的潜在表示
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0
        
        loss_config = self.config['loss']
        
        # L2损失
        l2 = self.l2_loss(sr_latent, hr_latent)
        loss_dict['l2'] = l2.item()
        total_loss += loss_config['l2_weight'] * l2
        
        # 频域损失
        if self.freq_loss is not None and loss_config['freq_weight'] > 0:
            freq = self.freq_loss(sr_latent, hr_latent)
            loss_dict['freq'] = freq.item()
            total_loss += loss_config['freq_weight'] * freq
        
        # LPIPS感知损失
        if self.lpips_loss is not None and loss_config.get('lpips_weight', 0) > 0:
            lpips_val = self.lpips_loss(sr_latent, hr_latent)
            loss_dict['lpips'] = lpips_val.item()
            total_loss += loss_config['lpips_weight'] * lpips_val
        
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
        
        total_loss_dict = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}")
        
        for step, batch in enumerate(pbar):
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
                      f"Step [{step+1}/{len(self.train_loader)}]")
                for key, value in loss_dict.items():
                    print(f"  {key}: {value:.4f}")
                print(f"  lr: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 记录到TensorBoard（每个step）
            for key, value in loss_dict.items():
                self.writer.add_scalar(f'Train_Step/{key}', value, self.global_step)
            self.writer.add_scalar('Train_Step/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss_dict = {key: value / num_batches for key, value in total_loss_dict.items()}
        
        # 记录epoch平均损失到TensorBoard
        for key, value in avg_loss_dict.items():
            self.writer.add_scalar(f'Train_Epoch/{key}', value, epoch)
        self.writer.add_scalar('Train_Epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_loss_dict
    
    @torch.no_grad()
    def validate(self, epoch=None):
        """
        验证
        
        Args:
            epoch: 当前epoch数（用于TensorBoard记录）
        
        Returns:
            avg_loss_dict: 平均损失字典
        """
        if self.val_loader is None:
            return {}
        
        if epoch is None:
            epoch = self.current_epoch
        
        self.noise_predictor.eval()
        
        total_loss_dict = {}
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            # 获取数据
            hr_images = batch['gt'].to(self.device)
            lr_images = batch['lq'].to(self.device)
            
            # 编码到潜在空间
            hr_images = hr_images * 2.0 - 1.0
            lr_images = lr_images * 2.0 - 1.0
            
            # HR图像直接编码
            hr_latent = self.vae.encode(hr_images)
            
            # LR图像需要先上采样到与HR相同尺寸，再编码
            scale_factor = self.config['data']['val']['scale']
            lr_images_upsampled = torch.nn.functional.interpolate(
                lr_images, scale_factor=scale_factor, mode='bicubic', align_corners=False
            )
            lr_latent = self.vae.encode(lr_images_upsampled)
            
            # 反向采样
            # 注意：lr_images是64x64的图像空间LR，用作UNet的lq条件
            num_steps = self.config['training']['sampling_steps']
            sr_latent = self.reverse_sampling_train(hr_latent, lr_latent, lr_images, num_steps)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(sr_latent, hr_latent)
            
            # 累积损失
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0.0
                total_loss_dict[key] += value
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
        
        # 计算平均损失
        num_batches = len(self.val_loader)
        avg_loss_dict = {key: value / num_batches for key, value in total_loss_dict.items()}
        
        # 记录验证损失到TensorBoard
        for key, value in avg_loss_dict.items():
            self.writer.add_scalar(f'Validation/{key}', value, epoch)
        
        return avg_loss_dict
    
    def train_step(self, hr_images, lr_images):
        """
        单步训练
        
        Args:
            hr_images: HR图像 [B, 3, H, W], 范围[0, 1]
            lr_images: LR图像 [B, 3, H, W], 范围[0, 1]
        
        Returns:
            loss_dict: 损失字典
        """
        self.noise_predictor.train()
        
        # 1. 编码到潜在空间（冻结的VAE）
        with torch.no_grad():
            # 转换到[-1, 1]
            hr_images = hr_images * 2.0 - 1.0
            lr_images = lr_images * 2.0 - 1.0
            
            # HR图像直接编码
            hr_latent = self.vae.encode(hr_images)
            
            # LR图像需要先上采样到与HR相同尺寸，再编码
            # 这样LR和HR在潜在空间中尺寸相同（都是64×64）
            scale_factor = self.config['data']['train']['scale']
            lr_images_upsampled = torch.nn.functional.interpolate(
                lr_images, scale_factor=scale_factor, mode='bicubic', align_corners=False
            )
            lr_latent = self.vae.encode(lr_images_upsampled)
        
        # 2. 反向采样（使用噪声预测器）
        num_steps = self.config['training']['sampling_steps']
        
        if self.config['training']['use_amp']:
            with autocast():
                # 注意：lr_images是64x64的图像空间LR，用作UNet的lq条件
                sr_latent = self.reverse_sampling_train(hr_latent, lr_latent, lr_images, num_steps)
                loss, loss_dict = self.compute_loss(sr_latent, hr_latent)
            
            # 反向传播（AMP）
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.config['training']['gradient_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.noise_predictor.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 注意：lr_images是64x64的图像空间LR，用作UNet的lq条件
            sr_latent = self.reverse_sampling_train(hr_latent, lr_latent, lr_images, num_steps)
            loss, loss_dict = self.compute_loss(sr_latent, hr_latent)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.noise_predictor.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
        
        # 更新EMA
        if self.ema is not None:
            self.ema.update()
        
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
        
        if self.ema is not None:
            checkpoint['ema'] = self.ema.shadow
        
        # 保存最新的checkpoint（包含完整训练状态）
        ckpt_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, ckpt_path)
        print(f"✓ Checkpoint已保存: {ckpt_path}")
        
        # 保存最佳模型（只保存噪声预测器权重）
        if is_best:
            best_path = self.exp_dir / 'checkpoints' / 'best_model.pth'
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
        
        if self.ema is not None and 'ema' in checkpoint:
            self.ema.shadow = checkpoint['ema']
        
        print(f"✓ Checkpoint已加载: {checkpoint_path}")
        print(f"  - Epoch: {self.current_epoch}")
        print(f"  - Global step: {self.global_step}")
        print(f"  - Best loss: {self.best_loss:.6f}")
        
        return checkpoint


def main():
    parser = argparse.ArgumentParser(description='训练噪声预测器（端到端）')
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
    
    print("\n" + "="*70)
    print("开始训练！")
    print("="*70 + "\n")
    
    # 训练循环
    num_epochs = trainer.config['training']['num_epochs']
    
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}\n")
            
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
            
            # 验证
            if trainer.val_loader is not None and epoch % trainer.config['experiment']['val_interval'] == 0:
                print(f"\n验证 Epoch {epoch}...")
                val_loss_dict = trainer.validate(epoch)
                print(f"验证损失:")
                for key, value in val_loss_dict.items():
                    print(f"  {key}: {value:.4f}")
            
            # 保存checkpoint
            if epoch % trainer.config['experiment']['save_interval'] == 0:
                is_best = avg_loss_dict['total'] < trainer.best_loss
                if is_best:
                    trainer.best_loss = avg_loss_dict['total']
                trainer.save_checkpoint(epoch, is_best=is_best)
        
        print("\n" + "="*70)
        print("训练完成！")
        print("="*70)
        
        # 关闭TensorBoard
        trainer.writer.close()
        print(f"\nTensorBoard日志已保存到: {trainer.exp_dir / 'tensorboard'}")
        print("使用以下命令查看: tensorboard --logdir=" + str(trainer.exp_dir / 'tensorboard'))
    
    except KeyboardInterrupt:
        print("\n\n训练被中断！")
        print("保存当前checkpoint...")
        trainer.save_checkpoint(epoch, is_best=False)
        print("Checkpoint已保存，可以使用--resume恢复训练")
        
        # 关闭TensorBoard
        trainer.writer.close()
        print(f"\nTensorBoard日志已保存到: {trainer.exp_dir / 'tensorboard'}")


if __name__ == '__main__':
    main()
