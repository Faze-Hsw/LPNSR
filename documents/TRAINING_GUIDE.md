# 噪声预测器训练指南

## 📋 概述

本指南介绍如何使用完整的训练脚本训练噪声预测器，用于替代ResShift反向采样中的随机高斯噪声。

## 🎯 训练目标

训练一个噪声预测器，使其能够：
1. 根据当前时间步 t 和潜在表示 x_t 预测合适的噪声
2. 根据LR图像的潜在表示 y 进行条件生成
3. 替代ResShift反向采样中的随机噪声，提升超分辨率质量

## 📁 文件结构

```
SR/
├── train_noise_predictor.py          # 训练脚本（已完成）
├── configs/
│   ├── train_noise_predictor.yaml    # 训练配置文件
│   ├── vqvae_config.yaml             # VAE配置
│   ├── unet_config.yaml              # UNet配置
│   └── realesrgan_degradation.yaml   # 退化配置
├── pretrained/
│   ├── autoencoder_vq_f4.pth         # VAE预训练权重
│   └── resshift_realsrx4_s4_v3.pth   # ResShift UNet预训练权重
├── traindata/                         # 训练数据（HR图像）
├── valdata/                           # 验证数据（HR图像）
├── datapipe/
│   ├── train_dataloader.py           # 数据加载器
│   └── realesrgan_degradation.py     # 退化管道
└── models/
    ├── noise_predictor.py            # 噪声预测器模型
    ├── unet.py                       # ResShift UNet
    └── autoencoder.py                # VAE
```

## 🚀 快速开始

### 1. 准备数据

将高质量图像放入训练和验证目录：

```bash
# 创建数据目录
mkdir -p SR/traindata SR/valdata

# 将HR图像复制到目录中
# 支持的格式：.png, .jpg, .jpeg, .bmp, .tiff, .webp
cp /path/to/your/hr/images/*.png SR/traindata/
cp /path/to/your/val/images/*.png SR/valdata/
```

**注意：**
- 训练数据会自动通过RealESRGAN退化管道生成LR-HR图像对
- 图像会被随机裁剪到配置文件中指定的大小（默认256x256）
- 建议准备至少1000张高质量图像用于训练

### 2. 检查配置文件

编辑 `SR/configs/train_noise_predictor.yaml`：

```yaml
# 数据设置
data:
  train:
    hr_dir: "SR/traindata"  # 训练数据目录
    crop_size: 256          # 裁剪大小
  val:
    hr_dir: "SR/valdata"    # 验证数据目录

# 训练设置
training:
  num_epochs: 200           # 训练轮数
  batch_size: 4             # 批次大小（根据显存调整）
  num_workers: 4            # 数据加载线程数
  sampling_steps: 3         # 反向采样步数
  
  # 优化技巧
  use_amp: true             # 混合精度训练
  use_ema: true             # 指数移动平均
  use_gradient_checkpointing: true  # 梯度检查点
  gradient_clip: 1.0        # 梯度裁剪

# 优化器设置
optimizer:
  type: "AdamW"
  lr: 1.0e-4
  weight_decay: 1.0e-4
```

### 3. 开始训练

```bash
# 从头开始训练
python -m SR.train_noise_predictor --config SR/configs/train_noise_predictor.yaml

# 从checkpoint恢复训练
python -m SR.train_noise_predictor --config SR/configs/train_noise_predictor.yaml --resume SR/experiments/noise_predictor/checkpoints/checkpoint_epoch_50.pth
```

### 4. 监控训练

训练过程中会显示：
- 每个batch的损失和学习率
- 每个epoch的平均损失
- 验证损失（如果配置了验证数据）

```
======================================================================
Epoch 1/200
======================================================================

Epoch [1/200]: 100%|████████| 250/250 [05:23<00:00,  1.29s/it, loss=0.0234, lr=1.00e-04]

Epoch 1 平均损失:
  total_loss: 0.0245
  l2_loss: 0.0198
  freq_loss: 0.0032
  stat_loss: 0.0015

当前学习率: 9.98e-05

验证 Epoch 1...
Validation: 100%|████████| 50/50 [01:12<00:00,  1.45s/it, loss=0.0221]
验证损失:
  total_loss: 0.0221
  l2_loss: 0.0178
  freq_loss: 0.0029
  stat_loss: 0.0014

✓ Checkpoint已保存: SR/experiments/noise_predictor_e2e/checkpoints/checkpoint_epoch_1.pth
✓ 最佳模型已保存: SR/experiments/noise_predictor_e2e/checkpoints/best_model.pth
```

## 📊 训练流程详解

### 完整训练流程

```
1. 数据加载
   ↓
   加载HR图像 → 随机裁剪 → 数据增强 → 应用退化 → 生成LR-HR对
   
2. 编码到潜在空间（冻结的VAE）
   ↓
   HR图像 → VAE编码 → hr_latent
   LR图像 → VAE编码 → lr_latent
   
3. ResShift反向采样（使用噪声预测器）
   ↓
   初始化：x_T = lr_latent + κ·√η_T·ε
   
   For t = T-1, T-2, ..., 1:
       ├─ 使用冻结的ResShift UNet预测x_0
       ├─ 计算后验分布参数（μ, σ²）
       ├─ 使用噪声预测器生成噪声（替代随机噪声）
       └─ 采样：x_{t-1} = μ + σ·noise
   
   得到：sr_latent = x_0
   
4. 计算损失
   ↓
   L_total = λ₁·L_L2 + λ₂·L_freq + λ₃·L_stat
   
   其中：
   - L_L2: MSE损失（sr_latent vs hr_latent）
   - L_freq: 频域感知损失
   - L_stat: 统计特征损失
   
5. 反向传播
   ↓
   只更新噪声预测器的参数
   （VAE和ResShift UNet保持冻结）
```

### 关键特性

#### 1. **ResShift扩散过程**

完全按照ResShift论文实现：
- ✅ 使用eta调度（指数增长）
- ✅ 残差偏移扩散：`x_t = (1-η_t)·x_0 + η_t·y + √η_t·κ·ε`
- ✅ 初始化：`x_T = y + κ·√η_T·ε`
- ✅ 后验分布：`μ = (η_{t-1}/η_t)·x_t + (α_t/η_t)·x_0`

#### 2. **数据加载器**

- ✅ 自动应用RealESRGAN退化管道
- ✅ 随机裁剪和数据增强
- ✅ 多线程加载（num_workers可配置）
- ✅ 支持验证数据集

#### 3. **训练优化**

- ✅ 混合精度训练（AMP）
- ✅ 梯度检查点（降低显存占用）
- ✅ 梯度裁剪（防止梯度爆炸）
- ✅ EMA（指数移动平均）
- ✅ 学习率调度（余弦退火）

#### 4. **Checkpoint管理**

- ✅ 自动保存最佳模型
- ✅ 保留最近N个checkpoint
- ✅ 支持恢复训练
- ✅ 保存完整训练状态

## ⚙️ 配置说明

### 关键配置参数

#### 训练参数

```yaml
training:
  num_epochs: 200              # 训练轮数
  batch_size: 4                # 批次大小
  num_workers: 4               # 数据加载线程数
  sampling_steps: 3            # 反向采样步数（3-5）
  
  # 扩散过程参数（ResShift）
  diffusion:
    num_timesteps: 1000        # 总扩散步数T
    kappa: 1.0                 # 方差控制参数κ
    eta_schedule: "exponential" # eta调度类型
    min_noise_level: 0.04      # η_1
    etas_end: 0.99             # η_T
    eta_power: 2.0             # 指数调度幂次
```

#### 噪声预测器参数

```yaml
noise_predictor:
  latent_channels: 3           # VAE潜在空间通道数
  model_channels: 160          # 基础通道数
  channel_mult: [1, 2, 3]      # 通道倍增因子
  num_res_blocks: 2            # 每层残差块数量
  attention_levels: 2          # 使用注意力的层数
  num_heads: 8                 # 注意力头数
  use_cross_attention: true    # 使用交叉注意力
  use_frequency_aware: true    # 使用频域感知
```

#### 损失函数权重

```yaml
loss:
  l2_weight: 1.0               # L2损失权重
  freq_weight: 0.1             # 频域损失权重
  stat_weight: 0.5             # 统计损失权重
```

#### 优化器参数

```yaml
optimizer:
  type: "AdamW"                # 优化器类型
  lr: 1.0e-4                   # 学习率
  beta1: 0.9                   # Adam beta1
  beta2: 0.999                 # Adam beta2
  weight_decay: 1.0e-4         # 权重衰减

scheduler:
  type: "CosineAnnealing"      # 调度器类型
  min_lr: 1.0e-6               # 最小学习率
```

## 💾 Checkpoint管理

### Checkpoint内容

每个checkpoint包含：
```python
{
    'epoch': 当前epoch,
    'global_step': 全局步数,
    'noise_predictor': 噪声预测器权重,
    'optimizer': 优化器状态,
    'scheduler': 学习率调度器状态,
    'best_loss': 最佳损失,
    'ema': EMA权重（如果启用）,
    'config': 训练配置
}
```

### 保存策略

- **定期保存**：每N个epoch保存一次（`save_interval`）
- **最佳模型**：验证损失最低时保存为`best_model.pth`
- **自动清理**：只保留最近N个checkpoint（`keep_recent_checkpoints`）

### 恢复训练

```bash
python -m SR.train_noise_predictor \
    --config SR/configs/train_noise_predictor.yaml \
    --resume SR/experiments/noise_predictor_e2e/checkpoints/checkpoint_epoch_50.pth
```

恢复训练会：
- ✅ 加载模型权重
- ✅ 恢复优化器状态
- ✅ 恢复学习率调度器
- ✅ 从上次的epoch继续训练

## 🔧 常见问题

### 1. 显存不足

**问题**：`RuntimeError: CUDA out of memory`

**解决方案**：
```yaml
training:
  batch_size: 2              # 减小批次大小
  use_gradient_checkpointing: true  # 启用梯度检查点
  gradient_accumulation_steps: 2    # 使用梯度累积
```

### 2. 训练速度慢

**问题**：训练速度太慢

**解决方案**：
```yaml
training:
  use_amp: true              # 启用混合精度训练
  num_workers: 8             # 增加数据加载线程
  use_gradient_checkpointing: false  # 禁用梯度检查点（如果显存充足）
```

### 3. 损失不收敛

**问题**：损失震荡或不下降

**解决方案**：
```yaml
optimizer:
  lr: 5.0e-5                 # 降低学习率

training:
  gradient_clip: 0.5         # 减小梯度裁剪阈值
  warmup_epochs: 10          # 增加预热轮数
```

### 4. 数据加载错误

**问题**：`ValueError: 在目录中没有找到任何图像文件`

**解决方案**：
- 检查数据目录路径是否正确
- 确保目录中有支持的图像格式
- 检查文件权限

## 📈 训练建议

### 数据准备

1. **数据量**：建议至少1000张高质量图像
2. **图像质量**：使用高分辨率、清晰的图像
3. **数据多样性**：包含不同场景、纹理、颜色

### 超参数调优

1. **学习率**：
   - 初始值：1e-4（默认）
   - 如果损失震荡：降低到5e-5
   - 如果收敛太慢：提高到2e-4

2. **批次大小**：
   - 显存充足：4-8
   - 显存不足：2-4（配合梯度累积）

3. **采样步数**：
   - 快速训练：3步
   - 平衡质量：4步
   - 最佳质量：5步

4. **损失权重**：
   - L2损失：1.0（主要损失）
   - 频域损失：0.1-0.2（细节增强）
   - 统计损失：0.3-0.5（纹理一致性）

### 训练监控

1. **损失曲线**：
   - 总损失应该稳定下降
   - 验证损失应该跟随训练损失

2. **学习率**：
   - 使用余弦退火逐渐降低
   - 最小学习率设为初始值的1/100

3. **梯度**：
   - 监控梯度范数
   - 如果梯度爆炸，降低学习率或减小梯度裁剪阈值

## 🎯 训练完成后

### 1. 评估模型

使用最佳模型进行推理：
```python
from SR.train_noise_predictor import NoisePredictorTrainer

# 加载训练器
trainer = NoisePredictorTrainer('SR/configs/train_noise_predictor.yaml')

# 加载最佳模型
trainer.load_checkpoint('SR/experiments/noise_predictor_e2e/checkpoints/best_model.pth')

# 进行推理
# ...
```

### 2. 导出模型

只导出噪声预测器权重：
```python
import torch

# 加载checkpoint
ckpt = torch.load('SR/experiments/noise_predictor_e2e/checkpoints/best_model.pth')

# 保存噪声预测器权重
torch.save(
    ckpt['noise_predictor'],
    'SR/pretrained/noise_predictor_best.pth'
)
```

### 3. 集成到推理流程

在ResShift推理中使用训练好的噪声预测器：
```python
# 加载噪声预测器
noise_predictor = create_noise_predictor(...)
noise_predictor.load_state_dict(torch.load('SR/pretrained/noise_predictor_best.pth'))

# 在反向采样中使用
# 替代：noise = torch.randn_like(x_t)
# 改为：noise, scale = noise_predictor(x_t, t, lr_latent)
```

## 📚 参考资料

- **ResShift论文**：[ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting](https://arxiv.org/abs/2307.12348)
- **训练实现文档**：`SR/RESSHIFT_TRAINING_FINAL.md`
- **扩散过程对比**：`SR/RESSHIFT_VS_DDPM.md`
- **Gradient Checkpointing**：`SR/GRADIENT_CHECKPOINTING.md`

## 🎉 总结

现在你已经有了一个完整的训练脚本，包括：

✅ **数据加载器**：自动应用退化管道生成LR-HR对  
✅ **ResShift扩散过程**：完全按照论文实现  
✅ **训练循环**：完整的epoch循环和验证  
✅ **Checkpoint管理**：自动保存和恢复  
✅ **优化技巧**：AMP、EMA、梯度检查点等  

开始训练吧！🚀
