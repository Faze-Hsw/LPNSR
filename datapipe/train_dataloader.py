"""
训练数据加载器
用于加载图像数据集并应用RealESRGAN退化管道生成训练数据对
"""

import os
import sys
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2

from LPNSR.datapipe.realesrgan_degradation import RealESRGANDegradation


class RealESRGANTrainDataset(Dataset):
    """
    RealESRGAN训练数据集
    加载图像文件夹，随机裁剪到指定大小，并应用退化管道生成LR-HR图像对
    """
    
    def __init__(
        self,
        data_dir: str,
        config_path: str,
        gt_size: int = 256,
        use_hflip: bool = True,
        use_rot: bool = False,
        image_extensions: List[str] = None
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 图像数据目录路径
            config_path: RealESRGAN退化配置文件路径
            gt_size: ground truth图像大小（裁剪后的大小）
            use_hflip: 是否使用水平翻转增强
            use_rot: 是否使用旋转增强
            image_extensions: 支持的图像扩展名列表
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.gt_size = gt_size
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # 默认支持的图像格式
        if image_extensions is None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        
        # 递归加载所有图像路径（搜索目录及其所有子目录）
        self.image_paths = []
        for ext in image_extensions:
            # 使用 ** 模式递归搜索所有子目录
            self.image_paths.extend(glob.glob(str(self.data_dir / f'**/*{ext}'), recursive=True))
            self.image_paths.extend(glob.glob(str(self.data_dir / f'**/*{ext.upper()}'), recursive=True))
        
        # 去重（以防有重复路径）
        self.image_paths = list(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在目录 {data_dir} 及其子目录中没有找到任何图像文件")
        
        print(f"数据集初始化完成：在 {data_dir} 及其子目录中找到 {len(self.image_paths)} 张图像")
        
        # 初始化退化管道
        self.degrader = RealESRGANDegradation(config_path)
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_paths)
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        加载图像
        
        Args:
            img_path: 图像路径
        
        Returns:
            RGB格式的numpy数组，值范围[0, 255]
        """
        # 使用cv2读取图像（BGR格式）
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _random_crop_or_pad(self, img: np.ndarray, crop_size: int) -> np.ndarray:
        """
        随机裁剪或填充图像到指定大小
        
        Args:
            img: 输入图像 [H, W, C]
            crop_size: 目标大小
        
        Returns:
            裁剪/填充后的图像 [crop_size, crop_size, C]
        """
        h, w = img.shape[:2]
        
        # 如果图像小于目标大小，先填充
        if h < crop_size or w < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            
            # 使用反射填充
            img = cv2.copyMakeBorder(
                img,
                pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                cv2.BORDER_REFLECT_101
            )
            h, w = img.shape[:2]
        
        # 随机裁剪
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        img_crop = img[top:top + crop_size, left:left + crop_size, :]
        return img_crop
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """
        数据增强
        
        Args:
            img: 输入图像 [H, W, C]
        
        Returns:
            增强后的图像
        """
        # 水平翻转
        if self.use_hflip and random.random() < 0.5:
            img = cv2.flip(img, 1)
        
        # 旋转（90度的倍数）
        if self.use_rot:
            rot_times = random.randint(0, 3)
            if rot_times > 0:
                img = np.rot90(img, rot_times)
        
        return img
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        Args:
            index: 样本索引
        
        Returns:
            包含'lq'（低质量图像）和'gt'（高质量图像）的字典
            两者都是torch.Tensor，形状为[C, H, W]，值范围[0, 1]
        """
        # 加载图像
        img_path = self.image_paths[index]
        img = self._load_image(img_path)
        
        # 随机裁剪或填充到目标大小
        img_gt = self._random_crop_or_pad(img, self.gt_size)
        
        # 数据增强
        img_gt = self._augment(img_gt)
        
        # 转换为tensor [H, W, C] -> [C, H, W]，值范围[0, 1]
        # 使用.copy()确保数组是连续的，避免负步长问题
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1).copy()).float() / 255.0
        
        # 添加batch维度 [C, H, W] -> [1, C, H, W]
        img_gt_batch = img_gt.unsqueeze(0)
        
        # 应用退化管道
        result = self.degrader.degrade(img_gt_batch)
        
        # 移除batch维度 [1, C, H, W] -> [C, H, W]
        img_lq = result['lq'].squeeze(0)
        img_gt_out = result['gt'].squeeze(0)
        
        return {
            'lq': img_lq,  # 低质量图像（退化后）
            'gt': img_gt_out,  # 高质量图像（ground truth）
            'lq_path': img_path,  # 图像路径（用于调试）
        }


def create_train_dataloader(
    data_dir: str,
    config_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    gt_size: int = 256,
    use_hflip: bool = True,
    use_rot: bool = False,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建训练数据加载器
    
    Args:
        data_dir: 图像数据目录路径
        config_path: RealESRGAN退化配置文件路径
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        gt_size: ground truth图像大小
        use_hflip: 是否使用水平翻转
        use_rot: 是否使用旋转
        shuffle: 是否打乱数据
        pin_memory: 是否将数据固定在内存中（加速GPU传输）
    
    Returns:
        DataLoader对象
    """
    dataset = RealESRGANTrainDataset(
        data_dir=data_dir,
        config_path=config_path,
        gt_size=gt_size,
        use_hflip=use_hflip,
        use_rot=use_rot
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后不完整的batch
    )
    
    return dataloader


def test_dataloader():
    """测试数据加载器"""
    import matplotlib.pyplot as plt
    
    # 配置路径
    data_dir = Path(__file__).parent.parent / 'traindata'
    config_path = Path(__file__).parent.parent / 'configs' / 'realesrgan_degradation.yaml'
    
    print(f"数据目录: {data_dir}")
    print(f"配置文件: {config_path}")
    
    # 创建数据加载器
    dataloader = create_train_dataloader(
        data_dir=str(data_dir),
        config_path=str(config_path),
        batch_size=4,
        num_workers=2,  # 测试时使用0避免多进程问题
        gt_size=256,
        use_hflip=True,
        use_rot=False,
        shuffle=True
    )
    
    print(f"数据加载器创建成功，共 {len(dataloader)} 个batch")
    
    # 获取一个batch
    batch = next(iter(dataloader))
    
    print(f"\nBatch信息:")
    print(f"  LQ形状: {batch['lq'].shape}")
    print(f"  GT形状: {batch['gt'].shape}")
    print(f"  LQ值范围: [{batch['lq'].min():.3f}, {batch['lq'].max():.3f}]")
    print(f"  GT值范围: [{batch['gt'].min():.3f}, {batch['gt'].max():.3f}]")
    
    # 可视化第一个样本
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for i in range(2):
        # 转换为numpy数组 [C, H, W] -> [H, W, C]
        img_gt = batch['gt'][i].permute(1, 2, 0).cpu().numpy()
        img_lq = batch['lq'][i].permute(1, 2, 0).cpu().numpy()
        
        # 显示GT
        axes[i, 0].imshow(img_gt)
        axes[i, 0].set_title(f'Sample {i+1} - GT (HR)', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # 显示LQ
        axes[i, 1].imshow(img_lq)
        axes[i, 1].set_title(f'Sample {i+1} - LQ (Degraded)', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 测试迭代多个batch
    print("\n测试迭代3个batch...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"Batch {i+1}: LQ shape={batch['lq'].shape}, GT shape={batch['gt'].shape}")
    
    print("\n数据加载器测试完成！")


if __name__ == '__main__':
    test_dataloader()
