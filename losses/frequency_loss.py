"""
频域感知损失（Focal Frequency Loss）

论文: Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)
论文链接: https://arxiv.org/abs/2012.12821

核心思想：
- 在频域中计算损失，关注不同频率成分的重建质量
- 使用自适应权重机制，动态调整不同频率的重要性
- 特别关注难以重建的频率成分（类似Focal Loss的思想）

优势：
- 能够更好地恢复高频细节和纹理
- 自适应地关注难以重建的频率成分
- 与空域损失互补，提升整体重建质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FocalFrequencyLoss(nn.Module):
    """
    频域感知损失（Focal Frequency Loss）
    
    该损失函数在频域中计算预测图像和目标图像的差异，并使用自适应权重
    机制来关注难以重建的频率成分。
    
    主要特点：
    1. 频域计算：使用FFT将图像转换到频域
    2. 自适应权重：根据频率成分的重建难度动态调整权重
    3. 多尺度支持：可以在多个尺度上计算损失
    4. 数值稳定性：添加了多重保护机制防止NaN和极值

    Args:
        loss_weight: 损失权重
        alpha: focal权重的指数，控制对难重建频率的关注程度（默认1.0）
        patch_factor: 将图像分块计算的因子（默认1，不分块）
        ave_spectrum: 是否对频谱取平均（默认False）
        log_matrix: 是否对频谱取对数（默认False）
        batch_matrix: 是否在batch维度上计算矩阵（默认False）
        eps: 数值稳定性的epsilon值（默认1e-6）
        max_weight: 权重矩阵的最大值裁剪（默认10.0）

    Examples:
        >>> criterion = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
        eps: float = 1e-6,
        max_weight: float = 10.0
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.eps = eps  # 更大的epsilon值，提升数值稳定性
        self.max_weight = max_weight  # 权重矩阵最大值裁剪

    def _safe_sqrt(self, x: torch.Tensor) -> torch.Tensor:
        """
        安全的平方根计算，防止负数和过小值

        Args:
            x: 输入tensor

        Returns:
            安全的平方根结果
        """
        return torch.sqrt(torch.clamp(x, min=self.eps))

    def _check_and_fix_nan(self, x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """
        检查并修复NaN和Inf值

        Args:
            x: 输入tensor
            name: tensor名称（用于调试）

        Returns:
            修复后的tensor
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            # 将NaN和Inf替换为0
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        return x

    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        """
        将图像转换到频域（带数值稳定性处理）

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            freq: 频域表示 [B, C, H, W, 2] (实部和虚部)
        """
        # 输入预处理：裁剪极端值防止FFT溢出
        x = torch.clamp(x, min=-100.0, max=100.0)

        # 使用2D FFT转换到频域
        freq = torch.fft.fft2(x, norm='ortho')

        # 将复数转换为实数表示 [real, imag]
        freq_real = self._check_and_fix_nan(freq.real, "freq_real")
        freq_imag = self._check_and_fix_nan(freq.imag, "freq_imag")
        freq = torch.stack([freq_real, freq_imag], dim=-1)

        return freq

    def _compute_amplitude(self, freq: torch.Tensor) -> torch.Tensor:
        """
        计算频谱幅度（带数值稳定性）

        Args:
            freq: 频域表示 [B, C, H, W, 2]

        Returns:
            amplitude: 幅度 [B, C, H, W]
        """
        # 使用安全的平方根计算
        amp_squared = freq[..., 0] ** 2 + freq[..., 1] ** 2
        amplitude = self._safe_sqrt(amp_squared)
        return amplitude

    def _compute_weight_matrix(
        self,
        pred_freq: torch.Tensor,
        target_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        计算自适应权重矩阵（带数值稳定性）

        Args:
            pred_freq: 预测频域表示 [B, C, H, W, 2]
            target_freq: 目标频域表示 [B, C, H, W, 2]

        Returns:
            matrix: 权重矩阵 [B, C, H, W]
        """
        # 计算频谱幅度
        pred_amp = self._compute_amplitude(pred_freq)
        target_amp = self._compute_amplitude(target_freq)

        # 计算自适应权重：频率成分差异越大，权重越高
        # 使用更稳定的计算方式：添加更大的epsilon，并裁剪分母
        diff = torch.abs(pred_amp - target_amp)
        # 确保分母不会太小
        denominator = torch.clamp(target_amp, min=self.eps * 10)
        matrix = diff / denominator

        # 裁剪权重矩阵，防止极端值
        matrix = torch.clamp(matrix, min=0.0, max=self.max_weight)

        # 检查并修复NaN
        matrix = self._check_and_fix_nan(matrix, "weight_matrix")

        if self.ave_spectrum:
            # 在通道维度上取平均
            matrix = torch.mean(matrix, dim=1, keepdim=True)

        if self.log_matrix:
            # 对权重取对数，压缩动态范围（使用安全的log）
            matrix = torch.log(matrix + 1.0)

        if self.batch_matrix:
            # 在batch维度上取平均
            matrix = torch.mean(matrix, dim=0, keepdim=True)

        return matrix

    def loss_formulation(
        self,
        recon_freq: torch.Tensor,
        real_freq: torch.Tensor,
        matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算频域损失（带数值稳定性）

        Args:
            recon_freq: 重建图像的频域表示 [B, C, H, W, 2]
            real_freq: 真实图像的频域表示 [B, C, H, W, 2]
            matrix: 频率权重矩阵 [B, C, H, W] 或 None

        Returns:
            loss: 频域损失值
        """
        # 计算频域差异
        diff = recon_freq - real_freq

        # 计算复数的模（使用安全的平方根）
        diff_squared = diff[..., 0] ** 2 + diff[..., 1] ** 2
        freq_distance = self._safe_sqrt(diff_squared)

        # 检查并修复NaN
        freq_distance = self._check_and_fix_nan(freq_distance, "freq_distance")

        # 应用focal权重
        if matrix is not None:
            # 确保权重矩阵是非负的
            matrix = torch.clamp(matrix, min=0.0)

            # 安全的幂运算：对于alpha != 1，先裁剪matrix避免极端值
            if self.alpha != 1.0:
                # 裁剪matrix到合理范围，防止幂运算产生极值
                matrix = torch.clamp(matrix, min=0.0, max=self.max_weight)
                # 对于非常小的值，直接设为eps避免0^alpha的问题
                matrix = torch.where(matrix < self.eps,
                                    torch.full_like(matrix, self.eps),
                                    matrix)

            weight_matrix = matrix ** self.alpha

            # 再次检查并修复NaN
            weight_matrix = self._check_and_fix_nan(weight_matrix, "weight_matrix_after_pow")

            freq_distance = freq_distance * weight_matrix

        # 最终检查
        freq_distance = self._check_and_fix_nan(freq_distance, "final_freq_distance")

        # 使用更稳定的mean计算
        loss = torch.mean(freq_distance)

        # 最终损失值检查
        if torch.isnan(loss) or torch.isinf(loss):
            # 如果最终损失仍然是NaN或Inf，返回0（不影响训练）
            return torch.tensor(0.0, device=recon_freq.device, dtype=recon_freq.dtype)

        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Focal Frequency Loss

        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            matrix: 可选的频率权重矩阵

        Returns:
            loss: 频域感知损失值
        """
        # 确保输入形状一致
        assert pred.shape == target.shape, \
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"

        # 输入预处理：检查并修复NaN
        pred = self._check_and_fix_nan(pred, "pred_input")
        target = self._check_and_fix_nan(target, "target_input")

        # 如果需要分块处理
        if self.patch_factor > 1:
            return self._forward_with_patches(pred, target, matrix)

        # 转换到频域
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # 如果没有提供权重矩阵，则计算自适应权重
        if matrix is None:
            matrix = self._compute_weight_matrix(pred_freq, target_freq)

        # 计算损失
        loss = self.loss_formulation(pred_freq, target_freq, matrix)

        return self.loss_weight * loss

    def _forward_with_patches(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        分块计算频域损失（带数值稳定性）

        将图像分成多个patch，分别计算频域损失后取平均。
        这样可以更好地捕捉局部频率特征。

        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            matrix: 可选的频率权重矩阵

        Returns:
            loss: 平均频域损失
        """
        b, c, h, w = pred.shape
        patch_size_h = h // self.patch_factor
        patch_size_w = w // self.patch_factor

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        num_patches = 0

        for i in range(self.patch_factor):
            for j in range(self.patch_factor):
                # 提取patch
                h_start = i * patch_size_h
                h_end = (i + 1) * patch_size_h if i < self.patch_factor - 1 else h
                w_start = j * patch_size_w
                w_end = (j + 1) * patch_size_w if j < self.patch_factor - 1 else w

                pred_patch = pred[:, :, h_start:h_end, w_start:w_end]
                target_patch = target[:, :, h_start:h_end, w_start:w_end]

                # 转换到频域
                pred_freq = self.tensor2freq(pred_patch)
                target_freq = self.tensor2freq(target_patch)

                # 计算patch的权重矩阵
                if matrix is None:
                    patch_matrix = self._compute_weight_matrix(pred_freq, target_freq)
                else:
                    patch_matrix = matrix[:, :, h_start:h_end, w_start:w_end]

                # 计算patch损失
                patch_loss = self.loss_formulation(pred_freq, target_freq, patch_matrix)

                # 只有当patch损失有效时才累加
                if not (torch.isnan(patch_loss) or torch.isinf(patch_loss)):
                    total_loss = total_loss + patch_loss
                    num_patches += 1

        # 防止除以0
        if num_patches == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 返回平均损失
        return self.loss_weight * (total_loss / num_patches)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"loss_weight={self.loss_weight}, "
                f"alpha={self.alpha}, "
                f"patch_factor={self.patch_factor}, "
                f"eps={self.eps}, "
                f"max_weight={self.max_weight})")