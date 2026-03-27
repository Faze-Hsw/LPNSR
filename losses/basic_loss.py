"""
Basic Loss Functions

Includes standard L2 loss (MSE Loss)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Loss(nn.Module):
    """
    L2 Loss (Mean Squared Error Loss)

    This is the most commonly used basic loss function for super-resolution tasks,
    used to measure pixel-level differences between predicted and real images.

    Advantages:
    - Simple and efficient to compute
    - Stable optimization
    - Ensures basic reconstruction quality

    Disadvantages:
    - Tends to produce overly smooth results
    - Limited ability to recover high-frequency details

    Args:
        reduction: Loss reduction method, optional 'mean', 'sum', 'none'
        loss_weight: Loss weight for multi-loss weighting

    Examples:
        >>> criterion = L2Loss(reduction='mean', loss_weight=1.0)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    """

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum' or 'none', got {reduction}"
            )

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate L2 loss

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            weight: Optional pixel weight [B, 1, H, W] or [B, C, H, W]

        Returns:
            loss: L2 loss value
        """
        # Calculate squared error
        loss = F.mse_loss(pred, target, reduction="none")

        # Apply weight (if provided)
        if weight is not None:
            loss = loss * weight

        # Reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # No reduction for 'none' mode

        return self.loss_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction}, loss_weight={self.loss_weight})"


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth version of L1 loss)

    Charbonnier loss is a differentiable approximation of L1 loss,
    also commonly used in super-resolution tasks.
    Compared to MSE loss, it is more robust to outliers.

    Formula: loss = sqrt((pred - target)^2 + eps^2)

    Args:
        eps: Smoothing parameter to avoid zero gradient
        reduction: Loss reduction method
        loss_weight: Loss weight

    Examples:
        >>> criterion = CharbonnierLoss(eps=1e-3)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    """

    def __init__(
        self, eps: float = 1e-3, reduction: str = "mean", loss_weight: float = 1.0
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Charbonnier loss

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            weight: Optional pixel weight

        Returns:
            loss: Charbonnier loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps}, reduction={self.reduction}, loss_weight={self.loss_weight})"
