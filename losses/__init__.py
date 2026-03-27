"""
Super-resolution training loss functions module

Includes the following loss functions:
1. L2Loss: Standard L2 loss (MSE)
2. GANLoss: GAN adversarial loss
3. LPIPSLoss: LPIPS perceptual loss (CVPR 2018)
"""

from .basic_loss import L2Loss
from .gan_loss import (
    GANLoss,
    NLayerDiscriminator,
    UNetDiscriminator,
    create_discriminator,
)
from .lpips_loss import LPIPSLoss

__all__ = [
    "L2Loss",
    "GANLoss",
    "NLayerDiscriminator",
    "UNetDiscriminator",
    "create_discriminator",
    "LPIPSLoss",
]
