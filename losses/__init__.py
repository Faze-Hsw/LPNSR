"""
Super-resolution training loss functions module

Includes the following loss functions:
1. GANLoss: GAN adversarial loss
2. LPIPSLoss: LPIPS perceptual loss (CVPR 2018)
"""

from .gan_loss import (
    GANLoss,
    NLayerDiscriminator,
    UNetDiscriminator,
    create_discriminator,
)
from .lpips_loss import LPIPSLoss

__all__ = [
    "GANLoss",
    "NLayerDiscriminator",
    "UNetDiscriminator",
    "create_discriminator",
    "LPIPSLoss",
]
