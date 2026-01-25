"""
Noise predictor based on EDSR-Unet architecture
Input is intermediate state, initial LR and current timestep, fully reconstructed network architecture
Preserves the original output distribution prediction and core function interfaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union
from dataclasses import dataclass

from LPNSR.ldm.modules.diffusionmodules.openaimodel import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
)

# Try importing xformers
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except ImportError:
    XFORMERS_IS_AVAILABLE = False
    print("xformers not available, using standard attention implementation")


@dataclass
class NoisePredictorOutput:
    noise: torch.Tensor
    latent_dist: Optional['DiagonalGaussianDistribution'] = None


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        eps = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: 'DiagonalGaussianDistribution' = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0]).to(self.parameters.device)

        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )


class EDSRResidualBlock(nn.Module):
    """EDSR-style residual block with dense connection feature fusion"""
    def __init__(self, channels: int, growth_rate: int = 32, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale

        # Dense connection convolution layers
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, channels, 3, padding=1)

        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Dense connections
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.activation(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))

        # Residual scaling
        return residual + self.res_scale * x4


class EDSRTimeModulatedBlock(nn.Module):
    """EDSR residual block with timestep modulation"""
    def __init__(self, channels: int, emb_channels: int, growth_rate: int = 32, res_scale: float = 0.1):
        super().__init__()
        self.res_block = EDSRResidualBlock(channels, growth_rate, res_scale)
        self.time_mlp = nn.Sequential(
            nn.ReLU(inplace=False),
            linear(emb_channels, channels),
            nn.Unflatten(1, (-1, 1, 1))  # Adjust to spatial dimensions
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Timestep modulation
        t_mod = self.time_mlp(t_emb)
        x = x + t_mod  # Time information injection

        # EDSR residual block
        return self.res_block(x)


class Downsample(nn.Module):
    """EDSR-style downsampling module"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class Upsample(nn.Module):
    """EDSR-style upsampling module"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample first then convolve
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.activation(self.conv(x))


class EDSRUnetNoisePredictor(nn.Module):
    """
    Noise predictor based on EDSR-Unet architecture
    Input: intermediate state (z_t), initial LR (lr_latent), timesteps
    """
    def __init__(
        self,
        latent_channels: int = 3,
        model_channels: int = 192,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        growth_rate: int = 32,
        res_scale: float = 0.1,
        double_z: bool = True
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_levels = len(channel_mult)
        self.double_z = double_z

        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.ReLU(inplace=False),
            linear(time_embed_dim, time_embed_dim)
        )

        # Input processing: fuse intermediate state and initial LR
        self.z_proj = nn.Conv2d(latent_channels, model_channels, 3, padding=1)
        self.lr_proj = nn.Conv2d(latent_channels, model_channels, 3, padding=1)
        self.input_fusion = nn.Conv2d(model_channels * 2, model_channels, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        current_channels = model_channels

        for level in range(self.num_levels):
            block = nn.ModuleList()
            # Add multiple residual blocks
            for _ in range(num_res_blocks):
                block.append(
                    EDSRTimeModulatedBlock(
                        current_channels,
                        time_embed_dim,
                        growth_rate,
                        res_scale
                    )
                )

            # Add downsampling (last layer does not downsample)
            if level < self.num_levels - 1:
                block.append(Downsample(current_channels))
                current_channels *= 2  # Double channels after downsampling

            self.encoder.append(block)

        # Bottleneck layer
        self.bottleneck = nn.ModuleList([
            EDSRTimeModulatedBlock(
                current_channels,
                time_embed_dim,
                growth_rate,
                res_scale
            ),
            EDSRTimeModulatedBlock(
                current_channels,
                time_embed_dim,
                growth_rate,
                res_scale
            )
        ])

        # Decoder
        self.decoder = nn.ModuleList()
        
        # Store input channel count for each decoder layer (for skip connections)
        decoder_channel_sizes = []
        
        for level in reversed(range(self.num_levels)):
            decoder_channel_sizes.append(current_channels)
            
            block = nn.ModuleList()
            
            # Add multiple residual blocks
            for _ in range(num_res_blocks):
                block.append(
                    EDSRTimeModulatedBlock(
                        current_channels,
                        time_embed_dim,
                        growth_rate,
                        res_scale
                    )
                )
            
            # Add upsampling (last layer does not upsample)
            if level > 0:
                block.append(Upsample(current_channels))
                current_channels = current_channels // 2  # Halve channels after upsampling

            self.decoder.append(block)
        
        # Create channel matching convolutions for skip connections
        self.skip_convs = nn.ModuleList()
        
        # List of channel counts saved by encoder (channels after each level processing)
        encoder_channels_list = []
        current_channels = model_channels
        for level in range(self.num_levels):
            # Channel count after each level processing
            encoder_channels_list.append(current_channels)
            if level < self.num_levels - 1:
                # Double channels after downsampling
                current_channels *= 2
        
        # Create corresponding skip connection convolutions for each decoder layer
        # Decoder goes from deep to shallow, encoder goes from shallow to deep
        # Therefore need reverse matching: decoder_level 0 uses encoder_level (num_levels-1)
        for i in range(self.num_levels):
            # Decoder layer i corresponds to encoder layer (num_levels-1-i)
            encoder_idx = self.num_levels - 1 - i
            encoder_channels = encoder_channels_list[encoder_idx]
            # Input channel count for corresponding decoder layer
            decoder_channels = decoder_channel_sizes[i]
            
            # Create channel matching convolution
            self.skip_convs.append(
                nn.Conv2d(encoder_channels, decoder_channels, kernel_size=1)
            )

        # Output layer
        output_channels = latent_channels * 2 if double_z else latent_channels
        # Use output channel count of the last decoder layer
        # The last decoder layer is the last element of decoder_channel_sizes
        final_decoder_channels = decoder_channel_sizes[-1]
        self.output = nn.Sequential(
            nn.Conv2d(final_decoder_channels, model_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            zero_module(nn.Conv2d(model_channels, output_channels, 3, padding=1))
        )

    def forward(
        self,
        z_t: torch.Tensor,          # Intermediate state
        lr_latent: torch.Tensor,    # Initial LR
        timesteps: torch.Tensor,    # Current timestep
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> Union[torch.Tensor, DiagonalGaussianDistribution, NoisePredictorOutput]:
        # Timestep embedding
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # Input fusion: intermediate state + initial LR
        z_feat = self.z_proj(z_t)
        lr_feat = self.lr_proj(lr_latent)
        x = self.input_fusion(torch.cat([z_feat, lr_feat], dim=1))

        # Encoder forward pass
        skip_connections = []
        for level in range(self.num_levels):
            # Pass through all residual blocks first
            for i, block in enumerate(self.encoder[level]):
                if isinstance(block, EDSRTimeModulatedBlock):
                    x = block(x, t_emb)
                else:  # Downsampling
                    # Save features before downsampling
                    skip_connections.append(x.clone())
                    x = block(x)
            
            # If it's the last layer, need to save features
            # Note: last layer has no downsampling, so save it here
            if level == self.num_levels - 1:
                skip_connections.append(x.clone())

        # Reverse skip_connections order to match decoder order
        # encoder: [level0, level1, level2, level3] -> [shallow to deep]
        # decoder needs: [level3, level2, level1, level0] -> [deep to shallow]
        skip_connections = skip_connections[::-1]

        # Bottleneck layer
        for block in self.bottleneck:
            x = block(x, t_emb)

        # Decoder forward pass
        # skip_connections is now [level3, level2, level1, level0], matching decoder order
        for level in range(self.num_levels):
            # Get corresponding skip connection
            skip = skip_connections[level]

            # Skip connection channel matching
            skip = self.skip_convs[level](skip)

            # Skip connection fusion (ensure size matching before processing any blocks)
            x = x + skip

            # Pass through all blocks of this layer (residual blocks and upsample)
            for block in self.decoder[level]:
                if isinstance(block, EDSRTimeModulatedBlock):
                    x = block(x, t_emb)
                elif isinstance(block, Upsample):
                    x = block(x)

        # Output processing
        h_out = self.output(x)

        if self.double_z:
            posterior = DiagonalGaussianDistribution(h_out)
            if sample_posterior:
                return posterior.sample(generator=generator)
            else:
                return posterior
        else:
            return h_out


def create_noise_predictor(
    latent_channels: int = 3, **kwargs
) -> nn.Module:
    return EDSRUnetNoisePredictor(latent_channels=latent_channels, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_noise_predictor(
        latent_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        growth_rate=32,
        res_scale=0.1
    ).to(device)

    # Test input: intermediate state, initial LR, timesteps
    batch_size = 2
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    z_t = torch.randn(batch_size, 3, 64, 64).to(device)  # Intermediate state
    lr_latent = torch.randn(batch_size, 3, 64, 64).to(device)  # Initial LR

    # Test forward pass
    print("=" * 50)
    print("EDSR-Unet Noise Predictor Test")
    print("=" * 50)

    predicted_noise = model(z_t, lr_latent, timesteps, sample_posterior=True)
    print(f"Intermediate state shape: {z_t.shape}")
    print(f"Initial LR shape: {lr_latent.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")

    posterior = model(z_t, lr_latent, timesteps, sample_posterior=False)
    print(f"Distribution mean shape: {posterior.mean.shape}")
    print(f"Distribution variance shape: {posterior.var.shape}")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")