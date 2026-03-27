"""
Noise predictor based on ResShift Swin-UNet architecture
Input: intermediate state (z_t), UNet prediction (pred_x0), LR image (lq), timesteps
Output: DiagonalGaussianDistribution (mean and logvar for probabilistic prediction)

输入拼接: z_t (3ch) + pred_x0 (3ch) + lq特征 (base_chn) → 预测采样噪声
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import basic operations from ldm modules
from ldm.modules.diffusionmodules.openaimodel import (
    avg_pool_nd,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)

# Import Swin Transformer layer (required - no fallback)
from models.swin_transformer import BasicLayer


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution for probabilistic noise prediction.
    Allows sampling and KL divergence computation.

    Args:
        parameters: Tensor of shape [B, 2*C, H, W] where first half is mean, second half is logvar
        deterministic: If True, std is set to 0 (deterministic mode)
    """

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
        """
        Sample from the distribution.

        Returns:
            Sampled tensor of shape [B, C, H, W]
        """
        if generator is not None:
            device = self.mean.device
            sample = torch.randn(
                self.mean.shape,
                generator=generator,
                device=device,
                dtype=self.mean.dtype,
            )
        else:
            sample = torch.randn_like(self.mean)

        return self.mean + self.std * sample

    def kl(
        self, other: Optional["DiagonalGaussianDistribution"] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(self || other).

        Args:
            other: Another distribution. If None, compute KL(self || N(0, I))

        Returns:
            KL divergence per batch element [B]
        """
        if self.deterministic:
            return torch.zeros(
                [self.mean.shape[0]], device=self.mean.device, dtype=self.mean.dtype
            )

        if other is None:
            # KL(self || N(0, I))
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            # KL(self || other)
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

    def partial_kl(
        self, other: Optional["DiagonalGaussianDistribution"] = None
    ) -> torch.Tensor:
        """
        Compute partial KL divergence (variance term only).

        Args:
            other: Another distribution. If None, compute partial KL relative to N(0, I)

        Returns:
            Partial KL divergence per batch element [B]
        """
        if self.deterministic:
            return torch.zeros(
                [self.mean.shape[0]], device=self.mean.device, dtype=self.mean.dtype
            )

        if other is None:
            return 0.5 * torch.sum(self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3],
            )

    def nll(
        self, sample: torch.Tensor, dims: Tuple[int, ...] = (1, 2, 3)
    ) -> torch.Tensor:
        """
        Compute negative log likelihood.

        Args:
            sample: Sample tensor
            dims: Dimensions to sum over

        Returns:
            NLL per batch element [B]
        """
        if self.deterministic:
            return torch.zeros(
                [self.mean.shape[0]], device=self.mean.device, dtype=self.mean.dtype
            )

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """
        Get the mode of the distribution (equivalent to mean for Gaussian).

        Returns:
            Mean tensor of shape [B, C, H, W]
        """
        return self.mean


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class SwinUNetNoisePredictor(nn.Module):
    """
    Noise predictor based on ResShift Swin-UNet architecture.

    This model takes intermediate state (z_t), UNet prediction (pred_x0), LR image (lq), and timesteps as input,
    and outputs the predicted noise (same shape as z_t).

    Input: z_t + pred_x0 + lq特征 → 拼接后输入UNet → 预测噪声

    Architecture follows ResShift's UNetModelSwin with Swin Transformer blocks.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 160,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: list = None,
        dropout: float = 0.0,
        channel_mult: tuple = (1, 2, 2, 4),
        conv_resample: bool = True,
        dims: int = 2,
        use_fp16: bool = False,
        num_heads: int = -1,
        num_head_channels: int = 32,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        swin_depth: int = 2,
        swin_embed_dim: int = 192,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        patch_norm: bool = False,
        cond_lq: bool = True,
        lq_size: int = 64,
        use_gradient_checkpointing: bool = False,
        double_z: bool = True,
    ):
        """
        Args:
            image_size: Size of the latent space (e.g., 64 for 256x256 image with 4x downsampled VAE)
            in_channels: Number of input channels in latent space (usually 3)
            model_channels: Base number of channels
            out_channels: Number of output channels (same as in_channels for noise prediction)
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Resolutions at which to apply attention
            dropout: Dropout probability
            channel_mult: Channel multiplier for each level
            conv_resample: Whether to use learned convolutions for up/downsampling
            dims: Number of dimensions (2 for images)
            use_fp16: Whether to use float16
            num_heads: Number of attention heads (-1 to use num_head_channels)
            num_head_channels: Number of channels per attention head
            use_scale_shift_norm: Whether to use scale-shift normalization
            resblock_updown: Whether to use residual blocks for up/downsampling
            swin_depth: Depth of Swin Transformer layers
            swin_embed_dim: Embedding dimension for Swin Transformer
            window_size: Window size for Swin Transformer
            mlp_ratio: MLP ratio for Swin Transformer
            patch_norm: Whether to use patch normalization in Swin
            cond_lq: Whether to condition on low-quality image
            lq_size: Size of the low-quality image (latent space size)
            use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
            double_z: If True, output 2*out_channels channels for probabilistic prediction (mean + logvar)
        """
        super().__init__()

        if attention_resolutions is None:
            attention_resolutions = [64, 32, 16, 8]

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)

        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0

        self.num_res_blocks = num_res_blocks
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.double_z = double_z

        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Feature extractor for LQ image
        if cond_lq and lq_size == image_size:
            self.feature_extractor = nn.Identity()
            base_chn = 3
        else:
            feature_extractor = []
            feature_chn = 3
            base_chn = 16
            num_down = int(math.log(lq_size / image_size) / math.log(2))
            for _ in range(num_down):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(
                    Downsample(base_chn, True, out_channels=base_chn * 2)
                )
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        # Input blocks (encoder)
        # 输入: z_t (3ch) + pred_x0 (3ch) + lr特征 (base_chn)
        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels_total = in_channels * 2 + base_chn  # z_t + pred_x0 + lq特征

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels_total, ch, 3, padding=1)
                )
            ]
        )
        input_block_chans = [ch]
        ds = image_size

        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)

                # Add Swin Transformer at specific resolutions
                if ds in attention_resolutions and jj == 0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads
                            if num_head_channels == -1
                            else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.0,
                            drop_path=0.0,
                            use_checkpoint=self.use_gradient_checkpointing,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        # Middle block
        middle_layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]
        middle_layers.append(
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads
                if num_head_channels == -1
                else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.0,
                drop_path=0.0,
                use_checkpoint=self.use_gradient_checkpointing,
                norm_layer=normalization,
                patch_norm=patch_norm,
            )
        )
        middle_layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        # Output blocks (decoder)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)

                # Add Swin Transformer at specific resolutions
                if ds in attention_resolutions and i == 0:
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads
                            if num_head_channels == -1
                            else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.0,
                            drop_path=0.0,
                            use_checkpoint=self.use_gradient_checkpointing,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output layer
        # If double_z=True, output 2*out_channels for probabilistic prediction (mean + logvar)
        final_out_channels = 2 * out_channels if double_z else out_channels
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, final_out_channels, 3, padding=1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        pred_x0: torch.Tensor,
        lr_image: torch.Tensor,
        timesteps: torch.Tensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
        return_dist: bool = False,
    ):
        """
        Forward pass of the noise predictor.

        Args:
            z_t: Intermediate state in latent space [B, C, H, W]
            pred_x0: UNet预测的x_0 [B, C, H, W]
            lr_image: LR image in image space [-1, 1] [B, C, H_lr, W_lr]
            timesteps: Current timestep [B]
            sample_posterior: If True and double_z=True, sample from distribution; otherwise return mode
            generator: Optional random generator for reproducible sampling
            return_dist: If True, return DiagonalGaussianDistribution instead of tensor

        Returns:
            Predicted noise tensor [B, C, H, W] (or DiagonalGaussianDistribution if return_dist=True)
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(
            self.dtype
        )

        # Process LQ condition
        lq = self.feature_extractor(lr_image.type(self.dtype))
        if lq.shape[2:] != z_t.shape[2:]:
            lq = F.pixel_unshuffle(lq, 2)

        # 拼接输入: z_t + pred_x0 + lq特征
        x = torch.cat([z_t, pred_x0, lq], dim=1)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        out = self.out(h)

        # Handle distribution output
        if self.double_z:
            dist = DiagonalGaussianDistribution(out)
            if return_dist:
                return dist
            elif sample_posterior:
                return dist.sample(generator=generator)
            else:
                return dist.mode()
        else:
            return out


def create_noise_predictor(
    image_size: int = 64, latent_channels: int = 3, **kwargs
) -> nn.Module:
    """
    Create a noise predictor with ResShift Swin-UNet architecture.

    Args:
        image_size: Size of the latent space
        latent_channels: Number of channels in latent space
        **kwargs: Additional arguments passed to SwinUNetNoisePredictor

    Returns:
        SwinUNetNoisePredictor model
    """
    return SwinUNetNoisePredictor(
        image_size=image_size, in_channels=latent_channels, **kwargs
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with ResShift-style config
    model = create_noise_predictor(
        image_size=64,
        latent_channels=3,
        model_channels=160,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[64, 32, 16, 8],
        num_head_channels=32,
        use_scale_shift_norm=True,
        swin_depth=2,
        swin_embed_dim=192,
        window_size=8,
        mlp_ratio=4,
        cond_lq=True,
        lq_size=64,
        double_z=True,  # Enable probabilistic output
    ).to(device)

    # Test input
    batch_size = 2
    timesteps = torch.randint(0, 4, (batch_size,)).to(device)
    z_t = torch.randn(batch_size, 3, 64, 64).to(device)  # Intermediate state (latent)
    pred_x0 = torch.randn(batch_size, 3, 64, 64).to(device)  # UNet预测的x_0
    lr_image = torch.randn(batch_size, 3, 64, 64).to(device)  # LR image (image space)

    # Test forward pass
    print("=" * 50)
    print("Swin-UNet Noise Predictor Test (Probabilistic Output)")
    print("=" * 50)

    print(f"Intermediate state shape: {z_t.shape}")
    print(f"UNet pred_x0 shape: {pred_x0.shape}")
    print(f"LR image shape: {lr_image.shape}")

    # Test 1: Default - sample from distribution
    sampled_noise = model(z_t, pred_x0, lr_image, timesteps, sample_posterior=True)
    print(f"\n[sample_posterior=True] Sampled noise shape: {sampled_noise.shape}")

    # Test 2: Return mode (deterministic)
    mode_noise = model(z_t, pred_x0, lr_image, timesteps, sample_posterior=False)
    print(f"[sample_posterior=False] Mode noise shape: {mode_noise.shape}")

    # Test 3: Return distribution object (for computing KL loss)
    dist = model(z_t, pred_x0, lr_image, timesteps, return_dist=True)
    print(f"\n[return_dist=True] Distribution mean shape: {dist.mean.shape}")
    print(f"Distribution logvar shape: {dist.logvar.shape}")

    # Test KL divergence
    kl = dist.kl()
    print(f"KL divergence shape: {kl.shape}, mean: {kl.mean().item():.4f}")

    print(
        f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )
