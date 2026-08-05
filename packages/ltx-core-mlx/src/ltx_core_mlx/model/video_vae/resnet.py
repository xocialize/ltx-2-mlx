"""Residual blocks for Video VAE.

Ported from ltx-core/src/ltx_core/model/video_vae/resnet.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.convolution import Conv3dBlock
from ltx_core_mlx.model.video_vae.normalization import pixel_norm


class ResBlock3d(nn.Module):
    """Pre-activation residual block: PixelNorm -> SiLU -> Conv.

    Reference: ltx-core ResnetBlock3D with NormLayerType.PIXEL_NORM.
    PixelNorm is parameterless so no norm weights in safetensors.

    Forward: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 + skip
    Produces keys: ``conv1.conv.{weight,bias}``, ``conv2.conv.{weight,bias}``
    """

    def __init__(self, channels: int, causal: bool = True, spatial_padding_mode: str = "zeros"):
        super().__init__()
        self.conv1 = Conv3dBlock(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = Conv3dBlock(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(nn.silu(pixel_norm(x)))
        x = self.conv2(nn.silu(pixel_norm(x)))
        return x + residual


class ResBlock3dAdapter(nn.Module):
    """Channel-changing residual block — PrunaVAED's ``conv_in``.

    Mirrors diffusers ``LTX2VideoResnetBlock3d`` in its ``in_channels != out_channels``
    configuration: the main path is the usual PixelNorm -> SiLU -> Conv pair, but the
    residual is routed through an *affine* LayerNorm over channels (``norm3``) and a
    1x1x1 ``conv_shortcut``.

    The stock LTX-2.3 decoder contains no such block — every stock ResBlock preserves its
    channel count and every stock norm is parameterless PixelNorm — so ``norm3`` is the
    only learned normalization anywhere in a pruned decoder. Pruna prunes *inside* each
    block while keeping the wider skip between blocks, and this block is what reconciles
    the two widths.

    Forward: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2  +  conv_shortcut(norm3(x))

    Produces keys: ``conv1.conv.{weight,bias}``, ``conv2.conv.{weight,bias}``,
    ``conv_shortcut.{weight,bias}``, ``norm3.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 1e-6,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv1 = Conv3dBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = Conv3dBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        # Reference uses a plain nn.Conv3d here, not the causal-padding wrapper.
        self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm3 = nn.LayerNorm(in_channels, eps=eps, affine=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(pixel_norm(x)))
        h = self.conv2(nn.silu(pixel_norm(h)))
        # norm3 normalizes over channels, which is already the last axis in BFHWC.
        return h + self.conv_shortcut(self.norm3(x))


class ResBlockStage(nn.Module):
    """A stage of N residual blocks at a fixed channel count.

    Produces keys: ``res_blocks.{i}.conv{1,2}.conv.{weight,bias}``
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.res_blocks = [
            ResBlock3d(channels, causal=causal, spatial_padding_mode=spatial_padding_mode) for _ in range(num_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.res_blocks:
            x = block(x)
        return x
