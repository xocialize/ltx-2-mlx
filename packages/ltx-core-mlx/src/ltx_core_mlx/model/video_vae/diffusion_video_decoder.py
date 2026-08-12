"""LTX-2.5 DiffVAE — the diffusion video decoder.

Ported from ltx-core `model/video_vae/diffusion_video_decoder.py` and the
`transformer/` package, following the COMBINED (full-volume) semantics. The
chunked/deferred path is deliberately NOT followed: its W-chunking
edge-replicates halo columns instead of shifting the NA window, which differs
from full-volume NATTEN at the outermost `kw//2` columns.

Shape/layout contract: the latent and the final pixels are channels-FIRST
``(B, C, T, H, W)``; everything between ``conv_in`` and ``conv_out`` is
channels-LAST ``(B, T, H, W, C)``.

Traps encoded here (each cost a careful read of the reference):

* **Per-block context re-injection.** Every one of the 8 stage-5 blocks adds
  its own ``context_proj(context)`` — the reference achieves this by having
  each block read an aliased ``[context | x]`` buffer. Dropping it after the
  first block is an easy and silent bug.
* **AdaLN chunk order is scale-then-shift**: ``scale_msa, shift_msa, gate_msa,
  scale_mlp, shift_mlp, gate_mlp, gate_ctx``, and the effective value is
  ``shared_adaln_chunk_i + scale_shift_table[i]``. The three gates are computed
  upstream but discarded — they are pre-folded into ``attn.proj`` /
  ``mlp.w_down`` / ``context_proj``. (Verified: neither shipped checkpoint
  carries gate tensors, so the fold is a no-op on both sides.)
* **Modulation is applied to the NORMED tensor**: ``norm(x) * (1 + scale) + shift``.
* **The two pixel-shuffles pack channels differently**: the upsample packs
  ``(c, t, h, w)`` while patchify/unpatchify pack ``(c, t, w, h)`` — W before H.
* **``norm_out`` has no modulation** (bare RMSNorm), unlike most DiT heads.
* **``forward`` returns ``[-1, 1]``**; the ``(x+1)*0.5`` clamp lives only in
  the video-writing helper.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.na3d import (
    apply_rope_3d,
    default_rope_dim_split,
    na3d,
    rope_inv_freqs,
)


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


class SwiGLU(nn.Module):
    """``w_down( silu(x @ w_gate.T) * (x @ w_up.T) )`` — all bias-free."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


def _mlp_hidden(dim: int, mult: float = 4.0) -> int:
    """Reference rounding: ``(int(dim*mult) + 15) // 16 * 16``."""
    return (int(dim * mult) + 15) // 16 * 16


class NAAttention(nn.Module):
    """Neighborhood attention with per-axis RoPE (channels-last volumes)."""

    def __init__(self, dim: int, kernel: tuple[int, int, int], head_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.kernel = tuple(kernel)
        self.scale = head_dim**-0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        d_t, d_h, d_w = default_rope_dim_split(head_dim)
        self.dim_split = (d_t, d_h, d_w)
        self._inv_t = rope_inv_freqs(d_t)
        self._inv_h = rope_inv_freqs(d_h)
        self._inv_w = rope_inv_freqs(d_w)

    def __call__(self, x: mx.array, tile: tuple[int | None, int | None] = (None, None)) -> mx.array:
        """x: (B, T, H, W, C) -> attention output (B, T, H, W, C), pre-``proj``.

        The reference applies ``proj`` outside, as part of the residual, so this
        returns the un-projected attention output.
        """
        B, T, H, W, C = x.shape
        NH, HD = self.num_heads, self.head_dim

        def heads(t: mx.array) -> mx.array:
            return t.reshape(B, T, H, W, NH, HD)

        q = self.q_norm(heads(self.to_q(x)))
        k = self.k_norm(heads(self.to_k(x)))
        v = heads(self.to_v(x))

        q = q * self.scale  # folded before RoPE, as upstream
        q = apply_rope_3d(q, self._inv_t, self._inv_h, self._inv_w, self.dim_split)
        k = apply_rope_3d(k, self._inv_t, self._inv_h, self._inv_w, self.dim_split)

        out = na3d(q, k, v, self.kernel, scale=1.0,
                   query_tile_t=tile[0], query_tile_h=tile[1])
        return out.reshape(B, T, H, W, C)


class NABlock(nn.Module):
    """Deterministic block (stages 1-4): pre-norm attn + pre-norm SwiGLU."""

    def __init__(self, dim: int, kernel: tuple[int, int, int], head_dim: int = 64):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NAAttention(dim, kernel, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _mlp_hidden(dim))

    def __call__(self, x: mx.array, tile=(None, None)) -> mx.array:
        x = x + self.attn.proj(self.attn(self.norm1(x), tile))
        return x + self.mlp(self.norm2(x))


class DiffusionNABlock(nn.Module):
    """Stage-5 block: context injection + modulated attn + modulated SwiGLU."""

    def __init__(self, dim: int, kernel: tuple[int, int, int], context_channels: int,
                 head_dim: int = 64):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim)
        self.scale_shift_table = mx.zeros((7, dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NAAttention(dim, kernel, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _mlp_hidden(dim))

    def modulation(self, shared: list[mx.array]) -> tuple[mx.array, ...]:
        """(scale_msa, shift_msa, scale_mlp, shift_mlp); gates are pre-folded."""
        eff = [shared[i] + self.scale_shift_table[i].reshape(1, 1, 1, 1, -1) for i in range(7)]
        return eff[0], eff[1], eff[3], eff[4]

    def __call__(self, x: mx.array, context: mx.array, shared: list[mx.array],
                 tile=(None, None)) -> mx.array:
        s_msa, sh_msa, s_mlp, sh_mlp = self.modulation(shared)
        x = x + self.context_proj(context)  # every block re-injects
        h = self.norm1(x) * (1.0 + s_msa) + sh_msa
        x = x + self.attn.proj(self.attn(h, tile))
        h = self.norm2(x) * (1.0 + s_mlp) + sh_mlp
        return x + self.mlp(h)


class LinearPixelShuffleUpsample(nn.Module):
    """Linear then depth-to-space over (t, h, w). Channel packing: (c, t, h, w)."""

    def __init__(self, in_channels: int, stride: tuple[int, int, int], reduction: int):
        super().__init__()
        self.stride = tuple(stride)
        self.out_channels = in_channels // reduction
        self.proj = nn.Linear(in_channels, self.out_channels * math.prod(stride))

    def __call__(self, x: mx.array, drop_leading_frame: bool = False) -> mx.array:
        B, T, H, W, _ = x.shape
        p1, p2, p3 = self.stride
        c = self.out_channels
        y = self.proj(x).reshape(B, T, H, W, c, p1, p2, p3)
        # "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c"
        y = y.transpose(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, T * p1, H * p2, W * p3, c)
        if p1 == 2 and drop_leading_frame:
            y = y[:, 1:]
        return y


def patchify_hw(x: mx.array, patch: int) -> mx.array:
    """(B,C,T,H,W) -> (B, C*patch*patch, T, H/patch, W/patch).

    Reference packing is ``(c p r q)`` with q=h_sub, r=w_sub — i.e. channel
    index ``c*patch^2 + w_sub*patch + h_sub``: **W before H**.
    """
    B, C, T, H, W = x.shape
    y = x.reshape(B, C, T, H // patch, patch, W // patch, patch)  # c,t,hq,q,wr,r
    y = y.transpose(0, 1, 6, 4, 2, 3, 5)  # b, c, r(w_sub), q(h_sub), t, h, w
    return y.reshape(B, C * patch * patch, T, H // patch, W // patch)


def unpatchify_hw(x: mx.array, patch: int) -> mx.array:
    """Inverse of :func:`patchify_hw`."""
    B, CP, T, H, W = x.shape
    C = CP // (patch * patch)
    y = x.reshape(B, C, patch, patch, T, H, W)  # c, r(w_sub), q(h_sub), t, h, w
    y = y.transpose(0, 1, 4, 5, 3, 6, 2)  # b, c, t, h, q, w, r
    return y.reshape(B, C, T, H * patch, W * patch)


def timestep_embedding(t: mx.array, dim: int = 256) -> mx.array:
    """PixArt/diffusers sinusoid with ``flip_sin_to_cos=True``, shift 0 -> [cos, sin]."""
    half = dim // 2
    exponent = -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
    freqs = mx.exp(exponent)
    emb = t.astype(mx.float32)[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    return mx.concatenate([emb[:, half:], emb[:, :half]], axis=-1)


class DiffusionVideoDecoder(nn.Module):
    """LTX-2.5 DiffVAE decoder (1-step x0 by default)."""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        t_emb_dim: int = 384,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        upsample_reductions: tuple[int, ...] = (2, 2, 1, 2),
        timestep_scale_multiplier: float = 1000.0,
        model_output_type: str = "x0",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.stage5_kernel = tuple(stage5_kernel)
        self.stage_kernels = stage_kernels
        self.upsample_strides = upsample_strides
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.context_channels = stage_channels[-1]
        c5 = stage_channels[-1]

        self.per_channel_statistics = {
            "mean": mx.zeros((in_channels,)),
            "std": mx.ones((in_channels,)),
        }
        self.conv_in = nn.Linear(in_channels, stage_channels[0])
        self.det_stages = [
            [NABlock(stage_channels[i], stage_kernels[i], head_dim) for _ in range(stage_depths[i])]
            for i in range(4)
        ]
        self.upsamples = [
            LinearPixelShuffleUpsample(stage_channels[i], upsample_strides[i], upsample_reductions[i])
            for i in range(4)
        ]
        self.t_embedder = nn.Sequential(
            nn.Linear(256, t_emb_dim), nn.SiLU(), nn.Linear(t_emb_dim, t_emb_dim)
        )
        self.conv_in_x_t = nn.Linear(out_channels * patch_size * patch_size, c5)
        self.shared_adaln = nn.Linear(t_emb_dim, 7 * c5)
        self.diff_blocks = [
            DiffusionNABlock(c5, self.stage5_kernel, self.context_channels, head_dim)
            for _ in range(stage_depths[4])
        ]
        self.norm_out = nn.RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, out_channels * patch_size * patch_size)
        self.type_emb = mx.zeros((in_channels,))

    # --- pieces, individually gateable ---

    def un_normalize(self, z: mx.array) -> mx.array:
        """(B,C,T,H,W) latent de-normalization (channels-first)."""
        m = self.per_channel_statistics["mean"].reshape(1, -1, 1, 1, 1)
        s = self.per_channel_statistics["std"].reshape(1, -1, 1, 1, 1)
        return z * s + m

    def context_volume(self, latent: mx.array, tile=(None, None)) -> mx.array:
        """Latent -> stage-4 context volume (channels-last), pre-crop."""
        x = self.un_normalize(latent).transpose(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for i in range(4):
            for blk in self.det_stages[i]:
                x = blk(x, tile)
                mx.eval(x)
            x = self.upsamples[i](x, drop_leading_frame=True)
            mx.eval(x)
        return x

    def diff_step(self, context: mx.array, x_t: mx.array, t: mx.array,
                  tile=(None, None)) -> mx.array:
        """Stage 5: noised pixels + context -> pixels (channels-first, [-1,1])."""
        patched = patchify_hw(x_t, self.patch_size)
        x = self.conv_in_x_t(patched.transpose(0, 2, 3, 4, 1))

        t_emb = self.t_embedder(timestep_embedding(t * self.timestep_scale_multiplier))
        chunks = self.shared_adaln(nn.silu(t_emb))
        c5 = chunks.shape[-1] // 7
        shared = [chunks[:, i * c5 : (i + 1) * c5].reshape(-1, 1, 1, 1, c5) for i in range(7)]

        for blk in self.diff_blocks:
            x = blk(x, context, shared, tile)
            mx.eval(x)

        x = self.conv_out(self.norm_out(x))
        x = x.transpose(0, 4, 1, 2, 3)
        return unpatchify_hw(x, self.patch_size)
