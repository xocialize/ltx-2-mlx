"""Duration head — predicts shot duration in seconds from Connector outputs.

Ported from ltx-core/src/ltx_core/duration_head/duration_head.py (LTX-2.5).

Consumes the *connector* outputs (the same context the DiT receives), so it runs
right after prompt encoding and before any latent work. No attention mask: the
connector substitutes learnable registers for padded positions and marks the
result fully attendable, so every token here is already valid.

The regression target is trained in log-seconds; ``exp`` is applied at the end
so callers always get seconds.

⚠️ Weight layout: upstream uses ``nn.MultiheadAttention``, whose checkpoint has a
FUSED ``in_proj_weight`` (3*D, D) + ``in_proj_bias``. The converter
(``scripts/convert_ltx25.py``) splits it into ``q_proj`` / ``k_proj`` / ``v_proj``
to match this module.

⚠️ Numerical note: the pooler softmaxes ONE query over ~2048 keys and the pooled
vector sits ~20x below the token scale, so ordinary fp32 accumulation
differences in the input projections (1.7e-6) amplify to ~1.6e-3 on ``pooled``
and ~2e-4 on the predicted seconds. That is inherent to the shape, not a kernel
choice — an explicit softmax and ``precise=True`` agree with
``mx.fast.scaled_dot_product_attention`` to 1e-6. It is far below the frame-grid
quantization the prediction feeds (8/24 s), so the shipped frame count is
unaffected. See ``scripts/gate_ltx25_duration.py``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class AttentionPooler(nn.Module):
    """Cross-attend ``num_queries`` learnable tokens against ``tokens``.

    Produces ``(B, num_queries, hidden_dim)`` regardless of input length.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_tokens = mx.zeros((num_queries, hidden_dim))
        self.cross_attn = _MultiheadAttention(hidden_dim, num_heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        B = tokens.shape[0]
        queries = mx.broadcast_to(self.query_tokens[None], (B, self.num_queries, self.hidden_dim))
        return self.cross_attn(queries, tokens, tokens)


class _MultiheadAttention(nn.Module):
    """Torch ``nn.MultiheadAttention`` equivalent (batch_first, no mask).

    Split q/k/v projections; the converter unpacks upstream's fused ``in_proj``.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        B, Tq, D = q.shape
        Tk = k.shape[1]
        H, hd = self.num_heads, self.head_dim

        qh = self.q_proj(q).reshape(B, Tq, H, hd).transpose(0, 2, 1, 3)
        kh = self.k_proj(k).reshape(B, Tk, H, hd).transpose(0, 2, 1, 3)
        vh = self.v_proj(v).reshape(B, Tk, H, hd).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=hd**-0.5)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, D)
        return self.out_proj(out)


class DurationHead(nn.Module):
    """Predict duration in seconds from one or both Connector outputs.

    Args:
        video_cross_attention_dim: Video connector output dim.
        audio_cross_attention_dim: Audio connector output dim.
        pooler_hidden_dim: Shared hidden dim both modalities project into.
        num_queries: Number of learnable pooling queries.
        num_pooler_heads: Attention heads used by the pooler.
        mlp_hidden_dim: Hidden width of the output MLP.
    """

    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()
        self.pooler_hidden_dim = pooler_hidden_dim
        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = mx.zeros((pooler_hidden_dim,))
        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = mx.zeros((pooler_hidden_dim,))
        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=num_queries,
            num_heads=num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden_dim)
        self.mlp_out = nn.Linear(mlp_hidden_dim, 1)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        """Returns duration in seconds, shape ``(B,)``."""
        if video_tokens is None and audio_tokens is None:
            raise ValueError("DurationHead requires at least one of video_tokens / audio_tokens")

        groups: list[mx.array] = []
        if video_tokens is not None:
            groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)

        tokens = mx.concatenate(groups, axis=1)
        pooled = self.attention_pooler(tokens)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        hidden = nn.gelu_approx(self.mlp_hidden(pooled_flat))
        log_duration = self.mlp_out(hidden).squeeze(-1)
        return mx.exp(log_duration)


def seconds_to_num_frames(seconds: float, frame_rate: float, min_s: float = 1.0, max_s: float = 20.0) -> int:
    """Clamp a predicted duration and snap it to the causal frame grid (8k+1).

    Mirrors upstream ``seconds_to_clamped_num_frames`` (helpers.py): clamp to
    ``[min_s, max_s]`` (pipeline defaults 1.0 / 20.0), then round to the nearest
    valid frame count with ``(frames - 1) % 8 == 0``.
    """
    clamped = max(min_s, min(max_s, float(seconds)))
    raw = clamped * frame_rate
    k = round((raw - 1) / 8)
    return max(9, int(8 * k + 1))
