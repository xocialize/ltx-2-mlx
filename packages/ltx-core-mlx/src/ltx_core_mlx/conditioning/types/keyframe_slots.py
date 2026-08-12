"""Generated keyframe slots — the LTX-2.5 multishot conditioning surface.

Ported from ltx-core `conditioning/types/keyframe_slots.py`.

A *slot* is a block of appended tokens the model DENOISES (unlike ordinary
keyframe conditioning, which appends clean reference tokens). Each slot asks the
model to invent one standalone keyframe at a chosen pixel-frame index; the DFR
pipeline then densifies between them. This is the mechanism behind "native
multishot generation".

Two things distinguish a slot from a regular latent frame, and both are
load-bearing:

* **RoPE span.** A normal latent frame spans ``temporal_scale_factor`` (8) pixel
  frames; a slot's temporal extent is forced to exactly ``[t, t+1)`` before the
  ``/fps`` division. ``causal_fix`` must NOT also be applied — it would collapse
  frame 0's span a second time.
* **keyframes_mask marked=True.** Slots receive the model's learned keyframes
  absolute-position embedding (already ported in the DiT). Ordinary
  ``VideoConditionByKeyframeIndex`` conditioning is NOT marked.

⚠️ Slots are denoised, so ``denoise_mask`` is 1 for their tokens and
``clean_latent`` is zeros — the noiser lerps the slot toward noise and the
clean latent is ignored.

⚠️ **Read generated keyframes back ONE FRAME AT A TIME.** A K-frame causal
decode would blend slots that were never temporally adjacent
(upstream ``types.py:271-273``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ltx_core_mlx.utils.positions import compute_video_positions


@dataclass(frozen=True)
class GeneratedKeyframeLayout:
    """Where the slot tokens live, so they can be read back after denoising."""

    pixel_frame_indices: tuple[int, ...]
    tokens_per_keyframe: int
    first_token: int

    def slot_span(self, i: int) -> tuple[int, int]:
        start = self.first_token + i * self.tokens_per_keyframe
        return start, start + self.tokens_per_keyframe


def slot_positions(
    pixel_frame_index: int,
    height: int,
    width: int,
    fps: float,
    spatial_scale: int = 32,
) -> mx.array:
    """(1, H*W, 3) point-coordinate positions for one slot.

    Full spatial grid at the target latent resolution; temporal coordinate is
    the slot's own pixel-frame index with a one-pixel-frame span, divided by
    fps. Our convention is the midpoint of upstream's ``[start, end)`` bounds
    (validated bit-exact in the DiT gate), so the midpoint of ``[t, t+1)`` is
    ``t + 0.5``.
    """
    base = compute_video_positions(1, height, width, fps)
    if base.ndim == 2:
        base = base[None]
    t_mid = (float(pixel_frame_index) + 0.5) / fps
    spatial = base[..., 1:]
    temporal = mx.full((base.shape[0], base.shape[1], 1), t_mid)
    return mx.concatenate([temporal, spatial], axis=-1)


def append_keyframe_slots(
    latent: mx.array,
    denoise_mask: mx.array,
    positions: mx.array,
    clean_latent: mx.array,
    keyframes_mask: mx.array | None,
    pixel_frame_indices: tuple[int, ...],
    height: int,
    width: int,
    fps: float,
    channels: int = 128,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, GeneratedKeyframeLayout]:
    """Append one denoised slot per entry in ``pixel_frame_indices``.

    Returns ``(latent, denoise_mask, positions, clean_latent, keyframes_mask,
    layout)``. Slot latents start at zero; the caller's noiser fills them.
    """
    B, N, C = latent.shape
    tokens_per = height * width
    n_new = tokens_per * len(pixel_frame_indices)

    slots = mx.zeros((B, n_new, C), dtype=latent.dtype)
    slot_pos = mx.concatenate(
        [slot_positions(t, height, width, fps) for t in pixel_frame_indices], axis=1
    )
    if slot_pos.shape[0] != B:
        slot_pos = mx.broadcast_to(slot_pos, (B, n_new, slot_pos.shape[-1]))

    # slots are DENOISED (mask 1) and carry the keyframes marker
    new_mask = mx.ones((B, n_new, 1), dtype=denoise_mask.dtype)
    base_kf = keyframes_mask if keyframes_mask is not None else mx.zeros((B, N, 1))
    new_kf = mx.ones((B, n_new, 1), dtype=base_kf.dtype)

    layout = GeneratedKeyframeLayout(
        pixel_frame_indices=tuple(pixel_frame_indices),
        tokens_per_keyframe=tokens_per,
        first_token=N,
    )
    return (
        mx.concatenate([latent, slots], axis=1),
        mx.concatenate([denoise_mask, new_mask], axis=1),
        mx.concatenate([positions, slot_pos], axis=1),
        mx.concatenate([clean_latent, mx.zeros_like(slots)], axis=1),
        mx.concatenate([base_kf, new_kf], axis=1),
        layout,
    )


def extract_keyframes(latent: mx.array, layout: GeneratedKeyframeLayout,
                      height: int, width: int) -> list[mx.array]:
    """Pull each denoised slot back out as its own (B, C, 1, H, W) latent.

    One latent per slot, deliberately: they must be decoded as standalone
    one-frame clips, never stacked into a K-frame video.
    """
    out = []
    for i in range(len(layout.pixel_frame_indices)):
        s, e = layout.slot_span(i)
        toks = latent[:, s:e, :]
        B, _, C = toks.shape
        out.append(toks.reshape(B, height, width, C).transpose(0, 3, 1, 2)[:, :, None])
    return out


def evenly_spaced_keyframe_positions(num_keyframes: int, num_frames: int) -> list[int]:
    """Interior pixel-frame positions for *num_keyframes* generated keyframes. Endpoints excluded.

    ⚠️ Upstream is ``torch.linspace(0, num_frames - 1, num_keyframes + 2).round()``, and
    ``torch.round`` is round-half-to-EVEN. Python's builtin ``round`` matches that, so the
    list-comprehension form below is bit-identical — but only because both are banker's
    rounding, not by coincidence of the arithmetic. Gated over a full sweep in
    ``scripts/gate_ltx25_slot_positions.py``; do not "simplify" to int(x + 0.5).
    """
    if num_keyframes < 0:
        raise ValueError(f"num_keyframes must be non-negative, got {num_keyframes}")
    if num_keyframes == 0:
        return []
    if num_frames < num_keyframes + 2:
        raise ValueError(
            f"Generated keyframes need at least num_keyframes + 2 target frames, got "
            f"num_keyframes={num_keyframes}, num_frames={num_frames}"
        )
    # float32 to match torch.linspace's dtype before rounding — at a .5 boundary the
    # float32 and float64 grids can land on different sides.
    grid = np.linspace(0, num_frames - 1, num_keyframes + 2, dtype=np.float32)
    return [int(v) for v in np.round(grid).astype(np.int64).tolist()][1:-1]


def has_generated_keyframes(generated_keyframes: int | Sequence[int]) -> bool:
    """Whether a ``generated_keyframes`` request asks for any slots.

    Callers must not test the argument's truthiness directly: a ``Sequence`` may be a tensor or an
    array, whose truth value is ambiguous and raises.
    """
    if isinstance(generated_keyframes, int):
        return generated_keyframes > 0
    return len(generated_keyframes) > 0


def resolve_generated_keyframes(generated_keyframes: int | Sequence[int], num_frames: int) -> list[int]:
    """Normalize the pipeline-level ``generated_keyframes`` argument to pixel-frame indices.

    An ``int`` requests that many evenly spaced interior keyframes; a sequence gives the target
    pixel-frame indices explicitly. ``0`` / empty means the feature is off.
    """
    if isinstance(generated_keyframes, int):
        return evenly_spaced_keyframe_positions(generated_keyframes, num_frames)

    positions = sorted({int(position) for position in generated_keyframes})
    if positions and (positions[0] < 0 or positions[-1] >= num_frames):
        raise ValueError(
            f"Generated keyframe positions must lie in [0, {num_frames}), got "
            f"{sorted(int(p) for p in generated_keyframes)}"
        )
    return positions
