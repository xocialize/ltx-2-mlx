"""Gate: generated keyframe slots survive every other conditioning type.

Slots are appended tokens carrying a ``GeneratedKeyframeLayout`` and a
``keyframes_mask``. Any conditioning that returns a NEW ``LatentState`` must
propagate both. Dropping them is SILENT — the slots stay in the sequence and
still denoise, they just become unlocatable and lose their keyframes marking, so
the failure looks like mildly worse output rather than an error. This gate
applies each conditioning type after slots and asserts both survive.

    uv run python scripts/gate_ltx25_slot_conditioning.py
"""

from __future__ import annotations

import sys

import mlx.core as mx

from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    VideoConditionByLatentIndex,
    noise_latent_state,
)
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.utils.positions import compute_video_positions

F, H, W, C, FPS = 4, 4, 6, 128, 24.0
TPF = H * W
N = F * TPF


def fresh_state() -> LatentState:
    pos = compute_video_positions(F, H, W, FPS)
    return LatentState(
        latent=mx.random.normal((1, N, C)),
        clean_latent=mx.zeros((1, N, C)),
        denoise_mask=mx.ones((1, N, 1)),
        positions=pos if pos.ndim == 3 else pos[None],
    )


def main() -> int:
    fails: list[str] = []

    slots = VideoGeneratedKeyframeSlots((8, 16), frame_rate=FPS, noise_seed=7)
    base = slots.apply(fresh_state(), (F, H, W))
    n_slots = 2 * TPF
    if base.latent.shape[1] != N + n_slots:
        fails.append(f"slot append: {base.latent.shape[1]} != {N + n_slots}")
    if base.generated_keyframe_layout is None or base.keyframes_mask is None:
        fails.append("slot apply did not set layout / keyframes_mask")
    slot_block = base.latent[:, N:, :]
    if float(mx.abs(slot_block).max()) == 0.0:
        fails.append("slot tokens are all zero — they would never be generated")
    if float(mx.abs(slot_block - base.latent[:, :n_slots, :]).max()) < 1e-6:
        fails.append("slot noise repeats the base noise (seed not independent)")
    print(f"  [1/4] slot append          +{n_slots} tokens, noised (std "
          f"{float(mx.sqrt(mx.mean(slot_block ** 2))):.3f})   "
          f"{'PASS' if not fails else 'FAIL'}")

    n0 = len(fails)
    try:
        slots2 = VideoGeneratedKeyframeSlots((24,), frame_rate=FPS)
        slots2.apply(base, (F, H, W))
        fails.append("second slot application did not raise")
    except ValueError:
        pass
    print(f"  [2/4] double-apply raises  {'PASS' if len(fails) == n0 else 'FAIL'}")

    n0 = len(fails)
    kf_lat = mx.random.normal((1, TPF, C))
    others = {
        "VideoConditionByKeyframeIndex": VideoConditionByKeyframeIndex(
            frame_idx=24, keyframe_latent=kf_lat, spatial_dims=(F, H, W), frame_rate=FPS, strength=1.0
        ),
        "VideoConditionByLatentIndex": VideoConditionByLatentIndex(
            frame_indices=[0], clean_latent=mx.random.normal((1, TPF, C)), strength=1.0
        ),
        "VideoConditionByReferenceLatent": VideoConditionByReferenceLatent(
            reference_latent=mx.random.normal((1, TPF, C)),
            reference_positions=(base.positions[:, :TPF, :] if base.positions is not None else None),
            strength=1.0,
        ),
    }
    # noise_latent_state is not a conditioning item but is the OTHER function that
    # rebuilds a LatentState wholesale (stage 2 / non-legacy_scalar_blend callsites),
    # so it can drop the fields exactly the same way.
    paths = {name: (lambda i=item: i.apply(base, (F, H, W))) for name, item in others.items()}
    paths["noise_latent_state"] = lambda: noise_latent_state(base, sigma=0.5, seed=3)
    for name, run in paths.items():
        try:
            after = run()
        except Exception as exc:
            fails.append(f"{name}: raised after slots — {type(exc).__name__}: {exc}")
            continue
        if after.generated_keyframe_layout is None:
            fails.append(f"{name}: DROPPED generated_keyframe_layout")
        if after.keyframes_mask is None:
            fails.append(f"{name}: DROPPED keyframes_mask")
        elif after.keyframes_mask.shape[1] != after.latent.shape[1]:
            # The invariant is mask length == SEQUENCE length. An earlier version of this
            # gate only checked it when the token count was unchanged, which let the
            # append-without-extending bug through until a real DFR run hit it.
            fails.append(
                f"{name}: keyframes_mask has {after.keyframes_mask.shape[1]} tokens but the "
                f"sequence has {after.latent.shape[1]}"
            )
    print(f"  [3/4] propagation          {len(paths)} state-rebuilding paths   "
          f"{'PASS' if len(fails) == n0 else 'FAIL'}")

    # The layout must still point at the slots after another item appends tokens.
    n0 = len(fails)
    appended = others["VideoConditionByKeyframeIndex"].apply(base, (F, H, W))
    lay = appended.generated_keyframe_layout
    if lay is not None:
        s, e = lay.slot_span(0)
        if float(mx.abs(appended.latent[:, s:e, :] - base.latent[:, N : N + TPF, :]).max()) > 0:
            fails.append("layout no longer points at the slot tokens after a later append")
    print(f"  [4/4] layout still locates {'PASS' if len(fails) == n0 else 'FAIL'}")

    if fails:
        print(f"\nFAIL — {len(fails)} problem(s):")
        for f in fails:
            print("  " + f)
        return 1
    print("\nPASS — slots survive every conditioning type; layout stays valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
