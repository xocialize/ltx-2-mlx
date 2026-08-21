"""LTX-2.5 duration-head parity gate vs the torch reference.

Golden from ``scripts/dump_ltx25_duration_golden.py`` (upstream ``DurationHead``
in fp32 on the banked connector goldens). Checks all three input modes plus the
frame-grid snap that actually ships.

## Tolerance rationale (measured, not guessed)

Residual agreement is ~2-6e-4 relative on the predicted seconds, and it is
amplification of ordinary fp32 accumulation, not a math difference:

  * stage-by-stage, ``tokens`` (the input projections) agree to **1.7e-6**;
    ``pooled`` jumps to 1.6e-3, ``hidden`` 6.2e-4, ``logd`` 1.5e-4.
  * the jump is at the pooler because a single query softmaxes over 2048 keys
    and the pooled vector sits ~20x below the token scale (cancellation), so a
    1e-6 token difference becomes a small softmax-weight shift times a large
    value magnitude.
  * it is NOT the attention kernel: ``mx.fast.scaled_dot_product_attention``,
    an explicit fp32 softmax, and ``precise=True`` softmax all agree with each
    other to 1e-6 (1.565e-3 / 1.564e-3 / 1.564e-3 vs the reference).
  * it is NOT dtype: running our head with fp32-cast weights moves the error
    by <1e-4 (the stored weights are bf16 on both sides).

So the numeric tolerance is 1e-3, and the *deliverable* check is exact: the
predicted duration must snap to the SAME frame count as the reference. At
24 fps one grid step is 8/24 = 0.333 s, four orders of magnitude above the
residual, so the frame count is insensitive to it by construction.

Usage:
    uv run --no-sync python scripts/gate_ltx25_duration.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

GOLDENS = Path(".work/ltx25-ref/goldens")
MODEL_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
REL_TOL = 1e-3


def main() -> None:
    from ltx_core_mlx.duration_head import DurationHead, seconds_to_num_frames
    from ltx_core_mlx.utils.weights import load_split_safetensors

    head = DurationHead()
    w = load_split_safetensors(MODEL_DIR / "duration_head.safetensors", prefix="duration_head.")
    head.load_weights(list(w.items()), strict=True)
    print(f"  strict load OK ({len(w)} tensors)")

    v = mx.array(np.load(GOLDENS / "connector_video_encoding_fp32.npy"))
    a = mx.array(np.load(GOLDENS / "connector_audio_encoding_fp32.npy"))

    cases = [
        ("both", (v, a), "duration_both"),
        ("video-only", (v, None), "duration_video_only"),
        ("audio-only", (None, a), "duration_audio_only"),
    ]
    ok = True
    for label, (vt, at), key in cases:
        ours = float(np.array(head(vt, at))[0])
        ref = float(np.load(GOLDENS / f"{key}.npy")[0])
        rel = abs(ours - ref) / abs(ref)
        status = "PASS" if rel < REL_TOL else "FAIL"
        if rel >= REL_TOL:
            ok = False
        print(f"  {label:11s}: ours {ours:9.6f}s  ref {ref:9.6f}s  rel {rel:.2e}  {status}")

    # the deliverable: identical frame count after clamp + causal-grid snap
    for label, (vt, at), key in cases:
        ours_s = float(np.array(head(vt, at))[0])
        ref_s = float(np.load(GOLDENS / f"{key}.npy")[0])
        for fps in (24.0, 30.0):
            of, rf = seconds_to_num_frames(ours_s, fps), seconds_to_num_frames(ref_s, fps)
            if of != rf:
                ok = False
                print(f"  FRAME MISMATCH {label} @{fps}fps: ours {of} ref {rf}")
        print(f"  {label:11s}: snaps to {seconds_to_num_frames(ours_s, 24.0)} frames @24fps "
              f"(ref {seconds_to_num_frames(ref_s, 24.0)}) MATCH")
    both = float(np.array(head(v, a))[0])
    frames = seconds_to_num_frames(both, 24.0)
    assert (frames - 1) % 8 == 0, "snapped frame count off the causal grid"
    for s, want_grid in ((0.2, True), (99.0, True)):
        f = seconds_to_num_frames(s, 24.0)
        assert (f - 1) % 8 == 0 and 9 <= f <= 481, f"clamp/grid broken at {s}s -> {f}"
    print("  clamp bounds honored at 0.2s and 99s")

    print(f"DURATION GATE: {'PASS' if ok else 'FAIL'}")
    assert ok, "duration head parity failed"


if __name__ == "__main__":
    main()
