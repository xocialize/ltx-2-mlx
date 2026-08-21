"""Claim C6 — "duration predictor: predicts a clip's length from the prompt".

Objective and cheap: the head is 1.9M params and consumes the connector
outputs, so no latent work is needed.

## Pre-registered expectations (declared before running)

E1 ORDERING     mean(short prompts) < mean(medium) < mean(long).
E2 DETERMINISM  identical prompt -> identical seconds (the head has no sampling).
E3 CLAMP        every snapped frame count lands in [1s, 20s] on the 8k+1 grid.
E4 EXPLICIT     prompts naming a duration ("a 3 second clip" vs "a 15 second
                clip") should order the same way. This one is a genuine test of
                whether the head reads duration language or only scene pacing —
                a null result here is informative, not a failure.

Usage:
    uv run --no-sync python scripts/measure_c6_duration.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

MODEL = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
FPS = 24.0

BATTERY = {
    "short": [
        "a quick glance over the shoulder",
        "a single blink of an eye",
        "a snap of the fingers",
        "a camera flash going off",
    ],
    "medium": [
        "a person walks across a room and sits down",
        "a car drives past the camera on a city street",
        "a barista pours milk into a coffee cup",
        "a dog catches a ball in mid air",
    ],
    "long": [
        "a slow steady pan across a vast valley at sunset, unhurried",
        "a long continuous shot of ocean waves rolling in over many minutes",
        "a time lapse of clouds drifting slowly across a mountain range",
        "an extended tracking shot following a train through the countryside",
    ],
    "explicit_short": ["a 3 second clip of a cat sitting on a windowsill"],
    "explicit_long": ["a 15 second clip of a cat sitting on a windowsill"],
}


def main() -> None:
    from ltx_core_mlx.duration_head import DurationHead, seconds_to_num_frames
    from ltx_core_mlx.utils.weights import load_split_safetensors
    from ltx_pipelines_mlx.utils.blocks import PromptEncoder

    head = DurationHead()
    head.load_weights(list(load_split_safetensors(
        MODEL / "duration_head.safetensors", prefix="duration_head.").items()), strict=True)

    enc = PromptEncoder(MODEL)
    results: dict[str, list[tuple[str, float, int]]] = {}
    try:
        for group, prompts in BATTERY.items():
            rows = []
            for p in prompts:
                v, a = enc.encode(p)
                mx.eval(v, a)
                secs = float(np.array(head(v, a).astype(mx.float32))[0])
                rows.append((p, secs, seconds_to_num_frames(secs, FPS)))
            results[group] = rows
        # E2: determinism — re-encode one prompt
        p0 = BATTERY["medium"][0]
        v, a = enc.encode(p0)
        mx.eval(v, a)
        repeat = float(np.array(head(v, a).astype(mx.float32))[0])
    finally:
        enc.free()

    print(f"\n  {'group':16s} {'seconds':>8s} {'frames':>7s}  prompt")
    print(f"  {'-'*16} {'-'*8} {'-'*7}  {'-'*46}")
    means = {}
    for group, rows in results.items():
        for p, s, f in rows:
            print(f"  {group:16s} {s:8.2f} {f:7d}  {p[:46]}")
        means[group] = float(np.mean([r[1] for r in rows]))

    print(f"\n  means: " + "  ".join(f"{g} {m:.2f}s" for g, m in means.items()))
    e1 = means["short"] < means["medium"] < means["long"]
    e2 = abs(repeat - results["medium"][0][1]) < 1e-6
    all_f = [f for rows in results.values() for _, _, f in rows]
    e3 = all((f - 1) % 8 == 0 and 24 <= f <= 481 for f in all_f)
    e4 = means["explicit_short"] < means["explicit_long"]
    print(f"\n  E1 ordering short<medium<long : {'PASS' if e1 else 'FAIL'}")
    print(f"  E2 determinism                : {'PASS' if e2 else 'FAIL'} "
          f"(repeat {repeat:.6f} vs {results['medium'][0][1]:.6f})")
    print(f"  E3 clamp + 8k+1 grid          : {'PASS' if e3 else 'FAIL'} "
          f"(range {min(all_f)}-{max(all_f)} frames = {min(all_f)/FPS:.1f}-{max(all_f)/FPS:.1f}s)")
    print(f"  E4 explicit duration language : {'responds' if e4 else 'NO RESPONSE'} "
          f"({means['explicit_short']:.2f}s vs {means['explicit_long']:.2f}s)")
    fixed = 121
    waste = float(np.mean([abs(f - fixed) for f in all_f])) / FPS
    print(f"\n  vs the fixed {fixed}-frame default: mean |delta| {waste:.2f}s per clip")


if __name__ == "__main__":
    main()
