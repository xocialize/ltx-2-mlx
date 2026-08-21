"""Build the C7 side-by-side artifact: LTX-2.3 vs LTX-2.5 distilled.

Matched seed, geometry, sampler settings and prompt; the only variable is the
checkpoint generation. Emits one PNG per prompt (2.3 left, 2.5 right) plus
no-reference proxies.

⚠️ The proxies are DESCRIPTIVE, not a verdict. C7 ("substantially improved
visual quality, prompt adherence, motion consistency") is a perceptual and
semantic claim; there is no ground truth for generated content, and sharpness
statistics cannot distinguish detail from grain. The PNGs are the deliverable —
they are for the operator's eye. Prompt adherence belongs to the C4 VLM-judge
battery, not here.

Usage:
    uv run --no-sync python scripts/make_c7_comparison.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

WORK = Path(".work/c7")
H, W, F = 512, 768, 9


def frames(p: Path) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H, W, 3).astype(np.float32) / 255.0


def hf(a: np.ndarray) -> float:
    g = a.mean(-1)
    lap = (-4 * g[:, 1:-1, 1:-1] + g[:, :-2, 1:-1] + g[:, 2:, 1:-1]
           + g[:, 1:-1, :-2] + g[:, 1:-1, 2:])
    return float(lap.var()) * 255.0**2


def flicker(a: np.ndarray) -> float:
    """Mean abs frame-to-frame delta — a crude temporal-consistency proxy."""
    return float(np.abs(np.diff(a, axis=0)).mean()) * 255.0


def main() -> None:
    print(f"\n  {'prompt':8s} {'model':6s} {'HF energy':>10s} {'flicker':>9s}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*9}")
    for i in (1, 2, 3):
        a_p, b_p = WORK / f"p{i}_23.mp4", WORK / f"p{i}_25.mp4"
        if not (a_p.exists() and b_p.exists()):
            print(f"  p{i}: missing ({a_p.exists()=}, {b_p.exists()=})")
            continue
        a, b = frames(a_p), frames(b_p)
        n = min(len(a), len(b))
        for tag, arr in (("2.3", a[:n]), ("2.5", b[:n])):
            print(f"  p{i:<7d} {tag:6s} {hf(arr):10.2f} {flicker(arr):9.3f}")
        mid = n // 2
        pair = np.concatenate([a[mid], b[mid]], axis=1)
        img = Image.fromarray((np.clip(pair, 0, 1) * 255).astype(np.uint8))
        out = WORK / f"c7_p{i}_2.3-left_2.5-right.png"
        img.save(out)
        print(f"           -> {out.name}")
    print("\n  PNGs are the deliverable: 2.3 LEFT, 2.5 RIGHT, identical seed/prompt/settings.")


if __name__ == "__main__":
    main()
