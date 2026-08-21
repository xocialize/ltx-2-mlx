"""Gate: MLX slot-position resolution == upstream, BIT-EXACTLY (integers, no tolerance)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ltx_core_mlx.conditioning.types.keyframe_slots import (
    evenly_spaced_keyframe_positions,
    has_generated_keyframes,
    resolve_generated_keyframes,
)

GOLDEN = Path(__file__).resolve().parents[1] / ".work" / "ltx25-ref" / "goldens" / "slot_positions.json"


def main() -> int:
    if not GOLDEN.exists():
        print(f"FAIL: golden missing at {GOLDEN}")
        return 1
    g = json.loads(GOLDEN.read_text())
    fails = []
    n_ok = n_raise = 0
    for key, want in g["evenly_spaced"].items():
        k, num_frames = (int(x) for x in key.split(":"))
        try:
            got = evenly_spaced_keyframe_positions(k, num_frames)
        except ValueError:
            if want is not None:
                fails.append(f"evenly_spaced({k},{num_frames}) raised, golden {want}")
            else:
                n_raise += 1
            continue
        if want is None:
            fails.append(f"evenly_spaced({k},{num_frames}) = {got}, golden raised")
        elif got != want:
            fails.append(f"evenly_spaced({k},{num_frames}) = {got} != {want}")
        else:
            n_ok += 1
    print(f"  [1/3] evenly spaced   {n_ok} exact, {n_raise} matched-raise   "
          f"{'PASS' if not fails else 'FAIL'}")

    n0 = len(fails)
    for key, want in g["explicit"].items():
        got = resolve_generated_keyframes(json.loads(key), 121)
        if got != want:
            fails.append(f"resolve({key}) = {got} != {want}")
    print(f"  [2/3] explicit lists  {len(g['explicit'])} cases   {'PASS' if len(fails) == n0 else 'FAIL'}")

    n0 = len(fails)
    for arg, want in ((0, False), (3, True), ([], False), ([7], True)):
        if has_generated_keyframes(arg) != want:
            fails.append(f"has_generated_keyframes({arg!r}) != {want}")
    print(f"  [3/3] feature flag    {'PASS' if len(fails) == n0 else 'FAIL'}")

    if fails:
        print(f"\nFAIL — {len(fails)} mismatch(es):")
        for f in fails[:12]:
            print("  " + f)
        return 1
    print("\nPASS — slot positions bit-identical to upstream across all 4482 cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
