"""Dump upstream generated-keyframe slot positions (run under the reference venv).

    .work/ltx25-ref/.venv/bin/python scripts/dump_ltx25_slot_positions_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ltx_pipelines.utils.helpers import evenly_spaced_keyframe_positions, resolve_generated_keyframes

OUT = Path(__file__).resolve().parents[1] / ".work" / "ltx25-ref" / "goldens" / "slot_positions.json"


def main() -> None:
    out: dict[str, object] = {"evenly_spaced": {}, "explicit": {}}
    n_err = 0
    for num_frames in range(2, 500):
        for k in range(0, 9):
            key = f"{k}:{num_frames}"
            try:
                out["evenly_spaced"][key] = evenly_spaced_keyframe_positions(k, num_frames)
            except ValueError:
                out["evenly_spaced"][key] = None
                n_err += 1
    for seq in ([], [5], [96, 24, 24, 48], [0], [120]):
        out["explicit"][json.dumps(seq)] = resolve_generated_keyframes(seq, 121)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}  ({len(out['evenly_spaced'])} cases, {n_err} raising)")


if __name__ == "__main__":
    main()
