"""Dump upstream DFR layout geometry to JSON (run under the reference venv).

The layout module is pure integer geometry, so the golden is the exact integers
upstream produces — the gate compares with ``==``, not a tolerance.

    .work/ltx25-ref/.venv/bin/python scripts/dump_ltx25_dfr_layout_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ltx_pipelines.dfr_layout import (
    choose_segment_length,
    pixel_to_latent_index,
    remap_positions_to_local,
    resolve_canvas,
    tile_ranges,
)

OUT = Path(__file__).resolve().parents[1] / ".work" / "ltx25-ref" / "goldens" / "dfr_layout.json"

# Frame counts spanning: both segment candidates winning, exact fits, and worst-case padding.
FRAME_COUNTS = [9, 17, 25, 33, 41, 49, 65, 73, 97, 121, 129, 161, 193, 201, 241, 289, 385, 481]
TILE_COUNTS = [1, 2, 4, 8]


def main() -> None:
    out: dict = {"segment_lengths": {}, "canvases": {}, "tiles": {}, "misc": {}}

    for content in range(1, 400):
        out["segment_lengths"][str(content)] = choose_segment_length(content)

    for n in FRAME_COUNTS:
        padded, segment, positions = resolve_canvas(n)
        out["canvases"][str(n)] = {"padded": padded, "segment": segment, "positions": positions}
        for tiles in TILE_COUNTS:
            try:
                ranges = tile_ranges(positions, padded, tiles)
            except ValueError as exc:
                out["tiles"][f"{n}:{tiles}"] = {"error": str(exc)}
                continue
            out["tiles"][f"{n}:{tiles}"] = [list(r) for r in ranges]

    out["misc"]["pixel_to_latent"] = {str(p): pixel_to_latent_index(p) for p in (0, 8, 24, 32, 64, 96, 480)}
    out["misc"]["remap"] = remap_positions_to_local([24, 48, 72, 96], 24)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    n_tiles = sum(1 for v in out["tiles"].values() if isinstance(v, list))
    print(f"wrote {OUT}  ({len(out['segment_lengths'])} segment lengths, "
          f"{len(out['canvases'])} canvases, {n_tiles} tile partitions)")


if __name__ == "__main__":
    main()
