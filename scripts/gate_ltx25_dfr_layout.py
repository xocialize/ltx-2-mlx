"""Gate: MLX DFR layout == upstream DFR layout, BIT-EXACTLY.

This module is integer geometry, so there is no numerics excuse and no
tolerance: every segment length, canvas, tile range, and error message must
match upstream exactly. Golden from ``dump_ltx25_dfr_layout_golden.py``.

    uv run python scripts/gate_ltx25_dfr_layout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

from ltx_pipelines_mlx.dfr_layout import (
    choose_segment_length,
    pixel_to_latent_index,
    remap_positions_to_local,
    resolve_canvas,
    stitch_tile_latents,
    tile_ranges,
)

GOLDEN = Path(__file__).resolve().parents[1] / ".work" / "ltx25-ref" / "goldens" / "dfr_layout.json"


def main() -> int:
    if not GOLDEN.exists():
        print(f"FAIL: golden missing at {GOLDEN}\n  run: .work/ltx25-ref/.venv/bin/python "
              f"scripts/dump_ltx25_dfr_layout_golden.py")
        return 1
    g = json.loads(GOLDEN.read_text())
    fails: list[str] = []

    for content, want in g["segment_lengths"].items():
        got = choose_segment_length(int(content))
        if got != want:
            fails.append(f"choose_segment_length({content}): {got} != {want}")
    print(f"  [1/5] segment lengths      {len(g['segment_lengths'])} content sizes  "
          f"{'PASS' if not fails else 'FAIL'}")

    n0 = len(fails)
    for n, want in g["canvases"].items():
        padded, segment, positions = resolve_canvas(int(n))
        if [padded, segment, positions] != [want["padded"], want["segment"], want["positions"]]:
            fails.append(f"resolve_canvas({n}): ({padded},{segment},{positions}) != "
                         f"({want['padded']},{want['segment']},{want['positions']})")
    print(f"  [2/5] canvases             {len(g['canvases'])} frame counts   "
          f"{'PASS' if len(fails) == n0 else 'FAIL'}")

    n0 = len(fails)
    n_ok = n_err = 0
    for key, want in g["tiles"].items():
        n, tiles = (int(x) for x in key.split(":"))
        padded, _segment, positions = resolve_canvas(n)
        try:
            got = [list(r) for r in tile_ranges(positions, padded, tiles)]
        except ValueError as exc:
            if not isinstance(want, dict) or want.get("error") != str(exc):
                fails.append(f"tile_ranges({key}): raised {exc!r}, golden {want!r}")
            else:
                n_err += 1
            continue
        if isinstance(want, dict):
            fails.append(f"tile_ranges({key}): returned {got}, golden expected error {want['error']!r}")
            continue
        # NamedTuple -> list round-trips tuples as lists; compare structurally.
        if json.dumps(got) != json.dumps(want):
            fails.append(f"tile_ranges({key}):\n    got  {got}\n    want {want}")
        else:
            n_ok += 1
    print(f"  [3/5] tile partitions      {n_ok} exact, {n_err} matched-raise   "
          f"{'PASS' if len(fails) == n0 else 'FAIL'}")

    n0 = len(fails)
    for p, want in g["misc"]["pixel_to_latent"].items():
        got = pixel_to_latent_index(int(p))
        if got != want:
            fails.append(f"pixel_to_latent_index({p}): {got} != {want}")
    if remap_positions_to_local([24, 48, 72, 96], 24) != g["misc"]["remap"]:
        fails.append("remap_positions_to_local mismatch")
    print(f"  [4/5] index helpers        {'PASS' if len(fails) == n0 else 'FAIL'}")

    # stitch: the one array op. Tag each tile's frames uniquely so a wrong prefix
    # drop, a wrong order, or a silently-overlapping seam is visible in the values.
    n0 = len(fails)
    padded, _s, positions = resolve_canvas(97)
    ranges = tile_ranges(positions, padded, 4)
    tiles_lat = []
    for r in ranges:
        t = r.latent_end_exclusive - r.latent_start
        tiles_lat.append(mx.broadcast_to(
            (mx.arange(t) + r.latent_start).reshape(1, 1, t, 1, 1).astype(mx.float32), (1, 2, t, 3, 3)))
    stitched = stitch_tile_latents(tiles_lat, ranges)
    want_t = ranges[-1].latent_end_exclusive - ranges[0].drop_latent_prefix - ranges[0].latent_start
    seq = [float(x) for x in stitched[0, 0, :, 0, 0].tolist()]
    expect = list(range(ranges[0].drop_latent_prefix, ranges[-1].latent_end_exclusive))
    if stitched.shape[2] != want_t:
        fails.append(f"stitch T={stitched.shape[2]} != {want_t}")
    if seq != [float(x) for x in expect]:
        fails.append(f"stitch is not a gapless, non-overlapping latent run:\n    {seq}\n    {expect}")
    print(f"  [5/5] stitch (gapless run) T={stitched.shape[2]}, latents "
          f"{expect[0]}..{expect[-1]}   {'PASS' if len(fails) == n0 else 'FAIL'}")

    if fails:
        print(f"\nFAIL — {len(fails)} mismatch(es):")
        for f in fails[:12]:
            print("  " + f)
        return 1
    print("\nPASS — MLX DFR layout is bit-identical to upstream on every probe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
