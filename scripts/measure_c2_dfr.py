"""Claim C2 — "diffusion fidelity rendering" (DFR).

DFR vs the plain distilled two-stage at matched prompt / seed / resolution /
frames. Both run the same distilled weights and the same 8+3 sigma schedule; the
difference is DFR's keyframe slots in both stages plus the seeded re-append of
the upsampled slots in stage 2.

    uv run python scripts/measure_c2_dfr.py --out .work/c2

Reported objectively: wall-clock, peak memory, token counts, and the decoded
output of both arms. Whether DFR looks *better* is an operator call — the C7
proxy trap (sharpness/flicker ranked 2.5 worse, opposite of judgement) applies
with full force here, since the two arms do not share a composition either.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

MODEL = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
PROMPT = (
    "A lone lighthouse on a rocky headland during a storm, waves exploding against the "
    "rocks, the beam sweeping through sheets of rain. Cinematic, high contrast, 35mm."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".work/c2")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.dfr import DFRPipeline
    from ltx_pipelines_mlx.distilled import DistilledPipeline

    report: dict = {"prompt": PROMPT, "config": vars(args), "arms": {}}

    # --- Arm A: plain distilled two-stage -------------------------------------
    aggressive_cleanup()
    mx.reset_peak_memory()
    pipe = DistilledPipeline(str(MODEL), low_memory=True)
    t0 = time.time()
    pipe.generate_and_save(
        prompt=PROMPT, output_path=str(out / "distilled.mp4"),
        height=args.height, width=args.width, num_frames=args.frames,
        frame_rate=args.frame_rate, seed=args.seed,
    )
    report["arms"]["distilled"] = {
        "seconds": round(time.time() - t0, 1),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "output": "distilled.mp4",
    }
    del pipe
    aggressive_cleanup()

    # --- Arm B: DFR -----------------------------------------------------------
    mx.reset_peak_memory()
    pipe = DFRPipeline(str(MODEL), low_memory=True)
    t0 = time.time()
    # Use generate_and_save, NOT generate_dfr + a hand-rolled decode: the former frees
    # the DiT before the decoders load. Skipping that free made a first run of this
    # script report +31 GB for DFR, all of which was the transformer still resident.
    pipe.generate_and_save(
        prompt=PROMPT, output_path=str(out / "dfr.mp4"),
        height=args.height, width=args.width, num_frames=args.frames,
        frame_rate=args.frame_rate, seed=args.seed, temporal_upsample_rounds=0,
    )
    canvas = dict(pipe.dfr_canvas)
    n_slots = len(pipe.generated_keyframe_positions)
    report["arms"]["dfr"] = {
        "seconds_incl_decode": round(time.time() - t0, 1),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "canvas": canvas,
        "num_slots": n_slots,
        "output": "dfr.mp4",
    }
    del pipe
    aggressive_cleanup()

    a, b = report["arms"]["distilled"], report["arms"]["dfr"]
    report["comparison"] = {
        "dfr_vs_distilled_seconds_ratio": round(b["seconds_incl_decode"] / a["seconds"], 3),
        "note": "both arms are generate_and_save, so both include decode and both free the DiT first",
        "peak_gb_delta": round(b["peak_gb"] - a["peak_gb"], 2),
        "slots_added": n_slots,
        "canvas_padding_frames": canvas["canvas_frames"] - canvas["requested_frames"],
    }
    (out / "c2_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nQuality is an OPERATOR call — compare distilled.mp4 vs dfr.mp4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
