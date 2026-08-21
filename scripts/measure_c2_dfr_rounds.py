"""Claim C2, temporal half — is DFR's densification cheaper than generating natively?

The efficiency claim only means something if the arms produce the SAME NUMBER OF
FRAMES. Each round maps N -> 2(N-1)+1, so to land on 97 frames:

    distilled native   97 frames generated directly
    DFR rounds=0       97 frames generated directly (spatial detailing only)
    DFR rounds=1       49 requested -> 97 out
    DFR rounds=2       25 requested -> 97 out

Every arm yields 97 frames at the same resolution, so wall-clock and peak are
directly comparable. Playback fps rises with rounds (24 / 48 / 96), which is the
point of the feature — the same motion, more temporal samples.

    uv run python scripts/measure_c2_dfr_rounds.py --out .work/c2r
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
TARGET_FRAMES = 97


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".work/c2r")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.dfr import DFRPipeline
    from ltx_pipelines_mlx.distilled import DistilledPipeline

    # (label, rounds, requested frames) — requested chosen so every arm emits 97.
    arms = [("distilled_native", None, TARGET_FRAMES),
            ("dfr_rounds0", 0, TARGET_FRAMES),
            ("dfr_rounds1", 1, 49),
            ("dfr_rounds2", 2, 25)]
    report: dict = {"prompt": PROMPT, "target_frames": TARGET_FRAMES, "config": vars(args), "arms": {}}

    for label, rounds, requested in arms:
        aggressive_cleanup()
        mx.reset_peak_memory()
        t0 = time.time()
        path = out / f"{label}.mp4"
        if rounds is None:
            pipe = DistilledPipeline(str(MODEL), low_memory=True)
            pipe.generate_and_save(
                prompt=PROMPT, output_path=str(path), height=args.height, width=args.width,
                num_frames=requested, frame_rate=args.frame_rate, seed=args.seed,
            )
            playback_fps = args.frame_rate
        else:
            pipe = DFRPipeline(str(MODEL), low_memory=True)
            pipe.generate_and_save(
                prompt=PROMPT, output_path=str(path), height=args.height, width=args.width,
                num_frames=requested, frame_rate=args.frame_rate, seed=args.seed,
                temporal_upsample_rounds=rounds,
            )
            playback_fps = args.frame_rate * 2**rounds
        report["arms"][label] = {
            "seconds": round(time.time() - t0, 1),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "requested_frames": requested,
            "rounds": rounds,
            "playback_fps": playback_fps,
            "output": path.name,
        }
        del pipe
        aggressive_cleanup()

    # Verify every arm really produced TARGET_FRAMES — the comparison is void otherwise.
    import subprocess
    for a in report["arms"].values():
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out / a["output"])],
            capture_output=True, text=True)
        a["actual_frames"] = int(r.stdout.strip() or 0)
        a["frames_ok"] = a["actual_frames"] == TARGET_FRAMES

    base = report["arms"]["distilled_native"]["seconds"]
    report["comparison"] = {
        label: {"vs_native_seconds_ratio": round(a["seconds"] / base, 3),
                "peak_gb_delta": round(a["peak_gb"] - report["arms"]["distilled_native"]["peak_gb"], 2)}
        for label, a in report["arms"].items()
    }
    report["all_arms_emitted_target_frames"] = all(a["frames_ok"] for a in report["arms"].values())
    (out / "c2_rounds_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
