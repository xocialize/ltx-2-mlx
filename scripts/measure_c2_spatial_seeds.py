"""C2 spatial detailing — same A/B across several seeds, to separate capability from roll.

One pair cannot tell "DFR renders better" from "DFR rolled a different scene": the
slot tokens divert the trajectory from step one, so the arms never share a
composition. If the operator's read (DFR looked better — e.g. lit windows on the
lighthouse where plain distilled had none) is a CAPABILITY difference it should
recur across seeds; if it was one lucky roll it will wash out.

    uv run python scripts/measure_c2_spatial_seeds.py --seeds 11 22 33
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
    ap.add_argument("--out", default=".work/c2seeds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.dfr import DFRPipeline
    from ltx_pipelines_mlx.distilled import DistilledPipeline

    report: dict = {"prompt": PROMPT, "config": vars(args), "pairs": {}}
    for seed in args.seeds:
        row: dict = {}
        for label, cls in (("distilled", DistilledPipeline), ("dfr", DFRPipeline)):
            aggressive_cleanup()
            mx.reset_peak_memory()
            pipe = cls(str(MODEL), low_memory=True)
            kw = dict(prompt=PROMPT, output_path=str(out / f"seed{seed}_{label}.mp4"),
                      height=args.height, width=args.width, num_frames=args.frames,
                      frame_rate=args.frame_rate, seed=seed)
            if label == "dfr":
                kw["temporal_upsample_rounds"] = 0
            t0 = time.time()
            pipe.generate_and_save(**kw)
            row[label] = {"seconds": round(time.time() - t0, 1),
                          "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
                          "output": f"seed{seed}_{label}.mp4"}
            del pipe
            aggressive_cleanup()
        row["ratio"] = round(row["dfr"]["seconds"] / row["distilled"]["seconds"], 3)
        report["pairs"][str(seed)] = row
        print(f"seed {seed}: distilled {row['distilled']['seconds']}s  "
              f"dfr {row['dfr']['seconds']}s  ({row['ratio']}x)")

    ratios = [r["ratio"] for r in report["pairs"].values()]
    report["ratio_mean"] = round(sum(ratios) / len(ratios), 3)
    report["ratio_range"] = [min(ratios), max(ratios)]
    (out / "c2_spatial_seeds.json").write_text(json.dumps(report, indent=2))
    print(f"\nratio mean {report['ratio_mean']}x, range {report['ratio_range']}")
    print("Quality across seeds is an OPERATOR call — compare seedN_distilled vs seedN_dfr.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
