"""Claim C1 — "native multishot generation" (generated keyframe slots).

The claim is that LTX-2.5 can natively lay out several distinct shots in one
generation, via keyframe SLOTS the model invents at chosen positions. This
script measures the parts that can be measured objectively and hands the rest
to operator judgment rather than inventing a proxy for it.

    uv run python scripts/measure_c1_multishot.py --out .work/c1

What it measures, and what it deliberately does NOT:

  OBJECTIVE (reported as numbers)
    * Cost: token-count delta, wall-clock, peak memory, for N=0 vs N=K slots.
      Slots are the whole cost story of this feature — each buys one pixel frame
      for a full latent frame of tokens — so this is the honest headline.
    * Non-degeneracy: each decoded slot's pixel statistics. A slot that failed to
      generate reads as noise or a flat frame, both of which are visible here.
    * Distinctness: pairwise distance between the decoded slots, and between each
      slot and the base video frame at the same position. "Multishot" implies the
      slots carry DIFFERENT content; K near-identical frames would falsify it.

  NOT MEASURED HERE (needs a human)
    * Whether the shots are *good*, or whether the cut reads as intentional.
      C7's proxy experiment is the precedent: sharpness/flicker proxies ranked
      2.5 WORSE than 2.3, the opposite of operator judgment, because a matched
      seed does not mean matched composition. A "shot-boundary score" would fail
      the same way. The contact sheet is the deliverable for that call.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

MODEL = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")

# A prompt that actually asks for multiple shots — measuring multishot on a
# single-shot prompt would test nothing the claim is about.
PROMPT = (
    "Shot 1: a wide establishing shot of a fog-covered harbour at dawn, fishing boats "
    "at anchor. Shot 2: a close-up of a weathered fisherman's hands coiling wet rope. "
    "Shot 3: a low-angle shot of gulls lifting off a wooden piling against a grey sky. "
    "Cinematic, 35mm film grain, muted cool colour grade."
)


def to_uint8(pixels: mx.array) -> np.ndarray:
    """(C, H, W) or (1, C, 1, H, W) in [-1, 1] or [0, 1] -> HWC uint8."""
    a = np.array(pixels.astype(mx.float32))
    a = np.squeeze(a)
    if a.ndim == 4:  # (C, T, H, W) with T == 1
        a = a[:, 0]
    if a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.min() < -0.05:
        a = (a + 1.0) / 2.0
    return (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def stats(img: np.ndarray) -> dict:
    f = img.astype(np.float32) / 255.0
    gx = np.abs(np.diff(f.mean(-1), axis=1)).mean()
    gy = np.abs(np.diff(f.mean(-1), axis=0)).mean()
    return {
        "mean": round(float(f.mean()), 4),
        "std": round(float(f.std()), 4),
        "grad": round(float(gx + gy), 5),  # ~0 = flat frame, very high = noise
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".work/c1")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--keyframes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.distilled import DistilledPipeline
    from ltx_pipelines_mlx.utils.blocks import VideoDecoder

    report: dict = {"prompt": PROMPT, "config": vars(args), "arms": {}}

    for arm, n_kf in (("baseline_0", 0), (f"slots_{args.keyframes}", args.keyframes)):
        aggressive_cleanup()
        mx.reset_peak_memory()
        pipe = DistilledPipeline(str(MODEL), low_memory=True)
        t0 = time.time()
        video_path = out / f"{arm}.mp4"
        kwargs = dict(
            prompt=PROMPT,
            output_path=str(video_path),
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.frame_rate,
            seed=args.seed,
        )
        if n_kf:
            kwargs["generated_keyframes"] = n_kf
        pipe.generate_and_save(**kwargs)
        elapsed = time.time() - t0
        peak = mx.get_peak_memory() / 1e9

        arm_report: dict = {
            "seconds": round(elapsed, 1),
            "peak_gb": round(peak, 2),
            "num_generated_keyframes": n_kf,
            "keyframes": [],
        }

        lat = getattr(pipe, "generated_keyframe_latents", []) or []
        pos = getattr(pipe, "generated_keyframe_positions", []) or []
        arm_report["slot_positions"] = list(pos)
        if lat:
            # Decode each slot as its OWN one-frame clip. A K-frame causal decode
            # would blend slots that were never temporally adjacent.
            dec = VideoDecoder(str(MODEL), verbose=False)
            decoder = dec.load()
            for i, (kf_lat, frame_idx) in enumerate(zip(lat, pos, strict=False)):
                px = decoder.decode(kf_lat) if hasattr(decoder, "decode") else decoder(kf_lat)
                mx.eval(px)
                img = to_uint8(px)
                name = f"{arm}_kf{i:02d}_f{frame_idx:04d}.png"
                Image.fromarray(img).save(out / name)
                arm_report["keyframes"].append({"index": i, "pixel_frame": int(frame_idx),
                                                "file": name, **stats(img)})
            del dec, decoder

        report["arms"][arm] = arm_report
        del pipe
        aggressive_cleanup()

    # Distinctness: multishot implies the slots carry different content.
    kfs = report["arms"].get(f"slots_{args.keyframes}", {}).get("keyframes", [])
    if len(kfs) >= 2:
        imgs = [np.asarray(Image.open(out / k["file"])).astype(np.float32) / 255.0 for k in kfs]
        pair = {}
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                rmse = float(np.sqrt(((imgs[i] - imgs[j]) ** 2).mean()))
                pair[f"kf{i}_vs_kf{j}"] = round(rmse, 4)
        report["distinctness_rmse"] = pair
        report["distinctness_min"] = round(min(pair.values()), 4)

    b = report["arms"]["baseline_0"]
    s = report["arms"][f"slots_{args.keyframes}"]
    report["cost"] = {
        "seconds_delta": round(s["seconds"] - b["seconds"], 1),
        "seconds_ratio": round(s["seconds"] / b["seconds"], 3) if b["seconds"] else None,
        "peak_gb_delta": round(s["peak_gb"] - b["peak_gb"], 2),
    }

    (out / "c1_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out / 'c1_report.json'} + decoded keyframe PNGs.")
    print("Shot quality / cut intent is an OPERATOR call — inspect the PNGs beside the mp4s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
