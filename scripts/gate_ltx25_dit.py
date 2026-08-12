"""LTX-2.5 DiT-forward parity gate vs the torch reference golden.

Goldens from ``scripts/dump_ltx25_dit_golden.py`` (upstream ``LTXModel`` built by
``LTXModelConfigurator.from_metadata`` on the official comfy checkpoint, MPS),
dumped at BOTH bf16 and fp32.

## Why the gate compares against fp32

A 48-block bf16 DiT has a precision noise floor far above the usual 0.999
per-op threshold, and it differs per stream (video is 4096-dim, audio 2048-dim).
Measured on the reference alone (torch-bf16 vs torch-fp32, same code, same
weights, same inputs): **video 0.996859, audio 0.999733**. Gating our bf16
output against the *bf16* golden therefore measures torch's own rounding, not
our error.

So the gate does two things:

  [B1] **accuracy** — ours vs the **fp32** reference (ground truth),
       cosine >= 0.999.
  [B2] **self-control** — our accuracy against fp32 must be at least as good
       as the *reference's own bf16 path* achieves against the same fp32
       truth. A relative check that adapts to the stream's precision
       characteristics and cannot be passed by loosening a constant.
       (mlx-porting #44: if the reference disagrees with itself more than we
       disagree with the reference, it is precision, not a port bug.)
       Note: ours-vs-torch-bf16 is reported as a diagnostic but is NOT an
       assertion — the more accurate our path is, the further it sits from
       torch's particular bf16 rounding, so that number floors out near the
       reference's own bf16-vs-fp32 figure by construction.

Plus an independent coordinate check, so a failure localizes:

  [A] **position builder** — our ``compute_video_positions`` /
      ``compute_audio_positions`` vs the midpoints of upstream's
      ``(B, n_dims, T, 2)`` bounds (latent bounds -> pixel coords ->
      causal fix -> /fps). The forward check [B] feeds the *reference's*
      positions, so [A] and [B] are independent.

Usage:
    uv run --no-sync python scripts/gate_ltx25_dit.py [--geometry small|prod]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

GOLDENS = Path(".work/ltx25-ref/goldens")
MODEL_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
THRESHOLD = 0.999


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="small")
    args = ap.parse_args()
    tag = args.geometry

    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
    from ltx_core_mlx.utils.weights import load_split_safetensors

    meta = json.loads((GOLDENS / f"dit_{tag}_meta.json").read_text())
    F, H, W, NA = meta["geometry"]
    fps, sigma = meta["fps"], meta["sigma"]

    def load(name: str, fp32: bool = False) -> np.ndarray:
        stem = f"dit_{tag}_fp32_{name}" if fp32 else f"dit_{tag}_{name}"
        return np.load(GOLDENS / f"{stem}.npy")

    inputs = {k: load(k) for k in (
        "video_latent", "audio_latent", "video_positions_mid",
        "audio_positions_mid", "keyframes_mask",
    )}
    v_ctx = np.load(GOLDENS / "connector_video_encoding_fp32.npy")
    a_ctx = np.load(GOLDENS / "connector_audio_encoding_fp32.npy")

    # ---- [A] position builder (independent of the forward check) ----
    from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions

    def pos_check(label: str, ours: np.ndarray, ref: np.ndarray) -> float:
        if ours.ndim == 2:
            ours = ours[None]
        if ours.shape != ref.shape:
            print(f"[A] {label}: SHAPE MISMATCH {ours.shape} vs {ref.shape}")
            return float("inf")
        d = float(np.abs(ours - ref).max())
        print(f"[A] {label} vs reference midpoints: maxAbs {d:.3e} "
              f"{'PASS' if d < 1e-5 else 'FAIL'}")
        return d

    dv = pos_check("video positions", np.array(compute_video_positions(F, H, W, fps)),
                   inputs["video_positions_mid"])
    da = pos_check("audio positions", np.array(compute_audio_positions(NA)),
                   inputs["audio_positions_mid"])

    # ---- run our forward on the reference's inputs ----
    cfg = LTXModelConfig.from_checkpoint_dir(MODEL_DIR)
    model = LTXModel(cfg)
    w = load_split_safetensors(MODEL_DIR / "transformer-distilled.safetensors", prefix="transformer.")
    model.load_weights(list(w.items()), strict=True)
    del w
    mx.clear_cache()

    v_out, a_out = model(
        mx.array(inputs["video_latent"]),
        mx.array(inputs["audio_latent"]),
        mx.array([sigma]),
        video_text_embeds=mx.array(v_ctx),
        audio_text_embeds=mx.array(a_ctx),
        video_positions=mx.array(inputs["video_positions_mid"]),
        audio_positions=mx.array(inputs["audio_positions_mid"]),
        keyframes_mask=mx.array(inputs["keyframes_mask"]),
    )
    mx.eval(v_out, a_out)
    ours = {"video": np.array(v_out.astype(mx.float32)), "audio": np.array(a_out.astype(mx.float32))}

    print(f"\n  geometry {tag}: video {F}x{H}x{W} = {F*H*W} tokens, audio {NA} tokens")
    print(f"  {'stream':7s} {'ours/fp32':>11s} {'ref-bf16/fp32':>14s}   {'ours/bf16 (diag)':>16s}")
    ok = True
    for stream in ("video", "audio"):
        t_bf16 = load(f"{stream}_out")
        t_fp32 = load(f"{stream}_out", fp32=True)
        c_acc = cosine(ours[stream], t_fp32)
        c_bf = cosine(ours[stream], t_bf16)
        floor = cosine(t_bf16, t_fp32)
        print(f"  {stream:7s} {c_acc:11.6f} {floor:14.6f}   {c_bf:16.6f}")
        if c_acc < THRESHOLD:
            print(f"    [B1] FAIL: accuracy vs fp32 below {THRESHOLD}")
            ok = False
        if c_acc < floor:
            print("    [B2] FAIL: less accurate than the reference's own bf16 path")
            ok = False

    assert max(dv, da) < 1e-5, "position builder diverges from the reference"
    assert ok, "DiT forward parity failed"
    print("\nDIT GATE: PASS (positions exact; accuracy >= 0.999 vs fp32 "
          "and >= the reference's own bf16 accuracy)")


if __name__ == "__main__":
    main()
