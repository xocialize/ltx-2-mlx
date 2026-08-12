"""LTX-2.5 DiffVAE component parity gate vs torch reference goldens.

Goldens from ``scripts/dump_ltx25_diffvae_golden.py`` (upstream
``DiffusionVideoDecoder`` in fp32 with the EAGER NA backend forced, loaded from
our converted weights with a hard 0-missing/0-unexpected assert).

Gates, in dependency order:
  [A] one deterministic NABlock  (det_stages.3.0, dim 512, kernel (3,5,5))
  [B] one LinearPixelShuffleUpsample (upsamples.3, stride (2,2,2), frame drop)
  [C] one stage-5 DiffusionNABlock  (diff_blocks.0, dim 256, kernel 11^3)
  [D] the full stage-5 stack        (8 blocks + norm_out + conv_out + unpatchify)

Threshold: cosine >= 0.9999. Absolute deltas land ~1e-3, which is Apple-GPU
fp32 matmul noise (documented in na3d.py and measured repeatedly on this
port), not a semantic gap.

Usage:
    uv run --no-sync python scripts/gate_ltx25_diffvae.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GOLDENS = Path(".work/ltx25-ref/goldens")
MODEL_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
THRESHOLD = 0.9999


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    from ltx_core_mlx.model.video_vae.diffusion_video_decoder import (
        DiffusionNABlock,
        LinearPixelShuffleUpsample,
        NABlock,
        patchify_hw,
        timestep_embedding,
        unpatchify_hw,
    )
    from ltx_core_mlx.utils.weights import load_split_safetensors

    w = {k: v.astype(mx.float32) for k, v in load_split_safetensors(
        MODEL_DIR / "vae_diffusion_decoder.safetensors",
        prefix="vae_diffusion_decoder.").items()}

    def sub(prefix: str) -> list:
        return [(k[len(prefix):], v) for k, v in w.items() if k.startswith(prefix)]

    ok = True

    def check(tag: str, got: mx.array, ref: np.ndarray) -> None:
        nonlocal ok
        c = cosine(np.array(got), ref)
        d = float(np.abs(np.array(got) - ref).max())
        good = c >= THRESHOLD
        ok &= good
        print(f"  [{tag}] cos {c:.6f}  maxAbs {d:.2e}  {'PASS' if good else 'FAIL'}")

    xa = mx.array(np.load(GOLDENS / "dv_nablock_in.npy"))

    blk = NABlock(512, (3, 5, 5), 64)
    blk.load_weights(sub("det_stages.3.0."), strict=True)
    check("A NABlock       ", blk(xa), np.load(GOLDENS / "dv_nablock_out.npy"))

    up = LinearPixelShuffleUpsample(512, (2, 2, 2), 2)
    up.load_weights(sub("upsamples.3."), strict=True)
    check("B upsample      ", up(xa, True), np.load(GOLDENS / "dv_upsample_out.npy"))

    ctx = mx.array(np.load(GOLDENS / "dv_diffblock_ctx.npy"))
    xt = mx.array(np.load(GOLDENS / "dv_diffblock_xt.npy"))

    conv_in_x_t = nn.Linear(48, 256)
    conv_in_x_t.load_weights(sub("conv_in_x_t."), strict=True)
    x0 = conv_in_x_t(patchify_hw(xt, 4).transpose(0, 2, 3, 4, 1))

    temb = nn.Sequential(nn.Linear(256, 384), nn.SiLU(), nn.Linear(384, 384))
    temb.load_weights([
        ("layers.0.weight", w["t_embedder.timestep_embedder.linear1.weight"]),
        ("layers.0.bias", w["t_embedder.timestep_embedder.linear1.bias"]),
        ("layers.2.weight", w["t_embedder.timestep_embedder.linear2.weight"]),
        ("layers.2.bias", w["t_embedder.timestep_embedder.linear2.bias"]),
    ], strict=True)
    adaln = nn.Linear(384, 7 * 256)
    adaln.load_weights([("weight", w["shared_adaln.proj.weight"]),
                        ("bias", w["shared_adaln.proj.bias"])], strict=True)
    ch = adaln(nn.silu(temb(timestep_embedding(mx.array([1.0]) * 1000.0))))
    shared = [ch[:, i * 256:(i + 1) * 256].reshape(-1, 1, 1, 1, 256) for i in range(7)]

    one = DiffusionNABlock(256, (11, 11, 11), 256, 64)
    one.load_weights(sub("diff_blocks.0."), strict=True)
    check("C diff block    ", one(x0, ctx, shared, tile=(2, None)),
          np.load(GOLDENS / "dv_diffblock_out.npy"))

    # [D] full stage-5 stack
    x = x0
    for i in range(8):
        b = DiffusionNABlock(256, (11, 11, 11), 256, 64)
        b.load_weights(sub(f"diff_blocks.{i}."), strict=True)
        x = b(x, ctx, shared, tile=(2, None))
        mx.eval(x)
    norm_out = nn.RMSNorm(256, eps=1e-6)
    norm_out.load_weights(sub("norm_out."), strict=True)
    conv_out = nn.Linear(256, 48)
    conv_out.load_weights(sub("conv_out."), strict=True)
    y = conv_out(norm_out(x)).transpose(0, 4, 1, 2, 3)
    check("D stage-5 stack ", unpatchify_hw(y, 4), np.load(GOLDENS / "dv_stage5_out.npy"))

    print(f"DIFFVAE COMPONENT GATE: {'PASS' if ok else 'FAIL'}")
    assert ok


if __name__ == "__main__":
    main()
