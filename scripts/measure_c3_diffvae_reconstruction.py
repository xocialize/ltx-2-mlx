"""Claim C3 — "new diffusion video decoder: sharper faces, textures, on-screen text".

The honest test: **reconstruction against real ground truth**, not sharpness
proxies on generated content (which conflate detail with grain and have no
reference to be right about).

    real clip -> VAE encode -> decode with BOTH decoders -> compare to the source

Both decoders share the same encoder and latent (the conv VAE is byte-identical
to LTX-2.3's), so the comparison is decoder-isolated by construction.

Metrics: PSNR and SSIM against the source frames, plus a high-frequency energy
ratio. The DiffVAE samples from noise, so a fixed seed is used and pixel-exact
reconstruction is not expected of either decoder — the question is which lands
closer to the truth.

Usage:
    uv run --no-sync python scripts/measure_c3_diffvae_reconstruction.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

MODEL = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
CLIPS = Path(".work/c3")
H, W, F = 512, 768, 9


def load_raw(p: Path) -> np.ndarray:
    return np.frombuffer(p.read_bytes(), np.uint8).reshape(F, H, W, 3).astype(np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on luma, 8x8 windows."""
    ga, gb = a.mean(-1), b.mean(-1)
    k1, k2 = 0.01**2, 0.03**2
    out = []
    for i in range(0, ga.shape[1] - 7, 8):
        for j in range(0, ga.shape[2] - 7, 8):
            x, y = ga[:, i:i + 8, j:j + 8], gb[:, i:i + 8, j:j + 8]
            mx_, my = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cxy = ((x - mx_) * (y - my)).mean()
            out.append(((2 * mx_ * my + k1) * (2 * cxy + k2)) /
                       ((mx_**2 + my**2 + k1) * (vx + vy + k2)))
    return float(np.mean(out))


def hf_energy(a: np.ndarray) -> float:
    g = a.mean(-1)
    lap = (-4 * g[:, 1:-1, 1:-1] + g[:, :-2, 1:-1] + g[:, 2:, 1:-1]
           + g[:, 1:-1, :-2] + g[:, 1:-1, 2:])
    return float(lap.var()) * 255.0**2  # report on an 8-bit scale


def main() -> None:
    from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
    from ltx_core_mlx.utils.weights import load_split_safetensors

    enc = VideoEncoder()
    ew = load_split_safetensors(MODEL / "vae_encoder.safetensors", prefix="vae_encoder.")
    # MLX treats leading-underscore attrs as private (utils/weights.remap rule)
    ew = {k.replace("._mean_of_means", ".mean_of_means")
           .replace("._std_of_means", ".std_of_means"): v for k, v in ew.items()}
    enc.load_weights(list(ew.items()))

    rows = []
    for raw in sorted(CLIPS.glob("*.raw")):
        src = load_raw(raw)                       # (F,H,W,3) in [0,1]
        x = mx.array(src.transpose(3, 0, 1, 2)[None] * 2.0 - 1.0)  # (1,3,F,H,W) in [-1,1]
        # ⚠️ encode() ALREADY returns a normalized latent. Calling
        # normalize_latent again divides by the per-channel std a second time
        # (std ~1 -> ~9, range +-74) and both decoders then reconstruct garbage
        # — measured as PSNR ~12 dB before this was caught.
        lat = enc.encode(x)
        mx.eval(lat)

        outs = {}
        for impl in ("conv", "diffusion"):
            if impl == "conv":
                from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder as ConvDec
                d = ConvDec()
                d.load_weights(list(load_split_safetensors(
                    MODEL / "vae_decoder.safetensors", prefix="vae_decoder.").items()))
                t0 = time.time()
                px = d.decode(lat) if hasattr(d, "decode") else d(lat)
                mx.eval(px)
                el = time.time() - t0
                arr = np.array(px)[0].transpose(1, 2, 3, 0)
                arr = np.clip((arr + 1) * 0.5, 0, 1)
            else:
                import importlib

                import ltx_core_mlx.model.video_vae.diffusion_video_decoder as DV
                importlib.reload(DV)
                d = DV.DiffusionVideoDecoder()
                d.load_weights([(k, v.astype(mx.float32)) for k, v in load_split_safetensors(
                    MODEL / "vae_diffusion_decoder.safetensors",
                    prefix="vae_diffusion_decoder.").items()], strict=True)
                t0 = time.time()
                px = d.decode_to_rgb(lat, key=mx.random.key(0))
                mx.eval(px)
                el = time.time() - t0
                arr = np.array(px)[0].transpose(1, 2, 3, 0)
            n = min(len(src), arr.shape[0])
            outs[impl] = (arr[:n], el)
            np.save(CLIPS / f"recon_{impl}_{raw.stem}.npy", arr[:n].astype(np.float32))

        s = src[: min(len(src), min(v[0].shape[0] for v in outs.values()))]
        row = {"clip": raw.stem[:28]}
        for impl, (arr, el) in outs.items():
            a = arr[: len(s)]
            row[impl] = (psnr(s, a), ssim(s, a), hf_energy(a), el)
        row["src_hf"] = hf_energy(s)
        rows.append(row)

    print(f"\n  {'clip':30s} {'decoder':10s} {'PSNR':>7s} {'SSIM':>7s} {'HF energy':>10s} {'time':>7s}")
    print(f"  {'-'*30} {'-'*10} {'-'*7} {'-'*7} {'-'*10} {'-'*7}")
    for r in rows:
        for impl in ("conv", "diffusion"):
            p_, s_, h_, t_ = r[impl]
            print(f"  {r['clip'] if impl=='conv' else '':30s} {impl:10s} "
                  f"{p_:7.2f} {s_:7.4f} {h_:10.2f} {t_:6.1f}s")
        print(f"  {'':30s} {'(source)':10s} {'':>7s} {'':>7s} {r['src_hf']:10.2f}")
    print()
    for metric, idx in (("PSNR", 0), ("SSIM", 1)):
        c = np.mean([r["conv"][idx] for r in rows])
        d = np.mean([r["diffusion"][idx] for r in rows])
        better = "DiffVAE" if d > c else "conv"
        print(f"  mean {metric}: conv {c:.3f}  DiffVAE {d:.3f}  -> {better} closer to truth")
    cs = np.mean([abs(r["conv"][2] - r["src_hf"]) for r in rows])
    ds = np.mean([abs(r["diffusion"][2] - r["src_hf"]) for r in rows])
    print(f"  mean |HF energy - source|: conv {cs:.2f}  DiffVAE {ds:.2f}"
          f"  -> {'DiffVAE' if ds < cs else 'conv'} closer to the source detail level")
    print("\n  ⚠️ PSNR/SSIM measure FIDELITY to the source, not perceptual quality. A\n"
          "     generative decoder that samples from noise can legitimately score lower\n"
          "     while looking sharper. This run establishes the fidelity ranking only;\n"
          "     the 'sharper faces/textures/on-screen text' claim still needs OCR\n"
          "     accuracy on text content + operator perceptual judgement (plan §V C3).")


if __name__ == "__main__":
    main()
