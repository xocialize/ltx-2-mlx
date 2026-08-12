"""Claim C3, on-screen-text half — Apple Vision OCR on decoder reconstructions.

The card claims the DiffVAE gives "sharper ... on-screen text". PSNR/SSIM
cannot answer that (they measure fidelity, and a generative decoder may
legitimately trade it). Text is the one part of the claim with an *objective*
readout: run OCR on the source frames and on each decoder's reconstruction of
the same latent, and compare what survives.

Uses Apple's Vision framework (VNRecognizeTextRequest) — on-device, no weights,
no port, no licence exposure; the same tooling the Forge net-clean track uses
for text masks. A VLM would be heavier and fuzzier for scoring accuracy.

Metrics per frame, treating the SOURCE's recognition as ground truth:
  * recall of source strings that are recovered exactly
  * mean OCR confidence on recovered strings
  * character-level similarity of the joined text

Usage:
    uv run --no-sync python scripts/measure_c3_ocr.py
"""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np

WORK = Path(".work/c3")
H, W, F = 512, 768, 9


def ocr(rgb: np.ndarray) -> list[tuple[str, float]]:
    """OCR a float[0,1] (H,W,3) frame -> [(text, confidence)]."""
    import Quartz
    import Vision
    from Foundation import NSData

    buf = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    h, w, _ = buf.shape
    provider = Quartz.CGDataProviderCreateWithCFData(NSData.dataWithBytes_length_(buf.tobytes(), buf.nbytes))
    cg = Quartz.CGImageCreate(
        w, h, 8, 24, w * 3, Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGBitmapByteOrderDefault, provider, None, False, Quartz.kCGRenderingIntentDefault)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    handler.performRequests_error_([req], None)
    out = []
    for o in req.results() or []:
        c = o.topCandidates_(1)
        if c:
            out.append((str(c[0].string()), float(c[0].confidence())))
    return out


def score(src: list[tuple[str, float]], got: list[tuple[str, float]]) -> tuple[float, float, float]:
    s_txt = [t for t, _ in src]
    g_txt = [t for t, _ in got]
    if not s_txt:
        return float("nan"), float("nan"), float("nan")
    recovered = [t for t in s_txt if t in g_txt]
    recall = len(recovered) / len(s_txt)
    conf = float(np.mean([c for t, c in got if t in recovered])) if recovered else 0.0
    sim = difflib.SequenceMatcher(None, " ".join(s_txt), " ".join(g_txt)).ratio()
    return recall, conf, sim


def main() -> None:
    recons = sorted(WORK.glob("recon_*.npy"))
    if not recons:
        raise SystemExit("no reconstructions — run measure_c3_diffvae_reconstruction.py with --save first")

    for raw in sorted(WORK.glob("*.raw")):
        src = np.frombuffer(raw.read_bytes(), np.uint8).reshape(F, H, W, 3).astype(np.float32) / 255.0
        arrs = {imp: np.load(WORK / f"recon_{imp}_{raw.stem}.npy") for imp in ("conv", "diffusion")
                if (WORK / f"recon_{imp}_{raw.stem}.npy").exists()}
        if len(arrs) < 2:
            continue
        print(f"\n  {raw.stem}")
        tot = {k: [] for k in arrs}
        n_src = 0
        for i in range(min(F, *(a.shape[0] for a in arrs.values()))):
            s = ocr(src[i])
            n_src += len(s)
            for imp, a in arrs.items():
                tot[imp].append(score(s, ocr(a[i])))
        print(f"    source strings detected: {n_src} over {i+1} frames")
        for imp in ("conv", "diffusion"):
            v = np.array([x for x in tot[imp] if not np.isnan(x[0])])
            if len(v):
                print(f"    {imp:10s} recall {v[:,0].mean():.3f}  conf {v[:,1].mean():.3f}  "
                      f"char-sim {v[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
