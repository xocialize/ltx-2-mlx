"""Gate: the audio-free forward works, and adding it changed NOTHING when audio is present.

The audio-free path (upstream ``audio=None``) is what DFR's temporal rounds denoise
their tiles with. Adding it meant gating four audio compute sites inside every
transformer block, which is exactly the kind of change that can perturb the normal
path by accident.

So the load-bearing assertion here is not that audio-free runs — it is that the
audio-PRESENT output is **bit-identical** to a baseline captured BEFORE the change
(``.work/audiofree/baseline.safetensors``, which also stores the weights so the
comparison is against the same parameters, not a re-initialised model).

    uv run python scripts/gate_ltx25_audio_free.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions

BASELINE = Path(__file__).resolve().parents[1] / ".work" / "audiofree" / "baseline.safetensors"


def build():
    cfg = LTXModelConfig(num_layers=3, video_dim=64, audio_dim=32, video_num_heads=4,
                         audio_num_heads=4, video_head_dim=16, audio_head_dim=8,
                         av_cross_num_heads=4, av_cross_head_dim=8)
    m = LTXModel(cfg)
    mx.random.seed(1)
    v = mx.random.normal((2, 24, 128))
    a = mx.random.normal((2, 10, 128))
    t = mx.array([0.7, 0.3])
    vt = mx.random.normal((2, 12, cfg.video_dim))
    at = mx.random.normal((2, 12, cfg.audio_dim))
    vp = compute_video_positions(3, 2, 4, 24.0)
    vp = vp if vp.ndim == 3 else vp[None]
    vp = mx.broadcast_to(vp, (2,) + vp.shape[1:])
    ap = compute_audio_positions(10)
    ap = ap if ap.ndim == 3 else ap[None]
    ap = mx.broadcast_to(ap, (2,) + ap.shape[1:])
    return m, dict(video_latent=v, audio_latent=a, timestep=t, video_text_embeds=vt,
                   audio_text_embeds=at, video_positions=vp, audio_positions=ap)


def main() -> int:
    if not BASELINE.exists():
        print(f"FAIL: baseline missing at {BASELINE}")
        return 1
    ref = mx.load(str(BASELINE))
    m, kw = build()
    # Restore the EXACT weights the baseline used — a re-initialised model would make
    # any comparison meaningless (this is the strict-load lesson from the DiT gate).
    weights = [(k[3:], v) for k, v in ref.items() if k.startswith("w::")]
    m.load_weights(weights, strict=True)
    fails: list[str] = []

    vo, ao = m(**kw)
    mx.eval(vo, ao)
    dv = float(mx.abs(vo.astype(mx.float32) - ref["video_out"].astype(mx.float32)).max())
    da = float(mx.abs(ao.astype(mx.float32) - ref["audio_out"].astype(mx.float32)).max())
    if dv != 0.0 or da != 0.0:
        fails.append(f"audio-present output CHANGED: video maxAbs {dv:g}, audio maxAbs {da:g} (must be 0)")
    print(f"  [1/4] audio present is BIT-IDENTICAL   video {dv:g}, audio {da:g}   "
          f"{'PASS' if not fails else 'FAIL'}")

    n0 = len(fails)
    kw_free = dict(kw)
    kw_free["audio_latent"] = None
    kw_free["audio_text_embeds"] = None
    kw_free["audio_positions"] = None
    vo_free, ao_free = m(**kw_free)
    mx.eval(vo_free)
    if ao_free is not None:
        fails.append(f"audio-free returned an audio tensor {tuple(ao_free.shape)}, expected None")
    if not bool(mx.all(mx.isfinite(vo_free))):
        fails.append("audio-free video output is not finite")
    if vo_free.shape != vo.shape:
        fails.append(f"audio-free video shape {tuple(vo_free.shape)} != {tuple(vo.shape)}")
    print(f"  [2/4] audio-free forward runs          video {tuple(vo_free.shape)}, audio None   "
          f"{'PASS' if len(fails) == n0 else 'FAIL'}")

    # Dropping audio must actually CHANGE video: the A2V cross-attention contribution is
    # gone. If it did not change, the gate would be blind to a2v being skipped wrongly.
    n0 = len(fails)
    delta = float(mx.abs(vo_free.astype(mx.float32) - vo.astype(mx.float32)).max())
    if delta == 0.0:
        fails.append(
            "audio-free video is identical to audio-present — A2V contributes nothing, so this "
            "gate cannot tell a correct skip from a broken one"
        )
    print(f"  [3/4] A2V really is dropped            video maxAbs delta {delta:.5f}   "
          f"{'PASS' if len(fails) == n0 else 'FAIL'}")

    # Audio-free must not depend on leftover audio kwargs being passed or not.
    n0 = len(fails)
    kw_partial = dict(kw)
    kw_partial["audio_latent"] = None
    vo_p, ao_p = m(**kw_partial)
    mx.eval(vo_p)
    d = float(mx.abs(vo_p.astype(mx.float32) - vo_free.astype(mx.float32)).max())
    if d != 0.0 or ao_p is not None:
        fails.append(f"audio-free depends on stale audio kwargs: video maxAbs {d:g}, audio {ao_p is not None}")
    print(f"  [4/4] stale audio kwargs ignored       maxAbs {d:g}   "
          f"{'PASS' if len(fails) == n0 else 'FAIL'}")

    if fails:
        print(f"\nFAIL — {len(fails)} problem(s):")
        for f in fails:
            print("  " + f)
        return 1
    print("\nPASS — audio-free forward works and the audio-present path is untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
