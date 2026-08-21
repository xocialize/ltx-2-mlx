"""LTX-2.5 denoise-loop parity gate — teacher-forced, per step.

Golden from ``scripts/dump_ltx25_loop_golden.py``: upstream's
``euler_ancestral_denoising_loop`` + ``SimpleDenoiser`` over the real DiT, with
every noise draw AND every per-step latent captured.

## Why this is teacher-forced rather than free-running

An 8-step ancestral loop at ``eta=1.0`` re-injects noise with a large
coefficient every step, which makes the trajectory chaotic: measured on the
reference alone, running the *same* loop in bf16 vs fp32 (identical noise to
cos 0.999999) lands on final latents that agree to only **video 0.828630 /
audio 0.995077**. A free-running final-latent cosine therefore cannot
distinguish a port bug from precision — the reference cannot reproduce itself.
(This is the fleet determinism doctrine: gate per-op, judge trajectories
perceptually.)

So each step is gated independently, re-synchronised to the reference:
for every step *i*, both sides start from the reference's ``x_i`` and the
reference's noise draw. Errors cannot compound, so a real mechanical divergence
(sigma pairing, x0 conversion, mask blending, renoise coefficients, keyframes
threading) shows up immediately and locally.

And — as in the DiT gate — the comparison target is **fp32 ground truth**, not
the bf16 golden: ``scripts/control_ltx25_loop_precision.py`` runs the fp32 model
teacher-forced on the same bf16 trajectory, giving both the truth tensors and
the reference's own bf16 error against them. The gate asserts our accuracy is
**at least as good as the reference's own bf16 path** at every step. Step 6 is
intrinsically the most sensitive (even fp32-vs-bf16 lands at 0.9932 there), so
an absolute 0.999 threshold is below the achievable floor for a bf16 model and
would be meaningless.

A free-running run is still executed and reported as a *diagnostic*, with the
reference's own bf16-vs-fp32 divergence printed beside it for scale.

Usage:
    uv run --no-sync python scripts/gate_ltx25_loop.py
"""

from __future__ import annotations

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
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState
    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
    from ltx_core_mlx.utils.weights import load_split_safetensors
    from ltx_pipelines_mlx.utils.samplers import denoise_loop

    meta = json.loads((GOLDENS / "loop_meta.json").read_text())
    F, H, W, NA = meta["geometry"]
    sigmas = meta["sigmas"]
    NV = F * H * W

    g = {k: np.load(GOLDENS / f"loop_{k}.npy") for k in (
        "video_positions_mid", "audio_positions_mid", "keyframes_mask",
        "noise_video", "noise_audio", "video_steps", "audio_steps",
        "video_final", "audio_final",
    )}
    v_ctx = mx.array(np.load(GOLDENS / "connector_video_encoding_fp32.npy"))
    a_ctx = mx.array(np.load(GOLDENS / "connector_audio_encoding_fp32.npy"))
    v_pos = mx.array(g["video_positions_mid"])
    a_pos = mx.array(g["audio_positions_mid"])
    kf = mx.array(g["keyframes_mask"])

    cfg = LTXModelConfig.from_checkpoint_dir(MODEL_DIR)
    model = LTXModel(cfg)
    w = load_split_safetensors(MODEL_DIR / "transformer-distilled.safetensors", prefix="transformer.")
    model.load_weights(list(w.items()), strict=True)
    del w
    mx.clear_cache()
    x0_model = X0Model(model)

    def states(v_lat: np.ndarray, a_lat: np.ndarray):
        return (
            LatentState(latent=mx.array(v_lat).astype(mx.bfloat16),
                        clean_latent=mx.zeros((1, NV, 128), dtype=mx.bfloat16),
                        denoise_mask=mx.ones((1, NV, 1)), positions=v_pos),
            LatentState(latent=mx.array(a_lat).astype(mx.bfloat16),
                        clean_latent=mx.zeros((1, NA, 128), dtype=mx.bfloat16),
                        denoise_mask=mx.ones((1, NA, 1)), positions=a_pos),
        )

    def run(v_state, a_state, sigma_pair, noise_idx: int | None):
        def provider(step_idx: int, modality: str, shape):
            assert noise_idx is not None, "noise requested on a terminal step"
            arr = (g["noise_video"] if modality == "video" else g["noise_audio"])[noise_idx]
            return mx.array(arr)

        out = denoise_loop(
            model=x0_model, video_state=v_state, audio_state=a_state,
            video_text_embeds=v_ctx, audio_text_embeds=a_ctx,
            sigmas=list(sigma_pair), keyframes_mask=kf,
            ancestral_eta=1.0, ancestral_s_noise=1.0,
            noise_provider=provider if noise_idx is not None else None,
            show_progress=False,
        )
        mx.eval(out.video_latent, out.audio_latent)
        return out

    # ---- [1] teacher-forced per-step parity vs fp32 ground truth ----
    tf_v = GOLDENS / "loop_fp32tf_video_next.npy"
    if not tf_v.exists():
        raise SystemExit("missing fp32 control tensors — run scripts/control_ltx25_loop_precision.py")
    truth_v = np.load(tf_v)
    truth_a = np.load(GOLDENS / "loop_fp32tf_audio_next.npy")

    n_steps = len(sigmas) - 1
    print(f"  teacher-forced vs fp32 truth: {n_steps} steps, {NV} video / {NA} audio tokens")
    print(f"    {'step':4s} {'ours V':>9s} {'refbf16 V':>10s} {'ours A':>9s} {'refbf16 A':>10s}")
    ok = True
    for i in range(n_steps):
        v_state, a_state = states(g["video_steps"][i], g["audio_steps"][i])
        terminal = sigmas[i + 1] == 0
        out = run(v_state, a_state, (sigmas[i], sigmas[i + 1]), None if terminal else i)
        ov = cosine(np.array(out.video_latent.astype(mx.float32)), truth_v[i])
        oa = cosine(np.array(out.audio_latent.astype(mx.float32)), truth_a[i])
        rv = cosine(g["video_steps"][i + 1], truth_v[i])
        ra = cosine(g["audio_steps"][i + 1], truth_a[i])
        mark = "" if (ov >= rv and oa >= ra) else "   <-- BELOW reference bf16"
        if ov < rv or oa < ra:
            ok = False
        print(f"    {i:<4d} {ov:9.6f} {rv:10.6f} {oa:9.6f} {ra:10.6f}{mark}")
    worst_v = worst_a = 1.0
    worst_i = -1

    # ---- [2] free-running diagnostic, against the reference's own dtype spread ----
    v_state, a_state = states(g["video_steps"][0], g["audio_steps"][0])
    out = denoise_loop(
        model=x0_model, video_state=v_state, audio_state=a_state,
        video_text_embeds=v_ctx, audio_text_embeds=a_ctx, sigmas=sigmas,
        keyframes_mask=kf, ancestral_eta=1.0, ancestral_s_noise=1.0,
        noise_provider=lambda s, m, shp: mx.array(
            (g["noise_video"] if m == "video" else g["noise_audio"])[s]),
        show_progress=False,
    )
    mx.eval(out.video_latent, out.audio_latent)
    fv = cosine(np.array(out.video_latent.astype(mx.float32)), g["video_final"])
    fa = cosine(np.array(out.audio_latent.astype(mx.float32)), g["audio_final"])
    ref_spread = ""
    if (GOLDENS / "loop_fp32_video_final.npy").exists():
        rv = cosine(np.load(GOLDENS / "loop_fp32_video_final.npy"), g["video_final"])
        ra = cosine(np.load(GOLDENS / "loop_fp32_audio_final.npy"), g["audio_final"])
        ref_spread = f"   [reference bf16-vs-fp32: video {rv:.6f} audio {ra:.6f}]"
    print(f"  free-running (diagnostic): video {fv:.6f} audio {fa:.6f}{ref_spread}")

    status = "PASS" if ok else "FAIL"
    print(f"\nLOOP GATE: {status} (every step at least as accurate as the reference's own bf16 path)")
    assert ok, "a step was less accurate than the reference's own bf16 execution"


if __name__ == "__main__":
    main()
