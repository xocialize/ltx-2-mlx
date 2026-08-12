"""Precision control for the teacher-forced loop gate.

Answers: how much does a *precision-only* change move a single ancestral step,
step by step, when teacher-forced onto the same trajectory?

Runs the fp32 model over the bf16 reference trajectory: for each step i, start
from the bf16 run's ``x_i``, take one ancestral step with the bf16 run's noise,
and compare against the bf16 run's ``x_{i+1}``. That is exactly the comparison
``gate_ltx25_loop.py`` performs for the MLX port, with torch-fp32 substituted
for the port — so the two tables are directly comparable and any step where the
port is no worse than fp32 is precision, not a port bug.

Run in the ref venv (needs ~88 GB for the fp32 model):
    .work/ltx25-ref/.venv/bin/python scripts/control_ltx25_loop_precision.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / ".work/ltx25-ref/goldens"
CKPT = Path(
    "/Volumes/Satechi/Models/Lightricks/LTX-2.5/diffusion_models/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
sys.path.insert(0, "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-core/src")
sys.path.insert(0, "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-pipelines/src")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    import safetensors
    from safetensors.torch import load_file

    from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.types import LatentState
    from ltx_pipelines.utils.denoisers import SimpleDenoiser

    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    DT = torch.float32

    meta = json.loads((GOLDENS / "loop_meta.json").read_text())
    F, H, W, NA = meta["geometry"]
    sigmas_list = meta["sigmas"]
    NV = F * H * W

    v_steps = np.load(GOLDENS / "loop_video_steps.npy")
    a_steps = np.load(GOLDENS / "loop_audio_steps.npy")
    v_noise = np.load(GOLDENS / "loop_noise_video.npy")
    a_noise = np.load(GOLDENS / "loop_noise_audio.npy")
    v_pos_b = np.load(GOLDENS / "loop_video_positions_bounds.npy") if (
        GOLDENS / "loop_video_positions_bounds.npy").exists() else None

    with safetensors.safe_open(str(CKPT), framework="pt") as f:
        metadata = {k: json.loads(v) if v.strip().startswith(("{", "[")) else v
                    for k, v in f.metadata().items()}
    model = LTXModelConfigurator.from_metadata(metadata)
    sd = load_file(CKPT)
    pref = "model.diffusion_model."
    model.load_state_dict({k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}, strict=False)
    del sd
    model = model.to(DT).to(DEV).eval()
    x0_model = X0Model(model)
    print(f"  fp32 model loaded on {DEV}")

    # rebuild the exact positions the golden used
    from ltx_core.components.patchifiers import (
        AudioPatchifier,
        VideoLatentPatchifier,
        get_pixel_coords,
    )
    from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape

    scale = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    v_bounds = VideoLatentPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=VideoLatentShape(batch=1, channels=128, frames=F, height=H, width=W), device="cpu")
    v_pos = get_pixel_coords(latent_coords=v_bounds, scale_factors=scale, causal_fix=True).float()
    v_pos[:, 0, ...] = v_pos[:, 0, ...] / meta["fps"]
    a_pos = AudioPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=1, channels=8, frames=NA, mel_bins=16), device="cpu").float()

    v_ctx = torch.tensor(np.load(GOLDENS / "connector_video_encoding_fp32.npy")).to(DT).to(DEV)
    a_ctx = torch.tensor(np.load(GOLDENS / "connector_audio_encoding_fp32.npy")).to(DT).to(DEV)
    kf = torch.tensor(np.load(GOLDENS / "loop_keyframes_mask.npy")).to(DEV)

    stepper = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
    denoiser = SimpleDenoiser(v_ctx, a_ctx)
    sigmas = torch.tensor(sigmas_list).to(DEV)

    print(f"  teacher-forced precision control ({len(sigmas_list)-1} steps)")
    rows = []
    v_next_all, a_next_all = [], []
    for i in range(len(sigmas_list) - 1):
        v_state = LatentState(
            latent=torch.tensor(v_steps[i]).to(DT).to(DEV),
            denoise_mask=torch.ones((1, NV, 1)).to(DEV),
            positions=v_pos.to(DEV),
            clean_latent=torch.zeros((1, NV, 128)).to(DT).to(DEV),
            keyframes_mask=kf,
        )
        a_state = LatentState(
            latent=torch.tensor(a_steps[i]).to(DT).to(DEV),
            denoise_mask=torch.ones((1, NA, 1)).to(DEV),
            positions=a_pos.to(DEV),
            clean_latent=torch.zeros((1, NA, 128)).to(DT).to(DEV),
        )
        with torch.no_grad():
            v_res, a_res = denoiser(x0_model, v_state, a_state, sigmas, i)
        terminal = sigmas_list[i + 1] == 0
        if terminal:
            v_next = v_res.denoised
            a_next = a_res.denoised
        else:
            v_next = stepper.step(v_state.latent, v_res.denoised, sigmas, i,
                                  noise=torch.tensor(v_noise[i]).to(DT).to(DEV))
            a_next = stepper.step(a_state.latent, a_res.denoised, sigmas, i,
                                  noise=torch.tensor(a_noise[i]).to(DT).to(DEV))
        vc = cosine(v_next.float().cpu().numpy(), v_steps[i + 1])
        ac = cosine(a_next.float().cpu().numpy(), a_steps[i + 1])
        rows.append((i, sigmas_list[i], sigmas_list[i + 1], vc, ac))
        v_next_all.append(v_next.float().cpu().numpy())
        a_next_all.append(a_next.float().cpu().numpy())
        print(f"    step {i} (s {sigmas_list[i]:.5f}->{sigmas_list[i+1]:.6f}): "
              f"video {vc:.6f} audio {ac:.6f}{'  <-- terminal' if terminal else ''}")

    np.save(GOLDENS / "loop_precision_control.npy",
            np.array([(r[3], r[4]) for r in rows], dtype=np.float64))
    # fp32 "ground truth" next-step tensors, teacher-forced on the bf16 trajectory
    np.save(GOLDENS / "loop_fp32tf_video_next.npy", np.stack(v_next_all))
    np.save(GOLDENS / "loop_fp32tf_audio_next.npy", np.stack(a_next_all))
    print(f"  saved fp32 teacher-forced next-step tensors: {np.stack(v_next_all).shape}")
    worst = min(min(r[3], r[4]) for r in rows)
    print(f"  worst precision-only step: {worst:.6f}")


if __name__ == "__main__":
    main()
