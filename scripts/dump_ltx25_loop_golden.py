"""Dump the LTX-2.5 denoise-loop golden (8-step distilled, ancestral stage 1).

Runs upstream's ``euler_ancestral_denoising_loop`` with ``SimpleDenoiser`` (the
distilled recipe: no CFG) over the real DiT, capturing every noise draw so the
MLX side can be fed the *identical* noise and compared on final latents. This
gates the loop mechanics — sigma schedule, x0 conversion, mask blending,
ancestral renoise, keyframes-mask threading — independently of RNG.

Noise draw order inside upstream's loop is video-then-audio per non-terminal
step (single seeded generator), which is what the MLX port replicates.

Run in the ref venv:
    .work/ltx25-ref/.venv/bin/python scripts/dump_ltx25_loop_golden.py
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
UPSTREAM = "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-core/src"
PIPELINES = "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-pipelines/src"
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, PIPELINES)

F, H, W, NA = 2, 4, 6, 8
FPS = 24.0
SEED = 42 + 10000  # upstream distilled: pipeline seed + ANCESTRAL_NOISE_SEED_OFFSET
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    DT = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    SUF = args.suffix

    import safetensors
    from safetensors.torch import load_file

    from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
    from ltx_core.components.patchifiers import (
        AudioPatchifier,
        VideoLatentPatchifier,
        get_pixel_coords,
    )
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.types import (
        AudioLatentShape,
        LatentState,
        SpatioTemporalScaleFactors,
        VideoLatentShape,
    )
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop

    DEV = "mps" if torch.backends.mps.is_available() else "cpu"

    with safetensors.safe_open(str(CKPT), framework="pt") as f:
        metadata = {k: json.loads(v) if v.strip().startswith(("{", "[")) else v
                    for k, v in f.metadata().items()}
    model = LTXModelConfigurator.from_metadata(metadata)
    sd = load_file(CKPT)
    pref = "model.diffusion_model."
    model.load_state_dict({k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}, strict=False)
    del sd
    model = model.to(DT).to(DEV).eval()
    print(f"  dtype {DT}")
    x0_model = X0Model(model)

    # positions (VideoLatentTools chain)
    scale = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    v_bounds = VideoLatentPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=VideoLatentShape(batch=1, channels=128, frames=F, height=H, width=W), device="cpu")
    v_pos = get_pixel_coords(latent_coords=v_bounds, scale_factors=scale, causal_fix=True).float()
    v_pos[:, 0, ...] = v_pos[:, 0, ...] / FPS
    a_pos = AudioPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=1, channels=8, frames=NA, mel_bins=16), device="cpu").float()

    NV = F * H * W
    rng = np.random.default_rng(777)
    v_init = rng.standard_normal((1, NV, 128)).astype(np.float32)
    a_init = rng.standard_normal((1, NA, 128)).astype(np.float32)

    v_ctx = torch.tensor(np.load(GOLDENS / "connector_video_encoding_fp32.npy")).to(DT).to(DEV)
    a_ctx = torch.tensor(np.load(GOLDENS / "connector_audio_encoding_fp32.npy")).to(DT).to(DEV)

    kf = torch.zeros((1, NV, 1))
    kf[:, : H * W] = 1.0

    v_state = LatentState(
        latent=torch.tensor(v_init).to(DT).to(DEV),
        denoise_mask=torch.ones((1, NV, 1)).to(DEV),
        positions=v_pos.to(DEV),
        clean_latent=torch.zeros((1, NV, 128)).to(DT).to(DEV),
        keyframes_mask=kf.to(DEV),
    )
    a_state = LatentState(
        latent=torch.tensor(a_init).to(DT).to(DEV),
        denoise_mask=torch.ones((1, NA, 1)).to(DEV),
        positions=a_pos.to(DEV),
        clean_latent=torch.zeros((1, NA, 128)).to(DT).to(DEV),
    )

    captured: list[np.ndarray] = []

    def capturing_noise(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
        captured.append(noise.float().cpu().numpy())
        return noise

    step_states: list[tuple[np.ndarray, np.ndarray]] = []
    step_x0: list[tuple[np.ndarray, np.ndarray]] = []

    class CapturingDenoiser(SimpleDenoiser):
        """Records the latent entering every step, so the gate can teacher-force."""

        def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
            step_states.append((
                video_state.latent.float().cpu().numpy(),
                audio_state.latent.float().cpu().numpy(),
            ))
            res = super().__call__(transformer, video_state, audio_state, sigmas, step_index)
            step_x0.append((
                res[0].denoised.float().cpu().numpy(),
                res[1].denoised.float().cpu().numpy(),
            ))
            return res

    sigmas = torch.tensor(DISTILLED_SIGMAS)
    stepper = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
    denoiser = CapturingDenoiser(v_ctx, a_ctx)

    print(f"running {len(DISTILLED_SIGMAS)-1}-step ancestral loop ({NV} video / {NA} audio tokens) ...")
    with torch.no_grad():
        v_final, a_final = euler_ancestral_denoising_loop(
            sigmas=sigmas.to(DEV),
            video_state=v_state,
            audio_state=a_state,
            stepper=stepper,
            transformer=x0_model,
            denoiser=denoiser,
            noise_seed=SEED,
            new_noise_fn=capturing_noise,
            model_dtype=DT,
        )

    print(f"captured {len(captured)} noise draws (expect 2 per non-terminal step)")
    shapes = [tuple(n.shape) for n in captured]
    print(f"  shapes: {shapes[:4]} ...")

    np.save(GOLDENS / f"loop{SUF}_video_init.npy", v_init)
    np.save(GOLDENS / f"loop{SUF}_audio_init.npy", a_init)
    np.save(GOLDENS / f"loop{SUF}_video_positions_mid.npy",
            ((v_pos[..., 0] + v_pos[..., 1]) / 2).permute(0, 2, 1).contiguous().numpy())
    np.save(GOLDENS / f"loop{SUF}_audio_positions_mid.npy",
            ((a_pos[..., 0] + a_pos[..., 1]) / 2).permute(0, 2, 1).contiguous().numpy())
    np.save(GOLDENS / f"loop{SUF}_keyframes_mask.npy", kf.numpy())
    # draws alternate video, audio per non-terminal step -> split into two stacks
    v_noise = np.stack(captured[0::2])  # (steps, B, Nv, C)
    a_noise = np.stack(captured[1::2])  # (steps, B, Na, C)
    assert v_noise.shape[2] == NV and a_noise.shape[2] == NA, "noise draw order is not video-then-audio"
    np.save(GOLDENS / f"loop{SUF}_noise_video.npy", v_noise)
    np.save(GOLDENS / f"loop{SUF}_noise_audio.npy", a_noise)
    print(f"  split noise: video {v_noise.shape} audio {a_noise.shape}")
    # per-step latents: x_0..x_{n-1} entering each step, plus the final x_n
    v_steps = np.stack([s[0] for s in step_states] + [v_final.latent.float().cpu().numpy()])
    a_steps = np.stack([s[1] for s in step_states] + [a_final.latent.float().cpu().numpy()])
    np.save(GOLDENS / f"loop{SUF}_video_steps.npy", v_steps)
    np.save(GOLDENS / f"loop{SUF}_audio_steps.npy", a_steps)
    print(f"  per-step latents: video {v_steps.shape} audio {a_steps.shape}")
    np.save(GOLDENS / f"loop{SUF}_video_x0.npy", np.stack([x[0] for x in step_x0]))
    np.save(GOLDENS / f"loop{SUF}_audio_x0.npy", np.stack([x[1] for x in step_x0]))
    print(f"  per-step x0: video {np.stack([x[0] for x in step_x0]).shape}")
    np.save(GOLDENS / f"loop{SUF}_video_final.npy", v_final.latent.float().cpu().numpy())
    np.save(GOLDENS / f"loop{SUF}_audio_final.npy", a_final.latent.float().cpu().numpy())
    (GOLDENS / f"loop{SUF}_meta.json").write_text(json.dumps({
        "geometry": [F, H, W, NA], "fps": FPS, "seed": SEED,
        "sigmas": DISTILLED_SIGMAS, "noise_shapes": shapes, "device": DEV,
    }, indent=2))
    print(f"saved loop goldens: video final {tuple(v_final.latent.shape)} "
          f"audio final {tuple(a_final.latent.shape)}")


if __name__ == "__main__":
    main()
