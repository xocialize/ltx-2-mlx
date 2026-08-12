"""Dump the LTX-2.5 DiT-forward golden from the upstream torch implementation.

Builds upstream ``LTXModel`` via ``LTXModelConfigurator.from_metadata`` on the
official comfy checkpoint, runs one forward at a small geometry, and saves every
input plus both velocity outputs.

Two position representations are saved so the MLX gate can isolate failures:
  * ``*_positions_bounds``  — upstream's ``(B, n_dims, T, 2)`` [start, end) form
  * ``*_positions_mid``     — ``(B, T, n_dims)`` midpoints, the convention the
    MLX port uses. RoPE is evaluated at the patch midpoint
    (``use_middle_indices_grid=True``), so feeding the midpoints to the MLX side
    reproduces the same effective positions and gates the DiT math alone.
    A separate check compares our own position builder against these.

Run in the ref venv (needs ~44 GB for the bf16 model):
    .work/ltx25-ref/.venv/bin/python scripts/dump_ltx25_dit_golden.py [--geometry small|prod]
"""

import argparse
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
sys.path.insert(0, UPSTREAM)

# (latent frames, latent H, latent W, audio tokens)
GEOMETRIES = {
    "small": (2, 4, 6, 8),
    # production grid: 512x768x25f -> F=4 latent frames, 16x24 latent HW
    "prod": (4, 16, 24, 40),
}
FPS = 24.0
SIGMA = 0.9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=list(GEOMETRIES), default="small")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--suffix", default="", help="tag suffix for the saved arrays")
    args = ap.parse_args()
    F, H, W, NA = GEOMETRIES[args.geometry]
    tag = args.geometry + args.suffix
    DT = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    import json

    import safetensors
    from safetensors.torch import load_file

    from ltx_core.components.patchifiers import (
        AudioPatchifier,
        VideoLatentPatchifier,
        get_pixel_coords,
    )
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import (
        AudioLatentShape,
        SpatioTemporalScaleFactors,
        VideoLatentShape,
    )

    DEV = "mps" if torch.backends.mps.is_available() else "cpu"

    with safetensors.safe_open(str(CKPT), framework="pt") as f:
        meta = f.metadata()
    metadata = {k: json.loads(v) if v.strip().startswith(("{", "[")) else v for k, v in meta.items()}
    tcfg = metadata.get("config", {}).get("transformer", {})
    print(f"checkpoint config: ff_bias={tcfg.get('ff_bias')} "
          f"keyframes={tcfg.get('use_keyframes_abs_pos_embedding')} "
          f"rope={tcfg.get('rope_type')} prec={tcfg.get('frequencies_precision')}")

    print("building model (torch, CPU) ...")
    model = LTXModelConfigurator.from_metadata(metadata)

    print("loading weights ...")
    sd = load_file(CKPT)
    pref = "model.diffusion_model."
    stripped = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    # connectors + projections live in the same file but belong to the TE stage
    unexpected_real = [
        k for k in unexpected
        if not k.startswith(("video_embeddings_connector.", "audio_embeddings_connector.",
                             "text_embedding_projection."))
    ]
    print(f"  loaded; missing={missing[:4]} unexpected(non-connector)={unexpected_real[:4]}")
    assert not missing and not unexpected_real, "state dict mismatch"
    del sd, stripped

    model = model.to(DT).to(DEV).eval()
    print(f"  model dtype: {DT}")

    # ---- geometry, positions (VideoLatentTools chain) ----
    scale = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    v_patch = VideoLatentPatchifier(patch_size=1)
    a_patch = AudioPatchifier(patch_size=1)

    v_shape = VideoLatentShape(batch=1, channels=128, frames=F, height=H, width=W)
    v_bounds = v_patch.get_patch_grid_bounds(output_shape=v_shape, device="cpu")
    v_pos = get_pixel_coords(latent_coords=v_bounds, scale_factors=scale, causal_fix=True).float()
    v_pos[:, 0, ...] = v_pos[:, 0, ...] / FPS

    a_shape = AudioLatentShape(batch=1, channels=8, frames=NA, mel_bins=16)
    a_pos = a_patch.get_patch_grid_bounds(output_shape=a_shape, device="cpu").float()

    NV = F * H * W
    print(f"geometry {tag}: video {F}x{H}x{W} = {NV} tokens, audio {NA} tokens")
    print(f"  video positions {tuple(v_pos.shape)} t-range [{float(v_pos[0,0].min()):.4f}, {float(v_pos[0,0].max()):.4f}]")
    print(f"  audio positions {tuple(a_pos.shape)} range [{float(a_pos.min()):.4f}, {float(a_pos.max()):.4f}]")

    # ---- inputs (seeded, saved for the MLX side) ----
    rng = np.random.default_rng(1234)
    v_latent = rng.standard_normal((1, NV, 128)).astype(np.float32) * 0.5
    a_latent = rng.standard_normal((1, NA, 128)).astype(np.float32) * 0.5

    v_ctx = np.load(GOLDENS / "connector_video_encoding_fp32.npy")
    a_ctx = np.load(GOLDENS / "connector_audio_encoding_fp32.npy")
    ctx_mask = np.load(GOLDENS / "connector_out_mask.npy")

    sigma = torch.full((1,), SIGMA)
    v_timesteps = torch.full((1, NV), SIGMA)
    a_timesteps = torch.full((1, NA), SIGMA)
    # first latent frame marked (VideoLatentTools._first_frame_keyframes_mask)
    kf = torch.zeros((1, NV, 1))
    kf[:, : H * W] = 1.0

    def to(x, dtype=None):
        dtype = DT if dtype is None else dtype
        return torch.as_tensor(x).to(dtype).to(DEV)

    video = Modality(
        latent=to(v_latent),
        sigma=sigma.to(DEV),
        timesteps=to(v_timesteps, torch.float32),
        positions=v_pos.to(DEV),
        context=to(v_ctx),
        context_mask=to(ctx_mask, torch.float32),
        keyframes_mask=to(kf, torch.float32),
    )
    audio = Modality(
        latent=to(a_latent),
        sigma=sigma.to(DEV),
        timesteps=to(a_timesteps, torch.float32),
        positions=a_pos.to(DEV),
        context=to(a_ctx),
        context_mask=to(ctx_mask, torch.float32),
    )

    print("forward ...")
    with torch.no_grad():
        v_out, a_out = model(video, audio, None)
    print(f"  outputs: video {tuple(v_out.shape)} audio {tuple(a_out.shape)}")

    # midpoints in the MLX convention (B, T, n_dims)
    v_mid = ((v_pos[..., 0] + v_pos[..., 1]) / 2).permute(0, 2, 1).contiguous()
    a_mid = ((a_pos[..., 0] + a_pos[..., 1]) / 2).permute(0, 2, 1).contiguous()

    out = {
        f"dit_{tag}_video_latent": v_latent,
        f"dit_{tag}_audio_latent": a_latent,
        f"dit_{tag}_video_positions_bounds": v_pos.numpy(),
        f"dit_{tag}_audio_positions_bounds": a_pos.numpy(),
        f"dit_{tag}_video_positions_mid": v_mid.numpy(),
        f"dit_{tag}_audio_positions_mid": a_mid.numpy(),
        f"dit_{tag}_keyframes_mask": kf.numpy(),
        f"dit_{tag}_video_out": v_out.float().cpu().numpy(),
        f"dit_{tag}_audio_out": a_out.float().cpu().numpy(),
    }
    for k, v in out.items():
        np.save(GOLDENS / f"{k}.npy", v)
    meta_out = {"geometry": [F, H, W, NA], "fps": FPS, "sigma": SIGMA, "device": DEV}
    (GOLDENS / f"dit_{tag}_meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"saved {len(out)} arrays + dit_{tag}_meta.json")


if __name__ == "__main__":
    main()
