"""Dump DiffVAE component goldens from the upstream torch decoder.

Builds the real decoder (COMBINED semantics, fp32, CPU) with our converted
weights and dumps, on small synthetic volumes:
  * one deterministic NABlock (det_stages.3.0, dim 512, kernel (3,5,5))
  * one upsample (upsamples.3)
  * one DiffusionNABlock (diff_blocks.0, dim 256, kernel (11,11,11))
  * the full stage-5 stack (all 8 diff blocks + head)

Run in the ref venv:
    .work/ltx25-ref/.venv/bin/python scripts/dump_ltx25_diffvae_golden.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
G = REPO / ".work/ltx25-ref/goldens"
SRC = Path("/Volumes/Satechi/Models/Lightricks/LTX-2.5/vae/ltx-2.5-video-vae-bf16.safetensors")
CFG = Path("/private/tmp/claude-501/-Volumes-Satechi-Development/ada6099d-cbd9-4eea-9386-4795fdfbfffb/scratchpad/ltx25/diffusion_decoder_config.json")
UP = "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-core/src"
sys.path.insert(0, UP)


def main() -> None:
    from safetensors.torch import load_file

    from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder

    c = json.loads(CFG.read_text())
    dec = DiffusionVideoDecoder(
        in_channels=c["latent_channels"], out_channels=c["out_channels"],
        patch_size=c["patch_size"], head_dim=c["decoder_head_dim"],
        t_emb_dim=c["decoder_t_emb_dim"],
        stage_channels=tuple(c["decoder_stage_channels"]),
        stage_depths=tuple(c["decoder_stage_depths"]),
        stage_kernels=tuple(tuple(k) for k in c["decoder_stage_kernels"]) + ((3, 3, 3),),
        upsamples=tuple((tuple(s), r) for s, r in zip(
            c["decoder_upsample_strides"], c["decoder_upsample_channel_reductions"])),
        stage5_kernel=tuple(c["decoder_stage5_kernel"]),
        timestep_scale_multiplier=c["decoder_timestep_scale_multiplier"],
        model_output_type=c["decoder_model_output_type"],
    )
    # Force the eager NA backend on every attention module. apply_diffvae_config
    # cannot be used here: its COMBINED branch hard-requires natten, which does
    # not exist on this host. The eager fallback computes the SAME full-volume
    # NA (shift-clamped windows) — it is the numerics we target; only the
    # chunked/deferred BLOCK mode would change semantics (W-chunk halo
    # replication), and we are not using it.
    from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
    from ltx_core.model.video_vae.transformer.fallback_na import EagerSdpaAttention

    n_patched = 0
    for m in dec.modules():
        if isinstance(m, NeighborhoodAttention3D):
            m.attention_function = EagerSdpaAttention()
            n_patched += 1
    print(f"  forced EagerSdpaAttention on {n_patched} NA modules")

    raw = load_file(SRC)
    sd = {}
    for k, v in raw.items():
        if not k.startswith("decoder."):
            continue
        nk = k[len("decoder."):]
        nk = nk.replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        nk = nk.replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        if nk.endswith((".qkv.weight", ".qkv.bias")):
            leaf = "weight" if nk.endswith(".weight") else "bias"
            base = nk[: -len(f"qkv.{leaf}")]
            d = v.shape[0] // 3
            # ⚠️ the module nests these under `attn.qkv.` (QKVProjections);
            # emitting `attn.to_q` silently leaves the model at RANDOM init
            # under strict=False. Always assert on missing/unexpected.
            sd[f"{base}qkv.to_q.{leaf}"] = v[:d].clone()
            sd[f"{base}qkv.to_k.{leaf}"] = v[d:2 * d].clone()
            sd[f"{base}qkv.to_v.{leaf}"] = v[2 * d:].clone()
        else:
            sd[nk] = v
    for k, v in raw.items():
        if k.startswith("per_channel_statistics."):
            sd[k] = v
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    # `type_emb` exists in the checkpoint but not in the reference decoder module
    # (unused by the decode path); everything else must match exactly.
    unexpected = [u for u in unexpected if u != "type_emb"]
    print(f"  loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        sys.exit(f"STATE DICT MISMATCH\n  missing={missing[:6]}\n  unexpected={unexpected[:6]}")
    dec = dec.to(torch.float32).eval()

    rng = np.random.default_rng(21)
    out = {}

    # A: one deterministic NABlock (stage 4, dim 512, kernel (3,5,5))
    blk = dec.det_stages[3][0]
    xa = rng.standard_normal((1, 4, 8, 8, 512)).astype(np.float32)
    with torch.no_grad():
        ya = blk(torch.tensor(xa))
    out["dv_nablock_in"], out["dv_nablock_out"] = xa, ya.numpy()

    # B: upsamples.3 (stride (2,2,2), 512 -> 256), with the leading-frame drop
    with torch.no_grad():
        yb = dec.upsamples[3](torch.tensor(xa), True)
    out["dv_upsample_out"] = yb.numpy()

    # C: one DiffusionNABlock + D: the whole stage-5 stack
    T5, H5, W5 = 12, 12, 12  # must be >= stage5 kernel (11,11,11): upstream guards
    ctx = rng.standard_normal((1, T5, H5, W5, 256)).astype(np.float32)
    xt = rng.standard_normal((1, 3, T5, H5 * 4, W5 * 4)).astype(np.float32)
    tt = torch.tensor([1.0])
    with torch.no_grad():
        cx = torch.cat([torch.tensor(ctx),
                        dec.conv_in_x_t(dec_patch := __import__("ltx_core.model.video_vae.ops",
                            fromlist=["patchify"]).patchify(torch.tensor(xt), patch_size_hw=4,
                            patch_size_t=1).permute(0, 2, 3, 4, 1))], dim=-1)
        t_emb = dec.t_embedder(dec.timestep_scale_multiplier * tt, hidden_dtype=torch.float32)
        mod = dec.shared_adaln(t_emb)
        one = dec.diff_blocks[0].forward_combined(cx, mod)
        full = dec.forward_diff_step(cx.clone(), tt)
    out["dv_diffblock_ctx"], out["dv_diffblock_xt"] = ctx, xt
    out["dv_diffblock_out"] = one.numpy()
    out["dv_stage5_out"] = full.numpy()

    # E: FULL end-to-end decode at the minimum legal latent (3,7,7).
    # Uses a fixed noise canvas so the MLX side can be fed the identical draw.
    lat = rng.standard_normal((1, 128, 3, 7, 7)).astype(np.float32)
    with torch.no_grad():
        padded = torch.tensor(lat)
        from ltx_core.model.video_vae.diffusion_tiling import (
            crop_trailing_context_natten_pad,
            pad_trailing_latent_for_natten_border,
        )
        # ghost-pad the RAW latent; forward_stages_1_to_3 un-normalizes internally
        z = pad_trailing_latent_for_natten_border(padded, dec._natten_trailing_pad_latent_frames)
        feat = dec.forward_stages_1_to_3(z, drop_leading_frame=True)
        ctx4 = dec.forward_stage_4(feat, drop_leading_frame=True)
        F5, H5, W5 = ctx4.shape[1], ctx4.shape[2] * dec.patch_size, ctx4.shape[3] * dec.patch_size
        noise = rng.standard_normal((1, 3, F5, H5, W5)).astype(np.float32)
        cx = torch.cat([ctx4, dec.conv_in_x_t(
            __import__("ltx_core.model.video_vae.ops", fromlist=["patchify"]).patchify(
                torch.tensor(noise), patch_size_hw=4, patch_size_t=1).permute(0, 2, 3, 4, 1))], dim=-1)
        pixels = dec.forward_diff_step(cx, torch.ones(1))
    out["dv_decode_latent"] = lat
    out["dv_decode_noise"] = noise
    out["dv_decode_context"] = ctx4.numpy()
    out["dv_decode_pixels"] = pixels.numpy()
    print(f"  E: latent {lat.shape} -> context {tuple(ctx4.shape)} -> pixels {tuple(pixels.shape)}")

    for k, v in out.items():
        np.save(G / f"{k}.npy", v)
    print("  saved:", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
