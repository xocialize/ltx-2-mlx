"""Convert PrunaAI/PrunaVAED (diffusers layout) to the ltx-core MLX video-decoder dialect.

PrunaVAED is a pruned, distillation-finetuned drop-in replacement for the LTX-2.3 video
VAE *decoder*. The encoder and latent format are unchanged, so only ``vae_decoder`` is
produced here.

It is NOT a pure width prune. Pruna keeps the *skip* width between blocks and prunes
*inside* each block, so two blocks gain a channel-adapter ResBlock (``conv_in``) that has
no counterpart in the stock decoder:

    diffusers up_blocks.1.conv_in : 512 -> 384
    diffusers up_blocks.2.conv_in : 384 -> 256

Those carry a 1x1x1 ``conv_shortcut`` and an *affine* ``norm3`` LayerNorm on the residual
path -- the stock decoder's only norms are parameterless PixelNorm, so ``norm3`` is the
one genuinely new parameter kind. See PrunaAI/PrunaVAED ``patch_diffusers.py``.

Key mapping (diffusers -> ltx-core flat up_blocks list, decoder only)::

    conv_in.conv.*                        -> conv_in.conv.*
    mid_block.resnets.{i}.*               -> up_blocks.0.res_blocks.{i}.*
    up_blocks.{n}.conv_in.*               -> up_blocks.{2n+1}.conv_in.*      [Pruna-only]
    up_blocks.{n}.upsamplers.0.conv.*     -> up_blocks.{2n+1}.conv.*
    up_blocks.{n}.resnets.{i}.*           -> up_blocks.{2n+2}.res_blocks.{i}.*
    conv_out.conv.*                       -> conv_out.conv.*
    latents_mean / latents_std            -> per_channel_statistics.{mean,std}

Conv weights are transposed from PyTorch ``(O, I, kT, kH, kW)`` to MLX
``(O, kT, kH, kW, I)``.

Usage::

    uv run python scripts/convert_pruna_vae_decoder.py \
        --src  /path/to/PrunaVAED/vae/diffusion_pytorch_model.safetensors \
        --config /path/to/PrunaVAED/vae/config.json \
        --out  /path/to/ltx-2.3-mlx/vae_decoder_pruna.safetensors \
        --reference /path/to/ltx-2.3-mlx/vae_decoder.safetensors
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx

PREFIX = "vae_decoder."


def derive_ladder(cfg: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the decoder ladder from config, mirroring Pruna's patched constructor.

    Returns the expected channel widths so every converted tensor can be shape-asserted
    instead of trusted. A checkpoint that disagrees fails loudly here rather than
    silently mis-mapping into a plausible-but-wrong architecture.
    """
    ch = tuple(reversed(cfg["decoder_block_out_channels"]))
    layers = tuple(reversed(cfg["decoder_layers_per_block"]))
    factor = tuple(reversed(cfg["upsample_factor"]))
    scaling = tuple(reversed(cfg["decoder_spatio_temporal_scaling"]))
    utype = tuple(cfg["upsample_type"])
    if len(utype) == len(ch) - 1:
        utype = tuple(reversed(utype))

    # (spatial_factor, temporal_factor) per upsample, matching the ltx-core convention.
    stride_of = {"spatial": (1, 2, 2), "temporal": (2, 1, 1), "spatiotemporal": (2, 2, 2)}

    width = ch[0]
    blocks: list[dict[str, Any]] = [
        {"kind": "res", "channels": width, "num_blocks": layers[0]},  # ltx-core up_blocks.0
    ]
    current = width
    for i in range(len(ch)):
        assert scaling[i], f"up_block {i} has no spatio-temporal scaling; unsupported"
        resnet_w = ch[i] // factor[i]
        pre = resnet_w * factor[i]
        st, sh, sw = stride_of[utype[i]]
        blocks.append(
            {
                "kind": "up",
                "in_channels": current,
                "adapter": None if current == pre else {"in": current, "out": pre},
                "conv_in": pre,
                "conv_out": (pre * st * sh * sw) // factor[i],
                "spatial_factor": sh,  # sh == sw for every LTX stride
                "temporal_factor": st,
            }
        )
        blocks.append({"kind": "res", "channels": resnet_w, "num_blocks": layers[i + 1]})
        current = resnet_w

    return {
        "blocks": blocks,
        "latent_channels": cfg["latent_channels"],
        "first_width": width,
        "final_width": current,
        "out_channels": cfg["out_channels"] * cfg["patch_size"] ** 2,
        "resnet_norm_eps": cfg["resnet_norm_eps"],
    }


def to_mlx_conv(w: mx.array) -> mx.array:
    """PyTorch (O, I, kT, kH, kW) -> MLX (O, kT, kH, kW, I)."""
    assert w.ndim == 5, f"expected a 5-D conv weight, got {w.shape}"
    return mx.transpose(w, (0, 2, 3, 4, 1))


def convert(src: Path, cfg: dict[str, Any]) -> tuple[dict[str, mx.array], list[str]]:
    raw = mx.load(str(src))
    ladder = derive_ladder(cfg)
    blocks = ladder["blocks"]
    out: dict[str, mx.array] = {}
    notes: list[str] = []

    # diffusers up_block n -> ltx-core flat indices (2n+1 = upsample, 2n+2 = res stage).
    def flat_up(n: int) -> int:
        return 2 * n + 1

    consumed: set[str] = set()

    def take(key: str) -> mx.array:
        consumed.add(key)
        if key not in raw:
            raise KeyError(f"missing expected tensor: {key}")
        return raw[key]

    def emit_conv(dst: str, src_key: str, expect: tuple[int, int]) -> None:
        w = to_mlx_conv(take(f"{src_key}.weight"))
        b = take(f"{src_key}.bias")
        o, i = expect
        assert w.shape == (o, 3, 3, 3, i), f"{src_key}: got {w.shape}, expected {(o, 3, 3, 3, i)}"
        assert b.shape == (o,), f"{src_key}.bias: got {b.shape}, expected {(o,)}"
        out[f"{PREFIX}{dst}.weight"] = w
        out[f"{PREFIX}{dst}.bias"] = b

    # --- conv_in -------------------------------------------------------------------
    emit_conv("conv_in.conv", "decoder.conv_in.conv", (ladder["first_width"], ladder["latent_channels"]))

    # --- flat up_blocks ------------------------------------------------------------
    for idx, blk in enumerate(blocks):
        if blk["kind"] == "res":
            # ltx-core index 0 is the diffusers mid_block; 2n+2 are the up-block resnets.
            src_stage = "decoder.mid_block" if idx == 0 else f"decoder.up_blocks.{(idx - 2) // 2}"
            c = blk["channels"]
            for j in range(blk["num_blocks"]):
                for which in (1, 2):
                    emit_conv(
                        f"up_blocks.{idx}.res_blocks.{j}.conv{which}.conv",
                        f"{src_stage}.resnets.{j}.conv{which}.conv",
                        (c, c),
                    )
        else:
            n = (idx - 1) // 2
            src_blk = f"decoder.up_blocks.{n}"
            if blk["adapter"]:
                a_in, a_out = blk["adapter"]["in"], blk["adapter"]["out"]
                emit_conv(f"up_blocks.{idx}.conv_in.conv1.conv", f"{src_blk}.conv_in.conv1.conv", (a_out, a_in))
                emit_conv(f"up_blocks.{idx}.conv_in.conv2.conv", f"{src_blk}.conv_in.conv2.conv", (a_out, a_out))
                # conv_shortcut is a plain nn.Conv3d(k=1) -- no causal/spatial padding.
                sw = take(f"{src_blk}.conv_in.conv_shortcut.weight")
                assert sw.shape == (a_out, a_in, 1, 1, 1), f"conv_shortcut: {sw.shape}"
                out[f"{PREFIX}up_blocks.{idx}.conv_in.conv_shortcut.weight"] = to_mlx_conv(sw)
                out[f"{PREFIX}up_blocks.{idx}.conv_in.conv_shortcut.bias"] = take(
                    f"{src_blk}.conv_in.conv_shortcut.bias"
                )
                # norm3: affine LayerNorm over the CHANNEL axis (last axis in MLX BFHWC).
                for p in ("weight", "bias"):
                    t = take(f"{src_blk}.conv_in.norm3.{p}")
                    assert t.shape == (a_in,), f"norm3.{p}: {t.shape}, expected {(a_in,)}"
                    out[f"{PREFIX}up_blocks.{idx}.conv_in.norm3.{p}"] = t
                notes.append(f"up_blocks.{idx}: channel adapter {a_in} -> {a_out} (norm3 + 1x1x1 shortcut)")
            emit_conv(
                f"up_blocks.{idx}.conv.conv",
                f"{src_blk}.upsamplers.0.conv.conv",
                (blk["conv_out"], blk["conv_in"]),
            )

    # --- conv_out + stats ----------------------------------------------------------
    emit_conv("conv_out.conv", "decoder.conv_out.conv", (ladder["out_channels"], ladder["final_width"]))
    out[f"{PREFIX}per_channel_statistics.mean"] = take("latents_mean")
    out[f"{PREFIX}per_channel_statistics.std"] = take("latents_std")

    leftovers = sorted(k for k in raw if k.startswith("decoder.") and k not in consumed)
    if leftovers:
        raise RuntimeError(
            "unconsumed decoder tensors -- the architecture has parts this converter does not "
            f"model:\n  " + "\n  ".join(leftovers)
        )

    return out, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="PrunaVAED vae/diffusion_pytorch_model.safetensors")
    ap.add_argument("--config", type=Path, required=True, help="PrunaVAED vae/config.json")
    ap.add_argument("--out", type=Path, required=True, help="destination vae_decoder_pruna.safetensors")
    ap.add_argument("--reference", type=Path, default=None, help="stock vae_decoder.safetensors to diff against")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    assert cfg["_class_name"] == "AutoencoderKLLTX2Video", f"unexpected class {cfg['_class_name']}"

    tensors, notes = convert(args.src, cfg)
    for n in notes:
        print(f"  [new] {n}")

    # MLX is lazy: unevaluated arrays serialize to safetensors as ZEROS, silently.
    mx.eval(list(tensors.values()))

    if args.reference is not None:
        ref = mx.load(str(args.reference))
        ours, theirs = set(tensors), set(ref)
        for p in ("mean", "std"):
            a, b = tensors[f"{PREFIX}per_channel_statistics.{p}"], ref[f"{PREFIX}per_channel_statistics.{p}"]
            delta = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
            print(f"  latent {p} vs stock: max|delta| = {delta:.3e}" + ("  [IDENTICAL]" if delta == 0 else ""))
        print(f"  keys: {len(ours)} pruna vs {len(theirs)} stock")
        print(f"  pruna-only: {len(ours - theirs)}   stock-only: {len(theirs - ours)}")
        shared = ours & theirs
        same = [k for k in shared if tensors[k].shape == ref[k].shape]
        print(f"  shared keys with identical shape: {len(same)}/{len(shared)}")

    params = sum(t.size for k, t in tensors.items() if "per_channel_statistics" not in k)
    print(f"  decoder params: {params / 1e6:.1f}M   tensors: {len(tensors)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out), tensors)
    print(f"  wrote {args.out}  ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
