"""Quantize the LTX-2.5 DiT to int8 group-64, replicating the LTX-2.3 q8 recipe exactly.

WHY THIS EXISTS
---------------
No quantized 2.5 DiT exists (AB-D-0013). It is the only lever that could bring
``compact24``/``balanced32`` back into range: with eviction on, peak is ~42-43 GB
*denoise*-bound on the bf16 DiT across the whole tested envelope (AB-R-0034/0035/0036),
so neither eviction nor encoder quant can reach those tiers. int4 is contraindicated
(weight-only int4 on a DiT is measurable quality damage that buys no speed, and 2.5's
int4 *encoder* already failed at 0.996728). q8 is the untested option.

⚠️ THIS IS AN EXPERIMENT, NOT AN EXPECTED WIN. On 2.3, per-forward doctrine said
"q8 ~= bf16" yet *end-to-end* at 121f it FAILED: q8-vs-bf16 pixel cosine 0.878
(q4 0.832) against a 0.9989 same-trajectory control. Per-forward faithfulness is not
trajectory faithfulness. And 2.3's ``--dit-q8-gate`` is a KNOWN FLAKY GATE (five runs
spanned 0.996388-0.999915), so a single cosine proves nothing -- repeat it and report
the spread.

THE RECIPE IS COPIED, NOT INVENTED
----------------------------------
Read off the shipped 2.3 q8 checkpoint rather than chosen by judgement:

  QUANTIZE  every ``transformer.transformer_blocks.N.*`` Linear ``.weight``
            -- attn1/attn2/audio_attn1/audio_attn2/audio_to_video_attn/
               video_to_audio_attn {to_q,to_k,to_v,to_out,to_gate_logits},
               ff.{proj_in,proj_out}, audio_ff.{proj_in,proj_out}
  KEEP bf16 everything outside the blocks -- the whole ``*adaln_single*`` family
            (incl. their ``emb.timestep_embedder.linear{1,2}``), ``patchify_proj``,
            ``proj_out``, ``audio_patchify_proj``, ``audio_proj_out``, and all norms.

⚠️ ``to_gate_logits`` IS quantized by the 2.3 recipe. Excluding it looks tempting --
0.09% of the parameters, and gate logits are disproportionately sensitive -- but
deviating would make the 2.5 result non-comparable to the 2.3 precedent, which is the
only reference point this experiment has. Replicate first; deviate only with a receipt.

Consumed with ZERO Swift changes: ``DiT.dense()`` detects a quantized Linear by the
presence of a ``.scales`` sibling and derives bits from the shapes
(``bits = weight_cols*32 / (scales_cols * groupSize)``), with groupSize fixed at 64.

Usage:
    .venv/bin/python scripts/quantize_ltx25_dit.py [--src DIR] [--out DIR] [--bits 8]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

SRC_DEFAULT = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
OUT_DEFAULT = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx-ditq8")
TRANSFORMER = "transformer-distilled.safetensors"
GROUP_SIZE = 64
BLOCK_PREFIX = "transformer.transformer_blocks."


def should_quantize(key: str, shape: list[int]) -> bool:
    """The 2.3 recipe: transformer-blocks-only, 2D Linear weights."""
    if not key.startswith(BLOCK_PREFIX):
        return False
    if not key.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    # MLX quantizes along the last (input) axis; it must divide the group size.
    return shape[-1] % GROUP_SIZE == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()

    src_tx = args.src / TRANSFORMER
    if not src_tx.exists():
        print(f"FAIL: {src_tx} not found", file=sys.stderr)
        return 2

    print(f"[dit-q{args.bits}] loading {src_tx}")
    t0 = time.time()
    w = mx.load(str(src_tx))
    print(f"[dit-q{args.bits}] loaded {len(w)} tensors in {time.time() - t0:.1f}s")

    out: dict[str, mx.array] = {}
    n_q = 0
    skipped_shape: list[str] = []
    t1 = time.time()
    for k, v in w.items():
        if should_quantize(k, list(v.shape)):
            wq, scales, biases = mx.quantize(v, group_size=GROUP_SIZE, bits=args.bits)
            out[k] = wq
            out[k[: -len("weight")] + "scales"] = scales
            out[k[: -len("weight")] + "biases"] = biases
            n_q += 1
        else:
            out[k] = v
            if k.startswith(BLOCK_PREFIX) and k.endswith(".weight") and len(v.shape) == 2:
                skipped_shape.append(k)
    mx.eval(list(out.values()))
    print(f"[dit-q{args.bits}] quantized {n_q} Linear modules in {time.time() - t1:.1f}s")
    if skipped_shape:
        # A block Linear that is NOT quantized is a recipe deviation -- surface it loudly
        # rather than letting it pass as an unnoticed bf16 island.
        print(f"[dit-q{args.bits}] ⚠️  {len(skipped_shape)} block Linear(s) skipped "
              f"(last dim not divisible by {GROUP_SIZE}): {skipped_shape[:5]}")

    # --- ACCEPTANCE: the recipe must match 2.3's families, not merely "have run" ---
    fams = {k.rsplit(".", 1)[0] for k in out if k.endswith(".scales")}
    leaf = sorted({f.split(".")[-1] for f in fams})
    expected = sorted(["to_q", "to_k", "to_v", "to_out", "to_gate_logits", "proj_in", "proj_out"])
    print(f"[dit-q{args.bits}] quantized leaf kinds: {leaf}")
    assert leaf == expected, f"recipe drift: {leaf} != {expected}"
    assert n_q == 48 * 34, f"expected 48 blocks x 34 Linears = 1632, got {n_q}"
    # Nothing outside the blocks may be quantized.
    stray = [f for f in fams if not f.startswith(BLOCK_PREFIX)]
    assert not stray, f"quantized outside transformer_blocks: {stray[:5]}"

    args.out.mkdir(parents=True, exist_ok=True)
    dst_tx = args.out / TRANSFORMER
    print(f"[dit-q{args.bits}] saving {dst_tx}")
    mx.save_safetensors(str(dst_tx), out)

    # Everything else symlinks to the parent tree -- same trick as the encoder sibling, so
    # `isLTX25` still keys off the in-dir gemma4 dir and no wiring changes.
    #
    # 🚨 DIRECTORIES ARE MIRRORED AS REAL DIRS OF FILE-SYMLINKS, NEVER AS A SYMLINKED DIR.
    # A symlinked `gemma4-12b-ltx-v1/` makes `#huggingFaceLoadModel` load an INCOMPLETE weight
    # set: the run dies with `MLXNN.UpdateError.keyNotFound` naming a DIFFERENT Gemma key each
    # time (`layers.0.post_attention_layernorm.weight`, then `model.norm.weight`) -- many keys
    # are missing and Swift's randomized dictionary iteration just reports a different one
    # first. The varying key is the tell: a genuine structural mismatch fails identically.
    # The encoder sibling never hit this because ITS gemma dir is a real directory (only the
    # transformer was symlinked) -- the opposite shape from this tree.
    for entry in sorted(os.listdir(args.src)):
        if entry == TRANSFORMER:
            continue
        src_entry = args.src / entry
        dst_entry = args.out / entry
        if dst_entry.exists() or dst_entry.is_symlink():
            continue
        if src_entry.is_dir():
            dst_entry.mkdir(parents=True, exist_ok=True)
            for child in sorted(os.listdir(src_entry)):
                child_link = dst_entry / child
                if not (child_link.exists() or child_link.is_symlink()):
                    os.symlink(src_entry / child, child_link)
        else:
            os.symlink(src_entry, dst_entry)

    src_gb = src_tx.stat().st_size / 1e9
    dst_gb = dst_tx.stat().st_size / 1e9
    print(f"[dit-q{args.bits}] disk {src_gb:.3f} -> {dst_gb:.3f} GB "
          f"({100 * (1 - dst_gb / src_gb):.1f}% smaller)")
    print(f"[dit-q{args.bits}] tree ready: {args.out}")
    (args.out / "QUANT-RECIPE.json").write_text(json.dumps({
        "source": str(src_tx), "bits": args.bits, "group_size": GROUP_SIZE,
        "scope": "transformer_blocks only (replicates the LTX-2.3 q8 recipe)",
        "quantized_modules": n_q, "disk_gb": round(dst_gb, 3),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
