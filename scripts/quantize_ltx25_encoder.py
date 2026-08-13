"""Quantize the LTX-2.5 tuned Gemma-4 text encoder into sibling model trees.

Recipe decided in bridge record AB-D-0010: **int8 group-64 with ``embed_tokens``
kept bf16**. ``embed_tokens`` is excluded because state 00 IS
``embed_tokens(ids) * embed_scale`` — one of the 49 states the connector consumes
directly, and the input to all 48 layers.

Three gotchas, each of which fails SILENTLY (AB-D-0010):

1. Must run from ``ltx-2-mlx/.venv`` (git-main mlx-lm, commit 254d153f).
   Released 0.31.3 lacks the ``gemma4_unified`` → ``gemma4`` remap in
   ``mlx_lm/utils.py``, so the checkpoint resolves to the wrong architecture.
2. The default ``-q`` quantizes ``embed_tokens`` too, and the ``mixed_*``
   recipes are worse — with ``tie_word_embeddings: true`` there is no
   ``lm_head``, so ``mixed_3_6`` puts the embedding table at 3 bits.
3. ``save_config`` pops ``vision_config`` and ``tokenizer.save_pretrained``
   REWRITES ``tokenizer.json``. The 2.5 tokenizer's no-BOS ``post_processor``
   (``single: [Sequence A]``) is load-bearing — it is why ``Gemma4Encoder.tokenize``
   prepends BOS by hand. Both are restored from the source below, byte-verified.

Output goes to SIBLING trees (``ltx-2.5-mlx-<tag>/gemma4-12b-ltx-v1/``) because
``LTX2Pipeline.gemma4Dir`` hardcodes that basename and ``isLTX25`` keys off its
existence — a sibling tree therefore needs zero wiring changes.

Usage::

    .venv/bin/python scripts/quantize_ltx25_encoder.py q8 q4 poison
"""

from __future__ import annotations

import filecmp
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SRC = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/gemma4-12b-ltx-v1")
DST_ROOT = Path("/Volumes/Satechi/Models/xocialize")
ENCODER_BASENAME = "gemma4-12b-ltx-v1"

#: Rewritten by ``tokenizer.save_pretrained`` / ``save_config``; restored verbatim.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


@dataclass(frozen=True)
class Arm:
    """One quantization arm."""

    tag: str
    bits: int
    group: int
    quantize_embeddings: bool
    note: str


ARMS: dict[str, Arm] = {
    "q8": Arm("q8", 8, 64, False, "the decided recipe"),
    "q4": Arm("q4", 4, 64, False, "lean-tier candidate — gate decides, do not ship on assumption"),
    # POISON CONTROL. Deliberately bad on BOTH axes the recipe cares about: a
    # coarser group (128 vs 64) and the embedding table quantized. If this passes
    # the gate, the gate is not discriminating and no other number here means
    # anything. This project has been bitten four times by measurements taken in
    # a regime where they could not fail.
    "poison": Arm("poison", 4, 128, True, "POISON CONTROL — must FAIL or the gate proves nothing"),
}


def dst_dir(tag: str) -> Path:
    return DST_ROOT / f"ltx-2.5-mlx-{tag}" / ENCODER_BASENAME


def convert_arm(arm: Arm) -> None:
    """Convert one arm, then repair what ``save()`` clobbered."""
    from mlx_lm.convert import convert

    out = dst_dir(arm.tag)
    if out.exists():
        print(f"[{arm.tag}] already exists at {out} — skipping conversion")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    if arm.quantize_embeddings:
        # Quantize everything mlx can (Linear AND Embedding) — the poison.
        predicate = None
    else:

        def predicate(path: str, module) -> bool:  # noqa: ANN001 - mlx-lm's signature
            return "embed_tokens" not in path

    print(
        f"[{arm.tag}] converting: {arm.bits}-bit group-{arm.group}, "
        f"embed_tokens {'QUANTIZED' if arm.quantize_embeddings else 'excluded (bf16)'} "
        f"— {arm.note}"
    )
    t0 = time.time()
    convert(
        str(SRC),
        str(out),
        quantize=True,
        q_bits=arm.bits,
        q_group_size=arm.group,
        quant_predicate=predicate,
    )
    print(f"[{arm.tag}] convert took {time.time() - t0:.1f}s")

    restore_clobbered(arm)


def restore_clobbered(arm: Arm) -> None:
    """Copy back the tokenizer files and ``vision_config`` that ``save()`` dropped.

    Byte-verified: a silent rewrite of ``tokenizer.json`` would change the
    ``post_processor`` and re-introduce BOS, which the Swift/oracle tokenizers
    both compensate for by hand.
    """
    out = dst_dir(arm.tag)

    for name in TOKENIZER_FILES:
        src, dst = SRC / name, out / name
        if not src.exists():
            print(f"[{arm.tag}] WARN source missing: {name}")
            continue
        rewritten = dst.exists() and not filecmp.cmp(src, dst, shallow=False)
        shutil.copy2(src, dst)
        assert filecmp.cmp(src, dst, shallow=False), f"{name} did not restore byte-identically"
        print(
            f"[{arm.tag}] restored {name}"
            + (" (was REWRITTEN by save_pretrained)" if rewritten else " (was unchanged)")
        )

    src_cfg = json.loads((SRC / "config.json").read_text())
    dst_cfg = json.loads((out / "config.json").read_text())
    for key in ("vision_config",):
        if key in src_cfg and key not in dst_cfg:
            dst_cfg[key] = src_cfg[key]
            print(f"[{arm.tag}] restored config.{key} (popped by save_config)")
    (out / "config.json").write_text(json.dumps(dict(sorted(dst_cfg.items())), indent=4))


def audit(arm: Arm) -> dict:
    """Prove the recipe actually landed — key-level, not by trusting the call."""
    import mlx.core as mx

    out = dst_dir(arm.tag)
    shards = sorted(out.glob("*.safetensors"))
    total_bytes = sum(p.stat().st_size for p in shards)

    quantized_linears, embed_quantized, embed_dtype = 0, False, None
    for shard in shards:
        w = mx.load(str(shard))
        for k, v in w.items():
            if k.endswith(".scales"):
                if "embed_tokens" in k:
                    embed_quantized = True
                else:
                    quantized_linears += 1
            if k.endswith("embed_tokens.weight"):
                embed_dtype = str(v.dtype)
        del w

    cfg = json.loads((out / "config.json").read_text())
    qcfg = cfg.get("quantization_config") or cfg.get("quantization") or {}

    result = {
        "tag": arm.tag,
        "dir": str(out),
        "gb_on_disk": round(total_bytes / 1e9, 3),
        "shards": len(shards),
        "quantized_modules": quantized_linears,
        "embed_tokens_quantized": embed_quantized,
        "embed_tokens_dtype": embed_dtype,
        "config_bits": qcfg.get("bits"),
        "config_group_size": qcfg.get("group_size"),
        "vision_config_present": "vision_config" in cfg,
        "tokenizer_files_identical": all(
            filecmp.cmp(SRC / n, out / n, shallow=False) for n in TOKENIZER_FILES if (SRC / n).exists()
        ),
    }

    # The recipe must be what we asked for, not what the defaults gave us.
    assert result["config_bits"] == arm.bits, f"{arm.tag}: bits {result['config_bits']} != {arm.bits}"
    assert result["config_group_size"] == arm.group, f"{arm.tag}: group != {arm.group}"
    assert result["embed_tokens_quantized"] == arm.quantize_embeddings, (
        f"{arm.tag}: embed_tokens quantized={result['embed_tokens_quantized']}, "
        f"wanted {arm.quantize_embeddings} — the predicate did not take effect"
    )
    assert result["tokenizer_files_identical"], f"{arm.tag}: tokenizer files diverge from source"
    return result


def main() -> None:
    tags = sys.argv[1:] or ["q8"]
    src_bytes = sum(p.stat().st_size for p in SRC.glob("*.safetensors"))
    print(f"[src] {SRC} — {src_bytes / 1e9:.3f} GB\n")

    results = []
    for tag in tags:
        arm = ARMS[tag]
        convert_arm(arm)
        r = audit(arm)
        r["source_gb"] = round(src_bytes / 1e9, 3)
        r["saving_gb"] = round((src_bytes - r["gb_on_disk"] * 1e9) / 1e9, 3)
        results.append(r)
        print(f"[{tag}] AUDIT {json.dumps(r, indent=2)}\n")

    Path("scripts/.quant_arms.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
