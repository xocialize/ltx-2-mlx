"""Convert the LTX-2.5 official split pack into the ltx-2-mlx weight layout.

Sources (comfy-aligned split pack + diffusers components), targets the same
per-component safetensors layout the LTX-2.3 conversion (mlx-forge) produced,
so the oracle's loaders work unchanged wherever the architecture is unchanged.

Verification strategy: LTX-2.5 re-ships several components byte-identical to
2.3 (conv video VAE, audio VAE + vocoder, both latent upscalers — receipted in
LTX_TESTING/LTX25-PORT-PLAN.md). For those, this converter converts the 2.5
source files and BIT-COMPARES the result against the existing dgrauet 2.3
conversion — validating this script against mlx-forge's behavior with free
fixtures. A mismatch is a converter bug, never accepted.

Usage (from the repo root, in the uv env):
    uv run python scripts/convert_ltx25.py --component upscalers
    uv run python scripts/convert_ltx25.py --component audio
    uv run python scripts/convert_ltx25.py --component vae
    uv run python scripts/convert_ltx25.py --component connector
    uv run python scripts/convert_ltx25.py --component te
    uv run python scripts/convert_ltx25.py --component duration
    uv run python scripts/convert_ltx25.py --component diffvae
    uv run python scripts/convert_ltx25.py --component dit --variant distilled
    uv run python scripts/convert_ltx25.py --component dit --variant dev
    uv run python scripts/convert_ltx25.py --component config
    uv run python scripts/convert_ltx25.py --component all
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import mlx.core as mx

SRC = Path("/Volumes/Satechi/Models/Lightricks/LTX-2.5")
DST = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
REF_23 = Path("/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx")


# ---------------------------------------------------------------------------
# safetensors helpers (raw header for metadata; mx.load for tensors)
# ---------------------------------------------------------------------------


def read_header(path: Path) -> tuple[dict, dict]:
    """Return (tensor_header, metadata) from a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {})
    return hdr, meta


def load_tensors(path: Path) -> dict[str, mx.array]:
    return dict(mx.load(str(path)))


def save_tensors(path: Path, tensors: dict[str, mx.array], metadata: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), tensors, metadata=metadata or {})
    mx.clear_cache()


def transpose_conv(v: mx.array, key: str = "") -> mx.array:
    """PyTorch conv layout -> MLX channels-last layout.

    Conv1d (O, I, K) -> (O, K, I); Conv2d (O, I, H, W) -> (O, H, W, I);
    Conv3d (O, I, D, H, W) -> (O, D, H, W, I). Linear/norm tensors pass through.

    ConvTranspose1d stores (I, O, K) in PyTorch and MLX wants (O, K, I), so the
    axis order differs: (1, 2, 0). BigVGAN's upsample stacks (`…ups.N.weight`)
    are the only transposed convs in the LTX family — detected by key.
    """
    is_conv_transpose = ".ups." in key and key.endswith(".weight")
    if v.ndim == 3:
        return mx.transpose(v, (1, 2, 0)) if is_conv_transpose else mx.transpose(v, (0, 2, 1))
    if is_conv_transpose:
        raise AssertionError(f"unexpected transposed-conv rank {v.ndim} for {key}")
    if v.ndim == 4:
        return mx.transpose(v, (0, 2, 3, 1))
    if v.ndim == 5:
        return mx.transpose(v, (0, 2, 3, 4, 1))
    return v


def bitcompare(converted: dict[str, mx.array], ref_path: Path, label: str) -> None:
    """Assert converted tensors are bit-identical to an existing reference file."""
    ref = load_tensors(ref_path)
    if set(converted) != set(ref):
        only_c = sorted(set(converted) - set(ref))[:8]
        only_r = sorted(set(ref) - set(converted))[:8]
        raise AssertionError(f"{label}: key sets differ.\n only converted: {only_c}\n only reference: {only_r}")
    bad = []
    for k in sorted(ref):
        a, b = converted[k], ref[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append(f"{k}: shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            continue
        if not mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item():
            bad.append(f"{k}: bytes differ")
        if len(bad) >= 8:
            break
    if bad:
        raise AssertionError(f"{label}: {len(bad)}+ tensor mismatches vs {ref_path.name}:\n  " + "\n  ".join(bad))
    print(f"  ✅ {label}: {len(ref)} tensors bit-identical to {ref_path.name}")


def apply_renames(key: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        key = key.replace(old, new)
    return key


# ---------------------------------------------------------------------------
# Components with a 2.3 bit-identity gate
# ---------------------------------------------------------------------------


def convert_upscalers() -> None:
    """Latent upscalers: bare keys -> versioned prefix, conv layout transposed.

    The 2.5-branded files are byte-identical to 2.3's (receipted), so outputs
    keep the 2.3 filenames the oracle already knows and are bit-compared.
    """
    jobs = [
        ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
         "spatial_upscaler_x2_v1_1.safetensors", "spatial_upscaler_x2_v1_1."),
        ("latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
         "temporal_upscaler_x2_v1_0.safetensors", "temporal_upscaler_x2_v1_0."),
    ]
    for src_rel, dst_name, prefix in jobs:
        raw = load_tensors(SRC / src_rel)
        out = {f"{prefix}{k}": transpose_conv(v) for k, v in raw.items()}
        bitcompare(out, REF_23 / dst_name, dst_name)
        save_tensors(DST / dst_name, out)
        print(f"  wrote {dst_name} ({len(out)} tensors)")
    # config sidecars are tiny json — copy from the 2.3 conversion (same models)
    for cfg in ("spatial_upscaler_x2_v1_1_config.json", "temporal_upscaler_x2_v1_0_config.json"):
        if (REF_23 / cfg).exists():
            shutil.copy2(REF_23 / cfg, DST / cfg)
            print(f"  copied {cfg}")


def convert_audio() -> None:
    """Audio bundle -> audio_vae.safetensors + vocoder.safetensors.

    2.5 bundles both under one file with hyphenated stats + doubled vocoder
    prefix; 2.3 split them. Emit the 2.3 split + names, bit-compare both.
    """
    raw = load_tensors(SRC / "vae/ltx-2.5-audio-vae-bf16.safetensors")
    audio: dict[str, mx.array] = {}
    voc: dict[str, mx.array] = {}
    for k, v in raw.items():
        if k.startswith("vocoder.vocoder."):
            nk = "vocoder." + k[len("vocoder.vocoder."):]
            voc[nk] = transpose_conv(v, nk)
        elif k.startswith("vocoder."):
            # bwe / mel-stft members keep single vocoder. prefix in 2.3? —
            # the 2.3 template decides; anything not vocoder.vocoder. maps 1:1
            voc[k] = transpose_conv(v, k)
        elif k.startswith("audio_vae."):
            nk = (k.replace(".per_channel_statistics.mean-of-means", ".per_channel_statistics._mean_of_means")
                   .replace(".per_channel_statistics.std-of-means", ".per_channel_statistics._std_of_means"))
            audio[nk] = transpose_conv(v)
        else:
            raise AssertionError(f"unexpected key in audio bundle: {k}")
    bitcompare(audio, REF_23 / "audio_vae.safetensors", "audio_vae")
    bitcompare(voc, REF_23 / "vocoder.safetensors", "vocoder")
    save_tensors(DST / "audio_vae.safetensors", audio)
    save_tensors(DST / "vocoder.safetensors", voc)
    print(f"  wrote audio_vae ({len(audio)}) + vocoder ({len(voc)})")


def convert_vae() -> None:
    """Conv video VAE -> vae_encoder.safetensors + vae_decoder.safetensors.

    2.5 conv file: encoder.* / decoder.* / top-level per_channel_statistics.
    2.3 layout: vae_encoder.* (stats as _mean_of_means/_std_of_means) and
    vae_decoder.* (stats as mean/std). Byte-identity gate against 2.3.
    """
    raw = load_tensors(SRC / "vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    enc: dict[str, mx.array] = {}
    dec: dict[str, mx.array] = {}
    stats: dict[str, mx.array] = {}
    for k, v in raw.items():
        if k.startswith("encoder."):
            enc["vae_encoder." + k[len("encoder."):]] = transpose_conv(v)
        elif k.startswith("decoder."):
            dec["vae_decoder." + k[len("decoder."):]] = transpose_conv(v)
        elif k.startswith("per_channel_statistics."):
            stats[k] = v
        else:
            raise AssertionError(f"unexpected key in conv VAE: {k}")
    # stats fan out into both files under different historical names
    mean = stats["per_channel_statistics.mean-of-means"]
    std = stats["per_channel_statistics.std-of-means"]
    enc["vae_encoder.per_channel_statistics._mean_of_means"] = mean
    enc["vae_encoder.per_channel_statistics._std_of_means"] = std
    dec["vae_decoder.per_channel_statistics.mean"] = mean
    dec["vae_decoder.per_channel_statistics.std"] = std
    bitcompare(enc, REF_23 / "vae_encoder.safetensors", "vae_encoder")
    bitcompare(dec, REF_23 / "vae_decoder.safetensors", "vae_decoder")
    save_tensors(DST / "vae_encoder.safetensors", enc)
    save_tensors(DST / "vae_decoder.safetensors", dec)
    print(f"  wrote vae_encoder ({len(enc)}) + vae_decoder ({len(dec)})")


# ---------------------------------------------------------------------------
# New / retrained components (template-checked, no bit gate possible)
# ---------------------------------------------------------------------------

DIT_RENAMES = [
    (".ff.net.0.proj.", ".ff.proj_in."),
    (".ff.net.2.", ".ff.proj_out."),
    (".audio_ff.net.0.proj.", ".audio_ff.proj_in."),
    (".audio_ff.net.2.", ".audio_ff.proj_out."),
    (".to_out.0.", ".to_out."),
    (".timestep_embedder.linear_1.", ".timestep_embedder.linear1."),
    (".timestep_embedder.linear_2.", ".timestep_embedder.linear2."),
    (".q_norm.", ".q_norm."),  # placeholder; norm names resolved below
]


def convert_dit(variant: str) -> None:
    """DiT comfy single-file -> transformer-<variant>.safetensors (oracle dialect).

    Validation: output key set must equal the 2.3 transformer template
    (from the dgrauet conversion) MINUS the 96 video-FF biases (ff_bias=false)
    PLUS keyframes_abs_pos_embedding. Any other delta aborts.
    """
    src = SRC / f"diffusion_models/ltx-2.5-22b-{variant}-transformer-bf16.safetensors"
    hdr, meta = read_header(src)
    print(f"  source: {src.name} ({len(hdr)} tensors)")

    # 2.5 comfy uses diffusers-style leaf names in places the oracle dialect
    # renames. norm naming: comfy `norm_q/norm_k` vs oracle `q_norm/k_norm`.
    renames = [
        (".ff.net.0.proj.", ".ff.proj_in."),
        (".ff.net.2.", ".ff.proj_out."),
        (".audio_ff.net.0.proj.", ".audio_ff.proj_in."),
        (".audio_ff.net.2.", ".audio_ff.proj_out."),
        (".to_out.0.", ".to_out."),
        (".timestep_embedder.linear_1.", ".timestep_embedder.linear1."),
        (".timestep_embedder.linear_2.", ".timestep_embedder.linear2."),
        (".norm_q.", ".q_norm."),
        (".norm_k.", ".k_norm."),
    ]

    raw = load_tensors(src)
    out: dict[str, mx.array] = {}
    bundled_connector: dict[str, mx.array] = {}
    for k, v in raw.items():
        if not k.startswith("model.diffusion_model."):
            raise AssertionError(f"unexpected DiT key prefix: {k}")
        bare = k[len("model.diffusion_model."):]
        # The comfy pack bundles the embeddings connectors (+ projections)
        # inside the transformer file; 2.3 conversion split them into
        # connector.safetensors. Set aside, verify identity below, drop here.
        if bare.startswith(("video_embeddings_connector.", "audio_embeddings_connector.",
                            "text_embedding_projection.")):
            bundled_connector[bare] = v
            continue
        nk = "transformer." + apply_renames(bare, renames)
        if nk in out:
            raise AssertionError(f"rename collision: {nk}")
        out[nk] = v  # DiT is Linear/norm/table-only — no conv transposes

    if bundled_connector:
        print(f"  dropped {len(bundled_connector)} bundled connector tensors "
              f"(connector.safetensors is built from this same bundle)")

    # Template validation against the 2.3 key set
    ref_hdr, _ = read_header(REF_23 / "transformer-distilled.safetensors")
    ref_keys = set(ref_hdr)
    new_keys = set(out)
    expected_gone = {k for k in ref_keys if (".ff.proj_in.bias" in k or ".ff.proj_out.bias" in k) and ".audio_ff." not in k}
    expected_new = {"transformer.keyframes_abs_pos_embedding"}
    unexplained_missing = (ref_keys - new_keys) - expected_gone
    unexplained_added = (new_keys - ref_keys) - expected_new
    if unexplained_missing or unexplained_added:
        raise AssertionError(
            f"DiT template mismatch.\n missing (unexplained, {len(unexplained_missing)}): "
            f"{sorted(unexplained_missing)[:10]}\n added (unexplained, {len(unexplained_added)}): "
            f"{sorted(unexplained_added)[:10]}"
        )
    print(f"  template OK: -{len(expected_gone)} video-FF biases, +keyframes_abs_pos_embedding")

    dst = DST / f"transformer-{variant}.safetensors"
    save_tensors(dst, out, metadata={k: v for k, v in meta.items() if isinstance(v, str)})
    print(f"  wrote {dst.name} ({len(out)} tensors)")


def convert_connector(projections: dict[str, mx.array] | None = None) -> None:
    """Comfy-bundled connectors -> connector.safetensors (oracle dialect).

    ⚠️ SOURCE CHOICE (2026-08-12): the comfy transformer files bundle the
    embeddings connectors, and those weights DIFFER from the diffusers
    connectors repo (learnable_registers hash e60b2b7c vs 5fc5081c — two
    training snapshots; the comfy pack is one day newer). The ltx-pipelines
    reference runtime loads connector blocks FROM the transformer file, so
    the comfy bundle is canonical for parity. Do not source from
    diffusers/connectors/ — it is the odd one out.

    Bundle keys are already in the 2.3 dialect (q_norm/k_norm, to_out.0
    Sequential kept, diffusers ff names kept): mapping is a prefix add.
    Projections come from the TE with-proj file (its aggregate_embed pair).
    Dev and distilled bundles are spot-verified identical.
    """
    dit_distilled = SRC / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    dit_dev = SRC / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    pref = "model.diffusion_model."

    def bundle_keys(path: Path) -> list[str]:
        hdr, _ = read_header(path)
        return sorted(k for k in hdr if "_embeddings_connector." in k)

    keys = bundle_keys(dit_distilled)
    raw = load_tensors(dit_distilled)
    out: dict[str, mx.array] = {}
    for k in keys:
        out["connector." + k[len(pref):]] = raw[k]

    # dev-vs-distilled identity spot check (6 tensors incl. registers)
    dev_raw = None
    probes = keys[:2] + keys[len(keys) // 2 : len(keys) // 2 + 2] + keys[-2:]
    dev_hdr, _ = read_header(dit_dev)
    import hashlib

    def _hash_at(path: Path, name: str) -> str:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            meta = hdr[name]
            st, en = meta["data_offsets"]
            f.seek(8 + n + st)
            return hashlib.sha256(f.read(en - st)).hexdigest()

    for k in probes:
        if _hash_at(dit_distilled, k) != _hash_at(dit_dev, k):
            raise AssertionError(f"dev/distilled connector bundles differ at {k} — per-variant connectors, re-derive")
    print(f"  {len(probes)} dev/distilled bundle identity probes OK")
    del raw
    mx.clear_cache()

    if projections is None:
        te_path = SRC / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
        te_hdr, _ = read_header(te_path)
        proj_keys = [k for k in te_hdr if k.startswith("text_embedding_projection.")]
        te_raw = load_tensors(te_path)
        projections = {k: te_raw[k] for k in proj_keys}
        del te_raw
        mx.clear_cache()
    for k, v in projections.items():
        out[f"connector.{k}"] = v

    # Template: exact match with the 2.3 connector layout (arch unchanged)
    ref_hdr, _ = read_header(REF_23 / "connector.safetensors")
    ref_keys, new_keys = set(ref_hdr), set(out)
    if ref_keys != new_keys:
        raise AssertionError(
            f"connector template mismatch.\n only 2.3: {sorted(ref_keys - new_keys)[:12]}\n"
            f" only 2.5: {sorted(new_keys - ref_keys)[:12]}"
        )
    for k in sorted(ref_keys):
        if tuple(out[k].shape) != tuple(ref_hdr[k]["shape"]):
            raise AssertionError(f"connector shape mismatch {k}: {out[k].shape} vs {ref_hdr[k]['shape']}")
    print(f"  template OK: {len(out)} tensors, keys+shapes exactly the 2.3 layout (weights retrained)")
    save_tensors(DST / "connector.safetensors", out)
    print("  wrote connector.safetensors (source: comfy bundle + TE projections)")


def convert_te() -> None:
    """with-proj checkpoint -> HF-format Gemma-4 dir loadable by mlx_lm.load().

    - comfy-flat LM keys -> HF names (model.language_model.*)
    - vision/audio tower tensors dropped (mlx-lm sanitize strips them anyway)
    - tokenizer.json extracted from the uint8 tensor
    - hf_asset__* sidecars extracted; config.json from `gemma_config` metadata
    - text_embedding_projection.* returned to the connector step via file scan
    """
    src = SRC / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    hdr, meta = read_header(src)
    te_dir = DST / "gemma4-12b-ltx-v1"
    te_dir.mkdir(parents=True, exist_ok=True)

    # config.json from metadata
    gemma_config = json.loads(meta["gemma_config"])
    (te_dir / "config.json").write_text(json.dumps(gemma_config, indent=2))
    print(f"  config.json: model_type={gemma_config.get('model_type')} "
          f"gemma_version={gemma_config.get('gemma_version')}")

    raw = load_tensors(src)

    # sidecar assets stored as uint8 tensors: `tokenizer_json` -> tokenizer.json,
    # `hf_asset__<filename>` -> <filename> (keys carry the extension)
    sidecar_keys = {k for k in raw if k == "tokenizer_json" or k.startswith("hf_asset__")}
    for key in sorted(sidecar_keys):
        fname = "tokenizer.json" if key == "tokenizer_json" else key[len("hf_asset__"):]
        data = bytes(bytearray(raw[key].tolist()))
        (te_dir / fname).write_bytes(data)
        print(f"  extracted {fname} ({len(data)/1e6:.1f} MB)")

    lm: dict[str, mx.array] = {}
    dropped_towers = 0
    for k, v in raw.items():
        if k in sidecar_keys or k.startswith("text_embedding_projection."):
            continue
        if k.startswith(("vision_model.", "multi_modal_projector.", "audio_projector.")):
            dropped_towers += 1
            continue
        if k.startswith("model.layers."):
            lm["model.language_model.layers." + k[len("model.layers."):]] = v
        elif k.startswith("model.embed_tokens."):
            lm["model.language_model.embed_tokens." + k[len("model.embed_tokens."):]] = v
        elif k.startswith("model.norm."):
            lm["model.language_model.norm." + k[len("model.norm."):]] = v
        else:
            raise AssertionError(f"unmapped TE key: {k}")
    print(f"  LM tensors: {len(lm)} (dropped {dropped_towers} vision/audio tower tensors)")

    # structural receipts: 49-state tap + k_eq_v global layers
    n_layers = gemma_config["text_config"]["num_hidden_layers"]
    layer_types = gemma_config["text_config"]["layer_types"]
    for i, lt in enumerate(layer_types):
        has_v = f"model.language_model.layers.{i}.self_attn.v_proj.weight" in lm
        if lt == "full_attention" and has_v:
            raise AssertionError(f"layer {i} is full_attention but has v_proj (k_eq_v expected)")
        if lt == "sliding_attention" and not has_v:
            raise AssertionError(f"layer {i} is sliding_attention but lacks v_proj")
    print(f"  k_eq_v layout verified across {n_layers} layers")

    # shard to ≤5 GB files with an index (HF convention, mlx_lm handles both)
    save_tensors(te_dir / "model.safetensors", lm)
    print(f"  wrote {te_dir.name}/model.safetensors")


def convert_duration() -> None:
    """Duration head: strip prefix, split the fused nn.MultiheadAttention in_proj."""
    raw = load_tensors(SRC / "model_patches/ltx-2.5-duration-head-bf16.safetensors")
    out: dict[str, mx.array] = {}
    for k, v in raw.items():
        if not k.startswith("duration_head."):
            raise AssertionError(f"unexpected duration-head key: {k}")
        k = k[len("duration_head."):]
        if k.endswith((".in_proj_weight", ".in_proj_bias")):
            leaf = "weight" if k.endswith("in_proj_weight") else "bias"
            base = k.rsplit(".in_proj_", 1)[0]
            d = v.shape[0] // 3
            out[f"duration_head.{base}.q_proj.{leaf}"] = v[:d]
            out[f"duration_head.{base}.k_proj.{leaf}"] = v[d : 2 * d]
            out[f"duration_head.{base}.v_proj.{leaf}"] = v[2 * d :]
        else:
            out[f"duration_head.{k}"] = v
    save_tensors(DST / "duration_head.safetensors", out)
    print(f"  wrote duration_head.safetensors ({len(out)} tensors)")
    for k in sorted(out):
        print(f"    {k}: {tuple(out[k].shape)}")


def convert_diffvae() -> None:
    """DiffVAE decoder -> vae_diffusion_decoder.safetensors.

    Replicates upstream load ops at conversion time (receipts:
    LTX25-REFERENCE-NOTES.md §C + upstream model_configurator.py:279-436):
      - keep only decoder.* from the comfy video-vae file (encoder comes from
        the conv file; stats pass through)
      - t_embedder.mlp.{0,2} -> t_embedder.timestep_embedder.linear{1,2}
        (oracle dialect: no underscore)
      - drop coarse_* (bundled preview head)
      - fold static gates (gate_msa->attn.proj, gate_mlp->mlp.w_down,
        gate_ctx->context_proj) into Linear weight+bias, then drop gates
      - split fused .qkv.{weight,bias} -> to_q/to_k/to_v
      - conv tensors -> MLX channels-last
    """
    src = SRC / "vae/ltx-2.5-video-vae-bf16.safetensors"
    raw = load_tensors(src)
    dec: dict[str, mx.array] = {}
    stats: dict[str, mx.array] = {}
    for k, v in raw.items():
        if k.startswith("decoder."):
            dec[k[len("decoder."):]] = v
        elif k.startswith("per_channel_statistics."):
            stats[k] = v
        elif k.startswith("encoder."):
            continue  # provided by the conv-VAE conversion
        else:
            raise AssertionError(f"unexpected key in DiffVAE file: {k}")

    renames = [
        ("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear1."),
        ("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear2."),
    ]
    gate_folds = [
        (".attn.proj.weight", ".gate_msa"), (".attn.proj.bias", ".gate_msa"),
        (".mlp.w_down.weight", ".gate_mlp"), (".mlp.w_down.bias", ".gate_mlp"),
        (".context_proj.weight", ".gate_ctx"), (".context_proj.bias", ".gate_ctx"),
    ]
    gates = {k: v for k, v in dec.items() if k.endswith((".gate_msa", ".gate_mlp", ".gate_ctx"))}
    out: dict[str, mx.array] = {}
    folded_n = split_n = dropped_coarse = 0
    for k, v in dec.items():
        if k.startswith("coarse_"):
            dropped_coarse += 1
            continue
        if k.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")):
            continue  # folded below, then dropped
        nk = apply_renames(k, renames)
        # gate fold (fp32 math, back to source dtype) — before layout transpose
        for leaf, gate_suffix in gate_folds:
            if nk.endswith(leaf):
                gk = nk[: -len(leaf)] + gate_suffix
                # gate keys were renamed too? gates dict uses pre-rename keys;
                # t_embedder rename never overlaps gate paths, so direct lookup:
                g = gates.get(gk)
                if g is not None:
                    gf = g.astype(mx.float32)
                    vf = v.astype(mx.float32)
                    v = (gf[:, None] * vf if v.ndim == 2 else gf * vf).astype(v.dtype)
                    folded_n += 1
                break
        if nk.endswith((".qkv.weight", ".qkv.bias")):
            leaf = "weight" if nk.endswith(".weight") else "bias"
            base = nk[: -len(f"qkv.{leaf}")]
            d = v.shape[0] // 3
            assert v.shape[0] % 3 == 0, f"fused qkv dim not /3: {nk} {v.shape}"
            out[f"vae_diffusion_decoder.{base}to_q.{leaf}"] = v[:d]
            out[f"vae_diffusion_decoder.{base}to_k.{leaf}"] = v[d : 2 * d]
            out[f"vae_diffusion_decoder.{base}to_v.{leaf}"] = v[2 * d :]
            split_n += 1
            continue
        out[f"vae_diffusion_decoder.{nk}"] = transpose_conv(v)
    for k, v in stats.items():
        # oracle-side stats names follow the decoder convention (mean/std)
        nk = (k.replace("per_channel_statistics.mean-of-means", "per_channel_statistics.mean")
               .replace("per_channel_statistics.std-of-means", "per_channel_statistics.std"))
        out[f"vae_diffusion_decoder.{nk}"] = v
    print(f"  folded {folded_n} gate targets, split {split_n} fused qkv, dropped {dropped_coarse} coarse tensors")
    save_tensors(DST / "vae_diffusion_decoder.safetensors", out)
    print(f"  wrote vae_diffusion_decoder.safetensors ({len(out)} tensors)")


def convert_config() -> None:
    """Extract embedded config from the DiT metadata; write config.json pair."""
    _, meta = read_header(SRC / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
    embedded = json.loads(meta["config"]) if "config" in meta else {}
    if embedded:
        (DST / "embedded_config.json").write_text(json.dumps(embedded, indent=2))
        print(f"  embedded_config.json: {list(embedded.keys())}")
    # oracle-facing config.json — 2.5 counterpart of the 2.3 file
    cfg23 = json.loads((REF_23 / "config.json").read_text())
    cfg = dict(cfg23)
    cfg.update(
        model_version="2.5.0",
        use_keyframes_abs_pos_embedding=True,
        ff_bias=False,
        audio_ff_bias=True,
    )
    (DST / "config.json").write_text(json.dumps(cfg, indent=2))
    print("  config.json written (2.3 base + 2.5 deltas)")
    # license + AUP travel with the conversion (LTX-2.x §3.2)
    for f in ("LICENSE.md", "ltx-acceptable-use-policy-snapshot-2026-08-12.pdf"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DST / f)
    print("  license + AUP copied")


COMPONENTS = {
    "upscalers": convert_upscalers,
    "audio": convert_audio,
    "vae": convert_vae,
    "connector": convert_connector,
    "te": convert_te,
    "duration": convert_duration,
    "diffvae": convert_diffvae,
    "config": convert_config,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component", required=True, choices=[*COMPONENTS, "dit", "all"])
    ap.add_argument("--variant", choices=["dev", "distilled"], help="dit only")
    args = ap.parse_args()

    DST.mkdir(parents=True, exist_ok=True)
    if args.component == "dit":
        if not args.variant:
            ap.error("--variant required for --component dit")
        print(f"[dit:{args.variant}]")
        convert_dit(args.variant)
    elif args.component == "all":
        for name, fn in COMPONENTS.items():
            print(f"[{name}]")
            fn()
        for variant in ("distilled", "dev"):
            print(f"[dit:{variant}]")
            convert_dit(variant)
    else:
        print(f"[{args.component}]")
        COMPONENTS[args.component]()


if __name__ == "__main__":
    main()
