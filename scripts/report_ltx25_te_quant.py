"""Full per-stage / per-state report for a quantized LTX-2.5 text encoder.

``scripts/gate_ltx25_te.py`` is the verdict — it ``assert``s at each of its four
stages and stops. That is right for a gate and wrong for a measurement: a failing
arm aborts at stage 2, so the number that actually decides the recipe (the
stage-4 connector-output cosine) is never produced for exactly the arms whose
behaviour we most need to characterise, including the poison control.

This runs the identical four stages against the identical torch goldens and
reports EVERY number without asserting: all 49 per-state cosines, the
pre-connector projections, and the full connector output. Use it alongside
``gate_ltx25_te.py``, not instead of it.

Why the per-state table rather than the mean: LTX consumes all 49 hidden states,
so the failure mode of interest is quantization error compounding with depth. A
mean hides monotone decay completely.

Usage::

    uv run --no-sync python scripts/report_ltx25_te_quant.py <goldens_dir> <model_dir> [tag]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

THRESHOLD = 0.999
PROMPT = "a cat playing piano, cinematic lighting"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    goldens = Path(sys.argv[1])
    model_dir = Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else model_dir.name

    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4LanguageModel
    from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
    from ltx_core_mlx.utils.weights import load_split_safetensors

    input_ids = np.load(goldens / "te_input_ids.npy")
    attn = np.load(goldens / "te_attention_mask.npy")
    golden = np.load(goldens / "te_hidden_states_fp32.npy")
    v_gold = np.load(goldens / "te_video_embeds_fp32.npy")
    a_gold = np.load(goldens / "te_audio_embeds_fp32.npy")
    assert np.abs(golden).max() > 0, "golden hidden states are zeros — re-dump"

    result: dict = {"tag": tag, "model_dir": str(model_dir), "threshold": THRESHOLD}

    enc = Gemma4LanguageModel(model_path=model_dir / "gemma4-12b-ltx-v1")
    enc.load()

    # --- stage 1: tokenization identity (exercises the RESTORED tokenizer files) ---
    my_ids, my_mask = enc.tokenize(PROMPT, max_length=input_ids.shape[1])
    ids_ok = bool((np.array(my_ids) == input_ids).all())
    mask_ok = bool((np.array(my_mask) == attn).all())
    result["tokenization_ids_exact"] = ids_ok
    result["tokenization_mask_exact"] = mask_ok
    print(f"[1/4] tokenization identity: ids={ids_ok} mask={mask_ok} {'PASS' if ids_ok and mask_ok else 'FAIL'}")

    # --- stage 2: all 49 hidden states, EVERY one reported ---
    states = enc.get_all_hidden_states(mx.array(input_ids), attention_mask=mx.array(attn))
    valid = attn[0].astype(bool)
    per_state = [cosine(np.array(s.astype(mx.float32))[0][valid], golden[i, 0][valid]) for i, s in enumerate(states)]
    result["per_state_cosine"] = per_state
    worst = min(per_state)
    result["states_worst"] = worst
    result["states_worst_index"] = int(np.argmin(per_state))
    result["states_mean"] = float(np.mean(per_state))
    result["states_below_threshold"] = [i for i, c in enumerate(per_state) if c < THRESHOLD]
    for i, c in enumerate(per_state):
        print(f"[2/4] state {i:02d} cosine={c:.6f}{'  <-- below threshold' if c < THRESHOLD else ''}")
    third = len(per_state) // 3
    result["depth_trend_shallow_mean"] = float(np.mean(per_state[:third]))
    result["depth_trend_deep_mean"] = float(np.mean(per_state[-third:]))
    print(
        f"[2/4] 49-state tap: mean {result['states_mean']:.6f} worst {worst:.6f} "
        f"at state {result['states_worst_index']}: {'PASS' if worst >= THRESHOLD else 'FAIL'}"
    )
    print(
        f"[2/4] depth trend: shallow-third {result['depth_trend_shallow_mean']:.6f} vs "
        f"deep-third {result['depth_trend_deep_mean']:.6f}"
    )

    fx = GemmaFeaturesExtractorV2()
    w = load_split_safetensors(model_dir / "connector.safetensors", prefix="connector.")
    fx.connector.load_weights(list(w.items()))

    # --- stage 3: pre-connector projections ---
    encoded = mx.stack(states, axis=-1)
    variance = mx.mean(encoded * encoded, axis=2, keepdims=True)
    normed = encoded * mx.rsqrt(variance + 1e-6)
    B, T, D, L = normed.shape
    stacked = (normed.reshape(B, T, D * L)) * mx.array(attn)[:, :, None].astype(normed.dtype)
    v_pre, a_pre = fx.connector.text_embedding_projection(stacked)
    result["proj_video"] = cosine(np.array(v_pre.astype(mx.float32))[0][valid], v_gold[0][valid])
    result["proj_audio"] = cosine(np.array(a_pre.astype(mx.float32))[0][valid], a_gold[0][valid])
    print(
        f"[3/4] pre-connector projections: video {result['proj_video']:.6f} "
        f"audio {result['proj_audio']:.6f}: "
        f"{'PASS' if min(result['proj_video'], result['proj_audio']) >= THRESHOLD else 'FAIL'}"
    )

    # --- stage 4: connector output ---
    #
    # ⚠️ The whole-array cosine that ``gate_ltx25_te.py`` reports CANNOT DISCRIMINATE, and
    # the reason is structural, not statistical. ``_replace_padding_with_registers`` re-sorts
    # to "valid tokens first, then registers", so with this 8-token prompt only positions
    # 0..7 of 1024 carry the prompt — the other 1016 are tiled ``learnable_registers``, which
    # are CONNECTOR weights and therefore bit-identical across every encoder quantization.
    # 99.2% of the compared tensor is fixed by construction, which drags every arm's cosine
    # toward 1.0 regardless of encoder quality.
    #
    # So the valid-position slice is reported alongside it, plus the register slice as the
    # control: if the register cosine is ~1.0 for an arm whose valid cosine is poor, the
    # whole-array number is provably dilution and not signal. This is the same check the
    # pad-token probe used — disable the thing that discards the error and confirm the
    # number moves — and it is why that probe is cited in CLAUDE.md.
    cv_gold = np.load(goldens / "connector_video_encoding_fp32.npy")
    ca_gold = np.load(goldens / "connector_audio_encoding_fp32.npy")
    v_emb, a_emb = fx(states, attention_mask=mx.array(attn))
    v_np = np.array(v_emb.astype(mx.float32))
    a_np = np.array(a_emb.astype(mx.float32))

    n_valid = int(attn.sum())
    result["n_valid_tokens"] = n_valid
    result["n_positions"] = int(cv_gold.shape[1])
    result["register_fraction"] = 1.0 - n_valid / cv_gold.shape[1]

    result["connector_video"] = cosine(v_np, cv_gold)
    result["connector_audio"] = cosine(a_np, ca_gold)
    result["connector_video_valid"] = cosine(v_np[:, :n_valid], cv_gold[:, :n_valid])
    result["connector_audio_valid"] = cosine(a_np[:, :n_valid], ca_gold[:, :n_valid])
    result["connector_video_registers"] = cosine(v_np[:, n_valid:], cv_gold[:, n_valid:])
    result["connector_audio_registers"] = cosine(a_np[:, n_valid:], ca_gold[:, n_valid:])

    diluted = min(result["connector_video"], result["connector_audio"])
    decisive = min(result["connector_video_valid"], result["connector_audio_valid"])
    result["connector_worst_whole_array"] = diluted
    result["connector_worst_valid"] = decisive

    result["verdict"] = (
        "PASS"
        if (
            decisive >= THRESHOLD
            and worst >= THRESHOLD
            and ids_ok
            and mask_ok
            and min(result["proj_video"], result["proj_audio"]) >= THRESHOLD
        )
        else "FAIL"
    )
    print(
        f"[4/4] connector, whole array ({result['n_positions']} pos, "
        f"{result['register_fraction']:.1%} registers): video {result['connector_video']:.6f} "
        f"audio {result['connector_audio']:.6f}  <- DILUTED, do not decide on this"
    )
    print(
        f"[4/4] connector, {n_valid} VALID positions: video {result['connector_video_valid']:.6f} "
        f"audio {result['connector_audio_valid']:.6f}: "
        f"{'PASS' if decisive >= THRESHOLD else 'FAIL'}"
    )
    print(
        f"[4/4] connector, register positions (control, ~1.0 => dilution confirmed): "
        f"video {result['connector_video_registers']:.6f} "
        f"audio {result['connector_audio_registers']:.6f}"
    )
    print(
        f"VERDICT [{tag}]: {result['verdict']}  (decisive valid-position connector cosine "
        f"{decisive:.6f}; whole-array would have read {diluted:.6f})"
    )

    result["encode_peak_gb"] = mx.get_peak_memory() / 1e9
    print(f"[mem] python encode peak {result['encode_peak_gb']:.2f} GB")

    out = Path("scripts/.te_report") / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
