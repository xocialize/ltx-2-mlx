"""LTX-2.5 text-encoder parity gate vs torch reference goldens.

Goldens are produced by ``.work/ltx25-ref/dump_te_golden.py`` (transformers
Gemma4Unified on MPS + upstream FeatureExtractorV2 — the exact implementation
the v1.2.0 reference wraps). This gate runs our Gemma-4 wrapper + extractor
on the identical tokens and checks:

  1. tokenization identity (ids + mask)
  2. all 49 hidden states, per-state cosine >= 0.999 on valid positions
     (adjudicates the post-final-norm tap convention, mask semantics,
     embed scale, dual-regime attention, k_eq_v)
  3. pre-connector projections (norm -> rescale -> aggregate_embed),
     cosine >= 0.999 vs the reference extractor+projection outputs

The post-connector stage (retrained weights, 2.3-identical code) has its own
golden TBD — this gate deliberately stops at the projection seam.

Usage:
    uv run --no-sync python scripts/gate_ltx25_te.py [goldens_dir] [model_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

GOLDENS = Path(sys.argv[1] if len(sys.argv) > 1 else ".work/ltx25-ref/goldens")
MODEL_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
PROMPT = "a cat playing piano, cinematic lighting"
THRESHOLD = 0.999


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4LanguageModel
    from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
    from ltx_core_mlx.utils.weights import load_split_safetensors

    input_ids = np.load(GOLDENS / "te_input_ids.npy")
    attn = np.load(GOLDENS / "te_attention_mask.npy")
    golden = np.load(GOLDENS / "te_hidden_states_fp32.npy")
    v_gold = np.load(GOLDENS / "te_video_embeds_fp32.npy")
    a_gold = np.load(GOLDENS / "te_audio_embeds_fp32.npy")
    assert np.abs(golden).max() > 0, "golden hidden states are zeros — re-dump (buffer seeding?)"

    enc = Gemma4LanguageModel(model_path=MODEL_DIR / "gemma4-12b-ltx-v1")
    enc.load()

    my_ids, my_mask = enc.tokenize(PROMPT, max_length=input_ids.shape[1])
    assert (np.array(my_ids) == input_ids).all() and (np.array(my_mask) == attn).all(), (
        "tokenization diverges from the reference"
    )
    print("[1/3] tokenization identity: PASS")

    states = enc.get_all_hidden_states(mx.array(input_ids), attention_mask=mx.array(attn))
    valid = attn[0].astype(bool)
    worst, worst_i = 1.0, -1
    for i, s in enumerate(states):
        c = cosine(np.array(s.astype(mx.float32))[0][valid], golden[i, 0][valid])
        if c < worst:
            worst, worst_i = c, i
    status = "PASS" if worst >= THRESHOLD else "FAIL"
    print(f"[2/3] 49-state tap: worst cosine {worst:.6f} at state {worst_i}: {status}")
    assert worst >= THRESHOLD

    fx = GemmaFeaturesExtractorV2()
    w = load_split_safetensors(MODEL_DIR / "connector.safetensors", prefix="connector.")
    fx.connector.load_weights(list(w.items()))
    encoded = mx.stack(states, axis=-1)
    variance = mx.mean(encoded * encoded, axis=2, keepdims=True)
    normed = encoded * mx.rsqrt(variance + 1e-6)
    B, T, D, L = normed.shape
    stacked = (normed.reshape(B, T, D * L)) * mx.array(attn)[:, :, None].astype(normed.dtype)
    v_pre, a_pre = fx.connector.text_embedding_projection(stacked)
    vc = cosine(np.array(v_pre.astype(mx.float32))[0][valid], v_gold[0][valid])
    ac = cosine(np.array(a_pre.astype(mx.float32))[0][valid], a_gold[0][valid])
    status = "PASS" if min(vc, ac) >= THRESHOLD else "FAIL"
    print(f"[3/3] pre-connector projections: video {vc:.6f} audio {ac:.6f}: {status}")
    assert min(vc, ac) >= THRESHOLD
    print("TE GATE: ALL PASS")


if __name__ == "__main__":
    main()
