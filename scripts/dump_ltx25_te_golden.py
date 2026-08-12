"""Dump the LTX-2.5 TE reference golden: 49 hidden states + extractor outputs.

Runs the transformers Gemma4Unified text stack (the exact implementation the
v1.2.0 reference wraps) on our converted HF-format dir, with the reference's
tokenization (strip + manual BOS + LEFT pad to 1024), output_hidden_states=True.
Then runs upstream ltx-core's FeatureExtractorV2 + projections for the
video/audio embeds golden.

Run in the ref venv:
    .venv/bin/python dump_te_golden.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

TE_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/gemma4-12b-ltx-v1")
OUT = Path(__file__).parent / "goldens"
OUT.mkdir(exist_ok=True)
PROMPT = "a cat playing piano, cinematic lighting"
MAX_LEN = 1024

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    config = AutoConfig.from_pretrained(TE_DIR)
    print("config:", config.model_type, getattr(config, "gemma_version", "?"))

    tok = AutoTokenizer.from_pretrained(TE_DIR)
    ids = tok.encode(PROMPT.strip(), add_special_tokens=False)
    bos = config.boa_token_id if False else 2  # text bos_token_id = 2
    if not ids or ids[0] != bos:
        ids = [bos, *ids]
    pad_id = 0
    pad = MAX_LEN - len(ids)
    input_ids = torch.tensor([[pad_id] * pad + ids])
    attn = torch.tensor([[0] * pad + [1] * len(ids)])
    np.save(OUT / "te_input_ids.npy", input_ids.numpy())
    np.save(OUT / "te_attention_mask.npy", attn.numpy())
    print(f"tokens: {len(ids)} valid, left-padded to {MAX_LEN}")

    print("building model on meta + loading weights ...")
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    model = model.to_empty(device="cpu")

    from safetensors.torch import load_file

    sd = load_file(TE_DIR / "model.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    # embedders for vision/audio were dropped in conversion; lm_head is tied
    real_missing = [
        m for m in missing
        if not m.startswith(("model.vision_embedder", "model.embed_audio", "model.embed_vision"))
        and m != "lm_head.weight"
    ]
    print(f"state dict: {len(sd)} loaded, missing(real)={real_missing[:6]}, unexpected={unexpected[:6]}")
    if real_missing:
        sys.exit(f"unexpected missing keys: {real_missing[:10]}")
    if hasattr(model, "lm_head") and model.lm_head.weight.device.type == "meta":
        model.lm_head.weight = torch.nn.Parameter(
            model.model.language_model.embed_tokens.weight.clone()
        )

    # --- seed non-persistent buffers (upstream encoder_configurator.py:362-400) ---
    # Meta-device construction leaves rotary inv_freq and embed_scale buffers
    # uninitialized (embed_scale=0 zeroes every hidden state — receipted).
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    l_model = model.model.language_model
    tcfg = config.text_config
    rope_emb = l_model.rotary_emb
    for layer_type in dict.fromkeys(tcfg.layer_types):
        rope_params = tcfg.rope_parameters[layer_type]
        if rope_params is None:
            continue
        rope_type = rope_params["rope_type"]
        if rope_type == "default":
            inv_freq, attn_scaling = rope_emb.compute_default_rope_parameters(tcfg, layer_type=layer_type)
        else:
            init_kwargs = {"layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                init_kwargs["head_dim_key"] = "global_head_dim"
            inv_freq, attn_scaling = ROPE_INIT_FUNCTIONS[rope_type](tcfg, **init_kwargs)
        rope_emb.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
        rope_emb.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
        setattr(rope_emb, f"{layer_type}_attention_scaling", attn_scaling)
    l_model.embed_tokens.register_buffer(
        "embed_scale", torch.tensor(tcfg.hidden_size**0.5, device="cpu"), persistent=False
    )
    print("seeded rotary + embed_scale buffers; embed_scale =", float(l_model.embed_tokens.embed_scale))

    model = model.to(torch.bfloat16).to(DEVICE).eval()

    with torch.no_grad():
        # reference base_encoder.py:68: model.model(input_ids, attention_mask,
        # output_hidden_states=True) — the inner model, skipping lm_head
        out = model.model(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attn.to(DEVICE),
            output_hidden_states=True,
        )
    hs = out.hidden_states
    print(f"hidden states: {len(hs)} × {tuple(hs[0].shape)} {hs[0].dtype}")

    stack = torch.stack([h.float().cpu() for h in hs], dim=0)  # (49, B, T, D)
    np.save(OUT / "te_hidden_states_fp32.npy", stack.numpy())
    print(f"saved te_hidden_states_fp32.npy {tuple(stack.shape)}")

    # ---- extractor + projections golden (upstream ltx-core, pure torch) ----
    sys.path.insert(0, "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-core/src")
    from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2

    from safetensors import safe_open

    te_file = Path("/Volumes/Satechi/Models/Lightricks/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    projs = {}
    with safe_open(te_file, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118
            if k.startswith("text_embedding_projection."):
                projs[k[len("text_embedding_projection."):]] = f.get_tensor(k)
    print("projection tensors:", {k: tuple(v.shape) for k, v in projs.items()})

    video_proj = torch.nn.Linear(188160, 4096, bias=True)
    audio_proj = torch.nn.Linear(188160, 2048, bias=True)
    video_proj.load_state_dict({"weight": projs["video_aggregate_embed.weight"], "bias": projs["video_aggregate_embed.bias"]})
    audio_proj.load_state_dict({"weight": projs["audio_aggregate_embed.weight"], "bias": projs["audio_aggregate_embed.bias"]})

    extractor = FeatureExtractorV2(
        video_aggregate_embed=video_proj.to(torch.bfloat16),
        embedding_dim=3840,
        audio_aggregate_embed=audio_proj.to(torch.bfloat16),
    )
    with torch.no_grad():
        v_emb, a_emb = extractor(
            hidden_states=tuple(h.cpu() for h in hs),
            attention_mask=attn,
        )
    np.save(OUT / "te_video_embeds_fp32.npy", v_emb.float().numpy())
    np.save(OUT / "te_audio_embeds_fp32.npy", a_emb.float().numpy())
    print("saved extractor+projection goldens:", tuple(v_emb.shape), tuple(a_emb.shape))


if __name__ == "__main__":
    main()
