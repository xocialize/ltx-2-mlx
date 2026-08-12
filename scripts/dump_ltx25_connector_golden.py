"""Dump the LTX-2.5 connector-stage golden via upstream's own modules.

Feeds the banked TE hidden-state goldens through upstream ltx-core's
FeatureExtractorV2 + Embeddings1DConnector×2 wrapped in EmbeddingsProcessor
(``process_hidden_states`` — extraction → right-pad re-sort → registers →
8-block connectors → mask re-application). Weights come from our converted
``connector.safetensors`` (bytes = the canonical comfy-bundle connectors).

Run in the ref venv:
    .work/ltx25-ref/.venv/bin/python scripts/dump_ltx25_connector_golden.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / ".work/ltx25-ref/goldens"
MODEL_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
sys.path.insert(0, "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-core/src")


def main() -> None:
    from safetensors.torch import load_file

    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
    from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2

    hs = np.load(GOLDENS / "te_hidden_states_fp32.npy")  # (49, 1, 1024, 3840)
    attn = torch.tensor(np.load(GOLDENS / "te_attention_mask.npy"))
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    states = tuple(torch.tensor(hs[i]).to(torch.bfloat16).to(DEV) for i in range(hs.shape[0]))

    w = load_file(MODEL_DIR / "connector.safetensors")

    def sub(prefix: str) -> dict[str, torch.Tensor]:
        return {k[len(prefix):]: v for k, v in w.items() if k.startswith(prefix)}

    video_proj = torch.nn.Linear(188160, 4096, bias=True)
    audio_proj = torch.nn.Linear(188160, 2048, bias=True)
    proj_sd = sub("connector.text_embedding_projection.")
    video_proj.load_state_dict({k[len("video_aggregate_embed."):]: v for k, v in proj_sd.items() if k.startswith("video_aggregate_embed.")})
    audio_proj.load_state_dict({k[len("audio_aggregate_embed."):]: v for k, v in proj_sd.items() if k.startswith("audio_aggregate_embed.")})

    extractor = FeatureExtractorV2(
        video_aggregate_embed=video_proj.to(torch.bfloat16),
        embedding_dim=3840,
        audio_aggregate_embed=audio_proj.to(torch.bfloat16),
    )

    # config: 2.5 connectors (diffusers connectors/config.json + 2.3-identical shape)
    common = dict(
        num_layers=8,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        causal_temporal_positioning=False,
        num_learnable_registers=128,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        apply_gated_attention=True,
        ff_bias=True,
    )
    video_conn = Embeddings1DConnector(attention_head_dim=128, num_attention_heads=32, **common)
    audio_conn = Embeddings1DConnector(attention_head_dim=64, num_attention_heads=32, **common)

    vc_missing, vc_unexpected = video_conn.load_state_dict(sub("connector.video_embeddings_connector."), strict=True), None
    audio_conn.load_state_dict(sub("connector.audio_embeddings_connector."), strict=True)
    print("connector weights loaded strict")

    proc = EmbeddingsProcessor(
        feature_extractor=extractor.to(DEV),
        video_connector=video_conn.to(torch.bfloat16).to(DEV),
        audio_connector=audio_conn.to(torch.bfloat16).to(DEV),
    )

    with torch.no_grad():
        out = proc.process_hidden_states(states, attn.to(DEV), padding_side="left")

    np.save(GOLDENS / "connector_video_encoding_fp32.npy", out.video_encoding.float().cpu().numpy())
    np.save(GOLDENS / "connector_audio_encoding_fp32.npy", out.audio_encoding.float().cpu().numpy())
    np.save(GOLDENS / "connector_out_mask.npy", out.attention_mask.cpu().numpy())
    print("saved connector goldens:",
          tuple(out.video_encoding.shape), tuple(out.audio_encoding.shape),
          "| out-mask valid:", int(out.attention_mask.sum()))


if __name__ == "__main__":
    main()
