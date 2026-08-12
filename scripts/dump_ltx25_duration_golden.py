"""Dump the LTX-2.5 duration-head golden from the upstream torch module.

Feeds the banked connector-stage goldens through upstream's ``DurationHead``
with our converted weights, and saves the predicted seconds.

Run in the ref venv:
    .work/ltx25-ref/.venv/bin/python scripts/dump_ltx25_duration_golden.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / ".work/ltx25-ref/goldens"
SRC = Path("/Volumes/Satechi/Models/Lightricks/LTX-2.5/model_patches/ltx-2.5-duration-head-bf16.safetensors")
sys.path.insert(0, "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120/packages/ltx-core/src")


def main() -> None:
    from safetensors.torch import load_file

    from ltx_core.duration_head.duration_head import DurationHead

    head = DurationHead()
    sd = {k[len("duration_head."):]: v for k, v in load_file(SRC).items()}
    missing, unexpected = head.load_state_dict(sd, strict=True), None
    head = head.to(torch.float32).eval()
    print(f"  loaded {len(sd)} tensors strict")

    v = torch.tensor(np.load(GOLDENS / "connector_video_encoding_fp32.npy"))
    a = torch.tensor(np.load(GOLDENS / "connector_audio_encoding_fp32.npy"))

    with torch.no_grad():
        both = head(v, a)
        video_only = head(v, None)
        audio_only = head(None, a)

    out = {
        "duration_both": both.numpy(),
        "duration_video_only": video_only.numpy(),
        "duration_audio_only": audio_only.numpy(),
    }
    for k, val in out.items():
        np.save(GOLDENS / f"{k}.npy", val)
        print(f"  {k}: {float(val[0]):.6f} s")


if __name__ == "__main__":
    main()
