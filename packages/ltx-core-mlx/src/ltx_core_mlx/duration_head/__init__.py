"""Duration head — prompt-conditioned shot-length prediction (LTX-2.5)."""

from ltx_core_mlx.duration_head.duration_head import (
    AttentionPooler,
    DurationHead,
    seconds_to_num_frames,
)

__all__ = ["AttentionPooler", "DurationHead", "seconds_to_num_frames"]
