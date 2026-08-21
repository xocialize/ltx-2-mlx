# ltx-2-mlx

Pure MLX port of [LTX-2](https://github.com/Lightricks/LTX-2) for Apple Silicon — a joint
**audio + video** diffusion transformer, running natively on Metal. Three-package monorepo
mirroring the reference structure: inference, pipelines, and training.

> **LTX-2.5 is the primary line.** The port is complete and parity-gated against the PyTorch
> reference. [LTX-2.3](docs/LTX-2.3.md) remains fully supported as the fallback and comparison arm.

## What 2.5 adds over 2.3

| | |
|---|---|
| **Gemma-4 text encoder** | `gemma4-12b-ltx-v1`, shipped **inside** the model tree. 49-state tap, parity-gated to mean cosine 0.999985. |
| **Generated keyframe slots** | Multi-shot generation in a single pass (`--num-generated-keyframes`). |
| **DFR pipeline** | Detail-First Rendering: half-res stage 1 → spatial upsample → stage-2 detailing, with optional temporal rounds. |
| **Diffusion VAE decoder** | A second decoder for the same latent, one x0 step from a noise canvas. |
| **Duration head** | Predicts shot duration in log-seconds, snapped to the causal frame grid. |
| **Ancestral stage-1 sampler** | η-scaled renoise on the first stage. |

## Measured, including what did not survive

Every vendor claim was measured rather than assumed. Reports and raw JSON live in
[`docs/claims/`](docs/claims/).

| Claim | Verdict |
|---|---|
| Multi-shot in one pass | ✅ **free** — no measurable cost over a single shot |
| Prompt enhancement | ✅ works; costs a **separate ~7 GB generative model** (the unified checkpoint is encode-only) |
| Duration head | ⚠️ **partial** — responds to pacing, not to explicit durations |
| DFR temporal rounds | ❌ **cost more, not less** — ×1.68 (r=1) / ×1.76 (r=2) vs native at matched output |
| Diffusion VAE decoder | ❌ **~10–12× slower, +24 GB, and less faithful** than the conv decoder |

The two ❌ rows are shipped and working — they are simply not worth using at these numbers, and we
would rather say so than let a feature list imply otherwise.

## Upstream gaps we work around

This port depends only on **released** packages — `uv sync` from a clean clone is enough. Two
upstream gaps are handled here rather than by pinning unreleased dependencies, and both are worth
knowing about if you read the code:

**`gemma4_unified` is not a model type released mlx-lm knows.** LTX-2.5's text encoder declares
`model_type: gemma4_unified`; mlx-lm registers the architecture as `gemma4`. Upstream added a
`MODEL_REMAPPING` entry for this, but it is not in any release (absent from v0.31.2 and v0.31.3,
present only on `main`). Rather than require a git dependency, `Gemma4LanguageModel.load` supplies
the mapping itself via `model_config={"model_type": "gemma4"}`. Remove it once a release carries the
mapping — nothing else depends on it.

**mlx-lm must be ≥ 0.31.2.** That is where `gemma4.py` first ships. On 0.31.0/0.31.1 the encoder
cannot load at all, so the floor is a hard requirement rather than a preference.

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- **ffmpeg** (video encoding)
- RAM: see *Weights* below. Weight streaming lowers the floor substantially.

## Installation

```bash
git clone https://github.com/xocialize/ltx-2-mlx.git
cd ltx-2-mlx
uv sync --all-extras
```

## Weights

LTX-2.5, published to `mlx-community`:

| Component | HuggingFace | Size |
|---|---|---|
| Base (bf16 DiT + encoder + VAEs + vocoder) | [`mlx-community/ltx-2.5-mlx`](https://huggingface.co/mlx-community/ltx-2.5-mlx) | ~100 GB |
| int8 DiT | [`mlx-community/ltx-2.5-mlx-ditq8`](https://huggingface.co/mlx-community/ltx-2.5-mlx-ditq8) | ~21 GB |
| int8 text encoder | [`mlx-community/ltx-2.5-mlx-q8`](https://huggingface.co/mlx-community/ltx-2.5-mlx-q8) | ~24 GB |
| Granules (weight streaming) | [`xocialize/ltx-2.5-granules`](https://huggingface.co/xocialize/ltx-2.5-granules) | ~59 GB |

⚠️ **The quantized DiT is `-ditq8`, not `-q8`.** On 2.5, `-q8` is the int8 **text encoder** — a
different axis. They can be mixed freely.

For 2.3 weights see [docs/LTX-2.3.md](docs/LTX-2.3.md).

## Quick Start

```bash
# Text-to-video with audio, LTX-2.5
ltx-2-mlx generate -p "a fox running down a beach at sunset, waves rolling in" \
  -m mlx-community/ltx-2.5-mlx --distilled \
  -W 704 -H 512 -f 121 --frame-rate 24 -o out.mp4

# Image-to-video — any mode accepts --image
ltx-2-mlx generate -p "the camera pushes in slowly" --image photo.jpg \
  -m mlx-community/ltx-2.5-mlx --distilled -W 704 -H 512 -f 121 --frame-rate 24 -o out.mp4

# int8 DiT — same components, quantized transformer
ltx-2-mlx generate -p "a fox in snow" -m mlx-community/ltx-2.5-mlx \
  --dit mlx-community/ltx-2.5-mlx-ditq8 --distilled \
  -W 704 -H 512 -f 121 --frame-rate 24 -o out.mp4
```

Frame counts follow an **8k+1** grid (9, 121, 161, 241 …); values in between are snapped.

### Python API

Pick the pipeline class matching your target — every public class
mirrors an upstream Lightricks/LTX-2 pipeline.

Two-stage (recommended for most use cases — dev model + CFG + upscale):

```python
from ltx_pipelines_mlx import TI2VidTwoStagesPipeline

pipe = TI2VidTwoStagesPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
pipe.generate_and_save(
    prompt="A sunset over the ocean with waves crashing",
    output_path="sunset.mp4",
    height=480,
    width=704,
    num_frames=97,
    seed=42,
    image="photo.jpg",  # optional I2V
)
```

For other modes:

- `DistilledPipeline` — fastest (distilled half-res + upscale).
- `TI2VidTwoStagesHQPipeline` — highest quality (res_2s + CFG + upscale).
- `TI2VidOneStagePipeline` — full-res CFG, no upscaler dependency.

Audio-to-Video:

```python
from ltx_pipelines_mlx import A2VidPipelineTwoStage

pipe = A2VidPipelineTwoStage(model_dir="dgrauet/ltx-2.3-mlx-q8")
pipe.generate_and_save(
    prompt="A musician performing",
    output_path="a2v.mp4",
    audio_path="music.wav",
)
```

Retake / Extend (single class — extend is folded into `RetakePipeline`):

```python
from ltx_pipelines_mlx import RetakePipeline

pipe = RetakePipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")

# Retake: regenerate latent frames 1-3
video_lat, audio_lat = pipe.retake_from_video(
    prompt="A different scene",
    video_path="source.mp4",
    start_frame=1,
    end_frame=3,
)

# Extend: add 2 latent frames after
video_lat, audio_lat = pipe.extend_from_video(
    prompt="Continue the motion",
    video_path="source.mp4",
    extend_frames=2,
    direction="after",
)
```

## CLI Reference

> **Full pipeline + flag matrix**: see [docs/PIPELINES.md](docs/PIPELINES.md) for a complete matrix of every CLI subcommand, the pipeline class behind it, supported sampler / model defaults, and which memory / perf flags apply where.

```
ltx-2-mlx generate   T2V / I2V / two-stage / HQ generation
  --prompt, -p        Text prompt (required)
  --output, -o        Output .mp4 path (required)
  --model, -m         Model weights (default: dgrauet/ltx-2.3-mlx-q8)
  --height, -H        Video height (default: 480)
  --width, -W         Video width (default: 704)
  --frames, -f        Number of frames (default: 97)
  --seed, -s          Random seed (-1 = random)
  --image, -i         Reference image for I2V
  --steps             Denoising steps for one-stage (default: 8)
  --two-stage         Enable two-stage pipeline (dev model + CFG)
  --two-stages-hq                Enable HQ pipeline (res_2s sampler)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30 standard, 15 HQ)
  --stage2-steps      Stage 2 steps (default: 3)
  --enhance-prompt    Enhance prompt with Gemma before generation
  --quiet, -q         Suppress progress output

ltx-2-mlx a2v        Audio-to-Video (two-stage, dev model + CFG)
  --audio, -a         Input audio file (required)
  --frame-rate        Output frame rate (required; LTX-2.3 trained at 24)
  --image, -i         Reference image for I2V (optional)
  --two-stages-hq                HQ mode (res_2s sampler for stage 1)
  --audio-start       Audio start time in seconds (default: 0)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30 standard, 15 HQ)
  --stage2-steps      Stage 2 steps (default: 3)

ltx-2-mlx retake     Regenerate a time segment (dev model + CFG)
  --video, -v         Source video file (required)
  --start             Start latent frame index (required)
  --end               End latent frame index (required)
  --steps             Denoising steps (default: 30)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)
  --no-regen-audio    Preserve original audio

ltx-2-mlx extend     Add frames before/after (dev model + CFG)
  --video, -v         Source video file (required)
  --extend-frames     Number of latent frames to add (required)
  --direction         "before" or "after" (default: after)
  --steps             Denoising steps (default: 30)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)

ltx-2-mlx keyframe   Keyframe interpolation (two-stage, dev model + CFG)
  --start             Start keyframe image (required)
  --end               End keyframe image (required)
  --frame-rate        Output frame rate (required; LTX-2.3 trained at 24)
  --cfg-scale         CFG scale (default: 3.0)
  --stg-scale         STG scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30)
  --stage2-steps      Stage 2 steps (default: 3)

ltx-2-mlx hdr-ic-lora HDR IC-LoRA (two-stage, LogC3 → linear HDR)
  --lora PATH STRENGTH       HDR LoRA (e.g. Lightricks/LTX-2.3-22b-IC-LoRA-HDR), repeatable
  --video-conditioning P S   Optional SDR ref video for V2V upgrade (omit for pure T2V)
  --image, -i                Optional I2V reference image
  --stage1-steps             Stage 1 steps (default: 8)
  --stage2-steps             Stage 2 steps (default: 3)
  --conditioning-strength    IC-LoRA attention strength (default: 1.0)
  --skip-stage-2             Skip upscale stage (half-res HDR output)
                  → saves <output>.mp4 + <output>.hdr.npz (fp32 (F,H,W,3) linear HDR)

ltx-2-mlx lipdub     [experimental] Lip-dub a reference video → audio
  --reference-video          Reference video providing visuals + target audio (required)
  --lora PATH STRENGTH       LipDub IC-LoRA (e.g. Lightricks/LTX-2.3-22b-IC-LoRA-LipDub), exactly one
  --reference-strength       Reference video conditioning strength (default: 1.0)
  --stage1-steps / --stage2-steps
                  Frame count auto-derived from the reference video (snapped to 8k+1)

ltx-2-mlx enhance    Prompt enhancement (no generation)
  --mode              "t2v" or "i2v" (default: t2v)

ltx-2-mlx info       Model info and memory estimate
```

### Environment variables

- `LTX2_GEMMA_EVAL_EVERY=N` — per-layer `mx.eval` cadence in the Gemma forward (default: `1`, i.e. eval every layer). Keeps each Metal command buffer below the macOS GPU watchdog (~10 s) deadline. Set to `0` on Mac Studio / M-series Ultra owners who never see the watchdog crash to recover full lazy-graph throughput.
- `LTX2_DIT_EVAL_EVERY=N` — flush the DiT block loop every N blocks (default: `8`, splits 48 blocks into 6 command buffers). Same trade-off as above; set to `0` on machines that don't crash to maximise throughput.
- `LTX2_GEMMA_MAX_LENGTH=N` — cap Gemma padded sequence length (default: `1024`). Last-resort knob; quality risk on lower values because left-padded RoPE positions drift outside the LTX training distribution.
- `LTX2_GEMMA_MAX_LENGTH=N` — cap padded Gemma sequence length (default 1024). Reducing to 512/256 speeds Gemma forward proportionally but **shifts left-padded RoPE positions** away from the LTX training distribution (quality risk). Last-resort knob.

## Frame Count Reference

The number of frames must be `8k + 1` (due to VAE temporal compression 8x). Common values at 24 fps:

| Frames | Duration | Latent frames | Notes |
|--------|----------|---------------|-------|
| 9 | 0.4s | 2 | Minimal, for quick tests |
| 25 | 1.0s | 4 | Short clip |
| 41 | 1.7s | 6 | |
| 49 | 2.0s | 7 | |
| 65 | 2.7s | 9 | |
| 81 | 3.4s | 11 | |
| 97 | 4.0s | 13 | **Default** |
| 121 | 5.0s | 16 | |
| 145 | 6.0s | 19 | |
| 161 | 6.7s | 21 | |
| 193 | 8.0s | 25 | Requires 64GB+ RAM |

Higher frame counts require more RAM. With int4 on 32GB, 97 frames at 512x320 is comfortable. Reduce resolution for longer videos.

## Packages

| Package | Description |
|---------|-------------|
| `ltx-core-mlx` | Model library: DiT, VAE, audio, text encoder, conditioning, guidance |
| `ltx-pipelines-mlx` | Generation pipelines: T2V, I2V, A2V, retake, extend, keyframe, two-stage |
| `ltx-trainer-mlx` | Training: LoRA fine-tuning with flow matching |

## Resources

- [LTX-2](https://github.com/Lightricks/LTX-2) — Lightricks reference (ltx-core + ltx-pipelines + ltx-trainer)
- [mlx-forge](https://github.com/dgrauet/mlx-forge) — weight conversion tool
- [LTX-2.5 weights](https://huggingface.co/mlx-community/ltx-2.5-mlx) · [LTX-2.3](docs/LTX-2.3.md)
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework

## License

**Two layers, and they differ.**

- **This port** — MIT (see `LICENSE`; copyright retained from the upstream
  [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) it is forked from).
- **The model weights** — [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2),
  which carries its own terms including a revenue gate and non-compete. Using or redistributing
  the weights binds you to them regardless of this repository's licence.
