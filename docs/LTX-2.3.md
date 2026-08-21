# LTX-2.3 on MLX

**LTX-2.5 is the primary line — see the [repository README](../README.md).** 2.3 remains fully
supported: every pipeline works, the weights are published, and it is the comparison arm whenever a
2.5 measurement needs a control. This page is what is specific to it.

## Weights

| Variant | HuggingFace | Size | RAM |
|---|---|---|---|
| bf16 | [`xocialize/ltx-2.3-mlx`](https://huggingface.co/xocialize/ltx-2.3-mlx) | ~42 GB | 64 GB+ |
| int8 | [`xocialize/ltx-2.3-mlx-q8`](https://huggingface.co/xocialize/ltx-2.3-mlx-q8) | ~21 GB | 32 GB+ |
| int4 | [`xocialize/ltx-2.3-mlx-q4`](https://huggingface.co/xocialize/ltx-2.3-mlx-q4) | ~12 GB | 16 GB+ |

Granules for weight streaming: [`xocialize/ltx-2.3-granules`](https://huggingface.co/xocialize/ltx-2.3-granules).

```bash
ltx-2-mlx generate -p "a fox in snow" -m xocialize/ltx-2.3-mlx-q8 \
  --distilled -W 704 -H 512 -f 121 --frame-rate 24 -o out.mp4
```

## What 2.3 does NOT have

Everything under *What 2.5 adds* in the main README: the Gemma-4 text encoder, generated keyframe
slots (multishot), the DFR pipeline, the diffusion VAE decoder, and the duration head. 2.3 uses a
4-bit Gemma-3 text encoder, which is why its memory profile differs so much from 2.5's — 2.5's
encoder is an unquantized bf16 12B.

## Text encoder

2.3 loads its encoder from a **separate repo**
([`mlx-community/gemma-3-12b-it-4bit`](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit)),
where 2.5 ships its encoder **inside** the model tree as `gemma4-12b-ltx-v1/`. Anything that
resolves paths must not assume one layout.

## Quantization ladder

- **q8 ≈ bf16** — reproduces the bf16 sample; the faithful tier.
- **q4 diverges but is artifact-free** — a *different but equally valid* composition rather than a
  degraded one, because the larger per-step error compounds over the distilled trajectory. The lean
  tier: it trades reproducibility, not quality.

## Upscaler variants

All three ship and are usable: `spatial_upscaler_x2_v1_1` (the upstream default),
`spatial_upscaler_x1_5_v1_0`, and `temporal_upscaler_x2_v1_0` (fps doubling). The x1.5 and temporal
two-stage compositions are ours — upstream has no consumer for them.

⚠️ 2.5 ships **no x1.5 upscaler**; that variant is 2.3-only.

## Provenance

These weights are a verbatim mirror of [`dgrauet/ltx-2.3-mlx`](https://huggingface.co/dgrauet/ltx-2.3-mlx),
converted with [mlx-forge](https://github.com/dgrauet/mlx-forge) — byte-identical on every shared
file. dgrauet's tree carries additional artifacts (`transformer-dev`, `transformer-distilled-1.1`,
the distilled LoRAs) that this mirror does not.
