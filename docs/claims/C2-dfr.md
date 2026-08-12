# Claim C2 — "diffusion fidelity rendering" (DFR)

**Status: pipeline PORTED and running; cost MEASURED; quality is an operator call.**

Run: `uv run python scripts/measure_c2_dfr.py`
Receipts: `docs/claims/C2-dfr-report.json`, artifacts in `.work/c2/`.
Config: 512×768×97, seed 4242, LTX-2.5 bf16, both arms `generate_and_save`.

## Cost — DFR vs the plain distilled two-stage

| arm | wall-clock (incl. decode) | peak |
|---|---|---|
| distilled two-stage | 52.8 s | 40.76 GB |
| DFR (3 slots, rounds=0) | 62.0 s | 41.66 GB |

**DFR costs ×1.174 time and +0.90 GB peak.** Both arms run the same distilled
weights and the same 8+3 sigma schedule at the same resolution and seed, so the
delta is the DFR structure itself: slots in both stages, plus the seeded
re-append of the upsampled slots in stage 2. The canvas chose segment 32 with
slots at pixel frames 32 / 64 / 96 and needed **zero** padding at 97 frames.

⚠️ **The first version of this measurement reported +31 GB, and it was wrong.**
`generate_dfr` loaded the DiT *before* encoding the prompt, so the bf16 Gemma-4
12B text encoder (~24 GB) sat on top of the 38 GB transformer. Tracing peak
per-phase localised the jump to stage 1 — half-res, ~1536 tokens, where 32 GB is
impossible — and a plain-distilled control traced the same way showed the
identical jump, which is what proved it was not a DFR property. This is the same
load-ordering mistake commit `1a30f74` fixed for the other pipelines. Fixed;
re-measured at +0.90 GB. **A control arm is what turned a plausible headline
finding into a bug report.**

## Quality — deliberately not scored here

The decoded output is coherent and well composed (a lighthouse on a headland in a
storm, beam through rain). Beyond that this doc reports no quality verdict.

A no-reference gradient proxy reads DFR *lower* (0.0141 vs 0.0232 at frame 48),
and that number is **not** evidence DFR is softer: the two arms do not share a
composition — DFR's stage 1 carries extra slot tokens, so the trajectory diverges
from the first step and the seed no longer pins the same scene. This is exactly
the C7 proxy trap, where sharpness and flicker proxies ranked 2.5 worse than 2.3
and the operator judged the opposite. Recorded so nobody re-derives it as a
finding.

## Scope actually shipped

**Shipped:** stage 1 (half-res base + slots on the x8 segment grid) → spatial
latent upsample of video *and* slots → stage 2 full-res detailing with the slots
re-appended **seeded** by their upsampled stage-1 content.

**Not shipped: `temporal_upsample_rounds > 0` raises `NotImplementedError`.** The
round and tile orchestration IS ported — tiling, seam anchors at strength 0.95,
per-tile ancestral seeds, the stitch, and the carry-forward merge — and the
canvas geometry under it is gated bit-exactly. What is missing is an audio-free
denoise path: upstream runs round tiles with `audio=None`, and our `denoise_loop`
dereferences the audio state unconditionally. Inventing tile-local audio
semantics that upstream has no counterpart for, and that no gate could check,
would be worse than the stated gap.

**Also not shipped: the spatial detailing IC-LoRA.** `detailing_reference`
defaults to False, matching upstream — the reference is appended only when a
detailing LoRA is loaded, that LoRA is opt-in with no default, and we do not host
one. Without it the reference tokens are out of distribution, because the adapter
that teaches the model to read them is precisely what is absent.

## Owed

1. Audio-free denoise path → then the temporal rounds, which is the half of the
   C2 efficiency claim about densifying time rather than detailing space.
2. Operator A/B on `distilled.mp4` vs `dfr.mp4`.
3. Re-measure cost with rounds enabled; the ×1.174 above covers rounds=0 only.
