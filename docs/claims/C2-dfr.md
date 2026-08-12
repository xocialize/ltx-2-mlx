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

## Quality — OPERATOR VERDICT (2026-08-12)

**Spatial detailing: WORTH IT.** The operator judged the DFR arm better —
specifically, lit windows on the lighthouse that the plain distilled arm did not
render, and a general read of "higher quality". Directional support for the
spatial half of C2 at ×1.279.

⚠️ **One pair cannot separate capability from roll.** The slot tokens divert the
trajectory from step one, so the two arms never share a composition — "DFR has
lights on the lighthouse" may be DFR rendering more, or may be DFR having rolled
a different scene. If it is a capability difference it should recur across seeds.
`scripts/measure_c2_spatial_seeds.py` runs the same A/B on seeds 11/22/33 for
exactly that check. Status until then: **directional, n=1 pair, one seed, one
observer, unblinded** — the same standing as the C7 verdict.

**Temporal rounds: NOT WORTH IT — the operator judged NATIVE better.** Combined
with native being 2.8× cheaper for the identical deliverable, rounds are
**dominated on both axes** at this clip length. Recommendation: do not use
`temporal_upsample_rounds` for 2-second 48 fps output. The one regime that could
still favour them is longer clips, where native high-fps generation runs out of
sequence length — untested.

Note this also settles the motion-regularity proxy against itself: it mildly
favoured DFR (std/mean 0.143 vs 0.198) and the operator went the other way. That
is the third time in this program a no-reference proxy has pointed opposite to
judgement (C7 sharpness/flicker, C1 position-RMSE, now this). Treat them as
hypothesis generators only.

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

## Temporal rounds — measured, and they are NOT an efficiency win

Rounds now work (audio-free denoise path shipped). Every arm below emits **97
frames** at 512×768, ffprobe-verified:

| arm | requested | out | playback fps | wall-clock | peak | vs native |
|---|---:|---:|---:|---:|---:|---:|
| distilled native | 97 | 97 | 24 | 50.2 s | 40.76 GB | 1.000× |
| DFR rounds=0 | 97 | 97 | 24 | 64.2 s | 41.66 GB | 1.279× |
| DFR rounds=1 | 49 | 97 | 48 | 138.9 s | 44.37 GB | 2.767× |
| DFR rounds=2 | 25 | 97 | 96 | 177.8 s | 44.37 GB | 3.542× |

**Read this carefully — the arms match on frame COUNT, not on duration.** Each
round maps N → 2(N−1)+1 at double the playback fps, so native is 4.0 s of 24 fps
video while rounds=2 is 1.0 s of 96 fps video. Same 97 frames, different clips.

**The decisive control settles it.** The deliverable of rounds=1 is *2 s at
48 fps*. Generating that natively — 97 frames with `frame_rate=48` — costs
**49.2 s / 40.75 GB**, versus **138.9 s** for DFR rounds=1. Same frame count,
same token count, same output duration and fps: **native is 2.8× cheaper.**

So on wall-clock, **temporal rounds are not an efficiency mechanism.** Reading
"more efficient generation" onto them is not supported by anything measured here.
The case for rounds has to be *quality* — staying inside the model's trained fps
distribution instead of asking it to lay out 8 pixel frames per latent token at a
compressed time base (the reason `_MAX_CONDITIONING_FPS` exists at all).

**That quality case is NOT settled here.** A per-frame motion-regularity proxy
mildly favours DFR (std/mean 0.143 vs 0.198; max/median 1.41 vs 1.67 — i.e.
native@48fps is a little more irregular, in the direction of the predicted
spike-then-stall). That is weak evidence from a no-reference proxy on clips with
different compositions, which is the trap C7 already sprung once. It is recorded
as a hint, not a verdict, and needs an operator A/B.

⚠️ **Muxing bug found while preparing the A/B, fixed:** `generate_and_save` wrote the
rounds output at the BASE frame rate, so a rounds=1 clip was a 4.04 s 24 fps
slow-motion version of a 2.02 s 48 fps result — right frames, wrong duration, no
error. Playback fps is `frame_rate * 2**rounds` and is decode-only; the
conditioning fps is a separate, capped quantity. The cost numbers above are
unaffected (they time generation, not muxing), but any A/B run before this fix
was comparing a slow-motion clip against a real-time one.

## Owed

1. Operator A/B: `distilled.mp4` vs `dfr.mp4` (spatial), and `dfr_rounds1.mp4`
   vs `native_48fps.mp4` (temporal) — the second is the one that decides whether
   rounds earn their 2.8×.
2. Longer clips: rounds may pay off where native high-fps generation runs out of
   sequence length, which is a regime this 97-frame test never enters.
