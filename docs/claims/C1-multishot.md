# Claim C1 — "native multishot generation"

**Status: mechanism VERIFIED, claim PARTIALLY MEASURED.** Generated keyframe
slots work end-to-end and cost ~10% wall-clock for 3 slots. But slots by
themselves did **not** produce multiple *shots* from a shot-listed prompt on the
distilled pipeline — that is DFR's job, and DFR is not yet ported.

Run: `uv run python scripts/measure_c1_multishot.py`
Receipts: `docs/claims/C1-multishot-report.json`, artifacts in `.work/c1/`.
Config: 512×768×97, distilled, seed 4242, LTX-2.5 bf16, 3 slots at evenly
spaced interior positions (pixel frames 24 / 48 / 72).

## What is established

**The mechanism is correct.** The decoded slots are coherent, correctly composed
frames of the generated scene — a fog-covered harbour with fishing boats and a
gull on a piling — at half resolution (384×256), because slots are stage-1 only,
matching upstream. Crucially, the slot at pixel frame 24 shows the gull *perched*
and the slot at frame 48 shows it *in flight*, which is exactly what the decoded
video does at those frames. The slots track the video's own timeline; they are
not re-rolls of the same instant, and not noise.

**Cost, on this box (numbers are objective):**

| arm | wall-clock | peak |
|---|---|---|
| baseline, 0 slots | 49.4 s | 40.76 GB |
| 3 slots | 54.7 s | 40.76 GB |

3 slots cost **+5.3 s (×1.107)** and **zero** extra peak memory. The peak is
unchanged because it is set by the 37.98 GB transformer weight load, not by
activations at this token count — so the slot cost lands entirely in time here.
That will not hold at resolutions where activations dominate; re-measure before
quoting the memory result anywhere else.

## What is NOT established, and why

**The prompt asked for three shots and got one continuous shot.** The prompt was
an explicit shot list ("Shot 1: wide establishing … Shot 2: close-up of hands …
Shot 3: low-angle gulls"). The generation is a single continuous harbour scene.
So on the distilled pipeline, slots are a **temporal-density mechanism, not a
shot-cutting mechanism** — they add a keyframe the model must satisfy at a
position, which is what DFR then densifies between. Reading "multishot" off this
run would be reading the wrong feature.

**The numeric position-alignment check is inconclusive, and is reported as such.**
Comparing each slot to the video frame at its own position vs another position
gave 2/3 "aligned" with margins of ~0.006 RMSE against a ~0.07 baseline. That is
not evidence: the scene is near-static, so every frame resembles every other
frame and RMSE cannot discriminate position at all. This is the
cannot-discriminate trap — a confident-looking number from a regime with no
signal. The *visual* check (gull perched vs in flight) discriminates cleanly and
is what the verdict rests on.

**Deliberately not measured: whether the shots are good.** C7's precedent is
explicit — sharpness/flicker proxies ranked 2.5 *worse* than 2.3, the opposite of
operator judgment, because a matched seed does not mean matched composition. A
"shot-boundary score" would fail the same way. That call needs a human.

## Owed

1. **Re-run on a high-motion prompt** so the position-alignment check lands in a
   regime that can actually discriminate. Until then that row stays inconclusive.
2. **Port the DFR pipeline** (`dfr_pipeline.py`) — it is the real consumer of
   slots and the actual multishot surface. Its canvas layout is already ported
   and bit-exact (`dfr_layout.py`, `scripts/gate_ltx25_dfr_layout.py`); what
   remains is the round/tile orchestration.
3. **Try the shot list through the prompt enhancer** (C5) — the enhancer is a
   separate generative Gemma checkpoint we do not host, so this is blocked on
   that, not on the port.
