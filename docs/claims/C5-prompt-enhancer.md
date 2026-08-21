# C5 prompt enhancer — "expands a short prompt into richer cinematic instructions at minimal extra compute"

## Measured (2026-08-12, oracle-side, M-series)

| item | value |
|---|---|
| enhancer model load (warm page cache) | 1.4 s |
| enhancement pass | **3.1 s** (7 words → 113 words) |
| total overhead | 4.5 s |
| as % of a 512×768×**9f** generation (14.1 s) | **32 %** |
| as % of a 512×768×**25f** generation (23.0 s) | ~20 % |

## Verdict: SUPPORTED at realistic clip lengths — and MY PREDICTION WAS WRONG

I predicted 15–30 s ("~600 greedy tokens through a 12B"), i.e. 5–10 % of a long run, and
flagged the claim as "suspect on-device". **The measurement falsified that by ~5×**: the pass
is 3.1 s. Recorded because a wrong prediction that gets quietly dropped is worse than never
having made one — the 4-bit quant and MLX's decode throughput are much better than I assumed.

The claim holds at production clip lengths (a 121-frame run is minutes, making 4.5 s ≈ 2–3 %).
It does *not* hold on very short clips, where overhead reaches ~32 % — but nobody generates
9-frame clips in production, so this is a measurement artifact, not a product concern.

## The real cost is architectural, not temporal

🔑 **LTX-2.5's own text encoder CANNOT enhance** — `gemma4_unified` is encode-only (confirmed:
our wrapper raises, matching upstream, which requires `--prompt-enhancer-gemma-root`). So
enhancement needs a **SECOND generative model resident** (we use Gemma-3-12B-it-4bit, ~7.5 GB).
That is a memory and deployment cost the "minimal extra compute" framing does not mention, and
it is the item that matters for a local-first product with a memory governor — not the 3 s.

## Caveats
- Load time was warm (the model was in page cache from C4 judging); a cold load is larger.
- Our path uses the Gemma-3 recipe (sampled, temp 0.7, ≤512 new tokens); the gemma4 recipe is
  greedy with ≤600 tokens, so the official enhancer may differ modestly.
- Quality delta (raw vs enhanced generations) not yet measured — that is the other half of C5.
