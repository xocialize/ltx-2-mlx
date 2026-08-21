# C4 length-degradation results (2026-08-12)

| elements | model | recall | decoy-FP |
|---:|---|---:|---:|
| 5 | 2.3 | 0.900 | 0.200 |
| 5 | 2.5 | **1.000** | 0.200 |
| 12 | 2.3 | 0.854 | 0.250 |
| 12 | 2.5 | **0.896** | 0.250 |
| 24 | 2.3 | 0.667 | 0.292 |
| 24 | 2.5 | **0.698** | 0.292 |

**[1] 2.5 scores higher at EVERY prompt length** (+0.100 / +0.042 / +0.031) — consistent
direction, consistent with the operator's C7 preference. Most defensible finding.

**[2] The slope claim is NOT supported.** "Holds complex prompts together instead of dropping
details across a longer sequence" predicts *slower* decay. 2.5 decays at least as fast: raw
−0.302 vs −0.233, and correcting for 2.5 starting at ceiling, relative retention is
**0.698 (2.5) vs 0.741 (2.3)**. Better everywhere, but not better *at holding up*.

**[3] ⚠️ The instrument is too noisy to settle [2].** The decoy control returns a **20–29%
false-positive rate** — same magnitude as the 2.3/2.5 gaps — and it **rises with checklist
length** (0.200 → 0.250 → 0.292), biasing precisely the slope [2] rests on.

Root cause in the per-element answers: some elements are generic truths the judge affirms
~100% regardless of content ("there is snow", "the setting is a forest", "warm light on her
from the lantern"); others read 0% for both models ("footprints are visible", "her breath is
visible as fog"). The checklist is bimodal rather than discriminating.

**Re-run fixes**: per-element decoys instead of one shared frame; judge prompted to cite visual
evidence before answering; screen checklist elements by pre-testing against unrelated frames and
drop any scoring >0; stronger judge.

The decoy control is why this is reported inconclusive rather than a clean "2.5 wins on prompt
adherence" — which the recall column alone would have supported.
