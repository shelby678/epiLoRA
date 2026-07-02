# DiscoTope-style XGBoost: ESM3 + ESM-IF1 concatenated embeddings

**Date:** 2026-06-19 · Runner: `run_bepipred_discotope_combo.py`

## Question
Do ESM3 (masked-LM, 1536-dim) and ESM-IF1 (inverse-folding, 512-dim) embeddings
carry **complementary** signal? Concatenate both into one XGBoost.

## Setup
- Features: [ESM3(1536) | ESM-IF1(512) | one-hot(20) | RSA(1)] = 2069-dim.
- Same XGBoost-100 recipe + 3-fold CV. 724 seqs present & length-aligned in both
  caches (vs 748 ESM3-only / 770 total).

## Results (test ROC-AUC)

| Features | fold1 | fold2 | fold3 | mean ± std |
|---|---|---|---|---|
| ESM-IF1 only (512) | 0.746 | 0.802 | 0.738 | 0.762 ± 0.029 |
| ESM3 only (1536) | 0.723 | 0.783 | 0.715 | 0.741 ± 0.030 |
| **ESM3 + ESM-IF1 (2069)** | 0.744 | 0.809 | 0.738 | **0.764 ± 0.032** |

## Conclusion
**No meaningful complementarity.** Combined = 0.7637 vs ESM-IF1 alone 0.7621:
+0.0016, far inside the ±0.03 fold std — a statistical tie. Adding 1536 ESM3
dims on top of ESM-IF1 neither helps nor hurts; ESM-IF1 already captures the
useful structural signal and the extra ESM3 features are largely redundant for
this task. (Combined also slightly *trails* ESM-IF1 on fold 1, only gains on
fold 2.)

Practical takeaway: **stay with ESM-IF1 + XGBoost (0.762)** — simpler, 4× fewer
features, same accuracy. The concatenation is not worth the cost.

## Where this leaves the project
Best results, all within ~0.02 of each other:
- ESM-IF1 + XGBoost = 0.762  (recommended)
- ESM3 + ESM-IF1 + XGBoost = 0.764 (tie, not worth complexity)
- ESM3 + LoRA = 0.754
- ESM3 + XGBoost = 0.741
- ESMC + LoRA = 0.71
Gains now look saturated around ~0.76 for structure-based features; further
progress likely needs a different angle (data, labels, or task framing) rather
than another embedding backbone.
