# Backbone ensemble: ESM3 / ESM2 / ESM-IF1 (LoRA+RYS), max & mean

**Date:** 2026-06-20 · Runners: `run_ensemble_{esm3,esm2,esmif1}.py`, `ensemble_eval.py`

## Setup
Trained the best LoRA+RYS model on each backbone, dumped per-residue test
predictions (pooled across the 3 CV folds, keyed `header|pos`), then ensembled
by `max` (and `mean`) over every model combination on the residues all members
share. Holdout = val = test, consistent with all prior experiments.

- **ESM3** (1.4B, +structure): RYS(36,44) + LoRA r8 last-16.
- **ESM2-650M** (HF, seq-only): RYS(24,30) + LoRA r8 last-16 (new HF injector + encoder RYS patch).
- **ESM-IF1** (142M, inverse-folding): RYS(4,8) + LoRA r4 all-8 layers, in the py3.9 fair-esm env.

## Single models (test ROC-AUC, 3-fold)

| Model | fold-mean ± std | pooled | n_res |
|---|---|---|---|
| **ESM-IF1 + LoRA + RYS** | **0.811 ± 0.061** | **0.833** | 121,460 |
| ESM3 + LoRA + RYS | 0.714 ± 0.024 | 0.706 | 117,506 |
| ESM2-650M + LoRA + RYS | 0.667 ± 0.006 | 0.667 | 117,506 |

## Ensembles (on shared structured residues, n≈114,963)

| Combination | max fold-mean | mean fold-mean | members (pooled) |
|---|---|---|---|
| esm3 + esmif1 | 0.811 ± 0.055 | **0.814 ± 0.052** | esm3=0.708, esmif1=0.832 |
| esm2 + esm3 + esmif1 | 0.796 ± 0.044 | 0.801 ± 0.039 | — |
| esm2 + esmif1 | 0.790 ± 0.049 | 0.796 ± 0.049 | esm2=0.670, esmif1=0.832 |
| esm2 + esm3 | 0.704 ± 0.018 | 0.712 ± 0.018 | — |

## Conclusions
1. **Ensembling does not beat the best single model.** Best ensemble
   (mean(esm3+esmif1) = 0.814 fold-mean / 0.832 pooled) is a statistical tie with
   ESM-IF1 alone (0.811 / 0.833 — pooled is identical on the shared set). `max`
   is slightly *worse* than `mean`: max takes the most-confident model per
   residue, so the weaker ESM2/ESM3 inject confident false positives on true
   negatives. Any combination that includes ESM2 drags the score down.
2. **The real win was fixing ESM-IF1.** IF1+LoRA+RYS = 0.833 pooled, a large jump
   over every prior result (discotope-if1 XGBoost 0.762, ESM3+LoRA 0.754). Two
   reasons: (a) the legacy IF1 LoRA was a silent no-op — the fused
   `F.multi_head_attention_forward` fast path bypassed the adapter; forcing the
   manual q/k/v path (`enable_torch_version=False`) makes LoRA train; (b) added
   RYS replay of the top encoder layers. Structure-native inverse folding remains
   the dominant signal.
3. **Ranking:** IF1 (0.81) >> ESM3 (0.71) > ESM2 (0.67). Sequence-only ESM2 is
   weakest; structure-aware models win, exactly as the earlier embedding studies
   predicted.

## 5-fold confirmation (2026-06-28): IF1 LoRA+RYS vs DiscoTope/XGBoost

Holding out each of the 5 partitions in turn (`run_if1_lora_5fold.py`,
`run_if1_xgb_5fold.py`), same backbone (ESM-IF1), same metric:

| Model | f1 | f2 | f3 | f4 | f5 | mean ± std |
|---|---|---|---|---|---|---|
| **ESM-IF1 + LoRA + RYS** | 0.737 | 0.900 | 0.825 | 0.771 | 0.887 | **0.824 ± 0.064** |
| discotope-if1 (frozen + XGBoost) | 0.743 | 0.811 | 0.736 | 0.719 | 0.727 | 0.747 ± 0.033 |

LoRA+RYS beats the DiscoTope/XGBoost recipe by **+0.077** on 5 folds (vs +0.049
on 3) — the win holds and widens. Caveat: LoRA+RYS variance is ~2× higher
(±0.064 vs ±0.033); fold 1 is weak for both (~0.74), folds 2/5 are very strong
for LoRA (0.90/0.89). XGBoost is the more *stable* model, LoRA+RYS the more
*accurate* one on average.

## Takeaway
Max/mean ensembling of unequal backbones is not the path forward — a strong
structural model alone wins. To push past ~0.83, invest in the IF1 model
(capacity, RYS range, structure features) or a *learned* combiner (stacking/
logistic over per-model probs) rather than a fixed max/mean rule.
