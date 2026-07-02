# ESMC (ESM Cambrian) + LoRA vs ESM3 + LoRA

**Date:** 2026-06-09 · **Commit:** added in `train_esmc.py` / `run_bepipred_esmc.py`

## Question
Does LoRA on the sequence-only **ESMC-600M** backbone beat our ESM3-based model?

## Setup
- ESMC-600M (1152-dim, 36 layers, 575M params), frozen + LoRA + LayerNorm→Linear head.
- ESMC shares ESM3's token vocab and block layout, so `id_map`, `LoRALinear`,
  `_inject_lora`, and the eval helpers were reused unchanged. **Sequence-only** —
  backbone coordinates are ignored (ESM3 uses them).
- Same 3-fold CV, LR (1e-3), batch (8), time budget (1200s). LoRA rank=8, alpha=8.

## Results (test ROC-AUC, mean ± std over 3 folds)

| Model | Config | fold1 | fold2 | fold3 | mean ± std |
|---|---|---|---|---|---|
| ESM3-sm (1.4B) +struct | LoRA r8 last-8 (`ls-rank-8`) | 0.738 | 0.713 | 0.715 | **0.722 ± 0.011** |
| ESM3-sm (1.4B) +struct | LoRA r8 all-48 (`ls-rank8-blocks-48`) | 0.743 | 0.781 | 0.737 | **0.754 ± 0.019** |
| ESMC-600M (seq-only) | LoRA r8 last-8 (`esmc-r8-last8`) | 0.722 | 0.768 | 0.637 | 0.709 ± 0.054 |
| ESMC-600M (seq-only) | LoRA r8 all-36 (`esmc-r8-all`) | 0.695 | 0.789 | 0.663 | 0.716 ± 0.053 |

## Conclusion
**ESMC-600M does not beat ESM3.** Matched last-8: 0.709 vs 0.722 (−0.013).
Best-vs-best: 0.716 vs 0.754 (−0.038). ESMC also shows ~3× higher fold variance
(±0.054 vs ±0.019), driven by a weak fold 3 (~0.65).

Plausible cause: ESMC-600M is ~2.4× smaller than ESM3-sm AND sequence-only,
whereas ESM3 also consumes antigen backbone coordinates — structural context
matters for (conformational) epitopes. Cheaper to train (~7 steps/s vs ~3), so
useful as a fast sequence-only baseline, but not a win on accuracy.

## Follow-ups (untested)
- ESMC + more LoRA capacity (rank 16) or +HiddenKey dropout (our ESM3 best recipe).
- ESMC as a *feature extractor* into the ESM-IF1+XGBoost track (best overall 0.762).
- Confirm whether structure (not just size) is the gap: ESM3 sequence-only ablation.
