# DiscoTope-style XGBoost: ESM3 embeddings vs ESM-IF1 embeddings

**Date:** 2026-06-19 · Runner: `run_bepipred_discotope.py` (`USE_ESMIF1=False`)

## Question
In the DiscoTope-3.0 recipe (per-residue embeddings + one-hot AA + RSA →
XGBoost-100 ensemble), how do **ESM3** embeddings compare to the **ESM-IF1**
embeddings that gave our best result?

## Setup (identical except the embedding backbone)
- Features: embeddings + one-hot(20) + RSA(1). ESM3=1536-dim, ESM-IF1=512-dim.
- Model: XGBoost ensemble of 100 (n_estimators=200, max_depth=4, lr=0.3,
  subsample=0.5), DiscoTope-3.0 pos/neg resampling.
- Same 3-fold BEPIPRED CV (holdout = test). ESM3 embeddings cached in
  `data/esm3_embed_cache/` (748/770 seqs; the rest exceed MAX_SEQ_LEN=512).

## Results (test ROC-AUC)

| Backbone (DiscoTope/XGBoost head) | fold1 | fold2 | fold3 | mean ± std |
|---|---|---|---|---|
| **ESM-IF1 (512), structure-native** | 0.746 | 0.802 | 0.738 | **0.762 ± 0.029** |
| ESM3 (1536) | 0.723 | 0.783 | 0.715 | 0.741 ± 0.030 |

## Conclusion
**ESM-IF1 embeddings beat ESM3 embeddings (+0.021) under the same classical
head**, despite ESM3 being 3× wider and a much larger model. Inverse-folding
representations (structure → sequence) appear better matched to epitope
prediction than ESM3's masked-LM-style embeddings.

Cross-head view of the ESM3 backbone:
- ESM3 + **LoRA neural head** = 0.754 (our `ls-rank8-blocks-48`)
- ESM3 + **XGBoost head** (this run) = 0.741
So for ESM3, the trainable LoRA head slightly beats frozen-embeddings+XGBoost.
For ESM-IF1 the opposite held (XGBoost 0.762 > LoRA-head 0.728): the value is in
the IF1 features, not in fine-tuning.

Overall project ranking unchanged: **ESM-IF1+XGBoost (0.762) > ESM3+LoRA (0.754)
> ESM3+XGBoost (0.741) > ESMC+LoRA (0.71)**. The common thread: more explicit
structural information → higher AUC.

## Follow-ups (untested)
- Concatenate ESM3 + ESM-IF1 embeddings into one XGBoost (complementary signal?).
- ESM3 embeddings WITH backbone coords vs sequence-only ESM3 embeddings, to
  isolate how much structure the cached ESM3 features actually carry.
