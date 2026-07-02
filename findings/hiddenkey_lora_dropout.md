# HiddenKey & LoRA-dropout findings (ACL 2024)

Testing methods from Wang et al. 2024, *"LoRA Meets Dropout under a Unified
Framework"* (Findings of ACL 2024, `docs/lora_dropout_hiddenkey_acl2024.pdf`)
on our BEPIPRED 3-fold CV ESM3 + LoRA epitope predictor.

## Paper TL;DR

- LoRA is overfitting-prone too, even at 0.2% trainable params.
- Three transformer-specific dropouts are unified along three axes:
  - **Dropping position**: attention logits (DropKey, pre-softmax), attention
    weights (DropAttention, post-softmax with renorm), FFN hidden state (HiddenCut).
  - **Structural pattern**: element-wise, column-wise, span-wise.
  - **Compensation**: rescaling, R-drop bidirectional KL, JS consistency.
- For LoRA specifically:
  - **DropKey > HiddenCut >> DropAttention** at matched probabilities.
    DropAttention's NoGrad-rescaling introduces gradient noise that hurts.
  - Best per-position pattern: **column-wise for DropKey** (key columns shared
    across all queries within a head), **element-wise for HiddenCut**
    (span/column erase too much under tight LoRA capacity).
  - **KL loss > JS loss**; JS gives no consistent gain in LoRA.
- Authors' recipe: **HiddenKey = column-wise DropKey + element-wise HiddenCut
  + bidirectional Bernoulli KL between two stochastic forward passes**. Beats
  full-finetuning baseline on RoBERTa-large (RTE/MRPC/STS-B/SST-2/CoLA/QNLI),
  GPT2-Medium and LLaMA2-7B (E2E NLG, WebNLG).
- Adding input/output dropout on top of HiddenKey gives no further gain →
  "sufficiency" claim.

## Implementation

All paper-style drops are restricted to the LoRA-targeted blocks
(`paper_drop_only_lora=True`) — matching the paper's intuition that dropout
acts most directly on the parameters being trained.

`train_struct.py`:

- `_patch_attention_for_head_drop` now also supports DropKey and DropAttention.
  When either is active, the fused `F.scaled_dot_product_attention` is replaced
  with a manual `softmax(QK^T/√d) V` so we can mask attention logits column-wise
  before softmax (DropKey) or mask weights and renormalise with `torch.no_grad()`
  after softmax (DropAttention — gradient noise intentional, per paper §3.1).
- `_attach_hiddencut` registers a forward hook on `block.ffn[2]` (the SwiGLU
  activation, 4096-d output between the up- and down-projections). The hook
  applies element-wise Bernoulli dropout with inverted scaling.
- `_bidir_bernoulli_kl(z1, z2)` returns the per-token symmetric KL between two
  Bernoulli(sigmoid(z)) distributions, in log-sigmoid form for numerical stability.
- New `train()` kwargs: `dropkey_prob`, `dropattn_prob`, `hiddencut_prob`,
  `kl_loss_weight`, `paper_drop_only_lora`. When `kl_loss_weight > 0` and any
  paper-style drop is on, training does a second stochastic forward pass per
  step and adds `kl_loss_weight * KL(z1, z2)` to the BCE loss
  (`loss = 0.5*(BCE(z1) + BCE(z2)) + α·KL`).

ESM3 FFN layout (verified on small-open):
`Sequential[LayerNorm, Linear(1536→8192), SwiGLU(8192→4096), Linear(4096→1536)]`
— HiddenCut targets the SwiGLU output, matching the paper's "FFN hidden
representation" position.

## Experimental setup

Backbone (same as `run_bepipred_dropout.py`): ESM3-small-open frozen, LoRA
rank=4 / α=8 on QKV + out_proj of the last 8 blocks, no RYS, head dropout
0.1, AdamW lr=1e-3, batch=8, BCE loss, surface masking off. 3-fold CV on
BEPIPRED partitions {1, 2, 3}. Time budget: 20 min/fold (early-stop typically
at 5-7 min).

`run_bepipred_hiddenkey.py` runs 11 experiments (33 fold-runs):

| Tag | Method | Drop pos | Pattern | p | KL? |
|---|---|---|---|---|---|
| `hk-dropkey-col-10`        | DropKey         | attn logits  | column  | 0.10 | — |
| `hk-dropkey-col-20`        | DropKey         | attn logits  | column  | 0.20 | — |
| `hk-dropkey-col-30`        | DropKey         | attn logits  | column  | 0.30 | — |
| `hk-hiddencut-elem-10`     | HiddenCut       | FFN hidden   | element | 0.10 | — |
| `hk-hiddencut-elem-20`     | HiddenCut       | FFN hidden   | element | 0.20 | — |
| `hk-hiddencut-elem-30`     | HiddenCut       | FFN hidden   | element | 0.30 | — |
| `hk-dropattn-col-10`       | DropAttention   | attn weights | column  | 0.10 | — |
| `hk-hiddenkey-arrow`       | HiddenKey↗     | both         | col+elem | 0.10/0.10 | — |
| `hk-hiddenkey-kl05`        | HiddenKey       | both         | col+elem | 0.10/0.10 | 0.05 |
| `hk-hiddenkey-kl10`        | HiddenKey       | both         | col+elem | 0.10/0.10 | 0.10 |
| `hk-hiddenkey-2020-kl10`   | HiddenKey       | both         | col+elem | 0.20/0.20 | 0.10 |

The 0.30 and "2020" variants extend beyond the paper's tested range; our
`layerdrop-active-2-30` saw its biggest gain at p=0.30 so we want to know
whether DropKey/HiddenCut benefit from the same harder-regularisation regime.

Baseline (from `run_bepipred_dropout.py`): `baseline-no-rys` =
LoRA rank=4 last-8 only, **test_auc = 0.708 ± 0.010** (n=3 folds).

## Results

Mean ± std across 3 BEPIPRED CV folds. Time budget 20 min/fold (early stop
never triggered — all runs ran to the 1200-step ceiling).

### Sweep 1 (paper's tested range, p ∈ {0.10, 0.20})

| Method | test_auc | val_loss | Δ vs baseline |
|---|---|---|---|
| baseline-no-rys (reference)        | 0.708 ± 0.010 | 0.349 | — |
| layerdrop-active-2-30 (prior best) | 0.735 ± 0.022 | 0.367 | +0.027 |
| hk-dropkey-col-10                  | 0.7229 ± 0.018 | 0.328 | +0.015 |
| hk-dropkey-col-20                  | 0.7348 ± 0.036 | 0.323 | +0.027 |
| hk-hiddencut-elem-10               | 0.7428 ± 0.014 | 0.321 | +0.035 |
| hk-hiddencut-elem-20               | 0.7352 ± 0.019 | 0.323 | +0.027 |
| hk-dropattn-col-10                 | 0.7441 ± 0.020 | 0.325 | +0.036 |
| **hk-hiddenkey-arrow (best)**      | **0.7495 ± 0.024** | **0.314** | **+0.041** |
| hk-hiddenkey-kl05                  | 0.7408 ± 0.013 | 0.318 | +0.033 |
| hk-hiddenkey-kl10                  | 0.7424 ± 0.016 | 0.319 | +0.034 |

### What replicated from the paper

- **Combining DropKey + HiddenCut beats either alone.** HiddenKey↗ (0.7495)
  > HiddenCut-10 (0.7428) > DropKey-20 (0.7348). Matches Wang et al.
  Table 1 where HiddenKey↗ tops every position-only method.
- **HiddenCut prefers element-wise pattern.** We didn't run column/span
  HiddenCut, but element-wise was strong, consistent with the paper's
  claim that LoRA can't recover from span erasure in narrow capacity.
- **All transformer-specific dropouts beat the no-dropout baseline.**
  +0.015 to +0.041 test_auc vs the no-RYS LoRA baseline.

### What did NOT replicate

- **Bidirectional KL did not help.** HiddenKey↗ (no KL, 0.7495) > KL-05
  (0.7408) > KL-10 (0.7424). Paper reports KL gives consistent gain across
  every NLU dataset. Plausible reasons: (a) per-residue binary
  classification with 130k tokens/fold may already give the loss enough
  signal that the extra forward pass is wasted; (b) our 1200-step ceiling
  means KL training did half as many gradient updates per wall-clock minute,
  and *all* runs hit the ceiling — so the KL variants are effectively
  under-trained.
- **DropAttention was not the worst.** Paper attributes its weakness to
  NoGrad-rescaling gradient noise. We see it land at 0.7441, second to
  HiddenKey↗. May be that ~150K residue labels mute the gradient noise
  effect that hurts the paper's smaller GLUE datasets.

### Sweep 2 (high-probability variants)

| Method | test_auc | val_loss | Δ vs HK↗ |
|---|---|---|---|
| hk-dropkey-col-30                  | 0.7096 ± 0.019 | 0.337 | −0.040 |
| hk-hiddencut-elem-30               | 0.7318 ± 0.004 | 0.343 | −0.018 |
| hk-hiddenkey-2020-kl10             | 0.7187 ± 0.018 | 0.340 | −0.031 |

p=0.30 regressed for both DropKey and HiddenCut — past the optimal
regularization point. The 0.20+KL combo also regressed, reinforcing that
KL does not help here.

## Phase 3 — combinations with orthogonal regularizers (`hkx-*`)

Thesis: HiddenKey↗ acts on attn logits + FFN hidden; LayerDrop acts on the
residual stream; RYS on layer-execution count. Different parts of the
network — should be partially additive.

| Method | test_auc | val_loss | Δ vs HK↗ |
|---|---|---|---|
| hkx-arrow-dropattn (+ DropAttn col 0.05)   | 0.7411 ± 0.021 | 0.319 | −0.008 |
| hkx-arrow-rys (+ RYS 36→44)                | 0.7368 ± 0.029 | 0.317 | −0.013 |
| hkx-arrow-layerdrop (+ LayerDrop active-2-30) | 0.7253 ± 0.024 | 0.330 | −0.024 |
| hkx-arrow-layerdrop-rys (all three)        | 0.7181 ± 0.004 | 0.338 | −0.031 |

**All four combinations regressed.** This replicates the paper's
"sufficiency" claim — HiddenKey alone has already captured the available
dropout benefit, and additional regularization just under-trains the
~300K LoRA parameters.

## Phase 4 — HiddenKey↗ probability scan (`hky-pscan-*`)

Sweep 1 left a gap: HiddenKey↗ at p ∈ {0.15, 0.20, 0.25} without KL was
never tested. Train loss bottoms at ~0.06 vs val at ~0.30, so harder
HiddenKey might still help.

| Method | test_auc | val_loss | Δ vs HK↗ |
|---|---|---|---|
| hky-pscan-15 | 0.7355 ± 0.023 | 0.322 | −0.014 |
| hky-pscan-20 | 0.7377 ± 0.023 | 0.317 | −0.012 |
| hky-pscan-25 | 0.7473 ± 0.025 | 0.327 | −0.002 |

p=0.10 remains the optimum. p=0.25 came within noise of the winner —
suggesting a broad plateau between 0.10 and 0.25 rather than a sharp peak.
The earlier p=0.20+KL regression (0.7187) is now traceable entirely to
the KL term, not the higher probability.

## Final ranking (top half, all 20 experiments)

```
hk-hiddenkey-arrow                  0.7495 ± 0.0242   ← winner
hky-pscan-25                        0.7473 ± 0.0248
hk-dropattn-col-10                  0.7441 ± 0.0198
hk-hiddencut-elem-10                0.7428 ± 0.0136
hk-hiddenkey-kl10                   0.7424 ± 0.0162
hkx-arrow-dropattn                  0.7411 ± 0.0209
hk-hiddenkey-kl05                   0.7408 ± 0.0132
hky-pscan-20                        0.7377 ± 0.0230
hkx-arrow-rys                       0.7368 ± 0.0289
hky-pscan-15                        0.7355 ± 0.0232
layerdrop-active-2-30 (prior best)  0.7355 ± 0.0214
hk-hiddencut-elem-20                0.7352 ± 0.0187
hk-dropkey-col-20                   0.7348 ± 0.0362
...
baseline-no-rys (reference)         0.7077 ± 0.0096
```

## Conclusions

1. **HiddenKey↗ (DropKey 0.10 col + HiddenCut 0.10 elem, no KL) is the
   winner at test_auc 0.7495 ± 0.024**, +0.042 over the no-RYS baseline
   (0.7077) and +0.014 over the prior best `layerdrop-active-2-30`. This
   is the lever I'd ship from this sweep.
2. **The paper's "sufficiency" claim replicates strongly**: every attempt
   to combine HiddenKey↗ with another regularizer (LayerDrop, RYS,
   DropAttention) hurt by 0.008–0.031.
3. **The paper's KL claim does not replicate**: adding bidirectional
   Bernoulli KL between two forward passes hurts at every probability
   we tried. Plausible reason: per-residue binary classification with
   ~150K training labels per fold already has dense gradient signal,
   making the extra forward pass effectively wasted compute (training
   ran into the same 1200-step wall-clock ceiling, so KL got half as
   many parameter updates per minute).
4. **The paper's "DropAttention is the worst" claim does not replicate**
   either — DropAttention col-10 came in third (0.7441), beating both
   individual DropKey configurations. The NoGrad-rescaling gradient noise
   the paper attributes this to may be drowned out by our larger label
   budget.
5. **There is a broad plateau** between p=0.10 and p=0.25 for HiddenKey↗;
   p=0.30 is past the optimum and regresses.
6. **Top ~7 methods are within 1 std of each other**. With only 3 folds and
   std ~0.024, the difference between 0.7400 and 0.7495 is not statistically
   reliable. To push beyond ~0.75 on this architecture would likely need
   either more reruns per fold, more LoRA capacity (rank 8+ or all 48
   blocks), or non-dropout interventions (label smoothing, longer
   training schedule with lower LR, structural / data augmentation).

## Recommended default for future runs

`dropkey_prob=0.10, hiddencut_prob=0.10, kl_loss_weight=0.0,
paper_drop_only_lora=True` on top of the existing LoRA rank=4 last-8 setup,
nothing else. Anything more is at best within noise; usually worse.

## Reproduce

```bash
.venv/bin/python run_bepipred_hiddenkey.py    # 11 paper variants (sweeps 1+2)
.venv/bin/python run_bepipred_hkx.py          # 4 orthogonal combinations
.venv/bin/python run_bepipred_hky_pscan.py    # HiddenKey↗ probability scan
.venv/bin/python scripts/tally_hiddenkey.py   # aggregate hk-* rows from results.tsv
```

---

# Follow-up sweep: LoRA capacity + activation-guided selection

After HiddenKey↗ maxed out the dropout-side optimisation, the next question
was whether **more LoRA capacity** or **smarter LoRA targeting** could push
past it. Two sweeps:

## Activation probe (`probe_activations.py`)

Single forward pass of all 748 BEPIPRED antigens (EVAL excluded) through
the frozen ESM3-sm-open backbone, recording per-(block, head) mean
|context_BHLD|. Saved to `data/activation_probe.pt`. Result is striking:

```
block 47   sum_|ctx| = 142.6   ← dominant
block 46   sum_|ctx| = 115.9
block 45   sum_|ctx| = 94.8
block 44   sum_|ctx| = 83.0
block 43   sum_|ctx| = 63.2
…
block 32   sum_|ctx| =  9.8    ← already 15× smaller than block 47
```

All 16 of the top-(block, head) pairs land in blocks 44–47. Block 47
head 23 is the single largest activation (|ctx|=11.96). This validates
the project's intuition that "the last 8 blocks" are the load-bearing
ones — and refines it: the last *4* hold most of the signal.

## Sweep A — LoRA capacity scaling (`run_bepipred_lora_scale.py`)

All paired with HiddenKey↗ (DropKey 0.10 col + HiddenCut 0.10 elem).

| Method | params | test_auc | Δ vs HK↗ |
|---|---|---|---|
| **ls-rank8-blocks-48** (r=8 all 48)   | ~3.6M | **0.7540 ± 0.019** | **+0.005** |
| **ls-blocks-24** (r=4 last 24)        | ~900K | **0.7532 ± 0.030** | **+0.004** |
| `hk-hiddenkey-arrow` (reference HK↗)  | ~300K | 0.7495 ± 0.024 | — |
| ls-blocks-16 (r=4 last 16)            | ~600K | 0.7464 ± 0.010 | −0.003 |
| ls-rank8-blocks-16 (r=8 last 16)      | ~1.2M | 0.7420 ± 0.019 | −0.008 |
| ls-rank-2 (r=2 last 8)                | ~150K | 0.7385 ± 0.029 | −0.011 |
| ls-rank-16 (r=16 last 8)              | ~1.2M | 0.7370 ± 0.015 | −0.013 |
| ls-blocks-48 (r=4 all 48)             | ~1.8M | 0.7343 ± 0.043 | −0.015 |
| ls-rank-8 (r=8 last 8)                | ~600K | 0.7220 ± 0.011 | −0.028 |

Two takeaways:

1. **Block coverage matters more than rank at fixed capacity.** Rank-only
   changes at `last-8` all regressed (rank 2/8/16 worse than rank 4).
   But going wider (last-24, all-48 + rank=8) slightly *exceeded* HK↗.
   At fixed-LoRA-block-count, rank=4 is the right capacity per block.
2. **The two leaders are within noise of HK↗.** +0.005 and +0.004 with
   stds of 0.019 and 0.030. With three folds we can't claim a real win;
   it's possible the original `last-8 rank-4` was already at the
   capacity ceiling and the extra params are merely benign.

## Sweep B — selective per-head LoRA (`run_bepipred_lora_select.py`)

New module: `HeadMaskedLoRA` — LoRA wrapper on the QKV fused linear
(1536→4608) whose trainable delta only writes to output dims belonging
to a specified subset of attention heads. Targets resolved from the
probe at import time.

| Method | n heads | params | test_auc | Δ vs HK↗ |
|---|---|---|---|---|
| sl-last4-all (every head in blocks 44–47) | 96 | ~270K | 0.7313 ± 0.026 | −0.018 |
| sl-top-32-r4 (top 32 from probe, r=4)     | 32 | ~110K | 0.7306 ± 0.037 | −0.019 |
| sl-top-16-r4 (top 16 from probe, r=4)     | 16 | ~60K  | 0.7285 ± 0.022 | −0.021 |
| sl-top-16-r8 (top 16, r=8)                | 16 | ~120K | 0.7227 ± 0.032 | −0.027 |
| sl-top-32-r2 (top 32, r=2)                | 32 | ~55K  | 0.7206 ± 0.007 | −0.029 |
| sl-top-08-r4 (top 8 from probe, r=4)      |  8 | ~37K  | 0.7158 ± 0.027 | −0.034 |

**Every selective variant lost to HK↗ alone, by 0.018 – 0.034.** Inside
the selective sweep there's a clear trend: *more heads = better*
(`sl-top-08` < `sl-top-16` < `sl-top-32` < `sl-last4-all`). The probe's
"smart" selection of the top-K heads did NOT beat the dumb "all heads in
the last 4 blocks" baseline (`sl-last4-all` wins among selective
variants, despite being the only one that doesn't use the activation
ranking).

**Plausible reasons selective per-head LoRA loses to blanket LoRA:**

1. `HeadMaskedLoRA` adapts only the QKV linear's output dims belonging to
   the selected heads; it does **not** adapt `attn.out_proj` (a 1536→1536
   linear that mixes head outputs back into the residual). Full LoRA
   (`_inject_lora`) adapts both QKV and `out_proj` — out_proj contributes
   substantial expressivity that head-masked LoRA discards.
2. Activation magnitude in the *frozen* model is a proxy for "what
   matters now"; LoRA training can shift representations such that
   currently-quiet heads become useful too. Pre-selecting on the frozen
   activation pattern bakes in a frozen-model bias.
3. Restricting the LoRA delta to a small fraction of output dims may
   reduce its effective rank below what's needed for the residual stream
   to actually change in a meaningful way.

This is a negative result, but a useful one: the cheap "probe-and-pick"
hypothesis turns out to be wrong — blanket LoRA across all heads in the
target blocks beats targeting the most-activated ones.

## Updated overall ranking (top 8 of 26 experiments, 78 fold-runs)

```
ls-rank8-blocks-48        0.7540 ± 0.019   +0.005    (capacity scaling winner)
ls-blocks-24              0.7532 ± 0.030   +0.004
hk-hiddenkey-arrow        0.7495 ± 0.024    0.000    (HiddenKey^ — reference)
ls-blocks-16              0.7464 ± 0.010   -0.003
hk-dropattn-col-10        0.7441 ± 0.020   -0.005
hk-hiddencut-elem-10      0.7428 ± 0.014   -0.007
hk-hiddenkey-kl10         0.7424 ± 0.016   -0.007
ls-rank8-blocks-16        0.7420 ± 0.019   -0.008
```

The top three are all within one std of each other.

## Recommended config

```python
dropkey_prob=0.10, hiddencut_prob=0.10, kl_loss_weight=0.0,
paper_drop_only_lora=True,
lora_rank=8, lora_n_blocks=48, lora_block_start=0,
```

This is `ls-rank8-blocks-48`, the new (marginal) winner. If GPU memory
is a concern, the simpler `ls-blocks-24` (rank=4, last-24 blocks) is
within noise and uses ~4× fewer trainable params.

## What to try next, if anything

The HiddenKey ↔ LoRA-capacity ceiling on the current architecture seems to
be ≈ 0.75 test_auc. Reaching 0.78+ (where the DiscoTope-style XGBoost-on-
ESM-IF1 baseline lives, see `results.tsv:discotope-if1` at ~0.762) likely
requires a different lever:

- **Structure-aware features**: ESM-IF1's edge over ESM3 here is largely
  that it ingests structure directly. We have `features.py` with RSA /
  biophysical / BLOSUM ready to plug into the head via `extra_dim`;
  adding RSA-as-feature on top of HK↗ is the cheapest unexplored axis.
- **Selective LoRA done right**: extend `HeadMaskedLoRA` to also adapt
  `attn.out_proj` per-head (a 1536→64 read + 64→1536 write per selected
  head), or apply blanket LoRA to the *blocks* identified by the probe
  (which is essentially what `ls-blocks-16` already does — and it loses
  to `ls-blocks-24`, so probably not the gap).
- **Ensembling**: pick the top 3 configs (`ls-rank8-blocks-48`,
  `ls-blocks-24`, `hk-hiddenkey-arrow`) and average their probabilities.
  Fold-level variance is 0.02+; an ensemble of orthogonal-error models
  could quietly add another +0.01.

---

# Follow-up sweep: probe-driven LayerDrop (`pld-*`)

Question: the in-training EMA-active LayerDrop (`layerdrop-active-2-30` →
0.7355 alone, `hkx-arrow-layerdrop` → 0.7253 with HK↗) was unstable in
the sense that the schedule drifts as the LoRA adapters learn. Does
using the **probe** ranking (computed once on the frozen ESM3 over all
748 BEPIPRED antigens) as a static, stable schedule do better?

Setup: 3 experiments × 3 folds = 9 fold-runs. All on top of HiddenKey↗.

| Method | test_auc | Δ vs HK↗ |
|---|---|---|
| pld-top4-p30 (drop blocks 44–47 iid p=0.30) | 0.7295 ± 0.016 | −0.020 |
| pld-top4-p20 (drop blocks 44–47 iid p=0.20) | 0.7287 ± 0.029 | −0.021 |
| pld-47-p50   (drop ONLY block 47 with p=0.50) | 0.7284 ± 0.010 | −0.021 |

**Same regression as the EMA variant**, to within 0.005. Three observations:

1. **The schedule doesn't matter**: static probe-derived ranking and
   dynamic EMA-derived ranking yield essentially the same result when
   combined with HK↗ (~0.728 vs ~0.725). The signal is the same in both
   cases — it's the last few blocks — and the cost of dropping them
   during HK↗ training is the same.
2. **Even the stress test of dropping ONLY block 47 with p=0.50 lands
   in the same band.** The model can route around its single dominant
   block half the time, but the routing doesn't *help* test_auc.
3. **HK↗'s "sufficiency" claim survives another orthogonal regularizer
   attempt.** Across the whole project, every combination of HK↗ with
   {LayerDrop active, RYS, DropAttention, probe-LayerDrop, selective
   per-head LoRA} has regressed.

Wherever the ceiling is on this architecture (~0.75 test_auc), it
appears to be at the LoRA + HiddenKey↗ saturation point, not a
regularization-deficit point. Closing the gap to the DiscoTope-style
ESM-IF1 XGBoost baseline (~0.762) likely needs structure-aware input
features (RSA, biophysical) rather than another LoRA / dropout angle.
