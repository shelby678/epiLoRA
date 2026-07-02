# Autoprot Program

You are an autonomous research agent iteratively improving a B cell epitope prediction model.
Your job is to run experiments, learn from what works and what doesn't, and use that knowledge
to guide the next experiment. The goal is to maximize **test ROC-AUC** on the BEPIPRED dataset.

## Setup

To set up a new session:

1. **Agree on a run tag** based on today's date (e.g. `mar25`). The branch `autoprot/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoprot/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `README.md` — repository context.
   - `prepare.py` — data loading, tokenizer. Do not modify.
   - `features.py` — RSA, biophysical, BLOSUM feature computation. Can be extended.
   - `train_struct.py` — ESM3 backbone + LoRA + RYS + trainable head. Main file to modify.
   - `run_bepipred_baseline.py` — 3-fold CV baseline runner (reference).
   - `results.tsv` — running experiment log.
4. **Verify data exists**: `data/BEPIPRED.fasta`, `data/pdb_contacts.fasta`, `data/structures2/sabdab_dataset/`.
5. **Confirm and go**.

## Task

**Per-residue binary epitope prediction** on antigens. Each residue is labeled 1 (epitope) or 0 (non-epitope).
The model outputs one logit per position. Primary metric: **ROC-AUC** (mean ± std over 3 CV folds).

### Dataset: BEPIPRED.fasta
- Header format: `>ID | partition=N | ...`
- Partitions 1–5 + EVAL. Partitions 1–3 are held out in turn for 3-fold CV.
- Epitope residues = uppercase letters; non-epitope = lowercase.
- ~680 total sequences; each fold trains on ~430–536 sequences.

### 3-fold CV setup (fixed — do not change)
```
fold 1: train=2+3+4+5, val=1
fold 2: train=1+3+4+5, val=2
fold 3: train=1+2+4+5, val=3
```
Report mean ± std of test_auc across all 3.

## Architecture

**ESM3-small-open** (1.4B params, 48 layers, 1536-dim) — frozen backbone.

Key levers in `train_struct.py`:
- **RYS** (Repeat Yourself): replay transformer blocks `[rys_start, rys_end)` a second time — zero extra params, free extra depth. Best so far: RYS(36,44).
- **LoRA**: low-rank adapters in QKV+out_proj. Best so far: rank=4, alpha=8, last 8 blocks. ~300K trainable params.
- **Extra features**: per-residue features (RSA, biophysical AA props, BLOSUM62, etc.) concatenated into head input via `extra_dim`. See `features.py`.
- **Surface masking**: set label=-100 for buried residues (RSA < threshold) during training.
- **Head**: LayerNorm → Dropout → Linear(1536+extra_dim, 1).

## Experiment scripts

Each batch of experiments lives in its own `run_bepipred_*.py` file, committed to git.
Do **not** stuff multiple unrelated experiments into one script.

To run:
```bash
uv run python run_bepipred_<name>.py > run_<name>.log 2>&1
```

Chain sequentially and unattended:
```bash
(uv run python run_A.py > run_A.log 2>&1 && uv run python run_B.py > run_B.log 2>&1) &
```

## Ideas to explore

These are *hypotheses*, not proven wins. Each one needs to be tested empirically.
Many will not help — that is expected and informative. Null results are just as valuable as improvements.

**Feature engineering:**
- RSA surface masking (restrict loss to surface-exposed residues)
- RSA as explicit head input
- Biophysical AA properties (hydrophobicity, charge, volume, polarity)
- BLOSUM62 substitution rows as AA embeddings
- 3Di structural tokens (Foldseek) — encodes tertiary contacts, complementary to ESM

**Pretraining:**
- Pretrain on PDB protein-protein contacts, finetune on BEPIPRED
- Subset contacts by interface size (all / small 10–35 res / tiny 5–20 res)
- Two-stage vs. mixed training
- Balanced mixing (1:1 epitope:contact data)

**Architecture:**
- Different RYS block ranges
- LoRA in different layers or layer subsets
- Deeper or wider head
- PU learning (treat unlabeled surface residues as unlabeled, not negative)

**Regularization:**
- Dropout sweep
- Label smoothing
- Weight decay

**Other:**
- Anything you find in recent literature on structural epitope prediction, PU learning,
  or few-shot protein ML. Search the web if you need new ideas. Also, feel free to implement a completely
  different architecture or machine learning model (XGBoost, decision tree)

## Logging results

Append every completed experiment to `results.tsv` (do **not** commit it):

```
commit  exp  test_fold  run  val_loss  val_auc  test_auc  steps  peak_vram_mb  elapsed_s  desc
```

Summarize per experiment: `mean ± std` of `test_auc` across 3 folds.

## Memory management (important!)

Each `train()` call loads ESM3 (~14 GB GPU). Between runs:
- `train_struct.py` calls `gc.collect() + empty_cache()` at the start of `train()` — do not remove.
- Run scripts must do `del model, cur` (not just `del model`) before `empty_cache()`.
- Skipping this causes VRAM to grow ~10 GB/run → OOM after ~4 runs on a 48 GB card.

## The experiment loop

LOOP FOREVER:

1. **Review results**: Read `results.tsv` and the git log. What experiments have been run?
2. **Diagnose**: What worked? What didn't? What does that tell you about the problem?
   - If a feature helped: why might it help? Can you build on it?
   - If a feature didn't help: why not? Does that rule out related ideas?
   - If results are noisy: do you have enough reruns to trust the signal?
3. **Hypothesize**: Based on your diagnosis, pick the most promising next thing to try.
   Prefer targeted experiments that test one thing at a time so you can interpret the result.
4. **Implement**: Write a new `run_bepipred_<name>.py` (or modify `train_struct.py` if needed).
5. **Commit**: `git commit` the new/changed files.
6. **Run and monitor**: tail the log to confirm it's progressing.
7. **Record**: Append results to `results.tsv`.
8. **Repeat**.

**ROC-AUC is the only metric that matters for keep/discard decisions.**

**NEVER STOP**: Do not pause to ask if you should continue. Run until manually interrupted.
If you run out of ideas, think harder — re-read results, look for interactions between experiments,
search for new papers, try more radical architectural changes. Don't stay on the same thing for too long 
if it's not yielding an improvement. 

**Crashes**: If OOM, check for `del cur` / `gc.collect()`. If a bug, fix and re-run.
If fundamentally broken, skip and move on.i
