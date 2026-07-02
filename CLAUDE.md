# CLAUDE.md

Orientation for Claude Code working in this repo. Read this first; then read
`program.md` for the research-loop instructions.

## What this project is

Autonomous research loop that iteratively improves a **B-cell epitope
prediction** model. Task: per-residue binary classification on antigen
sequences (epitope vs non-epitope). Primary metric: **ROC-AUC** averaged over
3 CV folds on `data/BEPIPRED.fasta`.

Architecture: **ESM3-small-open** (frozen, 48 layers, 1536-dim) + **LoRA** on
QKV / out_proj of the last 8 blocks + optional **RYS** (Repeat Yourself)
block-replay + trainable linear head. See `train_struct.py`.

## Layout

```
train_struct.py           main model + training loop (ESM3 + LoRA + RYS + head)
prepare.py                IMMUTABLE — tokenizer, FASTA loaders, PU loss
features.py               RSA / biophysical / BLOSUM extra features
run_bepipred_*.py         experiment runners; each script = one batch of
                          experiments; append rows to results.tsv

data/                     BEPIPRED.fasta, pdb_contacts.fasta, structures/, etc.
scripts/legacy/           old DiscoTope / Surf2Spot / ensemble research scripts
                          (kept for reference; not in active pipeline)
archive/                  stale logs, old results backups, large artifacts
docs/                     paper PDFs referenced from experiments
findings/                 markdown writeups of experiment findings
tests/                    pytest unit tests
program.md                the research-loop instructions (read this)
results.tsv               append-only experiment log (do not commit)
```

## Run an experiment

```bash
uv run python run_bepipred_<name>.py > /tmp/run_<name>.log 2>&1
```

Each fold burns ~7 min on this box; one experiment × 3 folds is ~22 min. Use
the existing fold + skip-if-already-logged pattern in `run_bepipred_dropout.py`
or `run_bepipred_lora_all_active.py` as the template.

## CV folds (do not change)

```
fold 1: test=part 1, val=part 1, train=parts 2+3+4+5
fold 2: test=part 2, val=part 2, train=parts 1+3+4+5
fold 3: test=part 3, val=part 3, train=parts 1+2+4+5
```

EVAL partition is excluded entirely. Report `mean ± std` over 3 folds.

## results.tsv schema

```
commit  exp  test_fold  run  val_loss  val_auc  test_auc  steps  peak_vram_mb  elapsed_s  desc
```

## Key levers in `train_struct.py`

| Lever | Notes |
|---|---|
| `lora_rank`, `lora_n_blocks`, `lora_block_start` | LoRA placement (default: rank=4, last 8 blocks) |
| `rys_start`, `rys_end` | RYS replay range (set `rys_end <= rys_start` to disable) |
| `head_drop_mode`, `head_drop_prob`, `head_drop_topk` | per-head attention output dropout (uniform / active top-k) |
| `layer_drop_mode`, `layer_drop_prob`, `layer_drop_topk` | LayerDrop on residual stream |
| `structure_dropout_prob` | per-sample, NaN out backbone coords |
| `rsa_surface_threshold`, `rsa_as_feature`, `bio_features`, `blosum_features` | feature variants |
| `pu_prior` | enable PU learning loss (vs BCE) |
| `dropkey_prob`, `hiddencut_prob`, `kl_loss_weight` | HiddenKey/DropKey/HiddenCut from ACL'24 paper (`docs/lora_dropout_hiddenkey_acl2024.pdf`) |

## Memory management (important)

Each `train()` call loads ESM3 (~14 GB GPU). Between runs the runner script
must `del model, cur` (NOT just `del model`) before `torch.cuda.empty_cache()`,
or VRAM grows ~10 GB/run and OOMs after a few experiments on a 48 GB card.
`train()` already does `gc.collect() + empty_cache()` at entry — keep it.

## Conventions

- One experiment batch = one `run_bepipred_<name>.py` script, committed to
  git before running.
- Append to `results.tsv` per fold per run; the skip-if-already-logged
  pattern lets you resume after a crash without re-running completed folds.
- Don't commit `results.tsv`, `.log` files, or anything under `archive/`.
- Don't edit `prepare.py`.
- Don't add features / refactors that aren't needed for the current
  experiment. Three repeated lines in a runner is fine; abstracting them is
  not unless it pays for itself across multiple runners.

## Paper-driven experiments

ACL 2024 Findings: *LoRA Meets Dropout under a Unified Framework* (Wang et
al.) — `docs/lora_dropout_hiddenkey_acl2024.pdf`. Recommends **HiddenKey** =
column-wise DropKey on attention logits + element-wise HiddenCut on FFN
hidden representations + bidirectional KL loss between two forward passes
(R-drop). Implementations live in `train_struct.py`; runners in
`run_bepipred_hiddenkey.py`.

## Today's working state

Active branch: `autoprot/mar25`. Recent focus: structure / layer / head
dropout sweep (`run_bepipred_dropout.py`), then full-coverage LoRA + active
LayerDrop (`run_bepipred_lora_all_active.py`). Best test_auc to date:
`layerdrop-active-2-30` at ~0.735 across 3 folds. Baseline (no RYS, LoRA
rank=4, last-8): ~0.708.
