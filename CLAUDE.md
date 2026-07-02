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
run_bepipred.py           config-driven ESM3 CV runner; picks a named
                          experiment set (baseline/rsa/dropout/hiddenkey/...)
run_bepipred_discotope*.py  DiscoTope-style XGBoost-on-embeddings runners
run_bepipred_esmc.py      ESMC-600M backbone variant
run_ensemble.py           ESM3/ESM2 ensemble members -> ensemble_preds/*.npz
run_ensemble_esmif1.py    ESM-IF1 ensemble member + IF1 library (py3.9 env)
run_if1_5fold.py          5-fold IF1 comparison: LoRA+RYS vs XGBoost
data/                     BEPIPRED.fasta, pdb_contacts.fasta, structures/, etc.
                          (+ the data-processing scripts; no data committed)
archive/                  stale logs, old results backups, large artifacts
docs/                     paper PDFs referenced from experiments
findings/                 markdown writeups of experiment findings
tests/                    pytest unit tests
program.md                the research-loop instructions (read this)
results.tsv               append-only experiment log (do not commit)
```

## Run an experiment

```bash
uv run python run_bepipred.py <set> > /tmp/run_<set>.log 2>&1
uv run python run_bepipred.py --list        # show all experiment sets
```

`<set>` is one of: `baseline`, `rsa`, `features`, `dropout`, `hiddenkey`,
`hky_pscan`, `hkx`, `lora_all_active`, `lora_scale`, `lora_select`,
`probe_ldrop`, `pretrain`, `ultra` (or `all`). Each set is a list of
`(name, desc, cfg)` entries defined in a `_set_<name>()` function near the top
of `run_bepipred.py`; add a new experiment batch by adding one such function
and registering it in the `SETS` dict.

Each fold burns ~7 min on this box; one experiment × 3 folds is ~22 min. The
runner appends per fold/run to `results.tsv` and skips any (exp, fold, run)
already logged, so a crashed run resumes cleanly.

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

- One experiment batch = one `_set_<name>()` function in `run_bepipred.py`,
  registered in `SETS`. Commit it before running.
- Append to `results.tsv` per fold per run; the skip-if-already-logged
  pattern lets you resume after a crash without re-running completed folds.
- Don't commit `results.tsv`, `.log` files, or anything under `archive/`.
- Don't edit `prepare.py`.
- Keep the runner config-driven: a new experiment should be a new entry in a
  `_set_*` list, not a new script. Genuinely different pipelines (XGBoost,
  a different backbone, a different env) still warrant their own file.

## Paper-driven experiments

ACL 2024 Findings: *LoRA Meets Dropout under a Unified Framework* (Wang et
al.) — `docs/lora_dropout_hiddenkey_acl2024.pdf`. Recommends **HiddenKey** =
column-wise DropKey on attention logits + element-wise HiddenCut on FFN
hidden representations + bidirectional KL loss between two forward passes
(R-drop). Implementations live in `train_struct.py`; run with
`run_bepipred.py hiddenkey` (also `hky_pscan`, `hkx`).

## Today's working state

Best test_auc to date: `layerdrop-active-2-30` (`run_bepipred.py dropout`) at
~0.735 across 3 folds. Baseline (no RYS, LoRA rank=4, last-8): ~0.708. The
ESM-IF1 LoRA+RYS member reaches ~0.82 (`run_if1_5fold.py lora`).
