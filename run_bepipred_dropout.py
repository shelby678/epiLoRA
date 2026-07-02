"""BEPIPRED dropout sweep: test structure dropout, LayerDrop, DropHead variants.

Experimental design (all on the same LoRA(rank=4, last-8) backbone, **no RYS**):

  baseline-no-rys        — LoRA only, no RYS, no extra dropout.
  struct-dropout-50      — per-sample, NaN out backbone coords with p=0.5.
  layerdrop-uniform-15   — standard LayerDrop: each LoRA block skipped iid p=0.15.
  layerdrop-active-2-30  — per-batch, drop top-2 LoRA blocks by recent activation EMA, p=0.3.
  drophead-uniform-15    — each attention head zeroed iid with p=0.15.
  drophead-active-2-30   — per-batch, drop top-2 most-activated heads, p=0.3.

Each method runs across all 3 BEPIPRED CV folds (single seed) and results are
appended to ``results.tsv``.  Time budget: 20 min per fold (early stop usually
kicks in at ~3-6 min).

Refs:
  - LayerDrop / structured dropout: Fan et al. 2019 (arXiv:1909.11556).
  - DropHead: Zhou et al. 2020 (arXiv:2004.13342).
  - Hard Attention Masking (top-k): Doe et al. 2024 (arXiv:2504.12088).
  - Inference-only / per-sample feature dropout: Greenman & Stafford 2025.

Run:
    .venv/bin/python run_bepipred_dropout.py > run_dropout.log 2>&1
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train_struct import (
    BATCH_SIZE, DROPOUT, LR, MAX_SEQ_LEN, WARMUP_STEPS, WEIGHT_DECAY,
    create_cv_datasets, train, compute_roc_auc,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV = Path("results.tsv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CV_FOLDS = [
    {"test": "1", "val": "1"},
    {"test": "2", "val": "2"},
    {"test": "3", "val": "3"},
]
N_RERUNS = 1
TIME_BUDGET = 1200  # 20 min ceiling per fold

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

# Shared base config: LoRA on last 8 blocks, NO RYS (rys_end <= rys_start disables).
BASE_CONFIG = dict(
    rys_start=0, rys_end=0,           # RYS disabled
    lora_rank=4, lora_alpha=8.0, lora_n_blocks=8, lora_block_start=-1,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    val_eval_interval=200,
    patience=5,
    compute_auc=True,
    device=DEVICE,
)


def _already_done() -> set[tuple[str, str, int]]:
    """Read RESULTS_TSV and return set of (exp, test_fold, run) tuples already logged."""
    if not RESULTS_TSV.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with open(RESULTS_TSV) as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                done.add((parts[1], parts[2], int(parts[3])))
            except ValueError:
                continue
    return done


def run_one(name: str, desc: str, hparams: dict) -> dict:
    """Run one experiment across all CV folds; append rows to RESULTS_TSV."""
    fold_test_aucs: list[float] = []
    fold_val_aucs: list[float] = []
    done = _already_done()
    print(f"\n{'=' * 60}\nEXPERIMENT: {name}\n  {desc}\n{'=' * 60}")

    for fold in CV_FOLDS:
        test_part, val_part = fold["test"], fold["val"]
        if all((name, test_part, r) in done for r in range(N_RERUNS)):
            print(f"  fold test={test_part}: SKIP (already in results.tsv)")
            continue
        logger.info(f"Loading fold: test={test_part}, val={val_part}")
        train_data, val_data, test_data = create_cv_datasets(
            BEPIPRED_FASTA, STRUCTURES_DIR,
            test_partition=test_part, val_partition=val_part, max_length=MAX_SEQ_LEN,
        )
        for run_idx in range(N_RERUNS):
            if (name, test_part, run_idx) in done:
                continue
            t0 = time.time()
            result = train(train_data, val_data, max_seconds=TIME_BUDGET, **hparams)
            elapsed = time.time() - t0

            # Rebuild model with same architecture to load best state for test eval.
            from train_struct import StructureEpitopePredictionModel
            model = StructureEpitopePredictionModel(
                dropout=hparams.get("dropout", DROPOUT),
                rys_start=hparams.get("rys_start", 0),
                rys_end=hparams.get("rys_end", 0),
                lora_rank=hparams.get("lora_rank", 0),
                lora_alpha=hparams.get("lora_alpha", 8.0),
                lora_n_blocks=hparams.get("lora_n_blocks", 8),
                lora_block_start=hparams.get("lora_block_start", -1),
            ).to(DEVICE)
            cur = model.state_dict()
            cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
            model.load_state_dict(cur)
            test_auc = compute_roc_auc(model, test_data, batch_size=BATCH_SIZE, device=DEVICE)
            del model, cur
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            val_auc = result["roc_auc"]
            fold_val_aucs.append(val_auc)
            fold_test_aucs.append(test_auc)
            print(
                f"  fold test={test_part} run={run_idx+1}/{N_RERUNS}:  "
                f"val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  "
                f"val_loss={result['val_loss']:.4f}  steps={result['steps']}  "
                f"elapsed={elapsed:.0f}s"
            )
            with open(RESULTS_TSV, "a") as f:
                f.write(
                    f"{COMMIT}\t{name}\t{test_part}\t{run_idx}\t"
                    f"{result['val_loss']:.6f}\t{val_auc:.6f}\t{test_auc:.6f}\t"
                    f"{result['steps']}\t{result['peak_vram_mb']}\t{elapsed:.1f}\t"
                    f"{desc}\n"
                )

    # Re-read all rows belonging to this experiment so summary spans skipped folds too.
    all_test: list[float] = []
    all_val: list[float] = []
    if RESULTS_TSV.exists():
        with open(RESULTS_TSV) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7 and parts[1] == name:
                    try:
                        all_val.append(float(parts[5]))
                        all_test.append(float(parts[6]))
                    except ValueError:
                        pass
    if not all_test:
        all_test = fold_test_aucs
        all_val = fold_val_aucs
    mean = float(np.mean(all_test))
    std = float(np.std(all_test))
    print(f"\n{'─' * 60}\nSUMMARY {name}")
    print(f"  test_auc = {mean:.4f} ± {std:.4f}   (val_auc={np.mean(all_val):.4f})  "
          f"[n={len(all_test)}]")
    print(f"{'─' * 60}")
    return {"name": name, "test_auc_mean": mean, "test_auc_std": std,
            "fold_test_aucs": all_test}


EXPERIMENTS = [
    ("baseline-no-rys",
     "no-RYS baseline: LoRA rank=4 last-8 only",
     dict(BASE_CONFIG)),

    ("struct-dropout-50",
     "structure dropout: NaN coords per-sample with p=0.5",
     {**BASE_CONFIG, "structure_dropout_prob": 0.50}),

    ("layerdrop-uniform-15",
     "LayerDrop uniform: skip each LoRA block iid p=0.15",
     {**BASE_CONFIG, "layer_drop_mode": "uniform", "layer_drop_prob": 0.15,
      "layer_drop_only_lora": True}),

    ("layerdrop-active-2-30",
     "LayerDrop active: drop top-2 LoRA blocks by EMA-activation, p=0.30",
     {**BASE_CONFIG, "layer_drop_mode": "active", "layer_drop_prob": 0.30,
      "layer_drop_topk": 2, "layer_drop_only_lora": True}),

    ("drophead-uniform-15",
     "DropHead uniform: each head zeroed iid p=0.15",
     {**BASE_CONFIG, "head_drop_mode": "uniform", "head_drop_prob": 0.15}),

    ("drophead-active-2-30",
     "DropHead active: per-batch zero top-2 highest-activation heads, p=0.30",
     {**BASE_CONFIG, "head_drop_mode": "active", "head_drop_prob": 0.30,
      "head_drop_topk": 2}),
]


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                    "steps\tpeak_vram_mb\telapsed_s\tdesc\n")

    summaries = []
    for name, desc, hparams in EXPERIMENTS:
        s = run_one(name, desc, hparams)
        summaries.append(s)

    print(f"\n{'=' * 60}\nFINAL SUMMARY (3-fold BEPIPRED CV)\n{'=' * 60}")
    print(f"{'method':<28}  test_auc")
    for s in summaries:
        print(f"{s['name']:<28}  {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f}")
