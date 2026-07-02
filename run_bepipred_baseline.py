"""BEPIPRED baseline: 3-fold CV with the best SAbDab config.

Trains one model per fold (holdout partition = 1, 2, 3).
Each fold uses ~500 training sequences, partition 5 as val, holdout as test.
Reports val ROC-AUC (per fold) and test ROC-AUC (per fold + mean ± std).

Best config from SAbDab experiments: RYS(36,44) + LoRA rank=4 last-8 + batch=4

Run:
    uv run python run_bepipred_baseline.py > run.log 2>&1
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
    BATCH_SIZE, DROPOUT, LR, MAX_SEQ_LEN, RYS_END, RYS_START,
    WARMUP_STEPS, WEIGHT_DECAY,
    create_cv_datasets, train, compute_roc_auc,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BEPIPRED_FASTA  = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR  = Path("data/structures2/sabdab_dataset")
RESULTS_TSV     = Path("results.tsv")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# 3-fold CV: hold out each of the three largest partitions in turn.
# Val partition is always "5" (148 seqs), giving ~430-540 train seqs per fold.
CV_FOLDS = [
    {"test": "1", "val": "1"},  # train on 2+3+4+5, holdout=1
    {"test": "2", "val": "2"},  # train on 1+3+4+5, holdout=2
    {"test": "3", "val": "3"},  # train on 1+2+4+5, holdout=3
]

# Number of random-seed reruns per fold (for mean/std stability)
N_RERUNS = 1

# Training budget per run (early stopping usually kicks in ~200s)
TIME_BUDGET = 1200  # 20 min ceiling

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

# ---------------------------------------------------------------------------
# Best-config hyperparameters (unchanged from SAbDab best)
# ---------------------------------------------------------------------------
BEST_CONFIG = dict(
    rys_start=36, rys_end=44,
    lora_rank=4, lora_alpha=8.0, lora_n_blocks=8, lora_block_start=-1,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    val_eval_interval=200,
    patience=5,          # slightly more patient on smaller dataset
    compute_auc=True,
    device=DEVICE,
)

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_cv(name: str, desc: str, hparams: dict) -> None:
    """Run one experiment across all CV folds, logging results to RESULTS_TSV."""

    fold_val_aucs:  list[float] = []
    fold_test_aucs: list[float] = []

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")

    for fold in CV_FOLDS:
        test_part = fold["test"]
        val_part  = fold["val"]

        logger.info(f"Loading fold: test={test_part}, val={val_part}")
        train_data, val_data, test_data = create_cv_datasets(
            BEPIPRED_FASTA, STRUCTURES_DIR,
            test_partition=test_part,
            val_partition=val_part,
            max_length=MAX_SEQ_LEN,
        )
        n_struct_tr = sum(1 for _, _, c, _ in train_data if c is not None)
        n_struct_ts = sum(1 for _, _, c, _ in test_data if c is not None)
        logger.info(
            f"  train={len(train_data)} ({n_struct_tr} w/ struct)  "
            f"val={len(val_data)}  test={len(test_data)} ({n_struct_ts} w/ struct)"
        )

        rerun_val_aucs:  list[float] = []
        rerun_test_aucs: list[float] = []

        for run_idx in range(N_RERUNS):
            t0 = time.time()
            result = train(
                train_data, val_data,
                max_seconds=TIME_BUDGET,
                **hparams,
            )
            elapsed = time.time() - t0

            # Evaluate on held-out test partition
            model_state = result["trainable_state"]
            # Rebuild model to evaluate on test set
            from train_struct import StructureEpitopePredictionModel, create_struct_dataloader
            model = StructureEpitopePredictionModel(
                dropout=hparams.get("dropout", DROPOUT),
                rys_start=hparams.get("rys_start", RYS_START),
                rys_end=hparams.get("rys_end", RYS_END),
                lora_rank=hparams.get("lora_rank", 0),
                lora_alpha=hparams.get("lora_alpha", 8.0),
                lora_n_blocks=hparams.get("lora_n_blocks", 8),
                lora_block_start=hparams.get("lora_block_start", -1),
            ).to(DEVICE)
            cur = model.state_dict()
            cur.update({k: v.to(DEVICE) for k, v in model_state.items() if k in cur})
            model.load_state_dict(cur)

            test_auc = compute_roc_auc(model, test_data, batch_size=BATCH_SIZE, device=DEVICE)
            del model, cur
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            val_auc = result["roc_auc"]
            rerun_val_aucs.append(val_auc)
            rerun_test_aucs.append(test_auc)

            print(
                f"  fold test={test_part} run={run_idx+1}/{N_RERUNS}:"
                f"  val_auc={val_auc:.4f}  test_auc={test_auc:.4f}"
                f"  val_loss={result['val_loss']:.4f}  steps={result['steps']}"
                f"  elapsed={elapsed:.0f}s"
            )

            # Write to results TSV
            with open(RESULTS_TSV, "a") as f:
                f.write(
                    f"{COMMIT}\t{name}\t{test_part}\t{run_idx}\t"
                    f"{result['val_loss']:.6f}\t{val_auc:.6f}\t{test_auc:.6f}\t"
                    f"{result['steps']}\t{result['peak_vram_mb']}\t{elapsed:.1f}\t"
                    f"{desc}\n"
                )

        fold_val_aucs.extend(rerun_val_aucs)
        fold_test_aucs.extend(rerun_test_aucs)
        print(
            f"  fold test={test_part} mean: val_auc={np.mean(rerun_val_aucs):.4f}"
            f"  test_auc={np.mean(rerun_test_aucs):.4f} ± {np.std(rerun_test_aucs):.4f}"
        )

    print(f"\n{'─'*60}")
    print(f"SUMMARY {name}")
    print(f"  val_auc  = {np.mean(fold_val_aucs):.4f} ± {np.std(fold_val_aucs):.4f}")
    print(f"  test_auc = {np.mean(fold_test_aucs):.4f} ± {np.std(fold_test_aucs):.4f}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Write TSV header if file doesn't exist yet
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(
                "commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                "steps\tpeak_vram_mb\telapsed_s\tdesc\n"
            )

    # Baseline: best SAbDab config (RYS+LoRA) on BEPIPRED with 3-fold CV
    run_cv(
        name="bepipred-baseline",
        desc="RYS(36,44)+LoRA rank4 last-8, best SAbDab config, no extra features",
        hparams=BEST_CONFIG,
    )
