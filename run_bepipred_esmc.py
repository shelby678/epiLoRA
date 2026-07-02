"""BEPIPRED ESMC experiment: LoRA on the ESM Cambrian sequence-only backbone.

Trains ESMC-600M (frozen) + LoRA + linear head on the same 3-fold CV as the
ESM3 experiments, so test ROC-AUC is directly comparable to the ESM3+LoRA
results in results.tsv (best ESM3+LoRA: ls-rank8-blocks-48 ~0.754).

ESMC is sequence-only: backbone coordinates loaded by create_cv_datasets are
ignored by the model. Same folds, LR, batch, and time budget as the baseline.

Two configs:
  esmc-r8-last8 : LoRA rank=8 on the last 8 blocks   (direct analog of ls-rank-8)
  esmc-r8-all   : LoRA rank=8 on all 36 blocks        (analog of ls-rank8-blocks-48)

Run:
    uv run python run_bepipred_esmc.py > /tmp/run_esmc.log 2>&1
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

from train_struct import BATCH_SIZE, MAX_SEQ_LEN, create_cv_datasets, compute_roc_auc
from train_esmc import ESMCEpitopeModel, train_esmc, _ESMC_LOADERS

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV = Path("results.tsv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ESMC_SIZE = "600m"
N_BLOCKS_TOTAL = _ESMC_LOADERS[ESMC_SIZE][2]

CV_FOLDS = [
    {"test": "1", "val": "1"},
    {"test": "2", "val": "2"},
    {"test": "3", "val": "3"},
]
TIME_BUDGET = 1200

try:
    COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    COMMIT = "unknown"


def run_cv(name: str, desc: str, hparams: dict) -> None:
    fold_val_aucs: list[float] = []
    fold_test_aucs: list[float] = []

    print(f"\n{'='*60}\nEXPERIMENT: {name}\n  {desc}\n{'='*60}")

    for fold in CV_FOLDS:
        test_part, val_part = fold["test"], fold["val"]

        # Skip-if-already-logged: resume after a crash without re-running folds.
        if RESULTS_TSV.exists():
            done = any(
                line.split("\t")[1] == name and line.split("\t")[2] == test_part
                for line in RESULTS_TSV.read_text().splitlines()[1:]
                if len(line.split("\t")) > 2
            )
            if done:
                logger.info(f"  skip {name} fold {test_part} (already logged)")
                continue

        train_data, val_data, test_data = create_cv_datasets(
            BEPIPRED_FASTA, STRUCTURES_DIR,
            test_partition=test_part, val_partition=val_part, max_length=MAX_SEQ_LEN,
        )
        logger.info(f"  fold test={test_part}: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

        t0 = time.time()
        result = train_esmc(
            train_data, val_data,
            size=ESMC_SIZE, max_seconds=TIME_BUDGET, device=DEVICE, **hparams,
        )
        elapsed = time.time() - t0

        model = ESMCEpitopeModel(
            size=ESMC_SIZE,
            dropout=hparams.get("dropout", 0.1),
            lora_rank=hparams.get("lora_rank", 8),
            lora_alpha=hparams.get("lora_alpha", 8.0),
            lora_n_blocks=hparams.get("lora_n_blocks", 8),
            lora_block_start=hparams.get("lora_block_start", -1),
            head_hidden_dim=hparams.get("head_hidden_dim", 0),
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
            f"  fold test={test_part}: val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  "
            f"val_loss={result['val_loss']:.4f}  steps={result['steps']}  elapsed={elapsed:.0f}s"
        )

        with open(RESULTS_TSV, "a") as f:
            f.write(
                f"{COMMIT}\t{name}\t{test_part}\t0\t{result['val_loss']:.6f}\t"
                f"{val_auc:.6f}\t{test_auc:.6f}\t{result['steps']}\t"
                f"{result['peak_vram_mb']}\t{elapsed:.1f}\t{desc}\n"
            )

    if fold_test_aucs:
        print(f"\n{'-'*60}\nSUMMARY {name}")
        print(f"  test_auc = {np.mean(fold_test_aucs):.4f} ± {np.std(fold_test_aucs):.4f}  (n={len(fold_test_aucs)})")
        print(f"{'-'*60}")


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\tsteps\tpeak_vram_mb\telapsed_s\tdesc\n")

    run_cv(
        name="esmc-r8-last8",
        desc=f"ESMC-{ESMC_SIZE} frozen + LoRA rank=8 last-8 blocks + linear head (seq-only)",
        hparams=dict(lora_rank=8, lora_alpha=8.0, lora_n_blocks=8, lora_block_start=-1),
    )

    run_cv(
        name="esmc-r8-all",
        desc=f"ESMC-{ESMC_SIZE} frozen + LoRA rank=8 all-{N_BLOCKS_TOTAL} blocks + linear head (seq-only)",
        hparams=dict(lora_rank=8, lora_alpha=8.0, lora_n_blocks=N_BLOCKS_TOTAL, lora_block_start=0),
    )
