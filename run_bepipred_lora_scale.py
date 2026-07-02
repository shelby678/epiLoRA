"""BEPIPRED LoRA scaling + rank sweep on top of HiddenKey↗.

Phase 4 verdict: HiddenKey↗ at p=0.10/0.10 (no KL) won the 11-experiment
paper sweep at test_auc 0.7495 ± 0.024 with LoRA rank=4 on the last 8 of
48 ESM3 blocks (~300K trainable params). This sweep tests whether more
LoRA capacity — wider coverage OR higher rank — yields stronger results
when paired with the best dropout config.

Experiments (8 × 3 folds = 24 fold-runs, ~3 hr):

  ## Coverage scaling (rank fixed at 4, vary number of LoRA blocks)
  ls-blocks-16: LoRA rank=4 on last 16 blocks (~600K params)
  ls-blocks-24: LoRA rank=4 on last 24 blocks (~900K params)
  ls-blocks-48: LoRA rank=4 on all 48 blocks  (~1.8M params)

  ## Rank scaling (blocks fixed at last-8, vary rank)
  ls-rank-2:    LoRA rank=2 on last 8 blocks  (~150K params)
  ls-rank-8:    LoRA rank=8 on last 8 blocks  (~600K params)
  ls-rank-16:   LoRA rank=16 on last 8 blocks (~1.2M params)

  ## Combined high capacity
  ls-rank8-blocks-16: rank=8 last-16          (~1.2M params)
  ls-rank8-blocks-48: rank=8 all-48           (~3.6M params)
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
TIME_BUDGET = 1200

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

# Base = HiddenKey↗ winner config from sweep 1.
BASE = dict(
    rys_start=0, rys_end=0,
    lora_alpha=8.0,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    val_eval_interval=200,
    patience=5,
    compute_auc=True,
    device=DEVICE,
    paper_drop_only_lora=True,
    dropkey_prob=0.10,
    hiddencut_prob=0.10,
)


def _already_done() -> set[tuple[str, str, int]]:
    if not RESULTS_TSV.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with open(RESULTS_TSV) as f:
        next(f, None)
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
    fold_test_aucs, fold_val_aucs = [], []
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

    all_test, all_val = [], []
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
        all_test, all_val = fold_test_aucs, fold_val_aucs
    mean = float(np.mean(all_test))
    std = float(np.std(all_test))
    print(f"\n{'─' * 60}\nSUMMARY {name}")
    print(f"  test_auc = {mean:.4f} ± {std:.4f}   (val_auc={np.mean(all_val):.4f})  "
          f"[n={len(all_test)}]")
    print(f"{'─' * 60}")
    return {"name": name, "test_auc_mean": mean, "test_auc_std": std}


EXPERIMENTS = [
    # Coverage scaling (rank=4 fixed)
    ("ls-blocks-16",
     "HK↗ + LoRA rank=4 last-16 blocks",
     {**BASE, "lora_rank": 4, "lora_n_blocks": 16, "lora_block_start": -1}),
    ("ls-blocks-24",
     "HK↗ + LoRA rank=4 last-24 blocks",
     {**BASE, "lora_rank": 4, "lora_n_blocks": 24, "lora_block_start": -1}),
    ("ls-blocks-48",
     "HK↗ + LoRA rank=4 all 48 blocks",
     {**BASE, "lora_rank": 4, "lora_n_blocks": 48, "lora_block_start": 0}),
    # Rank scaling (blocks=last-8 fixed)
    ("ls-rank-2",
     "HK↗ + LoRA rank=2 last-8 blocks",
     {**BASE, "lora_rank": 2, "lora_n_blocks": 8, "lora_block_start": -1}),
    ("ls-rank-8",
     "HK↗ + LoRA rank=8 last-8 blocks",
     {**BASE, "lora_rank": 8, "lora_n_blocks": 8, "lora_block_start": -1}),
    ("ls-rank-16",
     "HK↗ + LoRA rank=16 last-8 blocks",
     {**BASE, "lora_rank": 16, "lora_n_blocks": 8, "lora_block_start": -1}),
    # Combined high capacity
    ("ls-rank8-blocks-16",
     "HK↗ + LoRA rank=8 last-16 blocks",
     {**BASE, "lora_rank": 8, "lora_n_blocks": 16, "lora_block_start": -1}),
    ("ls-rank8-blocks-48",
     "HK↗ + LoRA rank=8 all 48 blocks",
     {**BASE, "lora_rank": 8, "lora_n_blocks": 48, "lora_block_start": 0}),
]


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                    "steps\tpeak_vram_mb\telapsed_s\tdesc\n")

    summaries = []
    for name, desc, hparams in EXPERIMENTS:
        summaries.append(run_one(name, desc, hparams))

    print(f"\n{'=' * 60}\nFINAL SUMMARY (LoRA scaling on HiddenKey↗)\n{'=' * 60}")
    print(f"{'method':<28}  test_auc")
    for s in summaries:
        print(f"{s['name']:<28}  {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f}")
