"""BEPIPRED HiddenKey-eXtended (hkx) sweep: combine the best HiddenKey config
with orthogonal regularizers.

Background (from `findings/hiddenkey_lora_dropout.md`):
  HiddenKey↗ at p=0.10/0.10 (DropKey col + HiddenCut elem, no KL) is the
  best paper-style dropout we found: test_auc 0.7495 ± 0.024 on BEPIPRED
  3-fold CV. Higher p (0.20/0.20 or 0.30) and added KL loss both regressed.

  layerdrop-active-2-30 (per-batch drop top-2 LoRA blocks by activation EMA)
  is the best LoRA-block-level regularizer from `run_bepipred_dropout.py`:
  0.7355 alone. RYS(36, 44) was the previous best architectural lever
  (sequence-level val_loss 0.265 before this sweep started).

The paper's regularizers and these residual-stream / depth regularizers
target different parts of the model — HiddenKey↗ on attn logits + FFN
hidden, LayerDrop on residual addition, RYS on layer-execution count.
They should be at least partially additive.

Experiments (4 × 3 folds = 12 fold-runs):
  hkx-arrow-layerdrop  — HiddenKey↗ + LayerDrop active-2-30
  hkx-arrow-rys        — HiddenKey↗ + RYS(36, 44)
  hkx-arrow-layerdrop-rys — HiddenKey↗ + LayerDrop + RYS (all three)
  hkx-arrow-dropattn   — HiddenKey↗ + DropAttention 0.05 (incorporate the
                          third paper drop at a low rate; surprisingly
                          strong alone at p=0.10 → 0.7441)

Run:
    .venv/bin/python run_bepipred_hkx.py >> /tmp/run_hiddenkey.log 2>&1
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

# Same backbone as hk-* sweep: LoRA rank=4 last-8, head dropout 0.1.
BASE_CONFIG = dict(
    rys_start=0, rys_end=0,
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
    paper_drop_only_lora=True,
    layer_drop_only_lora=True,
)

# HiddenKey↗ baseline (sweep 1 winner).
HK_ARROW = dict(BASE_CONFIG, dropkey_prob=0.10, hiddencut_prob=0.10)


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
    ("hkx-arrow-layerdrop",
     "HiddenKey↗ (DropKey 0.10 col + HiddenCut 0.10 elem) + LayerDrop active top-2 p=0.30",
     {**HK_ARROW,
      "layer_drop_mode": "active", "layer_drop_prob": 0.30, "layer_drop_topk": 2}),

    ("hkx-arrow-rys",
     "HiddenKey↗ + RYS(36, 44) — free depth boost on top of best dropout",
     {**HK_ARROW, "rys_start": 36, "rys_end": 44}),

    ("hkx-arrow-layerdrop-rys",
     "HiddenKey↗ + LayerDrop active-2-30 + RYS(36, 44) — all three regularizers",
     {**HK_ARROW,
      "rys_start": 36, "rys_end": 44,
      "layer_drop_mode": "active", "layer_drop_prob": 0.30, "layer_drop_topk": 2}),

    ("hkx-arrow-dropattn",
     "HiddenKey↗ + DropAttention col 0.05 — fold in the third paper drop at low rate",
     {**HK_ARROW, "dropattn_prob": 0.05}),
]


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                    "steps\tpeak_vram_mb\telapsed_s\tdesc\n")

    summaries = []
    for name, desc, hparams in EXPERIMENTS:
        summaries.append(run_one(name, desc, hparams))

    print(f"\n{'=' * 60}\nFINAL SUMMARY (HKX: HiddenKey + orthogonal regs)\n{'=' * 60}")
    print(f"{'method':<32}  test_auc")
    for s in summaries:
        print(f"{s['name']:<32}  {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f}")
