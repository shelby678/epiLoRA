"""BEPIPRED probe-driven LayerDrop sweep.

Tests the variant of "active LayerDrop" we hadn't tried: instead of
ranking blocks by their EMA residual norm *during training* (which the
earlier `layerdrop-active-2-30` did), this uses the **static activation
ranking from `data/activation_probe.pt`** — the mean |context_BHLD| each
block produces when the frozen ESM3 backbone forwards every BEPIPRED
antigen.

The probe ranking is stable, sequence-level, and computed before any
LoRA adaptation, so it isolates "blocks that dominate the frozen model
on antigen inputs" — which may or may not be the right thing to drop
during fine-tuning. The earlier in-training EMA active LayerDrop
(test_auc 0.7355) lost to HiddenKey↗ alone (0.7495), and combining
LayerDrop with HiddenKey↗ regressed further. This sweep tests whether
the probe-derived schedule does better than the in-training EMA.

All experiments pair with HiddenKey↗ (DropKey 0.10 + HiddenCut 0.10).

Experiments (3 × 3 folds = 9 fold-runs, ~70 min):
  pld-top4-p20  — drop each of blocks {44,45,46,47} iid p=0.20 (~0.8 layers/step dropped)
  pld-top4-p30  — same blocks, iid p=0.30 (~1.2 layers/step dropped)
  pld-47-p50    — drop ONLY block 47 (the single dominant layer) with p=0.50,
                  to force the model to route around its top-1 dependency
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
PROBE_PATH = Path("data/activation_probe.pt")
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


def top_k_blocks_from_probe(k: int) -> list[int]:
    """Return the indices of the k most-activated blocks (descending |context|)."""
    data = torch.load(PROBE_PATH, weights_only=False)
    block_mag = data["per_block_mag"]  # (n_blocks,) sum over heads
    return torch.topk(block_mag, k=k).indices.sort().values.tolist()


BASE = dict(
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
    dropkey_prob=0.10,
    hiddencut_prob=0.10,
    layer_drop_only_lora=False,  # we explicitly pass layer_drop_layers
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


# Probe-derived eligibility — passed via train()'s LayerDrop wiring. Note:
# train() currently picks layer_drop_layers from the LoRA range when
# layer_drop_only_lora=True; here we pass layer_drop_only_lora=False and
# need a way to set layer_drop_layers from outside. The simplest path is
# to override _configure_dropout's eligibility list by passing a custom
# kwarg through train() — but train() doesn't expose that today.
# Workaround: monkey-patch the model's _layer_drop_cfg after model init,
# OR add a new train() kwarg. We'll add the kwarg.

TOP_4 = top_k_blocks_from_probe(4)   # expect [44, 45, 46, 47]
TOP_1 = top_k_blocks_from_probe(1)   # expect [47]
print(f"[probe] top-4 blocks (sorted): {TOP_4}")
print(f"[probe] top-1 block:          {TOP_1}")


def run_one(name: str, desc: str, hparams: dict) -> dict:
    fold_test_aucs, fold_val_aucs = [], []
    done = _already_done()
    print(f"\n{'=' * 60}\nEXPERIMENT: {name}\n  {desc}\n{'=' * 60}")
    if "layer_drop_target_blocks" in hparams:
        print(f"  layer_drop_target_blocks = {hparams['layer_drop_target_blocks']}")

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
    ("pld-top4-p20",
     f"HK↗ + probe-uniform LayerDrop on blocks {TOP_4}, each iid p=0.20",
     {**BASE, "layer_drop_mode": "uniform", "layer_drop_prob": 0.20,
      "layer_drop_target_blocks": TOP_4}),
    ("pld-top4-p30",
     f"HK↗ + probe-uniform LayerDrop on blocks {TOP_4}, each iid p=0.30",
     {**BASE, "layer_drop_mode": "uniform", "layer_drop_prob": 0.30,
      "layer_drop_target_blocks": TOP_4}),
    ("pld-47-p50",
     f"HK↗ + drop ONLY block 47 (probe top-1) with p=0.50",
     {**BASE, "layer_drop_mode": "uniform", "layer_drop_prob": 0.50,
      "layer_drop_target_blocks": TOP_1}),
]


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                    "steps\tpeak_vram_mb\telapsed_s\tdesc\n")

    summaries = []
    for name, desc, hparams in EXPERIMENTS:
        summaries.append(run_one(name, desc, hparams))

    print(f"\n{'=' * 60}\nFINAL SUMMARY (Probe-driven LayerDrop)\n{'=' * 60}")
    print(f"{'method':<20}  test_auc")
    for s in summaries:
        print(f"{s['name']:<20}  {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f}")
