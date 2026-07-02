"""Train two ESM3+LoRA+RYS models on BEPIPRED, evaluate on sabdab holdouts.

Setup:
  - Architecture: RYS(36,44) + LoRA(rank=4, alpha=8, n_blocks=8) — best known config
  - Training data: BEPIPRED partitions 1+2+3+5, val = partition 4
  - Test (holdout 1): data/holdout1.fasta  (120 sabdab antigens, ≤30% sim to BEPIPRED)
  - Test (holdout 2): data/holdout2.fasta  (120 sabdab antigens, ≤30% sim to BEPIPRED)

Both models train on the same BEPIPRED data (no overlap with sabdab holdouts by construction).

Run:
    /path/to/python run_sabdab_holdout.py > run_sabdab_holdout.log 2>&1
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train_struct import (
    BATCH_SIZE, DROPOUT, LR, MAX_SEQ_LEN, WARMUP_STEPS, WEIGHT_DECAY,
    StructureEpitopePredictionModel, StructSample,
    _load_coords_rsa, train, compute_roc_auc,
)
from prepare import load_combined_fasta_partitioned, load_combined_fasta

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
HOLDOUT1_FASTA = Path("data/holdout1.fasta")
HOLDOUT2_FASTA = Path("data/holdout2.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV    = Path("results.tsv")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

VAL_PARTITION  = "4"   # held out from BEPIPRED for early stopping

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

HPARAMS = dict(
    rys_start=36, rys_end=44,
    lora_rank=4, lora_alpha=8.0, lora_n_blocks=8, lora_block_start=-1,
    dropout=DROPOUT, batch_size=BATCH_SIZE, lr=LR,
    weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS,
    val_eval_interval=200, patience=5,
    compute_auc=True, device=DEVICE,
    max_seconds=1200,
)


def load_bepipred_train_val() -> tuple[list[StructSample], list[StructSample]]:
    """Load BEPIPRED: train on all partitions except VAL_PARTITION and EVAL."""
    by_part = load_combined_fasta_partitioned(
        BEPIPRED_FASTA, exclude_partitions=frozenset({"EVAL"})
    )

    train_headers: list = []
    val_headers:   list = []
    for part_id, samples in by_part.items():
        if part_id == VAL_PARTITION:
            val_headers.extend(samples)
        else:
            train_headers.extend(samples)

    def _to_struct(header_samples: list) -> list[StructSample]:
        out: list[StructSample] = []
        for header, (token_ids, labels) in header_samples:
            if len(token_ids) > MAX_SEQ_LEN:
                continue
            seq_len = len(token_ids) - 2
            coords, rsa = _load_coords_rsa(header, seq_len, STRUCTURES_DIR)
            out.append((token_ids, labels, coords, rsa))
        return out

    return _to_struct(train_headers), _to_struct(val_headers)


def load_holdout(fasta_path: Path) -> list[StructSample]:
    """Load a holdout FASTA as StructSamples (with structures)."""
    headers, _ = load_combined_fasta(fasta_path, val_partition="__none__")
    out: list[StructSample] = []
    for header, (token_ids, labels) in headers:
        if len(token_ids) > MAX_SEQ_LEN:
            continue
        seq_len = len(token_ids) - 2
        coords, rsa = _load_coords_rsa(header, seq_len, STRUCTURES_DIR)
        out.append((token_ids, labels, coords, rsa))
    return out


def run_holdout(holdout_idx: int, holdout_fasta: Path,
                train_data: list[StructSample], val_data: list[StructSample]) -> None:
    name = f"sabdab-holdout-bepipred-{holdout_idx}"
    desc = (f"BEPIPRED train (val=part{VAL_PARTITION}), "
            f"test={holdout_fasta.name}")

    logger.info(f"Loading holdout {holdout_idx} test set ...")
    test_data = load_holdout(holdout_fasta)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {desc}")
    print(f"  train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = train(train_data, val_data, **HPARAMS)
    elapsed = time.time() - t0

    # Reload best weights into a fresh model for evaluation
    model = StructureEpitopePredictionModel(
        dropout=HPARAMS["dropout"],
        rys_start=HPARAMS["rys_start"], rys_end=HPARAMS["rys_end"],
        lora_rank=HPARAMS["lora_rank"], lora_alpha=HPARAMS["lora_alpha"],
        lora_n_blocks=HPARAMS["lora_n_blocks"],
        lora_block_start=HPARAMS.get("lora_block_start", -1),
    ).to(DEVICE)
    cur = model.state_dict()
    cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
    model.load_state_dict(cur)

    val_auc  = result["roc_auc"]
    test_auc = compute_roc_auc(model, test_data, batch_size=BATCH_SIZE, device=DEVICE)

    del model, cur
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(
        f"  val_auc={val_auc:.4f}  holdout_auc={test_auc:.4f}  "
        f"val_loss={result['val_loss']:.4f}  {elapsed:.0f}s"
    )

    with open(RESULTS_TSV, "a") as f:
        f.write(
            f"{COMMIT}\t{name}\t{holdout_idx}\t0\t"
            f"{result['val_loss']:.6f}\t{val_auc:.6f}\t{test_auc:.6f}\t"
            f"{result['steps']}\t{result['peak_vram_mb']}\t{elapsed:.1f}\t"
            f"{desc}\n"
        )


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(
                "commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                "steps\tpeak_vram_mb\telapsed_s\tdesc\n"
            )

    logger.info("Loading BEPIPRED training data ...")
    train_data, val_data = load_bepipred_train_val()
    logger.info(f"BEPIPRED: train={len(train_data)}  val={len(val_data)}")

    # Train once, evaluate on both holdouts
    logger.info("Training single model on BEPIPRED ...")
    t0 = time.time()
    result = train(train_data, val_data, **HPARAMS)
    elapsed = time.time() - t0

    model = StructureEpitopePredictionModel(
        dropout=HPARAMS["dropout"],
        rys_start=HPARAMS["rys_start"], rys_end=HPARAMS["rys_end"],
        lora_rank=HPARAMS["lora_rank"], lora_alpha=HPARAMS["lora_alpha"],
        lora_n_blocks=HPARAMS["lora_n_blocks"],
        lora_block_start=HPARAMS.get("lora_block_start", -1),
    ).to(DEVICE)
    cur = model.state_dict()
    cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
    model.load_state_dict(cur)

    val_auc = result["roc_auc"]
    print(f"\nval_auc={val_auc:.4f}  val_loss={result['val_loss']:.4f}  {elapsed:.0f}s")

    for holdout_idx, holdout_fasta in [(1, HOLDOUT1_FASTA), (2, HOLDOUT2_FASTA)]:
        test_data = load_holdout(holdout_fasta)
        test_auc = compute_roc_auc(model, test_data, batch_size=BATCH_SIZE, device=DEVICE)
        desc = f"BEPIPRED train (val=part{VAL_PARTITION}), test={holdout_fasta.name}"
        print(f"  holdout{holdout_idx}_auc={test_auc:.4f}")
        with open(RESULTS_TSV, "a") as f:
            f.write(
                f"{COMMIT}\tsabdab-holdout-bepipred\t{holdout_idx}\t0\t"
                f"{result['val_loss']:.6f}\t{val_auc:.6f}\t{test_auc:.6f}\t"
                f"{result['steps']}\t{result['peak_vram_mb']}\t{elapsed:.1f}\t"
                f"{desc}\n"
            )

    del model, cur
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
