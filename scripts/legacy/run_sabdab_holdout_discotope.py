"""Retrain DiscoTope-3.0 style model on BEPIPRED, evaluate on sabdab holdouts.

Architecture (identical to DiscoTope-3.0):
  ESM-IF1 (512-dim) + one-hot AA (20-dim) + RSA (1-dim) → XGBoost ensemble (100 models)

Training data: BEPIPRED partitions 1+2+3+5, val = partition 4
Test: holdout1.fasta and holdout2.fasta (120 sabdab antigens each, ≤30% sim to BEPIPRED)

Requires ESM-IF1 embeddings for holdout entries — run extract_esmif1_embeddings.py first.

Run:
    /path/to/python run_sabdab_holdout_discotope.py > run_sabdab_holdout_discotope.log 2>&1
"""

from __future__ import annotations

import gc
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Reuse helpers from run_bepipred_discotope
from run_bepipred_discotope import (
    one_hot_aa, extract_all_embeddings, build_feature_matrix,
    train_xgb_ensemble, predict_ensemble,
    BEPIPRED_FASTA, STRUCTURES_DIR, ESMIF1_CACHE_DIR, RESULTS_TSV,
    FEAT_DIM,
)
from prepare import load_combined_fasta_partitioned, load_combined_fasta
from train_struct import _parse_seq_id
from features import compute_rsa

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

HOLDOUT1_FASTA = Path("data/holdout1.fasta")
HOLDOUT2_FASTA = Path("data/holdout2.fasta")
VAL_PARTITION  = "4"

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"


def load_holdout_samples(fasta_path: Path) -> list:
    """Load holdout FASTA as a list of (header, (token_ids, labels)) matching prepare.py format."""
    headers, _ = load_combined_fasta(fasta_path, val_partition="__none__")
    return headers


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(
                "commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                "steps\tpeak_vram_mb\telapsed_s\tdesc\n"
            )

    name = "discotope3-sabdab-holdout"
    desc = "DiscoTope-3.0 retrained on BEPIPRED → evaluated on sabdab holdouts (≤30% sim)"

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")

    # ── Load BEPIPRED training data ────────────────────────────────────────────
    by_part = load_combined_fasta_partitioned(
        BEPIPRED_FASTA, exclude_partitions=frozenset({"EVAL"})
    )
    train_samples, val_samples = [], []
    for part_id, samples in by_part.items():
        if part_id == VAL_PARTITION:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

    logger.info(f"BEPIPRED: train={len(train_samples)}  val={len(val_samples)}")

    # ── Load holdout samples ───────────────────────────────────────────────────
    holdout1_samples = load_holdout_samples(HOLDOUT1_FASTA)
    holdout2_samples = load_holdout_samples(HOLDOUT2_FASTA)
    logger.info(f"Holdout1: {len(holdout1_samples)}  Holdout2: {len(holdout2_samples)}")

    # ── Extract / load ESM-IF1 embeddings ─────────────────────────────────────
    all_samples = train_samples + val_samples + holdout1_samples + holdout2_samples
    logger.info("Loading ESM-IF1 embeddings ...")
    embeddings = extract_all_embeddings(all_samples)
    logger.info(f"Embeddings loaded: {len(embeddings)}")

    # ── Build feature matrices ─────────────────────────────────────────────────
    logger.info("Building feature matrices ...")
    X_train, y_train = build_feature_matrix(train_samples, embeddings)
    X_val,   y_val   = build_feature_matrix(val_samples,   embeddings)
    X_h1,    y_h1    = build_feature_matrix(holdout1_samples, embeddings)
    X_h2,    y_h2    = build_feature_matrix(holdout2_samples, embeddings)

    logger.info(f"Train: {X_train.shape}  Val: {X_val.shape}")
    logger.info(f"Holdout1: {X_h1.shape}  Holdout2: {X_h2.shape}")

    # ── Train XGBoost ensemble ─────────────────────────────────────────────────
    logger.info("Training XGBoost ensemble (100 models) ...")
    t0 = time.time()
    models = train_xgb_ensemble(X_train, y_train)
    elapsed = time.time() - t0
    logger.info(f"Training done in {elapsed:.0f}s")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    val_preds = predict_ensemble(models, X_val)
    val_auc   = roc_auc_score(y_val, val_preds)

    h1_preds  = predict_ensemble(models, X_h1)
    h1_auc    = roc_auc_score(y_h1, h1_preds)

    h2_preds  = predict_ensemble(models, X_h2)
    h2_auc    = roc_auc_score(y_h2, h2_preds)

    print(f"\n  val_auc={val_auc:.4f}  holdout1_auc={h1_auc:.4f}  holdout2_auc={h2_auc:.4f}")
    print(f"  mean_holdout_auc={np.mean([h1_auc, h2_auc]):.4f}  elapsed={elapsed:.0f}s")

    for holdout_idx, auc in [(1, h1_auc), (2, h2_auc)]:
        with open(RESULTS_TSV, "a") as f:
            f.write(
                f"{COMMIT}\t{name}\t{holdout_idx}\t0\t"
                f"nan\t{val_auc:.6f}\t{auc:.6f}\t"
                f"nan\tnan\t{elapsed:.1f}\t{desc}\n"
            )
