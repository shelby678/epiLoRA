"""DiscoTope-style model on BEPIPRED: ESM3 embeddings + RSA + one-hot → XGBoost ensemble.

Replicates DiscoTope-3.0 architecture using our ESM3 backbone instead of ESM-IF1:
  - Per-residue features: ESM3 embeddings (1536-dim) + RSA (1-dim) + one-hot AA (20-dim) = 1557-dim
  - Model: XGBoost ensemble (100 models, same hyperparams as DiscoTope-3.0 args.json)
  - Same 3-fold CV as baseline (holdout = val = test)

This compares:
  Our baseline: ESM3 frozen backbone + trainable LoRA head (neural classifier)
  This script:  ESM3 frozen backbone + XGBoost ensemble  (gradient boosting)

Run:
    uv run python run_bepipred_discotope.py > run_discotope.log 2>&1
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from prepare import load_combined_fasta_partitioned, ID_TO_TOKEN
from features import compute_rsa
from train_struct import (
    MAX_SEQ_LEN, StructureEpitopePredictionModel,
    _parse_seq_id, parse_pdb_backbone,
)


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA     = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR     = Path("data/structures2/sabdab_dataset")
RESULTS_TSV        = Path("results.tsv")
ESM3_CACHE_DIR     = Path("data/esm3_embed_cache")
ESMIF1_CACHE_DIR   = Path("data/esmif1_embed_cache")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Set to True to use ESM-IF1 (512-dim, DiscoTope original backbone)
# Set to False to use ESM3 (1536-dim, our backbone)
USE_ESMIF1 = True

EMBED_DIM  = 512 if USE_ESMIF1 else 1536
FEAT_DIM   = EMBED_DIM + 20 + 1  # embeddings + one-hot AA + RSA

CV_FOLDS = [
    {"holdout": "1"},
    {"holdout": "2"},
    {"holdout": "3"},
]

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

# ── One-hot AA encoding ────────────────────────────────────────────────────────
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX   = {aa: i for i, aa in enumerate(_AA_ORDER)}

def one_hot_aa(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), 20), dtype=np.float32)
    for i, aa in enumerate(seq.upper()):
        if aa in _AA_IDX:
            arr[i, _AA_IDX[aa]] = 1.0
    return arr


# ── Embedding loading ──────────────────────────────────────────────────────────
# ESM-IF1 embeddings are pre-extracted by extract_esmif1_embeddings.py (discotope env).
# ESM3 embeddings are extracted on-demand below.

_last_hidden: list[torch.Tensor] = []

def _make_esm3() -> StructureEpitopePredictionModel:
    model = StructureEpitopePredictionModel(
        dropout=0.0, rys_start=36, rys_end=44,
        lora_rank=0, extra_dim=0,
    ).to(DEVICE)
    model.eval()
    def _hook(module, inp, out):
        _last_hidden.clear()
        _last_hidden.append(out.detach())
    model.head_ln.register_forward_hook(_hook)
    return model


def extract_all_embeddings(all_samples: list) -> dict[str, np.ndarray]:
    """Load per-residue embeddings for all samples.

    If USE_ESMIF1=True, reads from esmif1_embed_cache (pre-computed by extract_esmif1_embeddings.py).
    If USE_ESMIF1=False, computes ESM3 embeddings on-demand and caches to esm3_embed_cache.
    Returns header → (L, D) float32 array (BOS/EOS stripped).
    """
    cache_dir = ESMIF1_CACHE_DIR if USE_ESMIF1 else ESM3_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, np.ndarray] = {}

    if USE_ESMIF1:
        # Just load from pre-computed cache
        missing = 0
        for header, _ in all_samples:
            key  = header.replace(" ", "_").replace("/", "-")
            path = cache_dir / f"{key}.npy"
            if path.exists():
                results[header] = np.load(path)
            else:
                missing += 1
        if missing:
            logger.warning(f"{missing} ESM-IF1 embeddings missing from cache — run extract_esmif1_embeddings.py first")
        return results

    # ESM3: compute on-demand
    to_compute = [
        (h, tids) for h, (tids, _) in all_samples
        if len(tids) <= MAX_SEQ_LEN
        and not (cache_dir / f"{h.replace(' ', '_').replace('/', '-')}.npy").exists()
    ]

    if to_compute:
        logger.info(f"Computing {len(to_compute)} ESM3 embeddings ...")
        model = _make_esm3()
        for i, (header, token_ids) in enumerate(to_compute):
            seq_len = len(token_ids) - 2
            ids_t   = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            mask    = torch.ones_like(ids_t, dtype=torch.bool)
            coords  = torch.full((1, seq_len + 2, 3, 3), float("nan"),
                                 dtype=torch.float32, device=DEVICE)
            parsed  = _parse_seq_id(header)
            if parsed:
                pdb_id, chain = parsed
                pdb_path = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
                if pdb_path.exists():
                    flat = parse_pdb_backbone(pdb_path, chain)
                    if flat is not None and len(flat) == seq_len:
                        coords[0, 1:-1] = torch.tensor(flat, dtype=torch.float32,
                                                        device=DEVICE).view(seq_len, 3, 3)
            with torch.no_grad():
                model(ids_t, attention_mask=mask, structure_coords=coords)
            emb = _last_hidden[0][0, 1:-1, :].cpu().numpy()
            if emb.shape[0] == seq_len:
                key = header.replace(" ", "_").replace("/", "-")
                np.save(cache_dir / f"{key}.npy", emb)
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(to_compute)} done")
        del model; gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    for header, (token_ids, _) in all_samples:
        key  = header.replace(" ", "_").replace("/", "-")
        path = cache_dir / f"{key}.npy"
        if path.exists():
            results[header] = np.load(path)
    return results


# ── Feature + label assembly ───────────────────────────────────────────────────
def build_feature_matrix(
    samples: list,
    embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N_residues, 1557) feature matrix and binary labels."""
    X_list, y_list = [], []

    for header, (token_ids, labels) in samples:
        if header not in embeddings:
            continue

        emb     = embeddings[header]   # (L, D)
        seq_len = len(token_ids) - 2   # strip BOS/EOS

        if emb.shape[0] != seq_len:
            logger.warning(f"Length mismatch for '{header}': emb={emb.shape[0]} vs seq={seq_len}, skipping")
            continue

        # Decode AA sequence — only keep single-char tokens (multi-char = <unk> etc → 'X')
        aa_seq = "".join(
            t if len(t) == 1 else "X"
            for t in (ID_TO_TOKEN.get(tok, "X") for tok in token_ids[1:-1])
        )
        oh     = one_hot_aa(aa_seq)    # (L, 20)

        # RSA
        rsa_arr = np.zeros(seq_len, dtype=np.float32)
        parsed  = _parse_seq_id(header)
        if parsed:
            pdb_id, chain = parsed
            pdb_path = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
            if pdb_path.exists():
                rsa = compute_rsa(pdb_path, chain, seq_len)
                if rsa is not None:
                    rsa_arr = rsa

        rsa_col = rsa_arr.reshape(-1, 1)  # (L, 1)

        # Labels: strip BOS/EOS (-100), keep valid positions
        lbl   = np.array(labels[1:-1], dtype=np.float32)
        valid = lbl >= 0

        X = np.concatenate([emb, oh, rsa_col], axis=1)[valid]
        y = lbl[valid]

        X_list.append(X)
        y_list.append(y)

    if not X_list:
        return np.empty((0, FEAT_DIM), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.vstack(X_list), np.concatenate(y_list)


# ── XGBoost ensemble (DiscoTope-3.0 hyperparams) ──────────────────────────────
def train_xgb_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_models: int = 100,
    seed: int = 42,
) -> list:
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    logger.info(f"  Train residues: {len(pos_idx)} epitope, {len(neg_idx)} non-epitope")

    pos_ratio  = 0.7
    neg_factor = 2.5
    rng        = np.random.default_rng(seed)
    models     = []

    for i in range(n_models):
        n_pos = max(1, int(len(pos_idx) * pos_ratio))
        n_neg = min(len(neg_idx), int(n_pos * neg_factor / pos_ratio))

        s_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        s_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        idx   = np.concatenate([s_pos, s_neg])
        rng.shuffle(idx)

        m = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.3,
            subsample=0.5,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            random_state=int(rng.integers(0, 2**31)),
            device="cuda" if DEVICE == "cuda" else "cpu",
        )
        m.fit(X_train[idx], y_train[idx])
        models.append(m)

        if (i + 1) % 25 == 0:
            logger.info(f"  Trained {i+1}/{n_models} XGB models")

    return models


def predict_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    preds = np.zeros(len(X), dtype=np.float64)
    for m in models:
        preds += m.predict_proba(X)[:, 1]
    return preds / len(models)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(
                "commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                "steps\tpeak_vram_mb\telapsed_s\tdesc\n"
            )

    backbone = "ESM-IF1(512)" if USE_ESMIF1 else "ESM3(1536)"
    name = "discotope-if1" if USE_ESMIF1 else "discotope-esm3"
    desc = f"DiscoTope-style: {backbone} + one-hot(20) + RSA(1) -> XGBoost ensemble-100"

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")

    # Load BEPIPRED data
    by_part = load_combined_fasta_partitioned(
        BEPIPRED_FASTA, exclude_partitions=frozenset({"EVAL"})
    )
    all_samples = [s for part in by_part.values() for s in part]
    logger.info(f"Total sequences: {len(all_samples)}")

    # Extract/load ESM3 embeddings (cached)
    all_embeddings = extract_all_embeddings(all_samples)
    logger.info(f"Embeddings ready: {len(all_embeddings)}")

    fold_aucs: list[float] = []

    for fold in CV_FOLDS:
        holdout = fold["holdout"]
        t0 = time.time()

        train_samples = [s for k, v in by_part.items() if k != holdout for s in v]
        test_samples  = by_part.get(holdout, [])

        logger.info(f"\nFold holdout={holdout}: train={len(train_samples)}, test={len(test_samples)}")

        X_train, y_train = build_feature_matrix(train_samples, all_embeddings)
        X_test,  y_test  = build_feature_matrix(test_samples,  all_embeddings)

        models = train_xgb_ensemble(X_train, y_train)

        preds = predict_ensemble(models, X_test)
        auc   = float(roc_auc_score(y_test, preds))
        elapsed = time.time() - t0

        fold_aucs.append(auc)
        print(f"  fold={holdout}: test_auc={auc:.4f}  {elapsed:.0f}s")

        with open(RESULTS_TSV, "a") as f:
            f.write(
                f"{COMMIT}\t{name}\t{holdout}\t0\t"
                f"nan\t{auc:.6f}\t{auc:.6f}\t"
                f"0\t0\t{elapsed:.1f}\t{desc}\n"
            )

        del models

    print(f"\n{'─'*60}")
    print(f"SUMMARY {name}")
    print(f"  test_auc = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"{'─'*60}")
