"""ESM-IF1 per-residue embeddings + XGBoost classifier for BCR epitope prediction.

Loads pdb_chains.fasta, extracts ESM-IF1 encoder embeddings for each antigen chain,
and trains an XGBoost binary classifier on per-residue embeddings → epitope labels.

Evaluation uses the same metrics as the main pipeline:
  - val_loss: mean binary cross-entropy on labelled residue positions
  - roc_auc:  token-level ROC-AUC on the validation set

Usage:
    python train_esm_if1.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from prepare import load_combined_fasta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMBINED_FASTA = Path("data/pdb_chains.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV    = Path("results.tsv")
VAL_PARTITION  = "5"
MAX_SEQ_LEN    = 512        # max tokens incl. BOS/EOS; longer entries skipped

# XGBoost defaults
XGB_N_ESTIMATORS   = 500
XGB_MAX_DEPTH       = 6
XGB_LEARNING_RATE   = 0.05
XGB_SUBSAMPLE       = 0.8
XGB_COLSAMPLE       = 0.8
XGB_DEVICE          = "cpu"   # or "cuda" if GPU available

# ESM-IF1 is loaded via torch.hub (weights cached at
#   ~/.cache/torch/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt)
ESM_HUB_REPO = "facebookresearch/esm:main"
ESM_IF1_NAME = "esm_if1_gvp4_t16_142M_UR50"


# ---------------------------------------------------------------------------
# Structure loading helpers (re-using the pure-Python PDB parser from
# train_struct.py to avoid an extra BioPython dependency in this script)
# ---------------------------------------------------------------------------


def _parse_header(header: str) -> tuple[str, str] | None:
    """Return (pdb_id_lower, antigen_chain) from a combined-FASTA header."""
    if " " in header:
        parts = header.split()
        pdb_id = parts[0].split("_")[0].lower()
        antigen = parts[1] if len(parts) >= 2 else None
    else:
        parts = header.split("_")
        if len(parts) < 4:
            return None
        pdb_id = parts[0].lower()
        antigen = parts[3]
    return (pdb_id, antigen) if antigen else None


def _load_esm_if1() -> tuple:
    """Load ESM-IF1 model and alphabet via torch.hub."""
    logger.info("Loading ESM-IF1 model via torch.hub …")
    model, alphabet = torch.hub.load(
        ESM_HUB_REPO, ESM_IF1_NAME, force_reload=False, verbose=False
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"ESM-IF1 loaded on {device}")
    return model, alphabet


@torch.no_grad()
def _get_embedding(
    model,
    alphabet,
    pdb_path: Path,
    chain: str,
) -> np.ndarray | None:
    """Return ESM-IF1 encoder embeddings for one PDB chain.

    Calls ``esm.inverse_folding.util.load_coords`` to get backbone coords and
    ``esm.inverse_folding.util.get_encoder_output`` to get the embedding.

    Returns:
        Float32 array of shape ``(seq_len, emb_dim)`` or ``None`` on error.
    """
    # ESM-IF1 code lives in the torch.hub cache
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_esm_main"
    if str(hub_dir) not in sys.path:
        sys.path.insert(0, str(hub_dir))

    try:
        import esm.inverse_folding.util as esm_if_util  # type: ignore[import]
        coords, seq = esm_if_util.load_coords(str(pdb_path), chain)
        emb = esm_if_util.get_encoder_output(model, alphabet, coords)  # (L, D)
        return emb.cpu().float().numpy()
    except Exception as exc:
        logger.debug(f"ESM-IF1 embedding failed for {pdb_path} chain {chain}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_residue_dataset(
    header_samples: list,
    structures_dir: Path,
    model,
    alphabet,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ESM-IF1 embeddings and per-residue labels for a list of samples.

    Args:
        header_samples: List of ``(header, (token_ids, labels))`` from
                        ``load_combined_fasta``.
        structures_dir: Root of SAbDab PDB dataset.
        model, alphabet: Loaded ESM-IF1 model and alphabet.
        max_length:     Token-length cutoff (incl. BOS/EOS); longer entries skipped.

    Returns:
        ``(X, y)`` — feature matrix ``(N_residues, emb_dim)`` and label vector
        ``(N_residues,)`` of 0/1 values.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for header, (token_ids, labels) in header_samples:
        if len(token_ids) > max_length:
            continue

        parsed = _parse_header(header)
        if parsed is None:
            continue
        pdb_id, antigen = parsed

        pdb_path = structures_dir / pdb_id / "structure" / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue

        emb = _get_embedding(model, alphabet, pdb_path, antigen)
        if emb is None:
            continue

        # Align embedding length with label positions
        # labels: [-100, l1, l2, ..., ln, -100]  → AA positions are 1..n
        aa_labels = np.array(labels[1:-1])   # strip BOS/EOS
        if len(emb) != len(aa_labels):
            continue                           # length mismatch — skip

        X_parts.append(emb)
        y_parts.append(aa_labels)

    if not X_parts:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int32)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train(
    train_samples: list,
    val_samples: list,
    structures_dir: Path,
    max_length: int = MAX_SEQ_LEN,
    n_estimators: int = XGB_N_ESTIMATORS,
    max_depth: int = XGB_MAX_DEPTH,
    learning_rate: float = XGB_LEARNING_RATE,
    subsample: float = XGB_SUBSAMPLE,
    colsample_bytree: float = XGB_COLSAMPLE,
) -> dict:
    """Train XGBoost on ESM-IF1 embeddings and evaluate on the validation set.

    Returns:
        Dict with keys ``val_loss``, ``roc_auc``, ``n_train``, ``n_val``,
        ``emb_dim``, ``n_estimators``.
    """
    t0 = time.time()
    model, alphabet = _load_esm_if1()

    logger.info("Extracting ESM-IF1 embeddings for training set …")
    X_tr, y_tr = build_residue_dataset(
        train_samples, structures_dir, model, alphabet, max_length
    )
    logger.info(f"  train: {X_tr.shape[0]:,} residues, {X_tr.shape[1]} dims")

    logger.info("Extracting ESM-IF1 embeddings for validation set …")
    X_val, y_val = build_residue_dataset(
        val_samples, structures_dir, model, alphabet, max_length
    )
    logger.info(f"  val:   {X_val.shape[0]:,} residues")

    del model  # free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if X_tr.shape[0] == 0 or X_val.shape[0] == 0:
        logger.error("Empty embedding set — check structures_dir path and FASTA entries.")
        return {"val_loss": float("inf"), "roc_auc": float("nan")}

    # Class balance for pos_weight
    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    scale_pos_weight = n_neg / max(1, n_pos)
    logger.info(f"Training labels: {n_pos:,} positives / {n_neg:,} negatives  "
                f"(scale_pos_weight={scale_pos_weight:.2f})")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        device=device_str,
        verbosity=0,
        n_jobs=-1,
    )

    logger.info(f"Fitting XGBoost ({n_estimators} trees, max_depth={max_depth}) …")
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    val_probs = clf.predict_proba(X_val)[:, 1]

    # val_loss: mean binary cross-entropy
    eps = 1e-7
    val_loss = float(
        -np.mean(
            y_val * np.log(val_probs + eps)
            + (1 - y_val) * np.log(1 - val_probs + eps)
        )
    )

    if len(np.unique(y_val)) >= 2:
        roc_auc = float(roc_auc_score(y_val, val_probs))
    else:
        roc_auc = float("nan")

    elapsed = time.time() - t0
    logger.info(
        f"val_loss={val_loss:.6f}  roc_auc={roc_auc:.6f}  "
        f"elapsed={elapsed:.0f}s"
    )

    return {
        "val_loss": val_loss,
        "roc_auc": roc_auc,
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_val.shape[0]),
        "emb_dim": int(X_tr.shape[1]),
        "n_estimators": n_estimators,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        _commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        _commit = "unknown"

    logger.info(f"Commit: {_commit}")

    # Load data
    train_samples, val_samples = load_combined_fasta(
        COMBINED_FASTA, val_partition=VAL_PARTITION
    )
    logger.info(
        f"Loaded {len(train_samples)} train / {len(val_samples)} val sequences "
        f"from {COMBINED_FASTA}"
    )

    # Experiment suite — vary XGBoost + PU hyperparameters
    EXPERIMENTS = [
        dict(name="esm_if1-exp1",  desc="ESM-IF1 + XGBoost baseline",
             n_estimators=500, max_depth=6, learning_rate=0.05),
        dict(name="esm_if1-exp2",  desc="ESM-IF1 + XGBoost deeper trees",
             n_estimators=500, max_depth=8, learning_rate=0.05),
        dict(name="esm_if1-exp3",  desc="ESM-IF1 + XGBoost lower lr",
             n_estimators=1000, max_depth=6, learning_rate=0.02),
    ]

    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"EXPERIMENT: {name} — {exp['desc']}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        t0 = time.time()
        result = train(
            train_samples,
            val_samples,
            structures_dir=STRUCTURES_DIR,
            max_length=MAX_SEQ_LEN,
            n_estimators=exp.get("n_estimators", XGB_N_ESTIMATORS),
            max_depth=exp.get("max_depth", XGB_MAX_DEPTH),
            learning_rate=exp.get("learning_rate", XGB_LEARNING_RATE),
            subsample=exp.get("subsample", XGB_SUBSAMPLE),
            colsample_bytree=exp.get("colsample_bytree", XGB_COLSAMPLE),
        )
        elapsed = time.time() - t0

        # Write results in the same TSV format as train_struct.py
        row = "\t".join([
            _commit,
            name,
            f"{result['val_loss']:.6f}",
            f"{result['roc_auc']:.6f}",
            "0",           # peak_vram_mb (N/A for XGBoost)
            f"{elapsed:.0f}s",
            exp["desc"],
        ])
        print(row)

        if RESULTS_TSV.exists():
            with open(RESULTS_TSV, "a") as f:
                f.write(row + "\n")
