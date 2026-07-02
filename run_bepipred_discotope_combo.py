"""DiscoTope-style XGBoost on CONCATENATED ESM3 + ESM-IF1 embeddings.

Tests whether ESM3 (masked-LM, 1536-dim) and ESM-IF1 (inverse-folding, 512-dim)
embeddings carry complementary signal for epitope prediction. Same recipe and
3-fold CV as run_bepipred_discotope.py; the only change is the feature vector:

  [ ESM3(1536) | ESM-IF1(512) | one-hot AA(20) | RSA(1) ]  = 2069-dim

Both embedding sets are read from their existing caches (data/esm3_embed_cache,
data/esmif1_embed_cache). Only residues from sequences present and
length-aligned in BOTH caches are used (724/770 sequences).

Run:
    uv run python run_bepipred_discotope_combo.py > /tmp/run_discotope_combo.log 2>&1
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from prepare import load_combined_fasta_partitioned, ID_TO_TOKEN
from features import compute_rsa
from train_struct import _parse_seq_id
from run_bepipred_discotope import one_hot_aa, train_xgb_ensemble, predict_ensemble

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV = Path("results.tsv")
ESM3_CACHE_DIR = Path("data/esm3_embed_cache")
ESMIF1_CACHE_DIR = Path("data/esmif1_embed_cache")

FEAT_DIM = 1536 + 512 + 20 + 1  # ESM3 + ESM-IF1 + one-hot + RSA

CV_FOLDS = [{"holdout": "1"}, {"holdout": "2"}, {"holdout": "3"}]

try:
    COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    COMMIT = "unknown"


def _load_cache(cache_dir: Path, samples: list) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for header, _ in samples:
        key = header.replace(" ", "_").replace("/", "-")
        path = cache_dir / f"{key}.npy"
        if path.exists():
            out[header] = np.load(path)
    return out


def build_combined_matrix(
    samples: list,
    esm3: dict[str, np.ndarray],
    if1: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for header, (token_ids, labels) in samples:
        if header not in esm3 or header not in if1:
            continue
        e3, e1 = esm3[header], if1[header]
        seq_len = len(token_ids) - 2
        if e3.shape[0] != seq_len or e1.shape[0] != seq_len:
            continue

        aa_seq = "".join(
            t if len(t) == 1 else "X"
            for t in (ID_TO_TOKEN.get(tok, "X") for tok in token_ids[1:-1])
        )
        oh = one_hot_aa(aa_seq)  # (L, 20)

        rsa_arr = np.zeros(seq_len, dtype=np.float32)
        parsed = _parse_seq_id(header)
        if parsed:
            pdb_id, chain = parsed
            pdb_path = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
            if pdb_path.exists():
                rsa = compute_rsa(pdb_path, chain, seq_len)
                if rsa is not None:
                    rsa_arr = rsa
        rsa_col = rsa_arr.reshape(-1, 1)

        lbl = np.array(labels[1:-1], dtype=np.float32)
        valid = lbl >= 0
        X = np.concatenate([e3, e1, oh, rsa_col], axis=1)[valid]
        y = lbl[valid]
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        return np.empty((0, FEAT_DIM), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.vstack(X_list), np.concatenate(y_list)


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\tsteps\tpeak_vram_mb\telapsed_s\tdesc\n")

    name = "discotope-combo"
    desc = "DiscoTope-style: ESM3(1536)+ESM-IF1(512)+one-hot(20)+RSA(1) -> XGBoost ensemble-100"
    print(f"\n{'='*60}\nEXPERIMENT: {name}\n  {desc}\n{'='*60}")

    by_part = load_combined_fasta_partitioned(BEPIPRED_FASTA, exclude_partitions=frozenset({"EVAL"}))
    all_samples = [s for part in by_part.values() for s in part]
    esm3 = _load_cache(ESM3_CACHE_DIR, all_samples)
    if1 = _load_cache(ESMIF1_CACHE_DIR, all_samples)
    both = sum(1 for h, _ in all_samples if h in esm3 and h in if1)
    logger.info(f"Total seqs: {len(all_samples)}  |  in both caches: {both}")

    fold_aucs: list[float] = []
    for fold in CV_FOLDS:
        holdout = fold["holdout"]
        t0 = time.time()
        train_samples = [s for k, v in by_part.items() if k != holdout for s in v]
        test_samples = by_part.get(holdout, [])

        X_train, y_train = build_combined_matrix(train_samples, esm3, if1)
        X_test, y_test = build_combined_matrix(test_samples, esm3, if1)
        logger.info(f"\nFold holdout={holdout}: train={X_train.shape}, test={X_test.shape}")

        models = train_xgb_ensemble(X_train, y_train)
        preds = predict_ensemble(models, X_test)
        auc = float(roc_auc_score(y_test, preds))
        elapsed = time.time() - t0
        fold_aucs.append(auc)
        print(f"  fold={holdout}: test_auc={auc:.4f}  {elapsed:.0f}s")

        with open(RESULTS_TSV, "a") as f:
            f.write(f"{COMMIT}\t{name}\t{holdout}\t0\tnan\t{auc:.6f}\t{auc:.6f}\t0\t0\t{elapsed:.1f}\t{desc}\n")
        del models

    print(f"\n{'-'*60}\nSUMMARY {name}")
    print(f"  test_auc = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"{'-'*60}")
