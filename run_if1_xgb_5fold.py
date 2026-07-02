"""5-fold CV: DiscoTope-style XGBoost on frozen ESM-IF1 embeddings.

Same recipe as discotope-if1 (frozen IF1(512) + one-hot(20) + RSA(1) ->
XGBoost-100) but over all 5 partitions as holdout (1..5), so it is directly
comparable to run_if1_lora_5fold.py. Reuses the functions in
run_bepipred_discotope.py with the IF1 embedding cache.

Run:  uv run python run_if1_xgb_5fold.py > /tmp/run_if1_xgb_5f.log 2>&1
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from prepare import load_combined_fasta_partitioned

import run_bepipred_discotope as D
D.USE_ESMIF1 = True          # read by extract_all_embeddings (IF1 cache)
D.EMBED_DIM = 512
D.FEAT_DIM = 512 + 20 + 1

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

RESULTS_TSV = Path("results.tsv")
FOLDS = ["1", "2", "3", "4", "5"]
NAME = "if1-xgb-5f"
DESC = "5-fold: frozen ESM-IF1(512)+one-hot+RSA -> XGBoost-100 (discotope-if1 recipe)"

try:
    COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    COMMIT = "unknown"

if __name__ == "__main__":
    by_part = load_combined_fasta_partitioned(D.BEPIPRED_FASTA, exclude_partitions=frozenset({"EVAL"}))
    all_samples = [s for part in by_part.values() for s in part]
    emb = D.extract_all_embeddings(all_samples)
    logger.info(f"IF1 embeddings ready: {len(emb)}")

    fold_aucs = []
    for holdout in FOLDS:
        t0 = time.time()
        train_samples = [s for k, v in by_part.items() if k != holdout for s in v]
        test_samples = by_part.get(holdout, [])
        X_tr, y_tr = D.build_feature_matrix(train_samples, emb)
        X_te, y_te = D.build_feature_matrix(test_samples, emb)
        models = D.train_xgb_ensemble(X_tr, y_tr)
        preds = D.predict_ensemble(models, X_te)
        auc = float(roc_auc_score(y_te, preds))
        elapsed = time.time() - t0
        fold_aucs.append(auc)
        print(f"  fold={holdout}: test_auc={auc:.4f}  test={X_te.shape}  {elapsed:.0f}s")
        with open(RESULTS_TSV, "a") as f:
            f.write(f"{COMMIT}\t{NAME}\t{holdout}\t0\tnan\t{auc:.6f}\t{auc:.6f}\t0\t0\t{elapsed:.1f}\t{DESC}\n")
        del models

    print(f"\nSUMMARY {NAME}: test_auc = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}  (n={len(fold_aucs)})")
