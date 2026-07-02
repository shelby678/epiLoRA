"""5-fold CV on ESM-IF1 features: fine-tuned LoRA+RYS vs DiscoTope-style XGBoost.

Holdout = each of the 5 partitions in turn, so the two methods are directly
comparable.

    # LoRA+RYS fine-tune — run in the py3.9 fair-esm env:
    /home/sferrier/epitope_mapping/discotope3_web/env/bin/python run_if1_5fold.py lora

    # XGBoost on frozen IF1 embeddings — run in the uv env:
    uv run python run_if1_5fold.py xgb

Each method's heavy dependencies (esm / xgboost) are imported lazily inside its
branch, so the file loads in whichever environment you launch it from.
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_TSV = Path("results.tsv")
FOLDS = ["1", "2", "3", "4", "5"]


def _commit(repo: Path | None = None) -> str:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    if repo is not None:
        cmd = ["git", "-C", str(repo)] + cmd[1:]
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


def _write_row(results_tsv: Path, commit: str, name: str, holdout: str,
               auc: float, steps: int, elapsed: float, desc: str) -> None:
    with open(results_tsv, "a") as f:
        f.write(f"{commit}\t{name}\t{holdout}\t0\tnan\t{auc:.6f}\t{auc:.6f}\t"
                f"{steps}\t0\t{elapsed:.1f}\t{desc}\n")


def run_lora() -> None:
    """ESM-IF1 + LoRA + RYS fine-tune. Reuses model/training code from
    run_ensemble_esmif1.py (py3.9 fair-esm env)."""
    import torch
    import esm
    import run_ensemble_esmif1 as R

    name = "if1-lora-rys-5f"
    desc = "5-fold: ESM-IF1 + LoRA r4 all-8 + RYS(4,8) + linear head -> BCE"
    results_tsv = R.REPO / "results.tsv"
    commit = _commit(R.REPO)

    by_part = R.parse_bepipred(R.BEPIPRED_FASTA)
    esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_model = esm_model.eval()

    fold_aucs = []
    for holdout in FOLDS:
        train_entries = [e for k in FOLDS if k != holdout for e in by_part.get(k, [])]
        test_entries = by_part.get(holdout, [])
        logger.info(f"\n=== IF1 5-fold holdout={holdout}: loading structures ===")
        train_samples = R.load_samples(train_entries)
        test_samples = R.load_samples(test_entries)
        n_struct = sum(1 for *_, c in test_samples if c is not None)
        logger.info(f"  train={len(train_samples)} test={len(test_samples)} ({n_struct} w/ struct)")

        model = R.ESMIF1Model(esm_model, alphabet, R.LORA_RANK, R.LORA_ALPHA, R.LORA_LAYERS, 0.1).to(R.DEVICE)
        t0 = time.time()
        res = R.train_fold(model, train_samples, test_samples, R.MAX_SECONDS)
        auc = res["val_auc"]
        elapsed = time.time() - t0
        fold_aucs.append(auc)
        print(f"  fold={holdout}: test_auc={auc:.4f}  steps={res['steps']}  {elapsed:.0f}s")
        _write_row(results_tsv, commit, name, holdout, auc, res["steps"], elapsed, desc)
        del model
        gc.collect()
        if R.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nSUMMARY {name}: test_auc = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}  (n={len(fold_aucs)})")


def run_xgb() -> None:
    """DiscoTope-style XGBoost on frozen ESM-IF1 embeddings. Reuses the feature/
    training functions from run_bepipred_discotope.py with the IF1 cache (uv env)."""
    from sklearn.metrics import roc_auc_score
    from prepare import load_combined_fasta_partitioned
    import run_bepipred_discotope as D

    D.USE_ESMIF1 = True          # read by extract_all_embeddings (IF1 cache)
    D.EMBED_DIM = 512
    D.FEAT_DIM = 512 + 20 + 1

    name = "if1-xgb-5f"
    desc = "5-fold: frozen ESM-IF1(512)+one-hot+RSA -> XGBoost-100 (discotope-if1 recipe)"
    commit = _commit()

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
        _write_row(RESULTS_TSV, commit, name, holdout, auc, 0, elapsed, desc)
        del models

    print(f"\nSUMMARY {name}: test_auc = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}  (n={len(fold_aucs)})")


METHODS = {"lora": run_lora, "xgb": run_xgb}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=list(METHODS))
    args = parser.parse_args()
    METHODS[args.method]()


if __name__ == "__main__":
    main()
