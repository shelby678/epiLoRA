"""5-fold CV: ESM-IF1 + LoRA + RYS (the fine-tuned model). Holdout = each of the
5 partitions in turn, directly comparable to run_if1_xgb_5fold.py.

Reuses the model/training code in run_ensemble_esmif1.py. Run in the py3.9
fair-esm env:
    /home/sferrier/epitope_mapping/discotope3_web/env/bin/python run_if1_lora_5fold.py
"""

from __future__ import annotations

import gc
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import run_ensemble_esmif1 as R

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_TSV = R.REPO / "results.tsv"
FOLDS = ["1", "2", "3", "4", "5"]
NAME = "if1-lora-rys-5f"
DESC = "5-fold: ESM-IF1 + LoRA r4 all-8 + RYS(4,8) + linear head -> BCE"

try:
    COMMIT = subprocess.check_output(["git", "-C", str(R.REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    COMMIT = "unknown"

if __name__ == "__main__":
    import esm
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
        with open(RESULTS_TSV, "a") as f:
            f.write(f"{COMMIT}\t{NAME}\t{holdout}\t0\tnan\t{auc:.6f}\t{auc:.6f}\t{res['steps']}\t0\t{elapsed:.1f}\t{DESC}\n")
        del model
        gc.collect()
        if R.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nSUMMARY {NAME}: test_auc = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}  (n={len(fold_aucs)})")
