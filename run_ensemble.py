"""Ensemble members: train one epitope model per CV fold and dump per-residue
test predictions to data/ensemble_preds/<backbone>.npz for later ensembling.

    uv run python run_ensemble.py esm3
    uv run python run_ensemble.py esm2

Backbones:
    esm3   ESM3-small-open + LoRA rank=8 last-16 + RYS(36,44)   (train_struct)
    esm2   ESM2-650M       + LoRA rank=8 last-16 + RYS(24,30)   (train_esm2)

The ESM-IF1 ensemble member lives in run_ensemble_esmif1.py — it runs in the
py3.9 fair-esm environment and doubles as a library for the run_if1_5fold.py
5-fold comparison, so it is kept separate.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train_struct import MAX_SEQ_LEN
from ensemble_io import save_preds, cv_test_with_headers, predict_venv_model

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLDS = ["1", "2", "3"]
TIME_BUDGET = 1200


def _build_esm3():
    """Return (train_fn, model_factory, cfg) for the ESM3 backbone."""
    from train_struct import StructureEpitopePredictionModel, train
    cfg = dict(rys_start=36, rys_end=44,
               lora_rank=8, lora_alpha=8.0, lora_n_blocks=16, lora_block_start=-1,
               dropout=0.1)

    def train_fn(train_data, val_data):
        return train(train_data, val_data, max_seconds=TIME_BUDGET,
                     val_eval_interval=200, patience=5, compute_auc=True,
                     device=DEVICE, **cfg)

    def model_factory():
        return StructureEpitopePredictionModel(
            dropout=cfg["dropout"], rys_start=cfg["rys_start"], rys_end=cfg["rys_end"],
            lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            lora_n_blocks=cfg["lora_n_blocks"], lora_block_start=cfg["lora_block_start"],
        ).to(DEVICE)

    return train_fn, model_factory


def _build_esm2():
    """Return (train_fn, model_factory, cfg) for the ESM2-650M backbone."""
    from train_esm2 import ESM2EpitopeModel, train_esm2
    cfg = dict(lora_rank=8, lora_alpha=8.0, lora_n_blocks=16,
               rys_start=24, rys_end=30, dropout=0.1)

    def train_fn(train_data, val_data):
        return train_esm2(train_data, val_data, max_seconds=TIME_BUDGET, device=DEVICE,
                          compute_auc=True, val_eval_interval=200, patience=5, **cfg)

    def model_factory():
        return ESM2EpitopeModel(
            dropout=cfg["dropout"], lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            lora_n_blocks=cfg["lora_n_blocks"], rys_start=cfg["rys_start"], rys_end=cfg["rys_end"],
        ).to(DEVICE)

    return train_fn, model_factory


BACKBONES = {"esm3": _build_esm3, "esm2": _build_esm2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backbone", choices=list(BACKBONES))
    args = parser.parse_args()

    train_fn, model_factory = BACKBONES[args.backbone]()
    out = Path(f"data/ensemble_preds/{args.backbone}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for fold in FOLDS:
        train_data, val_data, test_h = cv_test_with_headers(
            BEPIPRED_FASTA, STRUCTURES_DIR, test_partition=fold, max_length=MAX_SEQ_LEN,
        )
        logger.info(f"\n=== {args.backbone.upper()} fold {fold}: "
                    f"train={len(train_data)} val={len(val_data)} test={len(test_h)} ===")
        t0 = time.time()
        result = train_fn(train_data, val_data)
        logger.info(f"fold {fold}: val_auc={result['roc_auc']:.4f} train_time={time.time() - t0:.0f}s")

        model = model_factory()
        cur = model.state_dict()
        cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
        model.load_state_dict(cur)

        recs = predict_venv_model(model, test_h, DEVICE, fold)
        all_records.extend(recs)
        logger.info(f"fold {fold}: {len(recs)} residue predictions")
        del model, cur
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_preds(out, all_records)
    logger.info(f"\nSaved {len(all_records)} predictions to {out}")


if __name__ == "__main__":
    main()
