"""Ensemble member: ESM3 + LoRA + RYS. Trains one model per CV fold and dumps
per-residue test predictions to data/ensemble_preds/esm3.npz.

Config: RYS(36,44) + LoRA rank=8 on the last 16 blocks (strong ESM3 config).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train_struct import (
    MAX_SEQ_LEN, StructureEpitopePredictionModel, train,
)
from ensemble_io import save_preds, cv_test_with_headers, predict_venv_model

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
OUT = Path("data/ensemble_preds/esm3.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLDS = ["1", "2", "3"]
TIME_BUDGET = 1200

CFG = dict(
    rys_start=36, rys_end=44,
    lora_rank=8, lora_alpha=8.0, lora_n_blocks=16, lora_block_start=-1,
    dropout=0.1, val_eval_interval=200, patience=5, compute_auc=True, device=DEVICE,
)

if __name__ == "__main__":
    all_records = []
    for fold in FOLDS:
        train_data, val_data, test_h = cv_test_with_headers(
            BEPIPRED_FASTA, STRUCTURES_DIR, test_partition=fold, max_length=MAX_SEQ_LEN,
        )
        logger.info(f"\n=== ESM3 fold {fold}: train={len(train_data)} val={len(val_data)} test={len(test_h)} ===")
        t0 = time.time()
        result = train(train_data, val_data, max_seconds=TIME_BUDGET, **CFG)
        logger.info(f"fold {fold}: val_auc={result['roc_auc']:.4f} train_time={time.time()-t0:.0f}s")

        model = StructureEpitopePredictionModel(
            dropout=CFG["dropout"], rys_start=CFG["rys_start"], rys_end=CFG["rys_end"],
            lora_rank=CFG["lora_rank"], lora_alpha=CFG["lora_alpha"],
            lora_n_blocks=CFG["lora_n_blocks"], lora_block_start=CFG["lora_block_start"],
        ).to(DEVICE)
        cur = model.state_dict()
        cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
        model.load_state_dict(cur)

        recs = predict_venv_model(model, test_h, DEVICE, fold)
        all_records.extend(recs)
        logger.info(f"fold {fold}: {len(recs)} residue predictions")
        del model, cur
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_preds(OUT, all_records)
    logger.info(f"\nSaved {len(all_records)} predictions to {OUT}")
