"""BEPIPRED RSA experiments: 3-fold CV.

Tests whether adding RSA (relative surface area) information improves epitope prediction.
Epitopes are almost exclusively surface-exposed, so RSA should be very informative.

Experiments:
  rsa-mask-15: Train/val loss only on surface residues (RSA > 0.15). Key discotope3 trick.
  rsa-mask-25: Stricter surface masking (RSA > 0.25).
  rsa-feat:    RSA as extra 1-dim input to head (no masking).
  rsa-feat-mask15: RSA as feature + surface masking at 0.15.
  bio-feat:    Biophysical properties (hydrophobicity, charge, volume, polarity) as head input.
  bio-rsa-mask15: Bio features + RSA masking.
  blosum-feat: BLOSUM62 rows (20-dim) as head input.
  all-feat-mask15: RSA + bio + BLOSUM + surface masking — kitchen sink.

All use the canonical best SAbDab config (RYS(36,44)+LoRA rank4 last-8).

Run:
    uv run python run_bepipred_rsa.py > run_rsa.log 2>&1
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train_struct import (
    BATCH_SIZE, DROPOUT, LR, MAX_SEQ_LEN,
    WARMUP_STEPS, WEIGHT_DECAY,
    StructureEpitopePredictionModel,
    create_cv_datasets, train, compute_roc_auc,
)
from features import compute_extra_features, extra_feature_dim

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV    = Path("results.tsv")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

CV_FOLDS = [
    {"test": "1", "val": "1"},
    {"test": "2", "val": "2"},
    {"test": "3", "val": "3"},
]
N_RERUNS    = 1
TIME_BUDGET = 1200

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

BASE_HPARAMS = dict(
    rys_start=36, rys_end=44,
    lora_rank=4, lora_alpha=8.0, lora_n_blocks=8, lora_block_start=-1,
    dropout=DROPOUT, batch_size=BATCH_SIZE, lr=LR,
    weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS,
    val_eval_interval=200, patience=5,
    compute_auc=True, device=DEVICE,
)

EXPERIMENTS = [
    dict(name="rsa-mask-15",
         desc="Surface masking RSA>0.15 (exclude buried from loss)",
         rsa_surface_threshold=0.15),
    dict(name="rsa-mask-25",
         desc="Surface masking RSA>0.25 (stricter surface filter)",
         rsa_surface_threshold=0.25),
    dict(name="rsa-feat",
         desc="RSA as 1-dim head input (no masking)",
         rsa_as_feature=True),
    dict(name="rsa-feat-mask15",
         desc="RSA as head feature + surface masking RSA>0.15",
         rsa_as_feature=True, rsa_surface_threshold=0.15),
    dict(name="bio-feat",
         desc="Biophysical AA properties (4-dim: hydrophobicity, charge, volume, polarity)",
         bio_features=True),
    dict(name="bio-rsa-mask15",
         desc="Biophysical features + RSA masking at 0.15",
         bio_features=True, rsa_surface_threshold=0.15),
    dict(name="blosum-feat",
         desc="BLOSUM62 rows (20-dim) as head input",
         blosum_features=True),
    dict(name="all-feat-mask15",
         desc="RSA feature + biophysical + BLOSUM62 + surface mask 0.15",
         rsa_as_feature=True, bio_features=True, blosum_features=True,
         rsa_surface_threshold=0.15),
]


def run_cv(name: str, desc: str, extra_hparams: dict) -> None:
    hparams = {**BASE_HPARAMS, **extra_hparams}

    fold_test_aucs: list[float] = []
    fold_val_aucs:  list[float] = []

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")

    rsa_feat   = hparams.pop("rsa_as_feature", False)
    bio_feat   = hparams.pop("bio_features", False)
    blosum_feat = hparams.pop("blosum_features", False)

    # Build extra_fn for test evaluation (consistent with training)
    def _extra_fn(batch, dev):
        return compute_extra_features(batch, dev,
                                      rsa_as_feature=rsa_feat,
                                      bio=bio_feat, blosum=blosum_feat)

    for fold in CV_FOLDS:
        test_part = fold["test"]
        val_part  = fold["val"]

        train_data, val_data, test_data = create_cv_datasets(
            BEPIPRED_FASTA, STRUCTURES_DIR,
            test_partition=test_part, val_partition=val_part,
            max_length=MAX_SEQ_LEN,
        )
        logger.info(
            f"fold test={test_part}: train={len(train_data)} val={len(val_data)} test={len(test_data)}"
        )

        for run_idx in range(N_RERUNS):
            t0 = time.time()
            result = train(
                train_data, val_data,
                max_seconds=TIME_BUDGET,
                rsa_as_feature=rsa_feat,
                bio_features=bio_feat,
                blosum_features=blosum_feat,
                **hparams,
            )
            elapsed = time.time() - t0

            # Rebuild model for test evaluation
            ed = extra_feature_dim(rsa_feat, bio_feat, blosum_feat)
            model = StructureEpitopePredictionModel(
                dropout=hparams.get("dropout", DROPOUT),
                rys_start=hparams["rys_start"], rys_end=hparams["rys_end"],
                lora_rank=hparams["lora_rank"], lora_alpha=hparams["lora_alpha"],
                lora_n_blocks=hparams["lora_n_blocks"],
                lora_block_start=hparams.get("lora_block_start", -1),
                extra_dim=ed,
            ).to(DEVICE)
            cur = model.state_dict()
            cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
            model.load_state_dict(cur)

            test_auc = compute_roc_auc(model, test_data, batch_size=BATCH_SIZE,
                                        device=DEVICE, extra_fn=_extra_fn)
            del model, cur
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            val_auc = result["roc_auc"]
            fold_val_aucs.append(val_auc)
            fold_test_aucs.append(test_auc)

            print(
                f"  fold={test_part} run={run_idx+1}: "
                f"val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  "
                f"val_loss={result['val_loss']:.4f}  steps={result['steps']}  {elapsed:.0f}s"
            )

            with open(RESULTS_TSV, "a") as f:
                f.write(
                    f"{COMMIT}\t{name}\t{test_part}\t{run_idx}\t"
                    f"{result['val_loss']:.6f}\t{val_auc:.6f}\t{test_auc:.6f}\t"
                    f"{result['steps']}\t{result['peak_vram_mb']}\t{elapsed:.1f}\t"
                    f"{desc}\n"
                )

    print(f"\n{'─'*60}")
    print(f"SUMMARY {name}")
    print(f"  val_auc  = {np.mean(fold_val_aucs):.4f} ± {np.std(fold_val_aucs):.4f}")
    print(f"  test_auc = {np.mean(fold_test_aucs):.4f} ± {np.std(fold_test_aucs):.4f}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(
                "commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                "steps\tpeak_vram_mb\telapsed_s\tdesc\n"
            )

    for exp in EXPERIMENTS:
        name = exp.pop("name")
        desc = exp.pop("desc")
        run_cv(name, desc, exp)
