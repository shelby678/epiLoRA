"""BEPIPRED feature experiments (round 2): RSA/bio/BLOSUM as head inputs.

RSA masking (0.15 and 0.25) both hurt vs baseline — they remove training signal
from buried residues that the model needs for context. Skipping masking variants.

Testing RSA / biophysical / BLOSUM as *additive features* (no masking), plus
a no-mask combination, to see if explicit residue descriptors help the head.

Experiments:
  rsa-feat:     RSA as 1-dim head input (no masking)
  bio-feat:     Biophysical AA props (hydrophob, charge, vol, polarity) — 4-dim
  blosum-feat:  BLOSUM62 substitution rows — 20-dim
  rsa-bio:      RSA + bio features combined
  all-feat:     RSA + bio + BLOSUM (no masking) — kitchen sink without masking

Run:
    uv run python run_bepipred_features.py > run_features.log 2>&1
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
    rsa_surface_threshold=0.0,  # no masking
)

EXPERIMENTS = [
    dict(name="rsa-feat",
         desc="RSA as 1-dim head input (no masking)",
         rsa_as_feature=True),
    dict(name="bio-feat",
         desc="Biophysical AA props (4-dim: hydrophob/charge/vol/polarity), no masking",
         bio_features=True),
    dict(name="blosum-feat",
         desc="BLOSUM62 rows (20-dim) as head input, no masking",
         blosum_features=True),
    dict(name="rsa-bio",
         desc="RSA (1-dim) + biophysical (4-dim) as head inputs, no masking",
         rsa_as_feature=True, bio_features=True),
    dict(name="all-feat",
         desc="RSA + biophysical + BLOSUM (25-dim total), no masking",
         rsa_as_feature=True, bio_features=True, blosum_features=True),
]


def run_cv(name: str, desc: str, rsa_as_feature=False, bio_features=False,
           blosum_features=False) -> None:
    hparams = dict(BASE_HPARAMS)
    ed = extra_feature_dim(rsa_as_feature, bio_features, blosum_features)

    def _extra_fn(batch, dev):
        return compute_extra_features(batch, dev,
                                      rsa_as_feature=rsa_as_feature,
                                      bio=bio_features, blosum=blosum_features)

    fold_test_aucs: list[float] = []
    fold_val_aucs:  list[float] = []

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")

    for fold in CV_FOLDS:
        test_part = fold["test"]
        val_part  = fold["val"]

        train_data, val_data, test_data = create_cv_datasets(
            BEPIPRED_FASTA, STRUCTURES_DIR,
            test_partition=test_part, val_partition=val_part,
            max_length=MAX_SEQ_LEN,
        )

        for run_idx in range(N_RERUNS):
            t0 = time.time()
            result = train(
                train_data, val_data,
                max_seconds=TIME_BUDGET,
                rsa_as_feature=rsa_as_feature,
                bio_features=bio_features,
                blosum_features=blosum_features,
                **hparams,
            )
            elapsed = time.time() - t0

            model = StructureEpitopePredictionModel(
                dropout=hparams["dropout"],
                rys_start=hparams["rys_start"], rys_end=hparams["rys_end"],
                lora_rank=hparams["lora_rank"], lora_alpha=hparams["lora_alpha"],
                lora_n_blocks=hparams["lora_n_blocks"],
                lora_block_start=hparams["lora_block_start"],
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
                f"val_loss={result['val_loss']:.4f}  {elapsed:.0f}s"
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
        run_cv(name, desc, **exp)
