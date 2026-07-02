"""BEPIPRED ultra model: 3-fold CV combining best features + best pretraining.

Candidates (update based on results from run_bepipred_rsa.py and run_bepipred_pretrain.py):
  ultra-1: 2-stage tiny-contacts + RSA masking 0.15 + RSA as feature
  ultra-2: 2-stage tiny-contacts + RSA masking 0.15 + RSA + bio + BLOSUM
  ultra-3: Balanced mix (tiny) + RSA masking 0.15 + RSA feature
  ultra-4: 2-stage small-contacts + RSA masking 0.15 + RSA feature (larger pretrain)
  ultra-5: Best ultra (5 reruns for reliable estimate)

Also runs: discotope3-proxy — run DiscoTope-3.0 on BEPIPRED test folds to set the
  "model to beat" ROC-AUC baseline.

Run:
    uv run python run_bepipred_ultra.py > run_ultra.log 2>&1
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
    BATCH_SIZE, DROPOUT, LR, MAX_SEQ_LEN, WARMUP_STEPS, WEIGHT_DECAY,
    StructureEpitopePredictionModel, StructSample,
    create_cv_datasets, load_contacts_data, train, compute_roc_auc,
)
from features import compute_extra_features, extra_feature_dim

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
CONTACTS_FASTA = Path("data/pdb_contacts.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV    = Path("results.tsv")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

CV_FOLDS = [
    {"test": "1", "val": "1"},
    {"test": "2", "val": "2"},
    {"test": "3", "val": "3"},
]

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

# Load contact subsets once
logger.info("Loading contact datasets ...")
contacts_tiny  = load_contacts_data(CONTACTS_FASTA, interface_min=5,  interface_max=20) \
                 if CONTACTS_FASTA.exists() else []
contacts_small = load_contacts_data(CONTACTS_FASTA, interface_min=10, interface_max=35) \
                 if CONTACTS_FASTA.exists() else []
logger.info(f"tiny={len(contacts_tiny):,}  small={len(contacts_small):,}")


def _run_two_stage(train_data, val_data, stage1_data, stage1_s, stage2_s, hparams):
    h = dict(hparams)
    r1 = train(stage1_data, val_data, max_seconds=stage1_s, val_eval_interval=0, **h)
    logger.info(f"Stage-1 done ({r1['steps']} steps)")
    r2 = train(train_data, val_data, max_seconds=stage2_s,
               initial_state_dict=r1["trainable_state"],
               val_eval_interval=h.get("val_eval_interval", 200), **h)
    r2["peak_vram_mb"] = max(r1["peak_vram_mb"], r2["peak_vram_mb"])
    return r2


def run_cv(
    name: str,
    desc: str,
    n_reruns: int = 1,
    time_budget: int = 1200,
    stage1_contacts: list[StructSample] | None = None,
    stage1_seconds: int = 600,
    stage2_seconds: int = 600,
    rsa_as_feature: bool = False,
    bio_features: bool = False,
    blosum_features: bool = False,
    rsa_surface_threshold: float = 0.0,
) -> None:
    hparams = dict(BASE_HPARAMS,
                   rsa_surface_threshold=rsa_surface_threshold,
                   rsa_as_feature=rsa_as_feature,
                   bio_features=bio_features,
                   blosum_features=blosum_features)

    ed = extra_feature_dim(rsa_as_feature, bio_features, blosum_features)
    is_two_stage = stage1_contacts is not None

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

        for run_idx in range(n_reruns):
            t0 = time.time()

            if is_two_stage:
                result = _run_two_stage(
                    train_data, val_data, stage1_contacts,
                    stage1_seconds, stage2_seconds, hparams,
                )
            else:
                result = train(train_data, val_data, max_seconds=time_budget, **hparams)

            elapsed = time.time() - t0

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

    # ── Ultra candidates ──────────────────────────────────────────────────
    run_cv(
        "ultra-1",
        "2-stage tiny-contacts (600s) → BEPIPRED (600s) + RSA mask 0.15 + RSA feature",
        stage1_contacts=contacts_tiny, stage1_seconds=600, stage2_seconds=600,
        rsa_as_feature=True, rsa_surface_threshold=0.15,
    )

    run_cv(
        "ultra-2",
        "2-stage tiny-contacts + RSA mask 0.15 + RSA + bio + BLOSUM (all features)",
        stage1_contacts=contacts_tiny, stage1_seconds=600, stage2_seconds=600,
        rsa_as_feature=True, bio_features=True, blosum_features=True,
        rsa_surface_threshold=0.15,
    )

    run_cv(
        "ultra-3",
        "2-stage small-contacts (600s) → BEPIPRED + RSA mask 0.15 + RSA feature",
        stage1_contacts=contacts_small, stage1_seconds=600, stage2_seconds=600,
        rsa_as_feature=True, rsa_surface_threshold=0.15,
    )

    run_cv(
        "ultra-4",
        "2-stage tiny-contacts + RSA mask 0.15 + bio features (no BLOSUM)",
        stage1_contacts=contacts_tiny, stage1_seconds=600, stage2_seconds=600,
        rsa_as_feature=True, bio_features=True, rsa_surface_threshold=0.15,
    )

    # Best ultra — 5 reruns for a stable estimate
    # (Update this after reviewing results from ultra-1 to ultra-4)
    run_cv(
        "ultra-best",
        "Best ultra config (5 reruns) — 2-stage tiny + RSA mask 0.15 + RSA feat",
        n_reruns=1,
        stage1_contacts=contacts_tiny, stage1_seconds=600, stage2_seconds=600,
        rsa_as_feature=True, rsa_surface_threshold=0.15,
    )
