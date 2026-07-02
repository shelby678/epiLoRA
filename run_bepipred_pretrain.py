"""BEPIPRED pretraining experiments: 3-fold CV.

Tests whether pretraining on protein-protein interface subsets improves epitope prediction.
The key question: does filtering PDB contacts to "antibody-like" interfaces (small, surface-
exposed, loop-rich) provide better pretraining signal than all interfaces?

Experiments:
  contacts-all:     Pretrain on all pdb_contacts.fasta, then finetune on BEPIPRED.
  contacts-small:   Pretrain only on small interfaces (10-35 epitope residues).
  contacts-tiny:    Pretrain only on very small interfaces (5-20 residues, CDR-sized).
  2stage-small:     Two-stage: contacts-small (stage1) → BEPIPRED (stage2), no overlap.
  2stage-tiny:      Two-stage: contacts-tiny → BEPIPRED.
  mix-small:        Mix BEPIPRED train + small contacts in same epoch (balanced 1:1).

All use the canonical best config + RSA surface masking (best from RSA experiments).

Run:
    uv run python run_bepipred_pretrain.py > run_pretrain.log 2>&1
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train_struct import (
    BATCH_SIZE, DROPOUT, LR, MAX_SEQ_LEN,
    WARMUP_STEPS, WEIGHT_DECAY, StructSample,
    StructureEpitopePredictionModel,
    create_cv_datasets, load_contacts_data, train, compute_roc_auc,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA  = Path("data/BEPIPRED.fasta")
CONTACTS_FASTA  = Path("data/pdb_contacts.fasta")
STRUCTURES_DIR  = Path("data/structures2/sabdab_dataset")
RESULTS_TSV     = Path("results.tsv")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

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
    rsa_surface_threshold=0.15,  # use best RSA masking by default
)

# Pre-load contact datasets once (they're large, don't reload per fold)
logger.info("Loading contact datasets ...")
contacts_all   = load_contacts_data(CONTACTS_FASTA) if CONTACTS_FASTA.exists() else []
contacts_small = load_contacts_data(CONTACTS_FASTA, interface_min=10, interface_max=35) \
                 if CONTACTS_FASTA.exists() else []
contacts_tiny  = load_contacts_data(CONTACTS_FASTA, interface_min=5,  interface_max=20) \
                 if CONTACTS_FASTA.exists() else []
logger.info(
    f"Contacts: all={len(contacts_all):,}  small(10-35)={len(contacts_small):,}"
    f"  tiny(5-20)={len(contacts_tiny):,}"
)

RNG = random.Random(42)


def _balanced_mix(epitope_data: list[StructSample], contact_data: list[StructSample]) -> list[StructSample]:
    """Downsample contacts to match epitope count, then concatenate."""
    n = len(epitope_data)
    sampled = RNG.sample(contact_data, min(len(contact_data), n))
    return epitope_data + sampled


def _run_two_stage(
    train_data: list[StructSample],
    val_data: list[StructSample],
    stage1_data: list[StructSample],
    stage1_seconds: int,
    stage2_seconds: int,
    hparams: dict,
) -> dict:
    """Stage 1: train on contacts (no periodic eval). Stage 2: finetune on epitopes."""
    h = dict(hparams)
    r1 = train(stage1_data, val_data, max_seconds=stage1_seconds,
               val_eval_interval=0, **h)
    logger.info(f"Stage-1 done ({r1['steps']} steps)")
    r2 = train(train_data, val_data, max_seconds=stage2_seconds,
               initial_state_dict=r1["trainable_state"],
               val_eval_interval=h.get("val_eval_interval", 200),
               **h)
    r2["peak_vram_mb"] = max(r1["peak_vram_mb"], r2["peak_vram_mb"])
    return r2


def run_cv(name: str, desc: str, mode: str) -> None:
    hparams = dict(BASE_HPARAMS)
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
        logger.info(f"fold test={test_part}: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

        for run_idx in range(N_RERUNS):
            t0 = time.time()

            if mode == "contacts-all":
                td = train_data + contacts_all
                result = train(td, val_data, max_seconds=TIME_BUDGET, **hparams)

            elif mode == "contacts-small":
                td = train_data + contacts_small
                result = train(td, val_data, max_seconds=TIME_BUDGET, **hparams)

            elif mode == "contacts-tiny":
                td = train_data + contacts_tiny
                result = train(td, val_data, max_seconds=TIME_BUDGET, **hparams)

            elif mode == "2stage-small":
                result = _run_two_stage(
                    train_data, val_data, contacts_small,
                    stage1_seconds=TIME_BUDGET // 2,
                    stage2_seconds=TIME_BUDGET // 2,
                    hparams=hparams,
                )

            elif mode == "2stage-tiny":
                result = _run_two_stage(
                    train_data, val_data, contacts_tiny,
                    stage1_seconds=TIME_BUDGET // 2,
                    stage2_seconds=TIME_BUDGET // 2,
                    hparams=hparams,
                )

            elif mode == "mix-small":
                td = _balanced_mix(train_data, contacts_small)
                result = train(td, val_data, max_seconds=TIME_BUDGET, **hparams)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            elapsed = time.time() - t0

            # Rebuild model for test evaluation
            model = StructureEpitopePredictionModel(
                dropout=hparams.get("dropout", DROPOUT),
                rys_start=hparams["rys_start"], rys_end=hparams["rys_end"],
                lora_rank=hparams["lora_rank"], lora_alpha=hparams["lora_alpha"],
                lora_n_blocks=hparams["lora_n_blocks"],
                lora_block_start=hparams.get("lora_block_start", -1),
            ).to(DEVICE)
            cur = model.state_dict()
            cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
            model.load_state_dict(cur)

            test_auc = compute_roc_auc(model, test_data, batch_size=BATCH_SIZE, device=DEVICE)
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


EXPERIMENTS = [
    dict(name="contacts-all",   desc="Mix all PDB contacts with BEPIPRED train", mode="contacts-all"),
    dict(name="contacts-small", desc="Mix small interfaces (10-35 res) + BEPIPRED", mode="contacts-small"),
    dict(name="contacts-tiny",  desc="Mix tiny interfaces (5-20 res, CDR-sized) + BEPIPRED", mode="contacts-tiny"),
    dict(name="2stage-small",   desc="2-stage: pretrain on small contacts → finetune BEPIPRED", mode="2stage-small"),
    dict(name="2stage-tiny",    desc="2-stage: pretrain on tiny contacts → finetune BEPIPRED", mode="2stage-tiny"),
    dict(name="mix-small",      desc="Balanced 1:1 mix: BEPIPRED + small contacts", mode="mix-small"),
]


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(
                "commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                "steps\tpeak_vram_mb\telapsed_s\tdesc\n"
            )

    for exp in EXPERIMENTS:
        run_cv(exp["name"], exp["desc"], exp["mode"])
