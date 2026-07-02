"""BEPIPRED selective LoRA: adapt only the top-K activated (block, head) pairs.

Uses ``data/activation_probe.pt`` (produced by ``probe_activations.py``):
the 48×24 mean |context_BHLD| matrix across all BEPIPRED antigens. The
probe revealed that activation is heavily concentrated in the last four
blocks (44–47), with block 47 alone holding ~50% of the top-16 head
magnitude.

Selective LoRA wraps the QKV linear with ``HeadMaskedLoRA`` — only the
output dims belonging to the chosen heads' Q, K, V slices receive a
trainable delta. Param count scales with the number of selected heads,
not with model size.

All experiments pair selective LoRA with HiddenKey↗ (DropKey 0.10 col +
HiddenCut 0.10 elem, no KL) — the dropout winner from earlier sweeps.

Experiments (6 × 3 folds = 18 fold-runs, ~2-2.5 hr):
  sl-top-08-r4   — LoRA rank=4 on top-8  (block, head) pairs from probe
  sl-top-16-r4   — LoRA rank=4 on top-16 (block, head) pairs from probe
  sl-top-32-r4   — LoRA rank=4 on top-32 (block, head) pairs from probe
  sl-top-16-r8   — rank=8 on top-16 (higher capacity per head)
  sl-top-32-r2   — rank=2 on top-32 (lower capacity, wider coverage)
  sl-last4-all   — every head in blocks 44, 45, 46, 47 (4 × 24 = 96 heads)
                   — control: "select the activated *layers*, not heads"
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
    create_cv_datasets, train, compute_roc_auc,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV = Path("results.tsv")
PROBE_PATH = Path("data/activation_probe.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CV_FOLDS = [
    {"test": "1", "val": "1"},
    {"test": "2", "val": "2"},
    {"test": "3", "val": "3"},
]
N_RERUNS = 1
TIME_BUDGET = 1200

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"


def load_probe_topk(k: int) -> dict[int, list[int]]:
    """Return {block_idx: [head_idx, ...]} for the top-k (block, head) pairs."""
    data = torch.load(PROBE_PATH, weights_only=False)
    flat = data["per_block_head_mag"].flatten()  # (n_blocks * n_heads,)
    n_heads = int(data["n_heads"])
    top_idx = torch.topk(flat, k=k).indices.tolist()
    out: dict[int, list[int]] = {}
    for idx in top_idx:
        b, h = divmod(idx, n_heads)
        out.setdefault(b, []).append(h)
    # Sort heads within each block for reproducibility
    return {b: sorted(hs) for b, hs in out.items()}


def all_heads_in_blocks(blocks: list[int]) -> dict[int, list[int]]:
    return {b: list(range(24)) for b in blocks}


BASE = dict(
    rys_start=0, rys_end=0,
    lora_rank=4, lora_alpha=8.0,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    val_eval_interval=200,
    patience=5,
    compute_auc=True,
    device=DEVICE,
    paper_drop_only_lora=True,
    dropkey_prob=0.10,
    hiddencut_prob=0.10,
)


def _already_done() -> set[tuple[str, str, int]]:
    if not RESULTS_TSV.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with open(RESULTS_TSV) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                done.add((parts[1], parts[2], int(parts[3])))
            except ValueError:
                continue
    return done


def run_one(name: str, desc: str, hparams: dict) -> dict:
    fold_test_aucs, fold_val_aucs = [], []
    done = _already_done()
    print(f"\n{'=' * 60}\nEXPERIMENT: {name}\n  {desc}\n{'=' * 60}")
    if "head_lora_targets" in hparams:
        print(f"  head_lora_targets = {hparams['head_lora_targets']}")

    for fold in CV_FOLDS:
        test_part, val_part = fold["test"], fold["val"]
        if all((name, test_part, r) in done for r in range(N_RERUNS)):
            print(f"  fold test={test_part}: SKIP (already in results.tsv)")
            continue
        logger.info(f"Loading fold: test={test_part}, val={val_part}")
        train_data, val_data, test_data = create_cv_datasets(
            BEPIPRED_FASTA, STRUCTURES_DIR,
            test_partition=test_part, val_partition=val_part, max_length=MAX_SEQ_LEN,
        )
        for run_idx in range(N_RERUNS):
            if (name, test_part, run_idx) in done:
                continue
            t0 = time.time()
            result = train(train_data, val_data, max_seconds=TIME_BUDGET, **hparams)
            elapsed = time.time() - t0

            from train_struct import StructureEpitopePredictionModel
            model = StructureEpitopePredictionModel(
                dropout=hparams.get("dropout", DROPOUT),
                rys_start=hparams.get("rys_start", 0),
                rys_end=hparams.get("rys_end", 0),
                lora_rank=hparams.get("lora_rank", 0),
                lora_alpha=hparams.get("lora_alpha", 8.0),
                lora_n_blocks=hparams.get("lora_n_blocks", 8),
                lora_block_start=hparams.get("lora_block_start", -1),
                head_lora_targets=hparams.get("head_lora_targets", None),
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
                f"  fold test={test_part} run={run_idx+1}/{N_RERUNS}:  "
                f"val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  "
                f"val_loss={result['val_loss']:.4f}  steps={result['steps']}  "
                f"elapsed={elapsed:.0f}s"
            )
            with open(RESULTS_TSV, "a") as f:
                f.write(
                    f"{COMMIT}\t{name}\t{test_part}\t{run_idx}\t"
                    f"{result['val_loss']:.6f}\t{val_auc:.6f}\t{test_auc:.6f}\t"
                    f"{result['steps']}\t{result['peak_vram_mb']}\t{elapsed:.1f}\t"
                    f"{desc}\n"
                )

    all_test, all_val = [], []
    if RESULTS_TSV.exists():
        with open(RESULTS_TSV) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7 and parts[1] == name:
                    try:
                        all_val.append(float(parts[5]))
                        all_test.append(float(parts[6]))
                    except ValueError:
                        pass
    if not all_test:
        all_test, all_val = fold_test_aucs, fold_val_aucs
    mean = float(np.mean(all_test))
    std = float(np.std(all_test))
    print(f"\n{'─' * 60}\nSUMMARY {name}")
    print(f"  test_auc = {mean:.4f} ± {std:.4f}   (val_auc={np.mean(all_val):.4f})  "
          f"[n={len(all_test)}]")
    print(f"{'─' * 60}")
    return {"name": name, "test_auc_mean": mean, "test_auc_std": std}


# Resolve targets from probe at import time so they're logged with the run.
TOP_8  = load_probe_topk(8)
TOP_16 = load_probe_topk(16)
TOP_32 = load_probe_topk(32)
print(f"[probe targets] top-8:  {TOP_8}")
print(f"[probe targets] top-16: {TOP_16}")
print(f"[probe targets] top-32: {TOP_32}")

EXPERIMENTS = [
    ("sl-top-08-r4",
     "Selective LoRA rank=4 on top-8 activated (block, head) pairs + HK↗",
     {**BASE, "lora_rank": 4, "head_lora_targets": TOP_8}),
    ("sl-top-16-r4",
     "Selective LoRA rank=4 on top-16 activated (block, head) pairs + HK↗",
     {**BASE, "lora_rank": 4, "head_lora_targets": TOP_16}),
    ("sl-top-32-r4",
     "Selective LoRA rank=4 on top-32 activated (block, head) pairs + HK↗",
     {**BASE, "lora_rank": 4, "head_lora_targets": TOP_32}),
    ("sl-top-16-r8",
     "Selective LoRA rank=8 on top-16 — higher per-head capacity + HK↗",
     {**BASE, "lora_rank": 8, "head_lora_targets": TOP_16}),
    ("sl-top-32-r2",
     "Selective LoRA rank=2 on top-32 — lower per-head capacity + HK↗",
     {**BASE, "lora_rank": 2, "head_lora_targets": TOP_32}),
    ("sl-last4-all",
     "Selective LoRA rank=4 on ALL heads of blocks 44-47 + HK↗ "
     "(layer-only selection control)",
     {**BASE, "lora_rank": 4,
      "head_lora_targets": all_heads_in_blocks([44, 45, 46, 47])}),
]


if __name__ == "__main__":
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                    "steps\tpeak_vram_mb\telapsed_s\tdesc\n")

    summaries = []
    for name, desc, hparams in EXPERIMENTS:
        summaries.append(run_one(name, desc, hparams))

    print(f"\n{'=' * 60}\nFINAL SUMMARY (Selective LoRA on activated heads)\n{'=' * 60}")
    print(f"{'method':<24}  test_auc")
    for s in summaries:
        print(f"{s['name']:<24}  {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f}")
