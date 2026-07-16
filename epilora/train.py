"""Train the epiLoRA model (ESM-IF1 + LoRA + RYS) and save a checkpoint.

    python train.py \
        --fasta data/BEPIPRED.fasta \
        --structures data/structures2/sabdab_dataset \
        --out weights/epilora_if1.pt

Trains on every non-EVAL partition, holding out one partition (``--val``) for
early stopping, then writes the trainable weights + config to ``--out``. Only
the LoRA adapters, the RYS-replayed encoder layers, and the head are saved
(~a few MB); the frozen ESM-IF1 backbone is re-downloaded at load time.

Must run in the fair-esm (py3.9) environment — see README / requirements.txt.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data import load_samples, parse_fasta
from model import (LORA_ALPHA, LORA_LAYERS, LORA_RANK, RYS_END, RYS_START,
                   build_model)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 200
VAL_INTERVAL = 200
PATIENCE = 10


@torch.no_grad()
def evaluate_auc(model, samples) -> float:
    model.eval()
    logits_all, labels_all = [], []
    for header, seq, labels, coords in samples:
        if coords is None:
            continue
        try:
            lg = model([coords], [seq])[0].cpu().numpy()
        except Exception:
            continue
        logits_all.append(lg)
        labels_all.append(labels)
    if not logits_all:
        return float("nan")
    y, s = np.concatenate(labels_all), np.concatenate(logits_all)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan")


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/torch RNGs (LoRA + head init draw from the global torch RNG)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_samples, val_samples, max_seconds: int, seed: int = 42) -> dict:
    """Train in-place with early stopping on val ROC-AUC; keep the best weights."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, WARMUP_STEPS)))

    best_auc, best_state, no_improve, step, tl = -1.0, None, 0, 0, 0.0
    rng = np.random.default_rng(seed)
    idxs = list(range(len(train_samples)))
    start = time.time()
    model.train()
    stop = False
    while not stop and time.time() - start < max_seconds:
        rng.shuffle(idxs)
        for idx in idxs:
            if time.time() - start >= max_seconds:
                break
            header, seq, labels, coords = train_samples[idx]
            if coords is None:
                continue
            try:
                logits = model([coords], [seq])[0]
            except Exception as e:
                logger.debug(f"skip {header}: {e}")
                continue
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.tensor(labels, dtype=torch.float32, device=model.device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            tl = 0.9 * tl + 0.1 * loss.item()
            if step % VAL_INTERVAL == 0:
                va = evaluate_auc(model, val_samples)
                model.train()
                logger.info(f"step={step} train_loss={tl:.4f} val_auc={va:.4f} {time.time()-start:.0f}s")
                if va > best_auc:
                    best_auc, no_improve = va, 0
                    best_state = model.trainable_state_dict()
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        logger.info(f"Early stop at step {step}")
                        stop = True
                        break
    if best_state is not None:
        model.load_trainable_state_dict(best_state)
    return {"steps": step, "val_auc": evaluate_auc(model, val_samples), "best_auc": best_auc}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, default=Path("data/BEPIPRED.fasta"))
    p.add_argument("--structures", type=Path, default=Path("data/structures2/sabdab_dataset"))
    p.add_argument("--out", type=Path, default=Path("weights/epilora_if1.pt"))
    p.add_argument("--val", default="5", help="partition held out for early stopping")
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42,
                   help="seed for dataset shuffling and LoRA/head weight init")
    args = p.parse_args()

    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    by_part = parse_fasta(args.fasta)
    if args.val not in by_part:
        p.error(f"val partition '{args.val}' not found; available: {sorted(by_part)}")

    logger.info(f"Loading structures ({DEVICE}) ...")
    train_entries = [e for k, v in by_part.items() if k != args.val for e in v]
    val_entries = by_part[args.val]
    train_samples = load_samples(train_entries, args.structures)
    val_samples = load_samples(val_entries, args.structures)
    n_tr = sum(1 for *_, c in train_samples if c is not None)
    n_va = sum(1 for *_, c in val_samples if c is not None)
    logger.info(f"train={len(train_samples)} ({n_tr} w/ struct)  "
                f"val={len(val_samples)} ({n_va} w/ struct)")
    if n_tr == 0:
        p.error("no structure-backed training samples found — check --structures path")

    model = build_model(device=DEVICE, rank=LORA_RANK, alpha=LORA_ALPHA,
                        n_lora_layers=LORA_LAYERS, rys_start=RYS_START, rys_end=RYS_END)
    t0 = time.time()
    res = train(model, train_samples, val_samples, args.max_seconds, seed=args.seed)
    logger.info(f"Done: best val_auc={res['best_auc']:.4f} steps={res['steps']} {time.time()-t0:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config(),
                "trainable_state": model.trainable_state_dict(),
                "val_auc": res["best_auc"]}, args.out)
    logger.info(f"Saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
