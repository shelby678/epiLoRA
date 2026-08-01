"""Train the epiLoRA model (ESM-IF1 + LoRA + RYS) and save a checkpoint.

    python train.py \
        --fasta data/train_test_eval/all_epitopes.fasta \
        --structures data/raw/all-structures-extracted \
        --fold 1 \
        --out weights/all_epitopes_fold1.pt

``--fasta`` is one of the ablation FASTAs from ``data/data_prep.smk``
(``data/train_test_eval/*_epitopes.fasta``). Each record carries a 5-fold CV
label ``i.j`` (see ``data/README.md``); ``--fold i`` trains on every record
whose fold != i.

Early stopping and test-set reporting are evaluated against a *fixed* shared
benchmark (``--eval-fastas``, default the homo-sapiens and
homo-sapiens+mus-musculus ablations) rather than ``--fasta``'s own fold-i
split, so every ablation's model is judged on the same held-out set — the
first ``--eval-fastas`` entry's ``i.0`` records are used for early stopping;
every entry's ``i.1`` records are reported as test AUC. Only the LoRA
adapters, the RYS-replayed encoder layers, and the head are saved (~a few
MB); the frozen ESM-IF1 backbone is re-downloaded at load time.

Must run in the fair-esm (py3.9) environment — see README / requirements.txt.
"""

from __future__ import annotations

import argparse
import json
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
from model import (DROPOUT, HEAD_DIM, LORA_ALPHA, LORA_LAYERS, LORA_RANK,
                   RYS_END, RYS_START, build_model)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent  # so defaults don't depend on cwd
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


def eval_name(path: Path) -> str:
    """Short label for an eval FASTA, e.g. allowed_species_homo_sapiens_epitopes.fasta -> homo_sapiens."""
    return path.stem.removeprefix("allowed_species_").removesuffix("_epitopes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path,
                   default=REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta",
                   help="ablation FASTA to train on (default recipe: human-only antibodies, <=10A resolution)")
    p.add_argument("--structures", type=Path, default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "weights/epilora_if1.pt")
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="held-out CV fold (trains on every other fold)")
    p.add_argument("--eval-fastas", type=Path, nargs="+", default=[
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta",
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
    ], help="fixed shared benchmark(s); fold i.0 of the first is used for early "
             "stopping, fold i.1 of every one is reported as test AUC")
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--seed", type=int, default=42,
                   help="seed for dataset shuffling and LoRA/head weight init")
    p.add_argument("--backbone", choices=["esmif1", "esm2", "esm3"], default="esmif1",
                   help="pretrained model to adapt: esmif1 (structure, champion), "
                        "esm2/esm3 (sequence-only, for the backbone ablation)")
    p.add_argument("--esm2-size", choices=["35M", "150M", "650M"], default="650M",
                   help="ESM2 checkpoint size (only used when --backbone esm2)")
    p.add_argument("--lora-rank", type=int, default=LORA_RANK)
    p.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    p.add_argument("--lora-layers", type=int, default=LORA_LAYERS,
                   help="number of top transformer layers to adapt with LoRA")
    p.add_argument("--rys-start", type=int, default=RYS_START,
                   help="replay window start (esmif1 backbone only; ignored otherwise)")
    p.add_argument("--rys-end", type=int, default=RYS_END,
                   help="replay window end, exclusive; rys_end<=rys_start disables RYS "
                        "(esmif1 backbone only; ignored otherwise)")
    p.add_argument("--head-dim", type=int, default=HEAD_DIM,
                   help="if set, use an MLP head Linear(hidden,head_dim)-GELU-Linear(.,1) "
                        "instead of the default direct Linear(hidden,1)")
    p.add_argument("--dropout", type=float, default=DROPOUT,
                   help="dropout applied in the head (and MLP head hidden layer, if used)")
    args = p.parse_args()

    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    val_label, test_label = f"{args.fold}.0", f"{args.fold}.1"

    by_part = parse_fasta(args.fasta)
    train_entries = [e for k, v in by_part.items() if int(k.split(".")[0]) != args.fold for e in v]

    eval_by_fasta = {ef: parse_fasta(ef) for ef in args.eval_fastas}
    val_entries = eval_by_fasta[args.eval_fastas[0]].get(val_label, [])
    if not val_entries:
        p.error(f"no '{val_label}' records in {args.eval_fastas[0]}")

    logger.info(f"Loading structures ({DEVICE}) ...")
    train_samples = load_samples(train_entries, args.structures)
    val_samples = load_samples(val_entries, args.structures)
    n_tr = sum(1 for *_, c in train_samples if c is not None)
    n_va = sum(1 for *_, c in val_samples if c is not None)
    logger.info(f"fold={args.fold}  train={len(train_samples)} ({n_tr} w/ struct)  "
                f"val={len(val_samples)} ({n_va} w/ struct)")
    if n_tr == 0:
        p.error("no structure-backed training samples found — check --structures path")

    if args.backbone == "esmif1":
        model = build_model(device=DEVICE, rank=args.lora_rank, alpha=args.lora_alpha,
                            n_lora_layers=args.lora_layers, rys_start=args.rys_start,
                            rys_end=args.rys_end, dropout=args.dropout, head_dim=args.head_dim)
    elif args.backbone == "esm2":
        from model_esm2 import build_model as build_model_esm2
        model = build_model_esm2(device=DEVICE, size=args.esm2_size, rank=args.lora_rank,
                                 alpha=args.lora_alpha, n_lora_layers=args.lora_layers,
                                 dropout=args.dropout, head_dim=args.head_dim)
    else:
        from model_esm3 import build_model as build_model_esm3
        model = build_model_esm3(device=DEVICE, rank=args.lora_rank, alpha=args.lora_alpha,
                                 n_lora_layers=args.lora_layers, dropout=args.dropout,
                                 head_dim=args.head_dim)
    t0 = time.time()
    res = train(model, train_samples, val_samples, args.max_seconds, seed=args.seed)
    logger.info(f"Done: best val_auc={res['best_auc']:.4f} steps={res['steps']} {time.time()-t0:.0f}s")

    test_aucs = {}
    for ef, by_part_eval in eval_by_fasta.items():
        test_entries = by_part_eval.get(test_label, [])
        test_samples = load_samples(test_entries, args.structures)
        test_aucs[eval_name(ef)] = evaluate_auc(model, test_samples)
    for name, auc in test_aucs.items():
        logger.info(f"test_auc[{name}]={auc:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": args.backbone,
                "config": model.config(),
                "trainable_state": model.trainable_state_dict(),
                "fasta": str(args.fasta),
                "fold": args.fold,
                "seed": args.seed,
                "val_auc": res["best_auc"],
                "test_auc": test_aucs}, args.out)
    logger.info(f"Saved checkpoint -> {args.out}")

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "backbone": args.backbone, "fasta": str(args.fasta), "fold": args.fold,
        "seed": args.seed, "steps": res["steps"],
        "seconds": round(time.time() - t0), "val_auc": res["best_auc"], "test_auc": test_aucs,
    }, indent=2))


if __name__ == "__main__":
    main()
