"""Ensemble ablation: train multiple seeds per fold, then average their predictions.

    python ablation/run_ensemble.py --max-seconds 1800 --n-seeds 3

For each CV fold (1-5) of ``--dataset`` (default: the champion data ablation,
allowed_species_homo_sapiens_epitopes.fasta), trains ``--n-seeds`` independently
seeded models (reusing an existing ``weights/ablation/{dataset}_fold{f}.pt`` as
seed 42 if present, instead of retraining it) with every other champion
hyperparameter held fixed. Then, per fold, loads all seeds' checkpoints,
averages their sigmoid probabilities on that fold's shared-benchmark val/test
split (same benchmark train.py always uses), and reports ensemble AUC next to
the mean single-model AUC -- answering "does averaging several seeds beat any
one of them."

Results land in ``--results`` (one row per fold: single-model mean/std AUC vs
ensemble AUC, per benchmark).

Must be run with epilora/env/bin/python3 (it imports data.py/predict.py/model.py
directly, which need fair-esm/biotite/torch-geometric).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = Path(__file__).resolve().parents[1] / "train.py"
ENV_PYTHON = REPO_ROOT / "epilora/env/bin/python3"
TRAIN_PYTHON = str(ENV_PYTHON) if ENV_PYTHON.exists() else sys.executable
sys.path.insert(0, str(TRAIN_PY.parent))

from data import load_samples, parse_fasta  # noqa: E402
from predict import load_model  # noqa: E402

FOLDS = (1, 2, 3, 4, 5)
BENCHMARKS = ("homo_sapiens", "homo_sapiens_mus_musculus")
DEFAULT_SEEDS = (42, 43, 44)  # 42 matches the champion sweep's own seed


def eval_name(path: Path) -> str:
    return path.stem.removeprefix("allowed_species_").removesuffix("_epitopes")


@torch.no_grad()
def ensemble_probs(models, samples, device) -> tuple[np.ndarray, np.ndarray]:
    """Average sigmoid probs across ``models`` for every structure-backed sample."""
    probs_by_model = [[] for _ in models]
    labels_all = []
    for header, seq, labels, coords, feats in samples:
        if coords is None or (feats is None and models[0].n_extra_feats):
            continue
        ok = True
        per_model = []
        for m in models:
            try:
                logits = m([coords], [seq], [feats])[0].cpu().numpy()
            except Exception:
                ok = False
                break
            per_model.append(1.0 / (1.0 + np.exp(-logits)))
        if not ok:
            continue
        for i, p in enumerate(per_model):
            probs_by_model[i].append(p)
        labels_all.append(labels)
    if not labels_all:
        return np.array([]), np.array([])
    y = np.concatenate(labels_all)
    stacked = np.stack([np.concatenate(p) for p in probs_by_model], axis=0)  # (n_seeds, N)
    return y, stacked


def train_seed(dataset: Path, fold: int, seed: int, out: Path, args, gpu_id: int) -> None:
    if out.exists():
        print(f"[fold={fold} seed={seed}] already trained -> {out}")
        return
    log_path = args.log_dir / f"{out.stem}.log"
    cmd = [TRAIN_PYTHON, str(TRAIN_PY),
           "--fasta", str(dataset.resolve()), "--structures", str(args.structures.resolve()),
           "--fold", str(fold), "--out", str(out.resolve()),
           "--max-seconds", str(args.max_seconds), "--seed", str(seed)]
    import os
    full_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    print(f"[fold={fold} seed={seed} gpu={gpu_id}] training -> {log_path}", flush=True)
    with open(log_path, "w") as logf:
        subprocess.run(cmd, cwd=str(TRAIN_PY.parent), env=full_env, stdout=logf, stderr=subprocess.STDOUT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path,
                   default=REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta")
    p.add_argument("--structures", type=Path, default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--eval-fastas", type=Path, nargs="+", default=[
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta",
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
    ])
    p.add_argument("--weights-dir", type=Path, default=REPO_ROOT / "weights/ensemble")
    p.add_argument("--baseline-weights-dir", type=Path, default=REPO_ROOT / "weights/ablation",
                   help="where to look for an existing seed=42 fold checkpoint to reuse")
    p.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "ensemble_results.csv")
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers or max(1, torch.cuda.device_count())
    stem = args.dataset.stem.removesuffix("_epitopes")

    # --- train every (fold, seed) not already on disk, spread across GPUs ---
    import queue
    import threading
    jobs = []
    for fold in args.folds:
        for seed in args.seeds:
            if seed == 42:
                baseline = args.baseline_weights_dir / f"{stem}_fold{fold}.pt"
                out = baseline if baseline.exists() else args.weights_dir / f"{stem}_fold{fold}_seed{seed}.pt"
            else:
                out = args.weights_dir / f"{stem}_fold{fold}_seed{seed}.pt"
            jobs.append((fold, seed, out))

    # A pool of free GPU ids (rather than gpu_id = i % n_workers) so a fast-finishing
    # job (e.g. an instant "already trained" skip) frees its GPU for immediate reuse
    # instead of leaving that GPU idle while another GPU gets double-booked.
    gpu_pool: queue.Queue = queue.Queue()
    for g in range(n_workers):
        gpu_pool.put(g)
    threads = []

    def worker(fold, seed, out):
        gpu_id = gpu_pool.get()
        try:
            train_seed(args.dataset, fold, seed, out, args, gpu_id)
        finally:
            gpu_pool.put(gpu_id)

    for fold, seed, out in jobs:
        t = threading.Thread(target=worker, args=(fold, seed, out))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # --- evaluate: per fold, ensemble vs single-model AUC on the shared benchmark ---
    from sklearn.metrics import roc_auc_score
    eval_by_fasta = {ef: parse_fasta(ef) for ef in args.eval_fastas}

    csv_fields = ["fold", "n_seeds", *[f"single_auc_mean_{b}" for b in BENCHMARKS],
                  *[f"single_auc_std_{b}" for b in BENCHMARKS], *[f"ensemble_auc_{b}" for b in BENCHMARKS]]
    rows = []
    for fold in args.folds:
        ckpts = []
        for seed in args.seeds:
            baseline = args.baseline_weights_dir / f"{stem}_fold{fold}.pt"
            out = baseline if (seed == 42 and baseline.exists()) else args.weights_dir / f"{stem}_fold{fold}_seed{seed}.pt"
            if out.exists():
                ckpts.append(out)
        if not ckpts:
            continue
        models = [load_model(c, args.device) for c in ckpts]

        row = {"fold": fold, "n_seeds": len(models)}
        for ef, by_part in eval_by_fasta.items():
            name = eval_name(ef)
            test_entries = by_part.get(f"{fold}.1", [])
            samples = load_samples(test_entries, args.structures,
                                   extra_feats=models[0].extra_feats)
            y, stacked = ensemble_probs(models, samples, args.device)
            if y.size == 0 or len(np.unique(y)) < 2:
                row[f"single_auc_mean_{name}"] = row[f"single_auc_std_{name}"] = row[f"ensemble_auc_{name}"] = ""
                continue
            single_aucs = [roc_auc_score(y, stacked[i]) for i in range(stacked.shape[0])]
            ensemble_pred = stacked.mean(axis=0)
            row[f"single_auc_mean_{name}"] = round(float(np.mean(single_aucs)), 4)
            row[f"single_auc_std_{name}"] = round(float(np.std(single_aucs)), 4)
            row[f"ensemble_auc_{name}"] = round(float(roc_auc_score(y, ensemble_pred)), 4)
        rows.append(row)
        print(f"[fold={fold}] n_seeds={len(models)} " +
              ", ".join(f"{b}: single={row[f'single_auc_mean_{b}']}+/-{row[f'single_auc_std_{b}']} "
                        f"ensemble={row[f'ensemble_auc_{b}']}" for b in BENCHMARKS))
        del models

    with open(args.results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.results}")


if __name__ == "__main__":
    main()
