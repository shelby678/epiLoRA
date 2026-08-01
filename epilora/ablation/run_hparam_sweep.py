"""Hyperparameter/backbone ablation sweep: 5-fold CV per config, champion dataset held fixed.

    python ablation/run_hparam_sweep.py --max-seconds 1800

Complements run_ablation.py (which varies the *data* ablation, holding the
champion recipe fixed) by varying one *training* hyperparameter axis at a
time -- LoRA rank/alpha, LoRA layer count, RYS replay window, head MLP
dimension, dropout, and pretrained backbone -- holding the data ablation
fixed at the champion dataset (allowed_species_homo_sapiens_epitopes.fasta
per ablation/results.csv). The champion setting itself (rank=4, alpha=8,
layers=8, rys=4-8, head=direct, dropout=0.1, backbone=esmif1) is NOT
retrained here -- its 5-fold numbers already exist in ablation/results.csv
and are pulled in at summarize time as the baseline every axis is measured
against.

Every job still early-stops/reports against the same shared benchmark as
run_ablation.py (--eval-fastas), so hparam configs stay comparable to each
other and to the data-ablation table.

Results land incrementally in --results (CSV, one row per config/fold).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

FOLDS = (1, 2, 3, 4, 5)
BENCHMARKS = ("homo_sapiens", "homo_sapiens_mus_musculus")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = Path(__file__).resolve().parents[1] / "train.py"
CHAMPION_DATASET = REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta"

# ESM3 (EvolutionaryScale's "esm" pip package) and fair-esm (ESM-IF1/ESM2, also imported
# as "esm") collide on the same top-level import name, so they live in two separate venvs;
# pick the right interpreter per job based on which backbone the config asks for.
ENV_PYTHON = REPO_ROOT / "epilora/env/bin/python3"
ENV_ESM3_PYTHON = REPO_ROOT / "epilora/env_esm3/bin/python3"


def python_for(config_name: str) -> str:
    if "--backbone" in CONFIGS[config_name] and "esm3" in CONFIGS[config_name]:
        return str(ENV_ESM3_PYTHON) if ENV_ESM3_PYTHON.exists() else sys.executable
    return str(ENV_PYTHON) if ENV_PYTHON.exists() else sys.executable

# name -> extra train.py CLI args (on top of champion defaults: rank=4 alpha=8 layers=8
# rys=4-8 head=direct backbone=esmif1). One axis varied at a time.
CONFIGS: dict[str, list[str]] = {
    # -- LoRA rank (alpha scaled 2x rank, matching the champion's own 4/8 ratio) --
    "lora_rank2_alpha4":  ["--lora-rank", "2", "--lora-alpha", "4"],
    "lora_rank8_alpha16": ["--lora-rank", "8", "--lora-alpha", "16"],
    "lora_rank16_alpha32": ["--lora-rank", "16", "--lora-alpha", "32"],
    # -- LoRA layer count (how many top encoder layers get adapters) --
    "lora_layers2": ["--lora-layers", "2"],
    "lora_layers4": ["--lora-layers", "4"],
    "lora_layers6": ["--lora-layers", "6"],
    # -- RYS replay window (encoder has 8 layers total) --
    "rys_none":     ["--rys-start", "8", "--rys-end", "8"],
    "rys_all":      ["--rys-start", "0", "--rys-end", "8"],
    "rys_last2":    ["--rys-start", "6", "--rys-end", "8"],
    "rys_last6":    ["--rys-start", "2", "--rys-end", "8"],
    # -- head MLP dimension (default: direct Linear(512,1), the champion) --
    "head_dim128":  ["--head-dim", "128"],
    "head_dim256":  ["--head-dim", "256"],
    "head_dim1024": ["--head-dim", "1024"],
    # -- head/MLP dropout (default: 0.1, the champion -- see baseline row) --
    "dropout_0.2": ["--dropout", "0.2"],
    "dropout_0.3": ["--dropout", "0.3"],
    "dropout_0.4": ["--dropout", "0.4"],
    # -- pretrained backbone (sequence-only ESM2/ESM3 vs. structure ESM-IF1) --
    "backbone_esm2_35M":  ["--backbone", "esm2", "--esm2-size", "35M"],
    "backbone_esm2_150M": ["--backbone", "esm2", "--esm2-size", "150M"],
    "backbone_esm2_650M": ["--backbone", "esm2", "--esm2-size", "650M"],
    "backbone_esm3":      ["--backbone", "esm3"],
}

CSV_FIELDS = ["config", "fold", "val_auc", *[f"test_auc_{b}" for b in BENCHMARKS],
              "steps", "seconds", "status"]


def gpu_count() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def run_job(config_name: str, fold: int, gpu_id: int, args, csv_lock, csv_path) -> None:
    out = (args.weights_dir / f"{config_name}_fold{fold}.pt").resolve()
    if out.exists():
        print(f"[{config_name} fold={fold}] already trained -> {out}, skipping", flush=True)
        return
    log_path = args.log_dir / f"{config_name}_fold{fold}.log"

    cmd = [python_for(config_name), str(TRAIN_PY),
           "--fasta", str(args.dataset.resolve()), "--structures", str(args.structures.resolve()),
           "--fold", str(fold), "--out", str(out),
           "--max-seconds", str(args.max_seconds), "--seed", str(args.seed),
           *CONFIGS[config_name]]
    full_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"[{config_name} fold={fold} gpu={gpu_id}] starting -> {log_path}", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(TRAIN_PY.parent), env=full_env,
                              stdout=logf, stderr=subprocess.STDOUT)

    row = {"config": config_name, "fold": fold, "val_auc": "", "steps": "", "seconds": "",
           "status": "ok" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"}
    for b in BENCHMARKS:
        row[f"test_auc_{b}"] = ""

    metrics_path = out.with_suffix(".metrics.json")
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        row["val_auc"] = m["val_auc"]
        row["steps"] = m["steps"]
        row["seconds"] = m["seconds"]
        for b in BENCHMARKS:
            row[f"test_auc_{b}"] = m["test_auc"].get(b, "")

    with csv_lock:
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    test_auc_str = ", ".join(f"{b}={row[f'test_auc_{b}']}" for b in BENCHMARKS)
    print(f"[{config_name} fold={fold} gpu={gpu_id}] done: {row['status']} "
          f"val_auc={row['val_auc']} test_auc={{{test_auc_str}}}", flush=True)


def load_champion_baseline(ablation_results: Path) -> dict[int, dict]:
    """Pull the champion dataset's already-trained per-fold rows out of
    ablation/results.csv, so the champion setting doesn't need retraining here."""
    if not ablation_results.exists():
        return {}
    stem = CHAMPION_DATASET.stem.removesuffix("_epitopes")
    out = {}
    with open(ablation_results, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == stem and r["status"] == "ok":
                out[int(r["fold"])] = r
    return out


def summarize(csv_path: Path, ablation_results: Path) -> None:
    baseline = load_champion_baseline(ablation_results)
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

    by_config: dict[str, list[dict]] = {}
    for r in rows:
        by_config.setdefault(r["config"], []).append(r)
    if baseline:
        by_config["champion (rank4/alpha8/layers8/rys4-8/head-direct/esmif1)"] = list(baseline.values())

    def mean_std(vals: list[float]) -> tuple[float, float]:
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return mean, var ** 0.5

    print(f"\n=== Hyperparameter/backbone sweep summary ({len(rows)} runs, {len(by_config)} configs) ===")
    for bench in BENCHMARKS:
        print(f"\n-- ranked by test_auc[{bench}] (mean +/- std across completed folds) --")
        summary = []
        for config, runs in by_config.items():
            vals = [float(r[f"test_auc_{bench}"]) for r in runs
                    if r["status"] == "ok" and r[f"test_auc_{bench}"] not in ("", "nan")]
            if vals:
                mean, std = mean_std(vals)
                summary.append((mean, std, config, len(vals)))
        summary.sort(reverse=True)
        for mean, std, config, n in summary:
            print(f"  {config:55s}  {mean:.4f} +/- {std:.4f}  (n={n}/{len(FOLDS)} folds)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=CHAMPION_DATASET)
    p.add_argument("--structures", type=Path, default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--weights-dir", type=Path, default=REPO_ROOT / "weights/hparam_sweep")
    p.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "hparam_results.csv")
    p.add_argument("--ablation-results", type=Path, default=Path(__file__).resolve().parent / "results.csv",
                   help="where to pull the champion baseline's already-trained folds from")
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    p.add_argument("--configs", nargs="+", default=None, choices=list(CONFIGS),
                   help="override config list (default: all)")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    if args.summarize_only:
        summarize(args.results, args.ablation_results)
        return

    configs = args.configs or list(CONFIGS)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers or gpu_count()

    jobs = [(c, f) for c in configs for f in args.folds]
    print(f"{len(jobs)} jobs ({len(configs)} configs x {len(args.folds)} folds), "
          f"{n_workers} parallel workers, results -> {args.results}")

    csv_lock = threading.Lock()
    # A pool of free GPU ids (rather than gpu_id = i % n_workers) so a fast-finishing
    # job (e.g. an instant "already trained" skip) frees its GPU for immediate reuse
    # instead of leaving that GPU idle while another GPU gets double-booked.
    gpu_pool: queue.Queue = queue.Queue()
    for g in range(n_workers):
        gpu_pool.put(g)
    threads = []

    def worker(config_name, fold):
        gpu_id = gpu_pool.get()
        try:
            run_job(config_name, fold, gpu_id, args, csv_lock, args.results)
        finally:
            gpu_pool.put(gpu_id)

    for config_name, fold in jobs:
        t = threading.Thread(target=worker, args=(config_name, fold))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    summarize(args.results, args.ablation_results)


if __name__ == "__main__":
    main()
