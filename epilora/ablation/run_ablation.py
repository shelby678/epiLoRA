"""Run the epiLoRA data ablation sweep: 5-fold CV on every data/train_test_eval/*_epitopes.fasta.

    python ablation/run_ablation.py --max-seconds 3600

For every ablation dataset and every CV fold (1-5) this shells out to
``train.py`` (so a crashed run doesn't take down the sweep), spreading jobs
across all visible GPUs. Every model is trained on its own dataset's fold-i
train split but early-stopped/tested against the *same* shared benchmark for
every dataset (see train.py's ``--eval-fastas``) -- the homo-sapiens and
homo-sapiens+mus-musculus ablations -- so the per-dataset ranking at the end
answers "which training recipe predicts human epitopes best," not "which
recipe looks best on its own idiosyncratic test split."

Results land incrementally in ``--results`` (CSV, one row per dataset/fold)
and per-job logs in ``--log-dir``; a ranked summary prints at the end (and
can be reprinted any time from an existing CSV with ``--summarize-only``).
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
# train.py needs fair-esm/biotite/torch-geometric, which live in this dedicated venv
# (the shared base env this repo used to rely on has proven mutable/unstable).
ENV_PYTHON = REPO_ROOT / "epilora/env/bin/python3"
TRAIN_PYTHON = str(ENV_PYTHON) if ENV_PYTHON.exists() else sys.executable

CSV_FIELDS = ["dataset", "fold", "val_auc", *[f"test_auc_{b}" for b in BENCHMARKS],
              "steps", "seconds", "status"]


def discover_datasets(tte_dir: Path) -> list[Path]:
    return sorted(tte_dir.glob("*_epitopes.fasta"))


def gpu_count() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def run_job(dataset: Path, fold: int, gpu_id: int, args, csv_lock, csv_path) -> None:
    stem = dataset.stem.removesuffix("_epitopes")
    out = (args.weights_dir / f"{stem}_fold{fold}.pt").resolve()
    log_path = args.log_dir / f"{stem}_fold{fold}.log"

    # Resolve to absolute paths: the subprocess below runs with cwd=TRAIN_PY.parent,
    # which would otherwise break any relative paths passed on argv.
    cmd = [TRAIN_PYTHON, str(TRAIN_PY),
           "--fasta", str(dataset.resolve()), "--structures", str(args.structures.resolve()),
           "--fold", str(fold), "--out", str(out),
           "--max-seconds", str(args.max_seconds), "--seed", str(args.seed)]
    full_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"[{stem} fold={fold} gpu={gpu_id}] starting -> {log_path}", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(TRAIN_PY.parent), env=full_env,
                              stdout=logf, stderr=subprocess.STDOUT)

    row = {"dataset": stem, "fold": fold, "val_auc": "", "steps": "", "seconds": "",
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
    print(f"[{stem} fold={fold} gpu={gpu_id}] done: {row['status']} "
          f"val_auc={row['val_auc']} test_auc={{{test_auc_str}}}", flush=True)


def summarize(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"no results yet at {csv_path}")
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    def mean_std(vals: list[float]) -> tuple[float, float]:
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return mean, var ** 0.5

    print(f"\n=== Ablation summary ({len(rows)} runs, {len(by_dataset)} datasets) ===")
    for bench in BENCHMARKS:
        print(f"\n-- ranked by test_auc[{bench}] (mean +/- std across completed folds) --")
        summary = []
        for dataset, runs in by_dataset.items():
            vals = [float(r[f"test_auc_{bench}"]) for r in runs
                    if r["status"] == "ok" and r[f"test_auc_{bench}"] not in ("", "nan")]
            if vals:
                mean, std = mean_std(vals)
                summary.append((mean, std, dataset, len(vals)))
        summary.sort(reverse=True)
        for mean, std, dataset, n in summary:
            print(f"  {dataset:45s}  {mean:.4f} +/- {std:.4f}  (n={n}/{len(FOLDS)} folds)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tte-dir", type=Path, default=REPO_ROOT / "data/train_test_eval")
    p.add_argument("--structures", type=Path, default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--weights-dir", type=Path, default=REPO_ROOT / "weights/ablation")
    p.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results.csv")
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    p.add_argument("--datasets", type=Path, nargs="+", default=None,
                   help="override dataset list (default: all *_epitopes.fasta in --tte-dir)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="parallel jobs (default: number of visible GPUs)")
    p.add_argument("--summarize-only", action="store_true",
                   help="skip training, just print the summary from --results")
    args = p.parse_args()

    if args.summarize_only:
        summarize(args.results)
        return

    datasets = args.datasets or discover_datasets(args.tte_dir)
    if not datasets:
        p.error(f"no *_epitopes.fasta found in {args.tte_dir}")

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers or gpu_count()

    jobs = [(d, f) for d in datasets for f in args.folds]
    print(f"{len(jobs)} jobs ({len(datasets)} datasets x {len(args.folds)} folds), "
          f"{n_workers} parallel workers, results -> {args.results}")

    csv_lock = threading.Lock()
    # A pool of free GPU ids (rather than gpu_id = i % n_workers) so a fast-finishing
    # job frees its GPU for immediate reuse instead of leaving that GPU idle while
    # another GPU gets double-booked.
    gpu_pool: queue.Queue = queue.Queue()
    for g in range(n_workers):
        gpu_pool.put(g)
    threads = []

    def worker(dataset, fold):
        gpu_id = gpu_pool.get()
        try:
            run_job(dataset, fold, gpu_id, args, csv_lock, args.results)
        finally:
            gpu_pool.put(gpu_id)

    for dataset, fold in jobs:
        t = threading.Thread(target=worker, args=(dataset, fold))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    summarize(args.results)


if __name__ == "__main__":
    main()
