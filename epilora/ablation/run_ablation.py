"""Run the epiLoRA data ablation sweep: 5-fold CV on every (dataset, model)
pair listed in ``ablation_list.csv``.

    python ablation/run_ablation.py --max-seconds 3600

The sweep is defined by ``--list`` (default ``ablation/ablation_list.csv``),
a CSV with columns ``name,dataset,model,config`` (``config`` optional):

    name             dataset                          model      config
    all_esmif1       all_epitopes.fasta               esmif1
    feats_mr5        min_resolution_5_epitopes.fasta  esmif1     feats_rsa_length.yaml
    rys_none_mr5     min_resolution_5_epitopes.fasta  esmif1     rys_none.yaml

- ``dataset`` is a filename in ``--tte-dir``.
- ``model`` is ``esmif1``, ``esm3``, or ``<esm2|esmc>_<size>`` (e.g. ``esm2_650M``);
  the matching ``configs/backbone_<model>.yaml`` is generated on first use.
- ``config`` (empty = derived from ``model``) points at a named recipe ablation
  under ``epilora/configs/`` (or an absolute path) for rows that vary a training
  axis ``model`` can't express -- extra head features, RYS off, ... (see
  ``ablation_list_feats_rys.csv``). ``model`` still picks the training env.

If ``--list`` doesn't exist yet, it's generated: every dataset under
``--tte-dir`` paired with ``esmif1``, plus the homo-sapiens/min-resolution-10
dataset paired with every other model -- then written out to be hand-edited.

Jobs whose weights + metrics already exist under ``--weights-dir`` are
skipped, with their cached metrics recorded in ``--results`` -- so a list can
be re-run any time to fill in only what's missing (``--force`` to retrain).

For every row x fold this shells out to ``train.py`` (so a crashed run
doesn't take down the sweep), spreading jobs across all visible GPUs. Every
model is trained on its own row's fold-i train split but early-stopped/tested
against the *same* shared benchmark for every row (see train.py's
``--eval-fastas``) -- the homo-sapiens and homo-sapiens+mus-musculus
ablations -- so the ranking at the end answers "which training recipe
predicts human epitopes best," not "which recipe looks best on its own
idiosyncratic test split."

Results land in ``--results`` (one row per sweep-row/fold, updated in place)
and per-job logs in ``--log-dir``; a ranked summary prints at the end (and
can be reprinted with ``--summarize-only``).

Pass ``--slurm`` to submit every job to Slurm (via ``sbatch
ablation/slurm_job.sh``) instead of running locally across this machine's
GPUs -- use when the GPUs live on a separate cluster/machine. Results/weights
land wherever your ``.env``'s ``sync_job_dir()`` ships them; aggregate with
``--summarize-only`` once they've synced back.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import yaml

FOLDS = (1, 2, 3, 4, 5)
BENCHMARKS = ("homo_sapiens", "homo_sapiens_mus_musculus")

REPO_ROOT = Path(__file__).resolve().parents[2]
EPILORA_DIR = Path(__file__).resolve().parents[1]
TRAIN_PY = EPILORA_DIR / "train.py"
CONFIGS_DIR = EPILORA_DIR / "configs"

# Default sweep: every dataset x esmif1, plus this one dataset x every other model.
DEFAULT_MODELS = ("esm2_35M", "esm2_150M", "esm2_650M", "esm3", "esmc_300M", "esmc_600M")
DEFAULT_MULTI_MODEL_DATASET = "allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta"

LIST_FIELDS = ["name", "dataset", "model", "config"]
CSV_FIELDS = ["name", "dataset", "model", "fold", "val_auc",
              *[f"test_auc_{b}" for b in BENCHMARKS], "steps", "seconds", "status"]

_MODEL_RE = re.compile(r"^(?P<backbone>esm2|esmc)_(?P<size>\w+)$")

# Machine-specific env locations -- edit epilora/machine_config.yaml, not this.
_machine_cfg = yaml.safe_load((EPILORA_DIR / "machine_config.yaml").read_text())
ENV_PYTHON = (EPILORA_DIR / _machine_cfg["env"]).resolve() / "bin" / "python3"
ENV_ESM3_PYTHON = (EPILORA_DIR / _machine_cfg["env_esm3"]).resolve() / "bin" / "python3"


def train_python_for(model: str) -> str:
    env_python = ENV_ESM3_PYTHON if (model == "esm3" or model.startswith("esmc")) else ENV_PYTHON
    return str(env_python) if env_python.exists() else sys.executable


def config_for_model(model: str) -> Path | None:
    """train.py's --config for a sweep row's `model` (None = the esmif1 champion).

    Generates and caches configs/backbone_<model>.yaml the first time a given
    model name is used, so new models only need to be added to the sweep
    list, not pre-authored as config files.
    """
    if model == "esmif1":
        return None
    if model == "esm3":
        cfg = {"backbone": "esm3"}
    else:
        m = _MODEL_RE.match(model)
        if not m:
            raise ValueError(f"unrecognized model {model!r} "
                              "(expected esmif1, esm3, esm2_<size>, or esmc_<size>; "
                              "for other backbones or recipe ablations, set the row's "
                              "config column to a configs/*.yaml)")
        cfg = {"backbone": m["backbone"], f"{m['backbone']}_size": m["size"]}

    path = CONFIGS_DIR / f"backbone_{model}.yaml"
    if not path.exists():
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"generated {path}")
    return path


def row_config_path(row: dict) -> Path | None:
    """The row's optional `config` column as a Path (under epilora/configs/,
    or absolute), or None if absent/empty."""
    name = (row.get("config") or "").strip()
    if not name:
        return None
    path = Path(name)
    return path if path.is_absolute() else CONFIGS_DIR / path


def config_for_row(row: dict) -> Path | None:
    """train.py's --config for a sweep row (None = the esmif1 champion): the
    row's `config` column when set, else derived from `model`."""
    explicit = row_config_path(row)
    if explicit is not None:
        return explicit
    return config_for_model(row["model"])


def discover_datasets(tte_dir: Path) -> list[Path]:
    return sorted(tte_dir.glob("*_epitopes.fasta"))


def generate_default_list(tte_dir: Path) -> list[dict]:
    rows = []
    for dataset in discover_datasets(tte_dir):
        stem = dataset.stem.removesuffix("_epitopes")
        rows.append({"name": f"{stem}_esmif1", "dataset": dataset.name, "model": "esmif1",
                     "config": ""})

    special_stem = Path(DEFAULT_MULTI_MODEL_DATASET).stem.removesuffix("_epitopes")
    for model in DEFAULT_MODELS:
        rows.append({"name": f"{special_stem}_{model}", "dataset": DEFAULT_MULTI_MODEL_DATASET,
                     "model": model, "config": ""})
    return rows


def load_list(list_path: Path, tte_dir: Path) -> list[dict]:
    if not list_path.exists():
        rows = generate_default_list(tte_dir)
        if not rows:
            raise SystemExit(f"no *_epitopes.fasta found in {tte_dir}, can't generate {list_path}")
        list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(list_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LIST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"generated default sweep list -> {list_path} ({len(rows)} rows)")
        return rows

    with open(list_path, newline="") as f:
        return list(csv.DictReader(f))


def gpu_count() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def wandb_run_name(job_name: str) -> str:
    """job_name, suffixed with the Slurm job id if this process is running under one.

    Covers both cases: run_ablation.py itself launched inside an salloc'd
    shell (SLURM_JOB_ID inherited here), and the plain-local/no-Slurm case
    (no suffix). The --slurm submission path handles its own suffix at the
    compute node instead, since the job id isn't known until sbatch runs.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    return f"{job_name}_{job_id}" if job_id else job_name


def result_row(row: dict, fold: int, status: str, metrics_path: Path) -> dict:
    """A --results CSV row for one (sweep-row, fold), filled from the job's
    metrics.json when it exists (blank on failure)."""
    out_row = {"name": row["name"], "dataset": row["dataset"], "model": row["model"],
               "fold": fold, "val_auc": "", "steps": "", "seconds": "", "status": status}
    for b in BENCHMARKS:
        out_row[f"test_auc_{b}"] = ""
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        out_row["val_auc"] = m["val_auc"]
        out_row["steps"] = m["steps"]
        out_row["seconds"] = m["seconds"]
        for b in BENCHMARKS:
            out_row[f"test_auc_{b}"] = m["test_auc"].get(b, "")
    return out_row


def record_result(out_row: dict, csv_lock, csv_path: Path) -> None:
    """Upsert out_row into --results, one row per (name, fold), so re-runs
    update in place instead of double-counting in summarize()."""
    with csv_lock:
        rows = []
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                rows = [r for r in csv.DictReader(f)
                        if not (r["name"] == out_row["name"] and int(r["fold"]) == out_row["fold"])]
        rows.append(out_row)
        rows.sort(key=lambda r: (r["name"], int(r["fold"])))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)


def run_job(row: dict, fold: int, gpu_id: int, args, csv_lock, csv_path) -> None:
    name, dataset_name, model = row["name"], row["dataset"], row["model"]
    dataset = (args.tte_dir / dataset_name).resolve()
    out = (args.weights_dir / f"{name}_fold{fold}.pt").resolve()
    log_path = args.log_dir / f"{name}_fold{fold}.log"
    job_name = f"{name}_fold{fold}"
    metrics_path = out.with_suffix(".metrics.json")

    if out.exists() and metrics_path.exists() and not args.force:
        msg = (f"[{name} fold={fold}] already trained -> {out}, skipping "
               f"({args.results.name} updated from cached metrics; --force to retrain)")
        if args.dry_run:
            print(msg, flush=True)
            return
        record_result(result_row(row, fold, "ok", metrics_path), csv_lock, csv_path)
        print(msg, flush=True)
        return

    # Resolve to absolute paths: the subprocess below runs with cwd=TRAIN_PY.parent,
    # which would otherwise break any relative paths passed on argv.
    cmd = [train_python_for(model), str(TRAIN_PY),
           "--fasta", str(dataset), "--structures", str(args.structures.resolve()),
           "--fold", str(fold), "--out", str(out),
           "--max-seconds", str(args.max_seconds), "--seed", str(args.seed)]
    cfg_path = config_for_row(row)
    if cfg_path is not None:
        cmd += ["--config", str(cfg_path)]
    if args.wandb:
        cmd += ["--wandb", "--wandb-project", args.wandb_project,
                "--wandb-run-name", wandb_run_name(job_name)]
    full_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    if args.dry_run:
        print(f"[{name} fold={fold} gpu={gpu_id}] DRY RUN: {' '.join(cmd)}", flush=True)
        return

    print(f"[{name} fold={fold} gpu={gpu_id}] starting -> {log_path}", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(TRAIN_PY.parent), env=full_env,
                              stdout=logf, stderr=subprocess.STDOUT)

    status = "ok" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    out_row = result_row(row, fold, status, metrics_path)
    record_result(out_row, csv_lock, csv_path)

    test_auc_str = ", ".join(f"{b}={out_row[f'test_auc_{b}']}" for b in BENCHMARKS)
    print(f"[{name} fold={fold} gpu={gpu_id}] done: {status} "
          f"val_auc={out_row['val_auc']} test_auc={{{test_auc_str}}}", flush=True)


def submit_slurm_job(row: dict, fold: int, args) -> None:
    """Submit one (row, fold) job to Slurm via ``sbatch args.slurm_script``.

    Fire-and-forget: the job trains and calls ``sync_job_dir()`` on the
    compute node itself (see slurm_job.sh), so there's no local result to
    record here -- unlike run_job(), this doesn't touch --results.
    """
    name, dataset_name, model = row["name"], row["dataset"], row["model"]
    dataset = (args.tte_dir / dataset_name).resolve()
    job_name = f"{name}_fold{fold}"

    env = {
        "TRAIN_PYTHON": train_python_for(model),
        "TRAIN_PY": str(TRAIN_PY),
        "FASTA": str(dataset),
        "STRUCTURES": str(args.structures.resolve()),
        "FOLD": str(fold),
        "OUT_NAME": f"{name}_fold{fold}.pt",
        "MAX_SECONDS": str(args.max_seconds),
        "SEED": str(args.seed),
        "WANDB": "1" if args.wandb else "0",
    }
    cfg_path = config_for_row(row)
    if cfg_path is not None:
        env["CONFIG_FILE"] = str(cfg_path)
    if args.wandb:
        env["WANDB_PROJECT"] = args.wandb_project
        env["WANDB_RUN_NAME"] = job_name

    cmd = ["sbatch", f"--job-name={job_name}",
           f"--export=ALL,{','.join(f'{k}={v}' for k, v in env.items())}"]
    cmd += shlex.split(args.slurm_args)
    cmd += [str(args.slurm_script)]

    if args.dry_run:
        print(f"[{job_name}] DRY RUN: {' '.join(cmd)}", flush=True)
        return

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{job_name}] sbatch FAILED: {proc.stderr.strip()}", flush=True)
    else:
        print(f"[{job_name}] {proc.stdout.strip()}", flush=True)


def summarize(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"no results yet at {csv_path}")
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)

    def mean_std(vals: list[float]) -> tuple[float, float]:
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return mean, var ** 0.5

    print(f"\n=== Ablation summary ({len(rows)} runs, {len(by_name)} sweep rows) ===")
    for bench in BENCHMARKS:
        print(f"\n-- ranked by test_auc[{bench}] (mean +/- std across completed folds) --")
        summary = []
        for name, runs in by_name.items():
            vals = [float(r[f"test_auc_{bench}"]) for r in runs
                    if r["status"] == "ok" and r[f"test_auc_{bench}"] not in ("", "nan")]
            if vals:
                mean, std = mean_std(vals)
                summary.append((mean, std, name, len(vals)))
        summary.sort(reverse=True)
        for mean, std, name, n in summary:
            print(f"  {name:55s}  {mean:.4f} +/- {std:.4f}  (n={n}/{len(FOLDS)} folds)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tte-dir", type=Path, default=REPO_ROOT / "data/train_test_eval")
    p.add_argument("--structures", type=Path, default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--weights-dir", type=Path, default=REPO_ROOT / "weights/ablation")
    p.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results.csv")
    p.add_argument("--list", type=Path, default=Path(__file__).resolve().parent / "ablation_list.csv",
                   help="CSV of name,dataset,model,config rows defining the sweep "
                        "(config optional; generated with a default sweep the first "
                        "time, if missing)")
    p.add_argument("--max-seconds", type=int, default=14400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    p.add_argument("--force", action="store_true",
                   help="retrain jobs whose weights + metrics already exist under "
                        "--weights-dir (by default those are skipped, with their "
                        "cached metrics recorded in --results)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="parallel jobs (default: number of visible GPUs)")
    p.add_argument("--summarize-only", action="store_true",
                   help="skip training, just print the summary from --results")
    p.add_argument("--dry-run", action="store_true",
                   help="print each job's train.py command (or sbatch command, with --slurm) "
                        "without actually training or submitting anything")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="log every job in the sweep to Weights & Biases (on by default; pass "
                        "--no-wandb to disable)")
    p.add_argument("--wandb-project", default="epilora", help="W&B project name (only used with --wandb)")
    p.add_argument("--slurm", action="store_true",
                   help="submit every job to Slurm (sbatch --slurm-script) instead of running "
                        "locally across this machine's GPUs -- use when GPUs live on a separate "
                        "cluster/machine")
    p.add_argument("--slurm-script", type=Path, default=Path(__file__).resolve().parent / "slurm_job.sh",
                   help="sbatch script submitted per job when --slurm is set")
    p.add_argument("--slurm-args", default="",
                   help="extra args appended to every `sbatch` call, e.g. "
                        "'--partition=other --gres=gpu:1' to override slurm_job.sh's #SBATCH "
                        "defaults (useful for running the same sweep against a second machine)")
    args = p.parse_args()

    if args.summarize_only:
        summarize(args.results)
        return

    rows = load_list(args.list, args.tte_dir)

    missing = [r["dataset"] for r in rows if not (args.tte_dir / r["dataset"]).exists()]
    if missing:
        p.error(f"dataset(s) referenced in {args.list} not found under {args.tte_dir}: {sorted(set(missing))}")

    bad_cfgs = sorted({str(c) for r in rows if (c := row_config_path(r)) is not None and not c.exists()})
    if bad_cfgs:
        p.error(f"config(s) referenced in {args.list} not found (relative names resolve "
                f"under {CONFIGS_DIR}): {bad_cfgs}")

    jobs = [(row, f) for row in rows for f in args.folds]

    if args.slurm:
        if not args.slurm_script.exists():
            p.error(f"--slurm-script not found: {args.slurm_script}")
        verb = "would submit" if args.dry_run else "submitting"
        print(f"{len(jobs)} jobs ({len(rows)} sweep rows x {len(args.folds)} folds), "
              f"{verb} to Slurm via {args.slurm_script}")
        for row, fold in jobs:
            submit_slurm_job(row, fold, args)
        if not args.dry_run:
            print(f"\nSubmitted {len(jobs)} jobs. Each trains and calls sync_job_dir() on its own "
                  f"compute node -- once results have synced back to --weights-dir, rerun with "
                  f"--summarize-only.")
        return

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers or gpu_count()

    print(f"{len(jobs)} jobs ({len(rows)} sweep rows x {len(args.folds)} folds), "
          f"{n_workers} parallel workers, results -> {args.results}")

    csv_lock = threading.Lock()
    # A pool of free GPU ids (rather than gpu_id = i % n_workers) so a fast-finishing
    # job frees its GPU for immediate reuse instead of leaving that GPU idle while
    # another GPU gets double-booked.
    gpu_pool: queue.Queue = queue.Queue()
    for g in range(n_workers):
        gpu_pool.put(g)
    threads = []

    def worker(row, fold):
        gpu_id = gpu_pool.get()
        try:
            run_job(row, fold, gpu_id, args, csv_lock, args.results)
        finally:
            gpu_pool.put(gpu_id)

    for row, fold in jobs:
        t = threading.Thread(target=worker, args=(row, fold))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if not args.dry_run:
        summarize(args.results)


if __name__ == "__main__":
    main()
