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

Every hyperparameter/backbone axis (LoRA rank/alpha/layers, RYS window, head
dim, dropout, backbone) lives in the ``TrainConfig`` dataclass below, not
on the CLI directly: ``--config`` points at one of the named ablations under
``configs/`` (default: none, i.e. the champion recipe, ``TrainConfig()``'s
own field defaults). ``--max-seconds``/``--eval-fastas``/``--wandb*`` are the
only config fields still overridable straight from the CLI, for one-off
tweaks without writing a new YAML file.

Early stopping and test-set reporting are evaluated against a *fixed* shared
benchmark (the config's ``eval_fastas``, default the homo-sapiens and
homo-sapiens+mus-musculus ablations) rather than ``--fasta``'s own fold-i
split, so every ablation's model is judged on the same held-out set — the
first ``eval_fastas`` entry's ``i.0`` records are used for early stopping;
every entry's ``i.1`` records are reported as test AUC. Only the LoRA
adapters, the RYS-replayed encoder layers, and the head are saved (~a few
MB); the frozen ESM-IF1 backbone is re-downloaded at load time.

Must run in the fair-esm (py3.9) environment — see README / requirements.txt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

from data import load_samples, parse_fasta
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent  # so defaults don't depend on cwd
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for training but torch.cuda.is_available() is False.")
DEVICE = "cuda"
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 200
VAL_INTERVAL = 200
PATIENCE = 10


@dataclasses.dataclass
class TrainConfig:
    """Every hyperparameter/backbone axis ablation/run_hparam_sweep.py varies,
    plus the eval benchmark and W&B settings -- see the module docstring above
    for how this fits with the CLI. Field defaults are the champion recipe.

    Job-identity args (--fasta/--structures/--out/--fold/--seed) are not part
    of this config; they stay plain train.py flags.
    """
    backbone: str = "esmif1"           # esmif1, esm2, esm3, or esmc
    esm2_size: str = "650M"            # 35M, 150M, or 650M (only used when backbone=esm2)
    esmc_size: str = "600M"            # 300M or 600M (only used when backbone=esmc)
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_layers: int = 8               # number of top transformer layers to adapt with LoRA
    rys_start: int = 4                 # replay window start
    rys_end: int = 8                   # replay window end, exclusive; rys_end<=rys_start disables RYS
    head_dim: Optional[int] = None     # if set, MLP head Linear(hidden,head_dim)-GELU-Linear(.,1)
    dropout: float = 0.1               # dropout applied in the head (and MLP hidden layer, if used)
    max_seconds: int = 14400
    eval_fastas: list[Path] = dataclasses.field(default_factory=lambda: [
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta",
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
    ])
    wandb: bool = False
    wandb_project: str = "epilora"
    wandb_run_name: Optional[str] = None


def print_config(cfg: TrainConfig) -> None:
    for k, v in dataclasses.asdict(cfg).items():
        logger.info(f"  {k}: {v}")


def load_config(path: Path | None) -> TrainConfig:
    """Load a named ablation's YAML overrides (see configs/) on top of the champion defaults."""
    cfg = TrainConfig()
    if path is None:
        logger.info("Using champion defaults:")
        print_config(cfg)
        return cfg
    overrides = yaml.safe_load(Path(path).read_text())  # raises FileNotFoundError if path doesn't exist
    if not overrides:
        raise ValueError(f"{path}: config file is empty")
    field_names = {f.name for f in dataclasses.fields(cfg)}
    unknown = set(overrides) - field_names
    if unknown:
        raise ValueError(f"{path}: unknown config key(s) {sorted(unknown)}")
    if "eval_fastas" in overrides:
        overrides["eval_fastas"] = [Path(p) for p in overrides["eval_fastas"]]
    cfg = dataclasses.replace(cfg, **overrides)
    logger.info(f"Loaded config from {path}:")
    print_config(cfg)
    return cfg


@torch.no_grad()
def evaluate_auc(model, samples) -> float:
    model.eval()
    logits_all, labels_all = [], []
    for header, seq, labels, coords in samples:
        if coords is None:
            continue
        try:
            lg = model([coords], [seq])[0].cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"evaluate_auc: forward pass failed on {header!r}: {e}") from e
        logits_all.append(lg)
        labels_all.append(labels)
    if not logits_all:
        raise RuntimeError(f"evaluate_auc: no structure-backed samples among {len(samples)} given")
    y, s = np.concatenate(labels_all), np.concatenate(logits_all)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan")


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/torch RNGs (LoRA + head init draw from the global torch RNG)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_samples, val_samples, max_seconds: int, seed: int = 42, wandb_run=None) -> dict:
    """Train in-place with early stopping on val ROC-AUC; keep the best weights."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, WARMUP_STEPS)))

    best_auc, best_state, no_improve, step, tl, n_skipped = -1.0, None, 0, 0, 0.0, 0
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
                n_skipped += 1
                logger.warning(f"train: skip {header}: {e}")
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
                logger.info(f"step={step} train_loss={tl:.4f} val_auc={va:.4f} "
                            f"skipped={n_skipped} {time.time()-start:.0f}s")
                if wandb_run is not None:
                    wandb_run.log({"train_loss": tl, "val_auc": va, "skipped": n_skipped}, step=step)
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
    else:
        logger.warning("training ended without ever reaching a validation checkpoint "
                        "(max-seconds too short, or every sample failed?); "
                        "reporting current, not best, weights")
    return {"steps": step, "val_auc": evaluate_auc(model, val_samples),
            "best_auc": best_auc, "skipped": n_skipped}


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
    p.add_argument("--seed", type=int, default=42,
                   help="seed for dataset shuffling and LoRA/head weight init")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML file overriding the champion hyperparameter/backbone recipe "
                        "(see configs/ for the named ablations); omit for the champion itself")
    p.add_argument("--max-seconds", type=int, default=None,
                   help="override the config's max_seconds")
    p.add_argument("--eval-fastas", type=Path, nargs="+", default=None,
                   help="override the config's eval_fastas -- fixed shared benchmark(s); fold "
                        "i.0 of the first is used for early stopping, fold i.1 of every one is "
                        "reported as test AUC")
    p.add_argument("--wandb", action="store_true", help="log this run to Weights & Biases")
    p.add_argument("--wandb-project", default=None, help="override the config's wandb_project")
    p.add_argument("--wandb-run-name", default=None,
                   help="override the config's wandb_run_name (default: <backbone>_<fasta stem>_fold<N>)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_seconds is not None:
        cfg.max_seconds = args.max_seconds
    if args.eval_fastas is not None:
        cfg.eval_fastas = args.eval_fastas
    if args.wandb:
        cfg.wandb = True
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        cfg.wandb_run_name = args.wandb_run_name

    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    wandb_run = None
    if cfg.wandb:
        import wandb
        run_name = cfg.wandb_run_name or f"{cfg.backbone}_{args.fasta.stem}_fold{args.fold}"
        def jsonable(v):
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, list):
                return [jsonable(x) for x in v]
            return v
        wandb_config = {k: jsonable(v) for k, v in
                        {**vars(args), **dataclasses.asdict(cfg)}.items()}
        wandb_run = wandb.init(project=cfg.wandb_project, name=run_name, config=wandb_config)

    val_label, test_label = f"{args.fold}.0", f"{args.fold}.1"

    by_part = parse_fasta(args.fasta)
    train_entries = [e for k, v in by_part.items() if int(k.split(".")[0]) != args.fold for e in v]

    eval_by_fasta = {ef: parse_fasta(ef) for ef in cfg.eval_fastas}
    val_entries = eval_by_fasta[cfg.eval_fastas[0]].get(val_label, []) # use the first fasta for evalution
    if not val_entries:
        p.error(f"no '{val_label}' records in {cfg.eval_fastas[0]}")

    logger.info(f"Loading structures ({DEVICE}) ...")
    train_samples = load_samples(train_entries, args.structures)
    val_samples = load_samples(val_entries, args.structures)
    n_tr = sum(1 for *_, c in train_samples if c is not None)
    n_va = sum(1 for *_, c in val_samples if c is not None)
    logger.info(f"fold={args.fold}  train={len(train_samples)} ({n_tr} w/ struct)  "
                f"val={len(val_samples)} ({n_va} w/ struct)")
    if n_tr == 0:
        p.error("no structure-backed training samples found — check --structures path")

    if cfg.backbone == "esmif1":
        model = build_model(device=DEVICE, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                            n_lora_layers=cfg.lora_layers, rys_start=cfg.rys_start,
                            rys_end=cfg.rys_end, dropout=cfg.dropout, head_dim=cfg.head_dim)
    elif cfg.backbone == "esm2":
        from model import build_model_esm2
        model = build_model_esm2(device=DEVICE, size=cfg.esm2_size, rank=cfg.lora_rank,
                                 alpha=cfg.lora_alpha, n_lora_layers=cfg.lora_layers,
                                 dropout=cfg.dropout, head_dim=cfg.head_dim)
    elif cfg.backbone == "esm3":
        from model import build_model_esm3
        model = build_model_esm3(device=DEVICE, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                                 n_lora_layers=cfg.lora_layers, dropout=cfg.dropout,
                                 head_dim=cfg.head_dim)
    else:
        from model import build_model_esmc
        model = build_model_esmc(device=DEVICE, size=cfg.esmc_size, rank=cfg.lora_rank,
                                 alpha=cfg.lora_alpha, n_lora_layers=cfg.lora_layers,
                                 dropout=cfg.dropout, head_dim=cfg.head_dim)
    t0 = time.time()
    res = train(model, train_samples, val_samples, cfg.max_seconds, seed=args.seed, wandb_run=wandb_run)
    logger.info(f"Done: best val_auc={res['best_auc']:.4f} steps={res['steps']} "
                f"skipped={res['skipped']} {time.time()-t0:.0f}s")

    test_aucs = {}
    for ef, by_part_eval in eval_by_fasta.items():
        test_entries = by_part_eval.get(test_label)
        test_samples = load_samples(test_entries, args.structures)
        test_aucs[eval_name(ef)] = evaluate_auc(model, test_samples)
    for name, auc in test_aucs.items():
        logger.info(f"test_auc[{name}]={auc:.4f}")

    if wandb_run is not None:
        wandb_run.summary["best_val_auc"] = res["best_auc"]
        wandb_run.summary["steps"] = res["steps"]
        wandb_run.summary["seconds"] = round(time.time() - t0)
        for name, auc in test_aucs.items():
            wandb_run.summary[f"test_auc_{name}"] = auc
        wandb_run.finish()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": cfg.backbone,
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
        "backbone": cfg.backbone, "fasta": str(args.fasta), "fold": args.fold,
        "seed": args.seed, "steps": res["steps"],
        "seconds": round(time.time() - t0), "val_auc": res["best_auc"], "test_auc": test_aucs,
    }, indent=2))


if __name__ == "__main__":
    main()
