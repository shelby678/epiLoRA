"""Evaluate a trained checkpoint (any backbone, or an ensemble of several) on
data/results/eval.fasta -- the final held-out benchmark, distinct from the CV
folds used during the ablation sweep -- and report per-residue ROC-AUC.

    python eval_final.py --weights weights/champion.pt
    python eval_final.py --weights weights/ensemble/*_seed42.pt weights/ensemble/*_seed43.pt --out preds.csv

Multiple --weights average their sigmoid probabilities (ensemble). Optionally
writes a per-residue CSV (instance, position, aa, prob, label) with --out, so
these predictions can be lined up against DiscoTope 3.0's own per-residue
scores on the same antigens.

Must be run with epilora/env/bin/python3 for esmif1/esm2 checkpoints, or
epilora/env_esm3/bin/python3 for esm3 checkpoints (see predict.load_model).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import load_samples, parse_fasta
from predict import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def predict_all(models, samples, device):
    """Return (headers, seqs, labels_list, probs_list) for structure-backed samples,
    ``probs_list[i]`` averaged sigmoid across ``models`` for sample i."""
    headers, seqs, labels_list, probs_list = [], [], [], []
    for header, seq, labels, coords in samples:
        if coords is None:
            continue
        per_model = []
        ok = True
        for m in models:
            try:
                logits = m([coords], [seq])[0].cpu().numpy()
            except Exception:
                ok = False
                break
            per_model.append(1.0 / (1.0 + np.exp(-logits)))
        if not ok:
            continue
        headers.append(header)
        seqs.append(seq)
        labels_list.append(labels)
        probs_list.append(np.mean(per_model, axis=0))
    return headers, seqs, labels_list, probs_list


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, nargs="+", required=True,
                   help="one checkpoint (single model) or several (ensemble average)")
    p.add_argument("--eval-fasta", type=Path, default=REPO_ROOT / "data/results/eval.fasta")
    p.add_argument("--structures", type=Path, default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--out", type=Path, default=None, help="optional per-residue CSV output")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_model(w, device) for w in args.weights]

    by_part = parse_fasta(args.eval_fasta)
    entries = [e for v in by_part.values() for e in v]
    samples = load_samples(entries, args.structures)
    n_struct = sum(1 for *_, c in samples if c is not None)
    print(f"eval.fasta: {len(entries)} records, {n_struct} with usable structure")

    headers, seqs, labels_list, probs_list = predict_all(models, samples, device)
    y = np.concatenate(labels_list)
    s = np.concatenate(probs_list)
    auc = roc_auc_score(y, s) if len(np.unique(y)) >= 2 else float("nan")
    print(f"n_models={len(models)}  n_residues={len(y)}  n_antigens={len(headers)}  ROC-AUC={auc:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance", "position", "aa", "prob", "label"])
            for header, seq, labels, probs in zip(headers, seqs, labels_list, probs_list):
                instance = header.split()[0]
                for i, (aa, lab, pr) in enumerate(zip(seq, labels, probs), start=1):
                    w.writerow([instance, i, aa, f"{pr:.4f}", lab])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
