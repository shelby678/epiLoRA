"""Evaluate a trained checkpoint (any backbone, or an ensemble of several) on
data/train_test_eval/eval/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta
-- the final held-out benchmark, distinct from the CV folds used during the
ablation sweep -- and report per-residue ROC-AUC.

    python eval_final.py --weights weights/champion.pt
    python eval_final.py --weights weights/ensemble/*_seed42.pt weights/ensemble/*_seed43.pt --out preds.csv

Multiple --weights average their sigmoid probabilities (ensemble). Optionally
writes a per-residue CSV (instance, position, aa, prob, label) with --out, so
these predictions can be lined up against DiscoTope 3.0's own per-residue
scores on the same antigens.

Must be run with epilora/env/bin/python3 for esmif1/esm2 checkpoints, or
epilora/env_esm3/bin/python3 for esm3/esmc checkpoints (see predict.load_model).
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
    """Return (headers, seqs, labels_list, probs_list) -- ``probs_list[i]``
    averaged sigmoid across ``models`` for sample i."""
    headers, seqs, labels_list, probs_list = [], [], [], []
    for header, seq, labels, coords, feats in samples:
        if coords is None:
            raise ValueError(f"no usable backbone coordinates for {header!r} -- "
                             "every eval-set antigen must be scorable for a fair comparison")
        if feats is None and models[0].n_extra_feats:
            raise ValueError(f"could not build head features {models[0].extra_feats} for "
                             f"{header!r} -- every eval-set antigen must be scorable for a "
                             "fair comparison")
        per_model = [1.0 / (1.0 + np.exp(-m([coords], [seq], [feats])[0].cpu().numpy()))
                     for m in models]
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
    p.add_argument("--eval-fasta", type=Path, default=REPO_ROOT /
                   "data/train_test_eval/eval/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta")
    p.add_argument("--structures", type=Path, default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--out", type=Path, default=None, help="optional per-residue CSV output")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_model(w, device) for w in args.weights]
    feat_names = {m.extra_feats for m in models}
    if len(feat_names) > 1:
        raise SystemExit("cannot ensemble checkpoints whose heads read different extra "
                         "features: " + "; ".join(str(sorted(f)) for f in feat_names))

    by_part = parse_fasta(args.eval_fasta)
    entries = [e for v in by_part.values() for e in v]
    samples = load_samples(entries, args.structures, extra_feats=models[0].extra_feats)
    n_struct = sum(1 for s in samples if s[3] is not None)
    print(f"eval.fasta: {len(entries)} records, {n_struct} with usable structure")

    headers, seqs, labels_list, probs_list = predict_all(models, samples, device)
    if labels_list:
        y = np.concatenate(labels_list)
        s = np.concatenate(probs_list)
        auc = roc_auc_score(y, s) if len(np.unique(y)) >= 2 else float("nan")
    else:
        y, auc = np.array([]), float("nan")
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
