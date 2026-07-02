"""Max-ensemble evaluation over backbone-model combinations.

Loads every data/ensemble_preds/<model>.npz, then for each non-empty combination
of models computes the max-ensemble prediction (prob = max over members per
residue) on the residues all members share, and reports pooled + per-fold
ROC-AUC. Single models are reported on their full residue set; multi-model
combos also show each member's AUC restricted to the shared set for a fair
read on whether the ensemble actually helps.

Run:  uv run python ensemble_eval.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ensemble_io import load_preds

PRED_DIR = Path("data/ensemble_preds")
FOLDS = ["1", "2", "3"]
OP = sys.argv[1] if len(sys.argv) > 1 else "max"  # "max" or "mean"


def _auc(labels, probs):
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _fold_mean_std(per_key_fold, keys, prob_map, label_map):
    aucs = []
    for f in FOLDS:
        ks = [k for k in keys if per_key_fold[k] == f]
        if not ks:
            continue
        y = np.array([label_map[k] for k in ks])
        p = np.array([prob_map[k] for k in ks])
        a = _auc(y, p)
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    models = {}
    for path in sorted(PRED_DIR.glob("*.npz")):
        name = path.stem
        d = load_preds(path)
        prob = {k: pr for k, pr in zip(d["key"], d["prob"])}
        label = {k: lb for k, lb in zip(d["key"], d["label"])}
        fold = {k: fo for k, fo in zip(d["key"], d["fold"])}
        models[name] = {"prob": prob, "label": label, "fold": fold, "keys": set(prob)}
        print(f"loaded {name}: {len(prob):,} residues")

    if not models:
        print("No prediction files found in", PRED_DIR)
        return

    names = sorted(models)
    # union fold map / label map (consistent across models for shared keys)
    fold_all = {}
    label_all = {}
    for m in models.values():
        fold_all.update(m["fold"])
        label_all.update(m["label"])

    print(f"\n{'combination':32s} {'n_res':>7s} {'pooled':>7s} {'fold_mean±std':>16s}  (members on shared set)")
    print("-" * 100)

    rows = []
    for r in range(1, len(names) + 1):
        for combo in combinations(names, r):
            shared = set.intersection(*(models[m]["keys"] for m in combo))
            if not shared:
                continue
            shared = sorted(shared)
            y = np.array([label_all[k] for k in shared])
            op = OP
            if op == "max":
                ens_prob = {k: max(models[m]["prob"][k] for m in combo) for k in shared}
            else:  # mean
                ens_prob = {k: float(np.mean([models[m]["prob"][k] for m in combo])) for k in shared}
            p = np.array([ens_prob[k] for k in shared])
            pooled = _auc(y, p)
            fm, fs = _fold_mean_std(fold_all, shared, ens_prob, label_all)
            # members restricted to shared set
            member_str = ""
            if len(combo) > 1:
                parts = []
                for m in combo:
                    pm = np.array([models[m]["prob"][k] for k in shared])
                    parts.append(f"{m}={_auc(y, pm):.4f}")
                member_str = "  " + " ".join(parts)
            label = f"{op}(" + "+".join(combo) + ")" if len(combo) > 1 else combo[0]
            print(f"{label:32s} {len(shared):7d} {pooled:7.4f} {fm:7.4f}±{fs:.4f}{member_str}")
            rows.append((label, len(shared), pooled, fm, fs))

    rows.sort(key=lambda x: x[3], reverse=True)
    print("\nRanked by fold-mean ROC-AUC:")
    for label, n, pooled, fm, fs in rows:
        print(f"  {fm:.4f} ± {fs:.4f}   {label}  (n={n}, pooled={pooled:.4f})")


if __name__ == "__main__":
    main()
