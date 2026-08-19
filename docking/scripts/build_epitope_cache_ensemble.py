#!/usr/bin/env python3
"""
Ensemble epiLoRA's own epitope probability with DiscoTope-3.0's raw score
into cache/epitope_ensemble/{stem}.csv (res_id,prob) -- same format as
cache/epitope/{stem}.csv, so generate_restraints.py needs no changes to
consume it.

DiscoTope-3.0's score is rank-normalized to [0, 1] (epiLoRA's prob already
is) then averaged 50/50 with epiLoRA's prob, per
benchmarking/ensemble_with_discotope.py's finding that this blend beats
either model alone (AUC 0.7892 vs 0.7734 epiLoRA-only / 0.7774
DiscoTope-3.0-only).

Only built for the predicted (`ag`, not `ag_gt`) antigen stems, since that's
the only conditioning source the main Snakefile's docking pipeline uses.

Run with plain stdlib python (no special env needed).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPILORA_CACHE = ROOT / "cache" / "epitope"
DISCO_DIR = ROOT / "scratch" / "discotope_out" / "antigens_ag"
OUT_CACHE = ROOT / "cache" / "epitope_ensemble"


def rank_normalize(values):
    """Map values to [0, 1] via their rank (ties broken by position order)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, i in enumerate(order):
        ranks[i] = rank / (len(values) - 1) if len(values) > 1 else 0.0
    return ranks


def main():
    OUT_CACHE.mkdir(parents=True, exist_ok=True)

    n = 0
    for epilora_csv in sorted(EPILORA_CACHE.glob("*_ag.csv")):
        stem = epilora_csv.stem  # e.g. "7so5_ag"
        disco_csv = DISCO_DIR / f"{stem}_A_discotope3.csv"
        if not disco_csv.exists():
            print(f"WARNING: no DiscoTope-3.0 output for {stem}, skipping")
            continue

        with open(epilora_csv) as f:
            epilora_probs = {int(r["res_id"]): float(r["prob"]) for r in csv.DictReader(f)}

        disco_scores = {}
        with open(disco_csv) as f:
            for r in csv.DictReader(f):
                disco_scores[int(r["res_id"])] = float(r["DiscoTope-3.0_score"])

        shared = sorted(set(epilora_probs) & set(disco_scores))
        n_epilora_only = len(set(epilora_probs) - set(disco_scores))
        n_disco_only = len(set(disco_scores) - set(epilora_probs))
        if n_epilora_only or n_disco_only:
            print(f"WARNING: {stem}: {n_disco_only} residues only in DiscoTope-3.0, "
                  f"{n_epilora_only} only in epiLoRA -- using the {len(shared)} shared residues")
        if not shared:
            print(f"WARNING: {stem}: no shared residues between epiLoRA and DiscoTope-3.0, skipping")
            continue

        disco_norm = rank_normalize([disco_scores[r] for r in shared])

        out_path = OUT_CACHE / f"{stem}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["res_id", "prob"])
            for res_id, d_norm in zip(shared, disco_norm):
                ensembled = 0.5 * epilora_probs[res_id] + 0.5 * d_norm
                w.writerow([res_id, f"{ensembled:.4f}"])
        n += 1
    print(f"Wrote {n} ensemble epitope cache files to {OUT_CACHE}")


if __name__ == "__main__":
    main()
