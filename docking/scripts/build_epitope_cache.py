#!/usr/bin/env python3
"""
Join scratch/residue_keys.csv (idx -> actual res_id, from fair-esm's own
residue walk) with scratch/epitope_preds/{stem}_A.csv (idx -> epiLoRA
probability) into one small cache/epitope/{stem}.csv (res_id,prob) per
antigen. This is what generate_restraints.py reads at Snakemake time --
computing epiLoRA predictions requires the epilora/env (torch+esm) which the
main pipeline's python (3.6) can't import, so predictions are precomputed
once here rather than shelled out to per docking pair.

Run with plain stdlib python (no special env needed) after
stage_antigens.py -> dump_residue_keys.py -> predict_ensemble.py.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"
CACHE = ROOT / "cache" / "epitope"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    res_ids = defaultdict(dict)
    with open(SCRATCH / "residue_keys.csv") as f:
        for row in csv.DictReader(f):
            res_ids[row["pdb_stem"]][int(row["idx"])] = int(row["res_id"])

    n = 0
    for stem, idx_to_resid in sorted(res_ids.items()):
        pred_path = SCRATCH / "epitope_preds" / f"{stem}_A.csv"
        with open(pred_path) as f:
            probs = [float(row["prob"]) for row in csv.DictReader(f)]
        if len(probs) != len(idx_to_resid):
            raise SystemExit(f"{stem}: {len(idx_to_resid)} residue keys but "
                              f"{len(probs)} predictions -- refusing to guess alignment")
        out_path = CACHE / f"{stem}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["res_id", "prob"])
            for i in range(len(probs)):
                w.writerow([idx_to_resid[i], f"{probs[i]:.4f}"])
        n += 1
    print(f"Wrote {n} epitope cache files to {CACHE}")


if __name__ == "__main__":
    main()
