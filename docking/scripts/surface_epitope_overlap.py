#!/usr/bin/env python3
"""
Combine per-antigen surface residues (haddock3-restraints calc_accessibility)
with epiLoRA champion-ensemble epitope predictions to answer: what fraction of
an antigen's surface residues clear a given epitope-probability threshold?

Inputs (already produced by stage_antigens.py / dump_residue_keys.py /
predict_ensemble.py):
  scratch/antigens/{pdb_stem}.pdb
  scratch/residue_keys.csv     -- pdb_stem,idx,chain,res_id,ins_code
  scratch/epitope_preds/{pdb_stem}_A.csv  -- pos,aa,prob,prob_std,epitope

Surface residues are computed directly here (rather than read from a cached
file) since it's just one `haddock3-restraints calc_accessibility` subprocess
call per antigen and no other script needs the result.
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"


def calc_surface_residues(haddock3_restraints, pdb_path, chain_id="A"):
    """Mirrors haddocking2/scripts/generate_restraints.py:calc_surface_residues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_pdb = os.path.join(tmpdir, "chain.pdb")
        with open(chain_pdb, "w") as out, open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")) and line[21] == chain_id:
                    out.write(line)
            out.write("END\n")
        subprocess.run(
            [haddock3_restraints, "calc_accessibility",
             "--export_to_actpass", chain_pdb],
            check=True, capture_output=True, cwd=tmpdir, timeout=120,
        )
        actpass = os.path.join(tmpdir, f"chain_passive_{chain_id}.actpass")
        if os.path.exists(actpass):
            with open(actpass) as f:
                return {int(x) for x in f.read().split() if x.strip().isdigit()}
    return set()


def load_residue_keys():
    """Return {pdb_stem: [res_id, ...]} indexed by idx (0-based, ascending)."""
    keys = defaultdict(dict)
    with open(SCRATCH / "residue_keys.csv") as f:
        for row in csv.DictReader(f):
            keys[row["pdb_stem"]][int(row["idx"])] = int(row["res_id"])
    return {stem: [d[i] for i in sorted(d)] for stem, d in keys.items()}


def load_probs(pdb_stem):
    """Return list of probs indexed by idx (0-based), aligned to residue_keys."""
    path = SCRATCH / "epitope_preds" / f"{pdb_stem}_A.csv"
    probs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            probs.append(float(row["prob"]))
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--haddock3-restraints",
                   default=str(ROOT / "env" / "bin" / "haddock3-restraints"))
    p.add_argument("--out-csv", default=str(SCRATCH / "surface_epitope_overlap.csv"))
    args = p.parse_args()

    residue_keys = load_residue_keys()
    rows = []
    for pdb_stem in sorted(residue_keys):
        res_ids = residue_keys[pdb_stem]
        probs = load_probs(pdb_stem)
        if len(probs) != len(res_ids):
            sys.exit(f"{pdb_stem}: {len(res_ids)} residue keys but {len(probs)} "
                      f"predictions -- refusing to guess an alignment")
        pdb_path = SCRATCH / "antigens" / f"{pdb_stem}.pdb"
        try:
            surface = calc_surface_residues(args.haddock3_restraints, pdb_path)
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL {pdb_stem}] {e.stderr.decode(errors='replace')[:300]}",
                  file=sys.stderr)
            continue
        print(f"  {pdb_stem}: {len(surface)} surface residues", file=sys.stderr)
        epitope_res_ids = {r for r, pr in zip(res_ids, probs) if pr > args.threshold}

        overlap = surface & epitope_res_ids
        pct = 100.0 * len(overlap) / len(surface) if surface else float("nan")
        rows.append({
            "pdb_stem": pdb_stem,
            "n_residues": len(res_ids),
            "n_surface": len(surface),
            "n_epitope_gt_thresh": len(epitope_res_ids),
            "n_surface_and_epitope": len(overlap),
            "pct_surface_is_epitope": round(pct, 2),
        })

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pcts = [r["pct_surface_is_epitope"] for r in rows]
    print(f"threshold = prob > {args.threshold}\n")
    print(f"{'pdb_stem':16s} {'n_res':>6s} {'n_surf':>7s} {'n_epi>thr':>10s} {'n_surf&epi':>11s} {'%surf=epi':>10s}")
    for r in rows:
        print(f"{r['pdb_stem']:16s} {r['n_residues']:6d} {r['n_surface']:7d} "
              f"{r['n_epitope_gt_thresh']:10d} {r['n_surface_and_epitope']:11d} "
              f"{r['pct_surface_is_epitope']:10.2f}")

    print(f"\nn antigens: {len(pcts)}")
    print(f"mean:   {statistics.mean(pcts):.2f}%")
    print(f"median: {statistics.median(pcts):.2f}%")
    print(f"stdev:  {statistics.pstdev(pcts):.2f}%")
    print(f"min:    {min(pcts):.2f}%")
    print(f"max:    {max(pcts):.2f}%")
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
