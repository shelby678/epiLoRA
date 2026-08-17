#!/usr/bin/env python3
"""
Discover unique merged antigen structures from the already-completed vanilla
HADDOCK pipeline (~/work/ab_MD/haddocking2) and symlink one representative
copy of each into scratch/antigens/{dname}_{variant}.pdb.

Each dname's antigen prep (prepare_pdbs.py -> input_ag.pdb, merged to chain A)
is independent of which antibody it's paired with, so any one ab_*_vs_ag{,_gt}
run directory has a byte-identical input_ag.pdb for that dname/variant. We
reuse those files directly (rather than re-deriving them) so the antigen
structures used here are guaranteed identical to the ones vanilla HADDOCK
docked against.
"""
import glob
import os

HADDOCKING2 = os.path.expanduser("~/work/ab_MD/haddocking2")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "scratch", "antigens")


def discover():
    """Return {(dname, variant): input_ag_pdb_path}."""
    found = {}
    for dname_dir in sorted(glob.glob(f"{HADDOCKING2}/runs/????")):
        dname = os.path.basename(dname_dir)
        for variant, pattern in [("ag", "*_vs_ag"), ("ag_gt", "*_vs_ag_gt")]:
            for pair_dir in sorted(glob.glob(f"{dname_dir}/{pattern}")):
                # "*_vs_ag" also matches "*_vs_ag_gt" -- exclude those explicitly
                if variant == "ag" and pair_dir.endswith("_gt"):
                    continue
                ag_pdb = os.path.join(pair_dir, "input_ag.pdb")
                if os.path.exists(ag_pdb):
                    found[(dname, variant)] = os.path.abspath(ag_pdb)
                    break
    return found


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    antigens = discover()
    print(f"Discovered {len(antigens)} unique (dname, variant) antigens")
    for (dname, variant), src in sorted(antigens.items()):
        link = os.path.join(OUT_DIR, f"{dname}_{variant}.pdb")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(src, link)
    print(f"Staged into {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
