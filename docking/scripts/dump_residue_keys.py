#!/usr/bin/env python3
"""
Print (pdb_stem, idx, chain, res_id, ins_code) for every residue of chain A in
each given PDB, in the same file-order walk that predict_ensemble.py's `pos`
index follows (fair-esm's load_structure + get_residue_starts). Never assume
`pos == res_id` -- gaps/offsets from prepare_pdbs.py's merge renumbering mean
they can diverge.

Must run in the fair-esm environment (epilora/env/bin/python).
"""
import biotite.structure as _bs

if not hasattr(_bs, "filter_backbone"):
    _bs.filter_backbone = _bs.filter_peptide_backbone

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "epilora"))
from color_by_prediction import residue_keys  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", type=Path, nargs="+", required=True)
    p.add_argument("--chain", default="A")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_stem", "idx", "chain", "res_id", "ins_code"])
        for pdb in args.pdb:
            keys = residue_keys(pdb, args.chain)
            for idx, (ch, res_id, icode, _res_name) in enumerate(keys):
                w.writerow([pdb.stem, idx, ch, res_id, icode])
            print(f"[dump] {pdb.name}: {len(keys)} residues", file=sys.stderr)


if __name__ == "__main__":
    main()
