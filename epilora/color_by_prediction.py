"""Write epiLoRA per-residue epitope probabilities into a PDB's B-factor column,
so a structure can be coloured continuously by predicted epitope score.

    python color_by_prediction.py --pdb ../data/raw/5IBV.pdb \
        --pred-dir ../data/results/champion_predictions --out-dir ../data/results/champion_pdb

Reads the ``<stem>_<chain>.csv`` files written by predict_ensemble.py and maps
each prediction back onto its residue. The mapping is index-based and only
valid because it replays fair-esm's own residue ordering: load_structure()
keeps backbone atoms of the requested chain, and extract_coords_from_structure()
walks residues in file order -- so prediction i is residue i of that same walk.
The residue identity (3->1 letter) is checked against the CSV as a guard.

B-factors are ``prob * --scale`` (default 100), a fixed linear map rather than
a per-chain min-max stretch, so scores stay comparable across chains and files.
Residues with no prediction (ligands, waters, other chains) get 0.

Rewriting is line-level: only columns 61-66 of ATOM/HETATM records change, so
everything else in the file is preserved byte-for-byte. Only .pdb input is
supported -- this fixed-column rewrite doesn't hold for mmCIF, even though
residue_keys() itself could load one. In a multi-MODEL file every model
receives the same scores -- predictions come from model 1, which is only
meaningful when the models share a residue composition (checked).

Must run in the fair-esm environment (epilora/env/bin/python).
"""

from __future__ import annotations

# See predict_ensemble.py: fair-esm 2.0 predates the biotite 1.0 rename.
import biotite.structure as _bs

if not hasattr(_bs, "filter_backbone"):
    _bs.filter_backbone = _bs.filter_peptide_backbone

import argparse
import csv
import sys
from pathlib import Path

from biotite.sequence import ProteinSequence
from biotite.structure import get_residue_starts

REPO_ROOT = Path(__file__).resolve().parent.parent


def residue_keys(pdb: Path, chain: str) -> list[tuple[str, int, str, str]]:
    """(chain_id, res_id, ins_code, res_name) per residue, in fair-esm's ordering."""
    from esm.inverse_folding.util import load_structure

    structure = load_structure(str(pdb), chain)
    starts = get_residue_starts(structure)
    return [
        (structure.chain_id[i], int(structure.res_id[i]),
         str(structure.ins_code[i]).strip(), str(structure.res_name[i]))
        for i in starts
    ]


def read_predictions(csv_path: Path) -> list[tuple[str, float]]:
    with open(csv_path, newline="") as f:
        return [(row["aa"], float(row["prob"])) for row in csv.DictReader(f)]


def check_model_consistency(pdb: Path) -> None:
    """Raise if a multi-MODEL PDB's models don't all share the same residue
    composition -- required since B-factors are computed once (from model 1
    via ``residue_keys``) and copied onto every model's matching atoms. A
    file with no MODEL/ENDMDL records (the common single-structure case) is
    trivially fine and returns immediately."""
    model_keys: dict[str, set[tuple[str, int, str]]] = {}
    current = "1"
    keys: set[tuple[str, int, str]] = set()
    with open(pdb) as f:
        for line in f:
            if line.startswith("MODEL"):
                current = line.split()[1]
                keys = set()
            elif line.startswith("ENDMDL"):
                model_keys[current] = keys
            elif line.startswith(("ATOM  ", "HETATM")):
                keys.add((line[21], int(line[22:26]), line[26].strip()))
    if len(model_keys) <= 1:
        return
    ref_model, ref_keys = next(iter(model_keys.items()))
    for model, keys in model_keys.items():
        if keys != ref_keys:
            raise ValueError(
                f"{pdb.name}: model {model} has a different residue composition than "
                f"model {ref_model} -- refusing to copy model {ref_model}'s scores onto every model"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pdb", type=Path, nargs="+", required=True)
    p.add_argument("--pred-dir", type=Path,
                   default=REPO_ROOT / "data/results/champion_predictions",
                   help="directory of <stem>_<chain>.csv files from predict_ensemble.py")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "data/results/champion_pdb")
    p.add_argument("--scale", type=float, default=100.0,
                   help="probability multiplier for the B-factor (default 100)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pdb in args.pdb:
        if pdb.suffix.lower() != ".pdb":
            p.error(f"{pdb.name}: only .pdb input is supported -- the B-factor rewrite "
                    f"below assumes fixed PDB column positions, which don't hold for "
                    f"mmCIF; convert to PDB first if you need this on a .cif structure")
        check_model_consistency(pdb)

        csvs = sorted(args.pred_dir.glob(f"{pdb.stem}_*.csv"))
        if not csvs:
            p.error(f"no prediction CSVs for {pdb.stem} in {args.pred_dir}")

        # (chain, res_id, ins_code) -> B-factor
        bfac: dict[tuple[str, int, str], float] = {}
        for csv_path in csvs:
            chain = csv_path.stem[len(pdb.stem) + 1:]
            preds = read_predictions(csv_path)
            keys = residue_keys(pdb, chain)
            if len(keys) != len(preds):
                p.error(f"{pdb.name} chain {chain}: {len(keys)} residues in structure "
                        f"but {len(preds)} predictions -- refusing to guess an alignment")
            for (ch, res_id, icode, res_name), (aa, prob) in zip(keys, preds):
                # If res_name were unconvertible, predict_ensemble.py would already
                # have crashed writing this same CSV -- no need to guess here.
                expect = ProteinSequence.convert_letter_3to1(res_name)
                if expect != aa:
                    p.error(f"{pdb.name} chain {chain} res {res_id}{icode}: "
                            f"structure has {res_name} ({expect}) but CSV has {aa}")
                bfac[(ch, res_id, icode)] = prob * args.scale
            print(f"[color] {pdb.name} chain {chain}: mapped {len(preds)} residues",
                  file=sys.stderr)

        out = args.out_dir / f"{pdb.stem}_epitope_bfactor.pdb"
        n_set = n_zero = 0
        with open(pdb) as fin, open(out, "w") as fout:
            for line in fin:
                if line.startswith(("ATOM  ", "HETATM")):
                    key = (line[21], int(line[22:26]), line[26].strip())
                    if key in bfac:
                        n_set += 1
                        b = bfac[key]
                    else:
                        n_zero += 1
                        b = 0.0
                    b_str = f"{b:6.2f}"
                    if len(b_str) > 6:
                        p.error(f"{pdb.name}: B-factor {b:.4f} at chain {key[0]} res "
                                f"{key[1]}{key[2]} doesn't fit the fixed 6-character PDB "
                                f"field -- use a smaller --scale")
                    line = f"{line[:60]}{b_str}{line[66:]}"
                fout.write(line)
        print(f"[color] wrote {out}  ({n_set} atoms scored, {n_zero} zeroed)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
