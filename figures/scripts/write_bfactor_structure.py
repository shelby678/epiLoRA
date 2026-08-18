"""Write a per-residue 'value' column (0-100) from a CSV (pos,aa,value --
the schema groundtruth_from_homology.py and predict_epitope.py both emit)
into a structure's B-factor column, for one chain group (or several, e.g.
a complex with multiple separate antigen copies each scored independently --
pass --chain/--values_csv more than once, paired in order, to merge them
all into ONE output file instead of one file per group).

    python write_bfactor_structure.py --pdb query.pdb --chain A \
        --values_csv prediction.csv --out out.pdb --log log
    python write_bfactor_structure.py --pdb query.cif --chain A|D \
        --values_csv prediction.csv --out out.cif --log log
    python write_bfactor_structure.py --pdb complex.cif \
        --chain A|D --values_csv groupAD_prediction.csv \
        --chain B|E|O --values_csv groupBEO_prediction.csv \
        --chain C|F|K --values_csv groupCFK_prediction.csv \
        --out complex_prediction_bfactor.cif --log log

--chain accepts a single chain id, or '|'-separated ids for a multi-chain
antigen (e.g. 'A|D'). Row i of a group's CSV is assigned to residue i of
that group's concatenated chain walk (chains in the order given) --
matching load_query()/residue_keys() elsewhere in this pipeline (residue
identity, from the CSV's 'aa' column, is checked against the structure as a
guard against a silent misalignment). Groups must not share a chain id.

Dispatches on --pdb's suffix, matching the two existing conventions
elsewhere in this repo rather than inventing a third:
  - .pdb: rewrite only B-factor columns (61-66) of the whole file in place
    (color_by_prediction.py's approach) -- every chain kept (including ones
    not covered by any group), atoms outside every --chain group zeroed.
  - .cif: re-serialize via Bio.PDB, restricted to the union of every
    --chain group (mark_bfactors.py's approach) -- needed for multi-
    character chain ids, which fixed-column PDB can't represent.

Must run in the fair-esm environment (epilora/env/bin/python3): the .pdb
path needs esm.inverse_folding.util (via epitope_pipeline_common) and the
.cif path needs Bio.PDB.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epitope_pipeline_common import parse_chains, read_value_csv, residue_keys  # noqa: E402


def _bfac_from_group(pdb_path: Path, chains: list[str], rows: list[dict]) -> dict:
    from Bio.Data.PDBData import protein_letters_3to1

    keys = residue_keys(str(pdb_path), chains)
    if len(keys) != len(rows):
        raise SystemExit(f"{pdb_path.name} chains {chains}: {len(keys)} residues in structure "
                          f"but {len(rows)} rows in --values_csv -- refusing to guess an alignment")
    bfac = {}
    for (ch, res_id, icode, res_name), row in zip(keys, rows):
        expect = protein_letters_3to1.get(res_name, "X")
        if expect != row["aa"]:
            raise SystemExit(f"{pdb_path.name} chain {ch} res {res_id}{icode}: "
                              f"structure has {res_name} ({expect}) but CSV has {row['aa']}")
        bfac[(ch, res_id, icode)] = float(row["value"])
    return bfac


def write_pdb(pdb_path: Path, groups: list[tuple[list[str], list[dict]]], out_path: Path) -> tuple[int, int]:
    bfac = {}
    for chains, rows in groups:
        bfac.update(_bfac_from_group(pdb_path, chains, rows))

    n_set = n_zero = 0
    with open(pdb_path) as fin, open(out_path, "w") as fout:
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
                    raise SystemExit(f"{pdb_path.name}: B-factor {b:.4f} doesn't fit the fixed "
                                      f"6-character PDB field at chain {key[0]} res {key[1]}{key[2]}")
                line = f"{line[:60]}{b_str}{line[66:]}"
            fout.write(line)
    return n_set, n_zero


def write_cif(pdb_path: Path, groups: list[tuple[list[str], list[dict]]], out_path: Path) -> tuple[int, int]:
    from Bio.PDB import MMCIFIO, MMCIFParser, Select
    from Bio.PDB.Polypeptide import is_aa

    structure = MMCIFParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(iter(structure))

    all_chains: list[str] = []
    bfac_by_id: dict[int, float] = {}
    for chains, rows in groups:
        for chain in chains:
            if chain not in model:
                raise SystemExit(f"{pdb_path.name}: chain {chain!r} not found")
            if chain in all_chains:
                raise SystemExit(f"{pdb_path.name}: chain {chain!r} appears in more than one --chain group")
            all_chains.append(chain)
        residues = [res for chain in chains for res in model[chain] if is_aa(res, standard=True)]
        for res, row in zip(residues, rows):
            from Bio.Data.PDBData import protein_letters_3to1
            expect = protein_letters_3to1.get(res.resname, "X")
            if expect != row["aa"]:
                raise SystemExit(f"{pdb_path.name} chain {res.get_parent().id} res {res.id}: "
                                  f"structure has {res.resname} ({expect}) but CSV has {row['aa']}")
            bfac_by_id[id(res)] = float(row["value"])

    n_set = n_zero = 0
    for chain in all_chains:
        for res in model[chain]:
            b = bfac_by_id.get(id(res), 0.0)
            n_set += int(id(res) in bfac_by_id)
            n_zero += int(id(res) not in bfac_by_id)
            for atom in res.get_atoms():
                atom.set_bfactor(b)

    chain_set = set(all_chains)

    class ChainSelect(Select):
        def accept_chain(self, c):
            return c.id in chain_set

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(out_path), select=ChainSelect())
    return n_set, n_zero


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True, type=Path)
    p.add_argument("--chain", required=True, action="append",
                    help="single chain id, or '|'-separated for a multi-chain antigen; "
                         "repeat (paired in order with --values_csv) to merge multiple groups into one output")
    p.add_argument("--values_csv", required=True, action="append", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path)
    args = p.parse_args()

    if len(args.chain) != len(args.values_csv):
        raise SystemExit(f"--chain given {len(args.chain)} times but --values_csv {len(args.values_csv)} times "
                          f"-- they're paired in order and must match")

    groups = [(parse_chains(chain), read_value_csv(csv_path))
              for chain, csv_path in zip(args.chain, args.values_csv)]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    suffix = args.pdb.suffix.lower()
    if suffix == ".pdb":
        n_set, n_zero = write_pdb(args.pdb, groups, args.out)
    elif suffix == ".cif":
        n_set, n_zero = write_cif(args.pdb, groups, args.out)
    else:
        raise SystemExit(f"unsupported structure suffix {suffix!r} (expected .pdb or .cif)")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w") as log:
        for chain, csv_path in zip(args.chain, args.values_csv):
            log.write(f"pdb: {args.pdb}  chain group: {chain}  values_csv: {csv_path}\n")
        log.write(f"wrote {args.out}  ({n_set} residues scored, {n_zero} zeroed)\n")


if __name__ == "__main__":
    main()
