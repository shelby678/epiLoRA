"""Shared helpers for the generic single-PDB epitope pipeline (Snakefile in
this directory). Every script below imports from here instead of
reimplementing structure loading, so all pipeline stages agree on exactly
the same residue ordering for a given (pdb, chain) -- this is what lets a
prediction CSV, a homology-transfer CSV and the final B-factor rewrite all
line up index-for-index without re-deriving the mapping three times.

Structure loading is Bio.PDB-based (PDBParser/MMCIFParser dispatched on
suffix) -- the same extraction epilora/data.py's load_backbone_coords and
data/scripts/structures.py use elsewhere in this repo, and handles .pdb and
.cif uniformly. predict.py's model-loading/inference functions
(load_model/predict) don't care how coords were produced, so this only
touches the extraction side.

Must run in the fair-esm environment (epilora/env/bin/python3): needs
biopython, and, transitively via predict.py, torch/esm if the caller goes
on to run the model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa

_pdb_parser = PDBParser(QUIET=True)
_cif_parser = MMCIFParser(QUIET=True)


def load_structure_model(path: str):
    """First model of a .pdb or .cif file, dispatched on suffix."""
    path = str(path)
    parser = _cif_parser if path.lower().endswith(".cif") else _pdb_parser
    return next(iter(parser.get_structure("x", path)))


def chain_residues(model, chain_id: str) -> list:
    """Standard amino-acid residues (in file order) for one chain."""
    if chain_id not in model:
        raise ValueError(f"chain {chain_id!r} not found (chains present: "
                          f"{[c.id for c in model]})")
    return [res for res in model[chain_id] if is_aa(res, standard=True)]


def default_chain(pdb_path: str) -> str:
    """First chain (in file order) with at least one standard amino-acid
    residue -- same default predict.py uses when --chain is omitted."""
    model = load_structure_model(pdb_path)
    for chain in model:
        if any(is_aa(res, standard=True) for res in chain):
            return chain.id
    raise ValueError(f"{pdb_path}: no chain with standard amino-acid residues")


def parse_chains(chain: str) -> list[str]:
    """A '|'-separated chain spec (e.g. 'A|D', matching SAbDab's own
    antigen_chain column convention -- see mark_bfactors.py) into an ordered
    list of chain ids. A multi-chain antigen (e.g. GP1+GP2 crystallized as
    separate chains) is treated as one concatenated sequence, in the order
    given, same as mark_bfactors.py/epilora/data.py's load_backbone_coords."""
    return chain.split("|")


def _multi_chain_residues(model, chains: list[str]) -> list:
    residues = []
    for chain_id in chains:
        residues.extend(chain_residues(model, chain_id))
    return residues


def load_query(pdb_path: str, chains: list[str]):
    """(coords, seq): (L, 3, 3) N/CA/C backbone coords (NaN where an atom is
    missing -- model.py tolerates this, same as training data) and the
    1-letter sequence, concatenated across ``chains`` in order."""
    residues = _multi_chain_residues(load_structure_model(pdb_path), chains)
    seq = "".join(protein_letters_3to1.get(res.resname, "X") for res in residues)
    coords = np.full((len(residues), 3, 3), np.nan, dtype=np.float32)
    for ri, res in enumerate(residues):
        for ai, atom_name in enumerate(("N", "CA", "C")):
            if atom_name in res:
                coords[ri, ai] = res[atom_name].coord
    return coords, seq


def residue_keys(pdb_path: str, chains: list[str]) -> list[tuple[str, int, str, str]]:
    """(chain_id, res_id, ins_code, res_name) per residue, in the same file
    order load_query() walks (each residue keeping its own chain id, so a
    multi-chain antigen's keys still map back onto the right chain)."""
    model = load_structure_model(pdb_path)
    keys = []
    for chain_id in chains:
        keys.extend((chain_id, res.id[1], res.id[2].strip(), res.resname)
                    for res in chain_residues(model, chain_id))
    return keys


def read_value_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_value_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_training_fasta(path: str) -> list[tuple[str, str]]:
    """Yield (header, sequence) for a data_prep.smk-style epitope fasta:
    one record per cluster, header's last whitespace-separated field is the
    fold label (e.g. '3.1'), sequence casing carries the epitope call
    (lowercase = epitope) -- see data/README.md."""
    header, seq = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq)
            header, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if header is not None:
        yield header, "".join(seq)


def fold_group_of(header: str) -> int:
    """The `i` in a trailing `i.j` fold label -- see data/README.md's Fold
    label scheme. fold `i`'s checkpoint was trained on every record whose
    label's `i` differs from this one (train.py: 'fold != --fold')."""
    return int(header.rsplit(" ", 1)[-1].split(".")[0])
