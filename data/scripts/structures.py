"""Shared structure-parsing helpers for the data-prep rules."""
from pathlib import Path

from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

_parser = MMCIFParser(QUIET=True)
_model_cache = {}


def cif_path(structures_dir, pdb_id):
    return Path(structures_dir) / pdb_id / f"{pdb_id}_sabdab.cif"


def load_model(structures_dir, pdb_id):
    if pdb_id not in _model_cache:
        path = cif_path(structures_dir, pdb_id)
        _model_cache.clear()  # rows are grouped by PDB; only ever need the last one
        _model_cache[pdb_id] = next(iter(_parser.get_structure(pdb_id, str(path))))
    return _model_cache[pdb_id]


def chain_residues(model, chain_id):
    """Amino-acid residues (in order) for a chain, skipping non-polymer hetero groups."""
    if chain_id not in model:
        return []
    return [res for res in model[chain_id] if is_aa(res, standard=True)]


def chain_sequence(model, chain_id):
    seq = []
    for res in chain_residues(model, chain_id):
        seq.append(protein_letters_3to1.get(res.resname, "X"))
    return "".join(seq)


def heavy_atoms(res):
    return [a for a in res.get_atoms() if a.element != "H"]
