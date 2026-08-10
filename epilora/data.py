"""Data loading for epiLoRA training.

Training data is a FASTA of antigen sequences (as produced by
``data/data_prep.smk``) plus their mmCIF structures.

Every entry's structure file must be present under ``structures_dir``; only
whether its residue count matches the sequence (chains concatenated in the
header's order) is tolerated as a per-entry skip — ESM-IF1 needs backbone
coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# A training example: (header, sequence, per-residue labels, backbone coords|None)
Sample = tuple


def parse_fasta(path: Path) -> dict[str, list]:
    """Parses labelled fasta sequences and labels by fold (e.g. 1.0, 1.1, 2.0, etc.)
    Returns a dict of format {fold_label: [(header, seq, labels), ...]}
    """
    by_part: dict[str, list] = {}

    def add(header: str, seq: str) -> None:
        fold_label = header.split()[-1]
        labels = [1 if c.islower() else 0 for c in seq]
        by_part.setdefault(fold_label, []).append((header, seq.upper(), labels))

    header, seq = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                add(header, "".join(seq))
            header, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if header is not None:
        add(header, "".join(seq))
    return by_part


def parse_seq_id(header: str):
    """Return (pdb_id, [chain, ...]) from a header."""
    fields = header.split()
    if len(fields) < 4:
        raise ValueError(f"malformed header (expected >=4 fields): {header!r}")
    instance = fields[0]
    if not instance.startswith("pdb_"):
        raise ValueError(f"malformed header (instance must start with 'pdb_'): {header!r}")
    pdb_id = instance[:len("pdb_") + 8]  # "pdb_" + 8-char PDB code
    chains = fields[3].split("|")
    return pdb_id, chains


def load_backbone_coords(cif_path: Path, chain_ids: list[str], seq_len: int):
    """Load (seq_len, 3, 3) N/CA/C coords for ``chain_ids`` (concatenated in
    order); None if the CIF can't be parsed, a chain is missing, or the
    residue count doesn't match ``seq_len`` -- every failure mode is treated
    the same way (skip + log a warning) so one bad structure file can't crash
    an entire training run the way the others are silently skipped."""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    try:
        model = next(iter(MMCIFParser(QUIET=True).get_structure("x", str(cif_path)))) # get the model
    except Exception as e:
        logger.warning(f"could not parse structure at {cif_path}: {e}")
        return None

    residues = []
    for chain_id in chain_ids:
        if chain_id not in model:
            logger.warning(f"chain {chain_id!r} not in model at {cif_path}")
            return None
        residues.extend(res for res in model[chain_id] if is_aa(res, standard=True))
    if len(residues) != seq_len:
        return None

    coords = np.full((seq_len, 3, 3), np.nan, dtype=np.float32)
    for ri, res in enumerate(residues):
        for ai, an in enumerate(["N", "CA", "C"]):
            if an in res:
                coords[ri, ai] = res[an].coord
    return coords


def load_samples(entries: list, structures_dir: Path) -> list:
    """Attach backbone coords to (header, seq, labels) entries."""
    structures_dir = Path(structures_dir)
    out = []
    for header, seq, labels in entries:
        pdb_id, chains = parse_seq_id(header)
        cp = structures_dir / pdb_id / f"{pdb_id}_sabdab.cif"
        if not cp.exists():
            raise FileNotFoundError(f"no structure file for {header!r}: {cp}")
        coords = load_backbone_coords(cp, chains, len(seq))
        out.append((header, seq, labels, coords))
    return out
