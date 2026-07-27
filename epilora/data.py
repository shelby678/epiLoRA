"""Data loading for epiLoRA training.

Training data is a FASTA of antigen sequences (as produced by
``data/data_prep.smk``) plus their mmCIF structures.

FASTA format (one record per antigen, one or more chains)::

    >pdb_000010bt-A-E 2026/01/10 1.99 I|J homo_sapiens homo_sapiens n=3 4.0
    ...ndklKRELtnkgqvADIYWL...

  * header fields (space-separated): ``instance``, ``date``, ``resolution``,
    ``antigen_chain`` (``|``-joined if the antigen spans multiple chains),
    ``heavy_species``, ``light_species``, ``n=<qualifying members>``,
    ``fold_label``.
  * ``instance`` starts with the PDB id (``pdb_XXXXXXXX``); the structure
    lives at ``<structures_dir>/<pdb_id>/<pdb_id>_sabdab.cif``.
  * sequence casing encodes the label: lowercase = epitope residue,
    UPPERCASE = non-epitope.
  * ``fold_label`` is ``{i}.{j}`` from the 5-fold CV scheme: ``i`` (1-5) is
    the cross-validation fold, ``j`` is that fold's role (``0`` = val/eval,
    ``1`` = test). See ``data/README.md``.

Only sequences whose structure is found (and whose residue count matches the
sequence, chains concatenated in the header's order) are usable — ESM-IF1
needs backbone coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# A training example: (header, sequence, per-residue labels, backbone coords|None)
Sample = tuple


def parse_fasta(path: Path) -> dict[str, list]:
    """Parse the labelled FASTA into {fold_label: [(header, seq, labels), ...]}."""
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
    """Return (pdb_id, [chain, ...]) from a header, or None if it can't be parsed."""
    fields = header.split()
    if len(fields) < 4:
        return None
    instance = fields[0]
    if not instance.startswith("pdb_"):
        return None
    pdb_id = instance[:len("pdb_") + 8]  # "pdb_" + 8-char PDB code
    chains = fields[3].split("|")
    return pdb_id, chains


def load_backbone_coords(cif_path: Path, chain_ids: list[str], seq_len: int):
    """Load (seq_len, 3, 3) N/CA/C coords for ``chain_ids`` (concatenated in
    order); None on mismatch."""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    try:
        model = next(iter(MMCIFParser(QUIET=True).get_structure("x", str(cif_path))))
    except Exception:
        return None

    residues = []
    for chain_id in chain_ids:
        if chain_id not in model:
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
        parsed = parse_seq_id(header)
        coords = None
        if parsed:
            pdb_id, chains = parsed
            cp = structures_dir / pdb_id / f"{pdb_id}_sabdab.cif"
            if cp.exists():
                coords = load_backbone_coords(cp, chains, len(seq))
        out.append((header, seq, labels, coords))
    return out
