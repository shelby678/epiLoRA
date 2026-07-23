"""Data loading for epiLoRA training.

Training data is a FASTA of antigen sequences plus their PDB structures.

FASTA format (one record per antigen chain)::

    >4qci_A 4qci A 1
    ...ndklKRELtnkgqvADIYWL...

  * header fields (space-separated): ``<id>_<chain>``, ``<pdb_id>``,
    ``<chain>``, ``<partition>``  (extra fields are ignored)
  * sequence casing encodes the label: lowercase = epitope residue,
    UPPERCASE = non-epitope. The partition ``EVAL`` is always held out.

Structures live at ``<structures_dir>/<pdb_id>/structure/<pdb_id>.pdb`` and only
sequences whose structure is found (and whose residue count matches the
sequence) are usable — ESM-IF1 needs backbone coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# A training example: (header, sequence, per-residue labels, backbone coords|None)
Sample = tuple


def parse_fasta(path: Path) -> dict[str, list]:
    """Parse the labelled FASTA into {partition: [(header, seq, labels), ...]}."""
    by_part: dict[str, list] = {}

    def add(header: str, seq: str) -> None:
        fields = header.split()
        part = fields[2] if len(fields) >= 3 else "?"
        if part == "EVAL":
            return
        labels = [1 if c.islower() else 0 for c in seq]
        by_part.setdefault(part, []).append((header, seq.upper(), labels))

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
    """Return (pdb_id, chain) from a header, or None if it can't be parsed."""
    fields = header.split()
    if len(fields) >= 2:
        return fields[0].split("_")[0].lower(), fields[1]
    return None


def load_backbone_coords(pdb_path: Path, chain_id: str, seq_len: int):
    """Load (seq_len, 3, 3) N/CA/C coords for ``chain_id``; None on mismatch."""
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
    try:
        f = pdb_io.PDBFile.read(str(pdb_path))
        st = pdb_io.get_structure(f, model=1)
    except Exception:
        return None
    st = st[(st.chain_id == chain_id) & struc.filter_amino_acids(st)]
    res_ids = struc.get_residues(st)[0]
    if len(res_ids) != seq_len:
        return None
    coords = np.full((seq_len, 3, 3), np.nan, dtype=np.float32)
    for ri, rid in enumerate(res_ids):
        ra = st[st.res_id == rid]
        for ai, an in enumerate(["N", "CA", "C"]):
            m = ra.atom_name == an
            if m.any():
                coords[ri, ai] = ra.coord[m][0]
    return coords


def load_samples(entries: list, structures_dir: Path) -> list:
    """Attach backbone coords to (header, seq, labels) entries."""
    structures_dir = Path(structures_dir)
    out = []
    for header, seq, labels in entries:
        parsed = parse_seq_id(header)
        coords = None
        if parsed:
            pdb_id, chain = parsed
            pp = structures_dir / pdb_id / "structure" / f"{pdb_id}.pdb"
            if pp.exists():
                coords = load_backbone_coords(pp, chain, len(seq))
        out.append((header, seq, labels, coords))
    return out
