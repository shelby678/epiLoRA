"""Data loading for epiLoRA training.

Training data is a FASTA of antigen sequences (as produced by
``data/data_prep.smk``) plus their mmCIF structures.

Every entry's structure file must be present under ``structures_dir``; only
whether its residue count matches the sequence (chains concatenated in the
header's order) is tolerated as a per-entry skip — ESM-IF1 needs backbone
coordinates.

Extracted coordinates are cached in a directory next to ``structures_dir``
(see load_backbone_coords), since the same structure otherwise gets
re-parsed by every ablation dataset/fold/backbone that includes it. Caching
next to each structure's own .cif file isn't an option: the extracted
structures are read-only on disk (root-squashed NFS), so the cache needs
its own writable directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel shape written to the coords cache to mean "load_backbone_coords
# returned None" (parse failure / missing chain / length mismatch) -- distinct
# from any real (seq_len, 3, 3) result since seq_len is always >= 1.
_NO_COORDS_SENTINEL_SHAPE = (0, 3, 3)

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


def _coords_cache_path(cache_dir: Path, cif_path: Path, chain_ids: list[str], seq_len: int) -> Path:
    """Cache path for load_backbone_coords' (deterministic) output -- a flat
    file per (structure, chains, length) under ``cache_dir``."""
    key = f"{'-'.join(chain_ids)}_{seq_len}"
    return cache_dir / f"{cif_path.stem}.{key}.coords.npy"


def _write_cache(cache_path: Path, arr: np.ndarray) -> None:
    """Write via temp-file + atomic rename, so concurrent training jobs
    sharing a structure (e.g. two ablation-sweep folds racing on the same
    pdb_id) never observe a partially-written cache file."""
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp{os.getpid()}")
    with open(tmp_path, "wb") as f:
        np.save(f, arr)  # file-object target -> numpy won't append another .npy
    os.replace(tmp_path, cache_path)


def load_backbone_coords(cif_path: Path, chain_ids: list[str], seq_len: int, cache_dir: Path):
    """Load (seq_len, 3, 3) N/CA/C coords for ``chain_ids`` (concatenated in
    order); None if the CIF can't be parsed, a chain is missing, or the
    residue count doesn't match ``seq_len`` -- every failure mode is treated
    the same way (skip + log a warning) so one bad structure file can't crash
    an entire training run the way the others are silently skipped.

    Cached to a .npy file under ``cache_dir`` (see _coords_cache_path): this
    is a pure function of its arguments, the Bio.PDB parse is slow (~350ms/
    structure), and the same structure gets reloaded by every ablation
    dataset/fold/backbone that happens to include it. Delete ``cache_dir``
    (or a structure's ``*.coords.npy`` file within it) to force a re-parse."""
    cache_path = _coords_cache_path(cache_dir, cif_path, chain_ids, seq_len)
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
        except Exception as e:
            logger.warning(f"could not read cache {cache_path}: {e}; re-parsing")
        else:
            return None if cached.shape == _NO_COORDS_SENTINEL_SHAPE else cached

    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    try:
        model = next(iter(MMCIFParser(QUIET=True).get_structure("x", str(cif_path)))) # get the model
    except Exception as e:
        logger.warning(f"could not parse structure at {cif_path}: {e}")
        _write_cache(cache_path, np.zeros(_NO_COORDS_SENTINEL_SHAPE, dtype=np.float32))
        return None

    residues = []
    for chain_id in chain_ids:
        if chain_id not in model:
            logger.warning(f"chain {chain_id!r} not in model at {cif_path}")
            _write_cache(cache_path, np.zeros(_NO_COORDS_SENTINEL_SHAPE, dtype=np.float32))
            return None
        residues.extend(res for res in model[chain_id] if is_aa(res, standard=True))
    if len(residues) != seq_len:
        _write_cache(cache_path, np.zeros(_NO_COORDS_SENTINEL_SHAPE, dtype=np.float32))
        return None

    coords = np.full((seq_len, 3, 3), np.nan, dtype=np.float32)
    for ri, res in enumerate(residues):
        for ai, an in enumerate(["N", "CA", "C"]):
            if an in res:
                coords[ri, ai] = res[an].coord
    _write_cache(cache_path, coords)
    return coords


def load_samples(entries: list, structures_dir: Path, cache_dir: Path | None = None) -> list:
    """Attach backbone coords to (header, seq, labels) entries.

    ``cache_dir`` defaults to a sibling of ``structures_dir`` (e.g.
    ``all-structures-extracted`` -> ``all-structures-extracted_coords_cache``),
    since ``structures_dir`` itself is read-only."""
    structures_dir = Path(structures_dir)
    if cache_dir is None:
        cache_dir = structures_dir.parent / f"{structures_dir.name}_coords_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for header, seq, labels in entries:
        pdb_id, chains = parse_seq_id(header)
        cp = structures_dir / pdb_id / f"{pdb_id}_sabdab.cif"
        if not cp.exists():
            raise FileNotFoundError(f"no structure file for {header!r}: {cp}")
        coords = load_backbone_coords(cp, chains, len(seq), cache_dir)
        out.append((header, seq, labels, coords))
    return out
