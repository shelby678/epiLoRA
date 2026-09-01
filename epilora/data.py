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

Optionally (``load_samples(..., extra_feats=(...))``) each sample also carries
extra per-residue head features -- RSA, antigen length, an amino-acid one-hot;
see build_extra_feats below. RSA is cached like the coordinates are.
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

# Same idea for the RSA cache: a (0,)-shaped file means "returned None".
_NO_RSA_SENTINEL_SHAPE = (0,)

# A training example: (header, sequence, per-residue labels, backbone
# coords|None, extra feats|None) -- feats are None unless asked for, or when
# they couldn't be computed (skipped like missing coords).
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


def parse_structure_model(path: Path):
    """First model of an mmCIF/PDB file, or None if it can't be parsed (logged)."""
    from Bio.PDB import MMCIFParser, PDBParser
    parser = (PDBParser(QUIET=True) if Path(path).suffix.lower() in (".pdb", ".ent")
              else MMCIFParser(QUIET=True))
    try:
        return next(iter(parser.get_structure("x", str(path))))
    except Exception as e:
        logger.warning(f"could not parse structure at {path}: {e}")
        return None


def select_residues(model, chain_ids: list[str], path: Path):
    """Standard-amino-acid residues of ``chain_ids``, in chain order -- the
    single source of truth for which residues the FASTA sequence maps to, so
    coords and RSA line up with the labels. None if a chain is absent."""
    from Bio.PDB.Polypeptide import is_aa
    residues = []
    for chain_id in chain_ids:
        if chain_id not in model:
            logger.warning(f"chain {chain_id!r} not in model at {path}")
            return None
        residues.extend(res for res in model[chain_id] if is_aa(res, standard=True))
    return residues


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

    model = parse_structure_model(cif_path)
    if model is None:
        _write_cache(cache_path, np.zeros(_NO_COORDS_SENTINEL_SHAPE, dtype=np.float32))
        return None
    residues = select_residues(model, chain_ids, cif_path)
    if residues is None or len(residues) != seq_len:
        _write_cache(cache_path, np.zeros(_NO_COORDS_SENTINEL_SHAPE, dtype=np.float32))
        return None

    coords = np.full((seq_len, 3, 3), np.nan, dtype=np.float32)
    for ri, res in enumerate(residues):
        for ai, an in enumerate(["N", "CA", "C"]):
            if an in res:
                coords[ri, ai] = res[an].coord
    _write_cache(cache_path, coords)
    return coords


# ==== extra per-residue features for the prediction head =====================
#
# Scalar features concatenated onto the backbone embedding right before the
# per-residue head (see model.EpitopeModel.forward):
#
#   rsa     freesasa's side-chain-inclusive ``relativeTotal``, computed on the
#           antigen chain(s) alone (antibody excluded) -- the same convention
#           as pct_non_epitope_surface_hist.py / surface_epitope_overlap.py.
#   length  log10-scaled antigen length, identical for every residue of an
#           entry (a per-antigen scale term).
#   aa      20-way amino-acid one-hot -- the only lane carrying information
#           the ESM-IF1 encoder cannot see (it reads coords only), so without
#           it the model is strictly sequence-blind.
#
# rsa/length are standardised to ~zero mean / unit variance with the fixed
# constants below (measured once over the fold-1 training split of the
# default ablation, not learned), so a checkpoint means the same thing on any
# antigen. The unit variance matters: on raw-scale features the head's feature
# weights barely move next to the LayerNorm'd embedding path, and the model
# trains as if it had no features at all.

# name -> how many head inputs it contributes ("aa" is a block of 20, the rest
# single scalars); extra_feats_width() is the only thing that computes it.
EXTRA_FEATURE_WIDTHS = {"rsa": 1, "length": 1, "aa": 20}
EXTRA_FEATURE_NAMES = tuple(EXTRA_FEATURE_WIDTHS)

RSA_MEAN, RSA_STD = 0.327, 0.278
LENGTH_LOG10_MEAN, LENGTH_LOG10_STD = 2.611, 0.305

# One-hot alphabet. Fixed order (a checkpoint's head weights are indexed by
# it); residues outside these 20 get an all-zero row. Left as raw 0/1: an
# active one-hot is already on par with the LayerNorm'd embedding dims it
# sits beside, so standardising would only hurt.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def extra_feats_width(names) -> int:
    """Total number of head inputs contributed by ``names``."""
    unknown = [n for n in names if n not in EXTRA_FEATURE_WIDTHS]
    if unknown:
        raise ValueError(f"unknown extra feature(s) {unknown} "
                         f"(known: {', '.join(EXTRA_FEATURE_NAMES)})")
    return sum(EXTRA_FEATURE_WIDTHS[n] for n in names)


def aa_one_hot(seq: str) -> np.ndarray:
    """(len(seq), 20) one-hot over AA_ALPHABET; all-zero rows for non-standard residues."""
    out = np.zeros((len(seq), len(AA_ALPHABET)), dtype=np.float32)
    for i, aa in enumerate(seq.upper()):
        j = AA_INDEX.get(aa)
        if j is not None:
            out[i, j] = 1.0
    return out

# relativeTotal is ~[0, 1] but can exceed 1 for unusually exposed residues;
# clipped only to bound outliers.
RSA_CLIP_MAX = 2.0

# Fill for residues freesasa reports no relative area for -- treated as
# buried rather than dropping the antigen.
RSA_MISSING_FILL = 0.0


def build_extra_feats(names, seq: str, rsa=None) -> np.ndarray:
    """Assemble the (len(seq), extra_feats_width(names)) matrix the head
    reads, concatenating the named blocks in order (the order stored in the
    checkpoint's config)."""
    seq_len = len(seq)
    blocks = []
    for name in names:
        if name == "rsa":
            if rsa is None:
                raise ValueError("build_extra_feats: 'rsa' requested but no RSA array given")
            if len(rsa) != seq_len:
                raise ValueError(f"build_extra_feats: RSA has {len(rsa)} values for a "
                                 f"{seq_len}-residue sequence")
            # A NaN here would silently poison the head's input for that residue.
            clipped = np.clip(np.nan_to_num(np.asarray(rsa, dtype=np.float32),
                                            nan=RSA_MISSING_FILL), 0.0, RSA_CLIP_MAX)
            blocks.append(((clipped - RSA_MEAN) / RSA_STD).reshape(seq_len, 1))
        elif name == "length":
            z = (np.log10(seq_len) - LENGTH_LOG10_MEAN) / LENGTH_LOG10_STD
            blocks.append(np.full((seq_len, 1), z, dtype=np.float32))
        elif name == "aa":
            blocks.append(aa_one_hot(seq))
        else:
            raise ValueError(f"unknown extra feature {name!r} "
                             f"(known: {', '.join(EXTRA_FEATURE_NAMES)})")
    return np.concatenate(blocks, axis=1).astype(np.float32)


# Feature blocks that change with the sequence but not the structure -- what
# a substitution scan has to recompute (see swap_sequence_feats).
SEQUENCE_DEPENDENT_FEATS = ("aa",)


def swap_sequence_feats(names, feats: np.ndarray, seq: str) -> np.ndarray:
    """Return ``feats`` with its sequence-dependent blocks rebuilt for ``seq``.

    Callers that mutate sequences must go through this -- reusing the
    parent's feature matrix would score the mutant with the parent residue's
    identity.
    """
    if not any(n in SEQUENCE_DEPENDENT_FEATS for n in names):
        return feats
    out = np.array(feats, dtype=np.float32, copy=True)
    off = 0
    for name in names:
        w = EXTRA_FEATURE_WIDTHS[name]
        if name == "aa":
            out[:, off:off + w] = aa_one_hot(seq)
        off += w
    return out


# freesasa labels chains with a single character, but SAbDab ids can be
# multi-character ("AAA", "K2") -- relabelled from this pool.
_FREESASA_CHAIN_LABELS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          "abcdefghijklmnopqrstuvwxyz0123456789")


def _pdb_atom_name(atom) -> str:
    """PDB-column-formatted atom name (e.g. " CA ", "FE  ") -- freesasa reads
    the element from the name's leading columns and only guesses (noisily) if
    it isn't padded; mmCIF-parsed atoms carry no padding."""
    name = atom.get_name()
    return f"{name:<4}" if len(name) >= 4 or len(atom.element) == 2 else f" {name:<3}"


def residue_rsa(residues, path: Path) -> np.ndarray:
    """Relative solvent accessibility of each residue in ``residues``.

    Runs freesasa on exactly the heavy atoms of ``residues`` (the antigen
    chain(s) alone for a training entry); unscored residues get
    RSA_MISSING_FILL. The structure is assembled atom by atom rather than via
    a temporary PDB file (PDBIO can't write multi-character chain ids); the
    two routes agree to <1e-4 RSA.
    """
    import freesasa

    chain_ids = list(dict.fromkeys(res.get_parent().id for res in residues))
    # Chains with a usable single-char label keep it; the rest take unused ones.
    labels = {c: c for c in chain_ids if len(c) == 1}
    free = (c for c in _FREESASA_CHAIN_LABELS if c not in labels.values())
    for chain_id in chain_ids:
        if chain_id not in labels:
            labels[chain_id] = next(free)

    structure = freesasa.Structure()
    keys = []
    for res in residues:
        label = labels[res.get_parent().id]
        res_number = f"{res.id[1]}{res.id[2].strip()}"  # freesasa's residue key
        for atom in res.get_atoms():
            if atom.element == "H":  # freesasa's own PDB reader skips hydrogens too
                continue
            x, y, z = (float(v) for v in atom.coord)
            structure.addAtom(_pdb_atom_name(atom), res.resname, res_number, label, x, y, z)
        keys.append((label, res_number))
    areas = freesasa.calc(structure).residueAreas()

    rsa = np.full(len(residues), RSA_MISSING_FILL, dtype=np.float32)
    n_missing = 0
    for ri, (label, res_number) in enumerate(keys):
        area = areas.get(label, {}).get(res_number)
        if area is not None and area.hasRelativeAreas:
            rsa[ri] = area.relativeTotal
        else:
            n_missing += 1
    if n_missing:
        logger.warning(f"{path}: no relative SASA for {n_missing}/{len(residues)} residues; "
                       f"filled with {RSA_MISSING_FILL}")
    return rsa


def _rsa_cache_path(cache_dir: Path, cif_path: Path, chain_ids: list[str], seq_len: int) -> Path:
    key = f"{'-'.join(chain_ids)}_{seq_len}"
    return cache_dir / f"{cif_path.stem}.{key}.rsa.npy"


def load_rsa(cif_path: Path, chain_ids: list[str], seq_len: int, cache_dir: Path):
    """Load (seq_len,) per-residue RSA for ``chain_ids``; None on the same
    failure modes load_backbone_coords tolerates. Cached like the coords are
    (the parse + SASA calculation costs ~0.3s per structure)."""
    cache_path = _rsa_cache_path(cache_dir, cif_path, chain_ids, seq_len)
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
        except Exception as e:
            logger.warning(f"could not read cache {cache_path}: {e}; re-computing")
        else:
            return None if cached.shape == _NO_RSA_SENTINEL_SHAPE else cached

    model = parse_structure_model(cif_path)
    residues = None if model is None else select_residues(model, chain_ids, cif_path)
    if residues is None or len(residues) != seq_len:
        _write_cache(cache_path, np.zeros(_NO_RSA_SENTINEL_SHAPE, dtype=np.float32))
        return None
    try:
        rsa = residue_rsa(residues, cif_path)
    except Exception as e:
        logger.warning(f"could not compute RSA for {cif_path} chains {chain_ids}: {e}")
        _write_cache(cache_path, np.zeros(_NO_RSA_SENTINEL_SHAPE, dtype=np.float32))
        return None
    _write_cache(cache_path, rsa)
    return rsa


def rsa_for_structure_file(path: Path, chain_ids: list[str], seq_len: int) -> np.ndarray:
    """Uncached per-residue RSA for an arbitrary PDB/mmCIF file (the
    prediction-time counterpart of load_rsa). Raises rather than returning
    None: there is no other sample to fall back to at prediction time."""
    model = parse_structure_model(path)
    if model is None:
        raise ValueError(f"could not parse structure {path}")
    residues = select_residues(model, chain_ids, path)
    if residues is None:
        raise ValueError(f"chain(s) {chain_ids} not found in {path}")
    if len(residues) != seq_len:
        raise ValueError(
            f"{path} chain(s) {'|'.join(chain_ids)}: found {len(residues)} standard amino-acid "
            f"residues but the scored sequence has {seq_len} -- cannot line up per-residue "
            f"features with the sequence")
    return residue_rsa(residues, path)


def load_samples(entries: list, structures_dir: Path, cache_dir: Path | None = None,
                 extra_feats=()) -> list:
    """Attach backbone coords (and optionally extra head features) to
    (header, seq, labels) entries, returning Sample 5-tuples. ``extra_feats``
    names the features the model's head reads (see EXTRA_FEATURE_NAMES); when
    empty, each sample's feature slot is None. ``cache_dir`` defaults to a
    sibling of ``structures_dir``, which itself is read-only."""
    structures_dir = Path(structures_dir)
    if cache_dir is None:
        cache_dir = structures_dir.parent / f"{structures_dir.name}_coords_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extra_feats = tuple(extra_feats)
    extra_feats_width(extra_feats)   # validates the names
    out = []
    for header, seq, labels in entries:
        pdb_id, chains = parse_seq_id(header)
        cp = structures_dir / pdb_id / f"{pdb_id}_sabdab.cif"
        if not cp.exists():
            raise FileNotFoundError(f"no structure file for {header!r}: {cp}")
        coords = load_backbone_coords(cp, chains, len(seq), cache_dir)
        feats = None
        if extra_feats:
            rsa = load_rsa(cp, chains, len(seq), cache_dir) if "rsa" in extra_feats else None
            # rsa is None only where coords is too, so the caller skips the
            # sample either way.
            if "rsa" not in extra_feats or rsa is not None:
                feats = build_extra_feats(extra_feats, seq, rsa)
        out.append((header, seq, labels, coords, feats))
    return out
