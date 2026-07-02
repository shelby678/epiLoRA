"""Per-residue feature extraction for epitope prediction.

Provides:
- RSA (relative surface area) computation from PDB files via biotite.
- Biophysical amino-acid property lookup tables (hydrophobicity, charge, volume, polarity).
- BLOSUM62 per-residue embeddings via BioPython.
- A unified `compute_extra_features` helper used by the training loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Vocabulary mapping helpers (mirrors prepare.py without importing it)
# ---------------------------------------------------------------------------

# Token IDs 5-24 are amino acids in alphabetical order: A C D E F G H I K L M N P Q R S T V W Y
_TOKEN_TO_AA: dict[int, str] = {
    5: "A", 6: "C", 7: "D", 8: "E", 9: "F", 10: "G", 11: "H", 12: "I",
    13: "K", 14: "L", 15: "M", 16: "N", 17: "P", 18: "Q", 19: "R",
    20: "S", 21: "T", 22: "V", 23: "W", 24: "Y",
}

# Inverse: AA → token ID
_AA_TO_TOKEN: dict[str, int] = {v: k for k, v in _TOKEN_TO_AA.items()}

_PAD_VOCAB_SIZE = 32  # matches prepare.PAD_VOCAB_SIZE


# ---------------------------------------------------------------------------
# RSA computation (Shrake-Rupley via biotite, Sander-scale normalisation)
# ---------------------------------------------------------------------------

# Sander scale max SASA values (Å²) — same as DiscoTope-3.0
_SANDER_MAX: dict[str, float] = {
    "A": 106.0, "R": 248.0, "N": 157.0, "D": 163.0, "C": 135.0,
    "Q": 198.0, "E": 194.0, "G": 84.0,  "H": 184.0, "I": 169.0,
    "L": 164.0, "K": 205.0, "M": 188.0, "F": 197.0, "P": 136.0,
    "S": 130.0, "T": 142.0, "W": 227.0, "Y": 222.0, "V": 142.0,
    "X": 169.55,
}

_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def compute_rsa(pdb_path: Path, chain_id: str, seq_len: int) -> np.ndarray | None:
    """Compute per-residue RSA for one chain using Shrake-Rupley + Sander normalisation.

    Args:
        pdb_path:  Path to the PDB file (full structure, all chains present for context).
        chain_id:  Chain to extract RSA for.
        seq_len:   Expected number of residues (must match residue count in structure).

    Returns:
        float32 ndarray of shape ``(seq_len,)`` clipped to [0, 1], or ``None``
        if parsing fails or residue count mismatches.
    """
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb_io

        with open(pdb_path) as fh:
            pdbf = pdb_io.PDBFile.read(fh)
        atom_array = pdbf.get_structure(model=1)

        # Keep only standard amino-acid atoms for the target chain
        aa_mask = struc.filter_amino_acids(atom_array) & (atom_array.chain_id == chain_id)
        chain_atoms = atom_array[aa_mask]
        if len(chain_atoms) == 0:
            return None

        # Shrake-Rupley SASA per atom
        atom_sasa = struc.sasa(chain_atoms, vdw_radii="ProtOr")

        # Sum per residue
        res_sasa = struc.apply_residue_wise(chain_atoms, atom_sasa, np.nansum)

        if len(res_sasa) != seq_len:
            return None

        # Residue names for Sander normalisation
        starts = struc.get_residue_starts(chain_atoms)
        res_names = chain_atoms.res_name[starts]
        aa_1letter = [_THREE_TO_ONE.get(rn, "X") for rn in res_names]
        sander_vals = np.array([_SANDER_MAX[aa] for aa in aa_1letter], dtype=np.float32)

        rsa = (res_sasa / sander_vals).astype(np.float32)
        return np.clip(rsa, 0.0, 1.0)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Biophysical amino-acid properties
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity (divided by 4.5 → [-1, 1])
_KD: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
# Net charge at pH 7 (integer scale, divided by 1 → [-1, 1])
_CHARGE: dict[str, float] = {
    "R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
    **{aa: 0.0 for aa in "ANCQGILMFPSTVWY"},
}
# Van der Waals volume (Pontius et al.), normalised by 227.8 (Trp) → [0, 1]
_VOLUME: dict[str, float] = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}
# Zimmerman polarity index, normalised by 52 (Arg) → [0, 1]
_POLARITY: dict[str, float] = {
    "A": 0.0, "R": 52.0, "N": 3.38, "D": 40.7, "C": 1.48,
    "Q": 3.53, "E": 49.9, "G": 0.0, "H": 51.6, "I": 0.13,
    "L": 0.13, "K": 49.5, "M": 1.43, "F": 0.35, "P": 1.58,
    "S": 1.67, "T": 1.66, "V": 0.13, "W": 2.10, "Y": 1.61,
}

# Pre-built lookup tables indexed by token_id (shape [PAD_VOCAB_SIZE, 4])
def _build_bio_table() -> np.ndarray:
    table = np.zeros((_PAD_VOCAB_SIZE, 4), dtype=np.float32)
    kd_scale = 4.5
    vol_scale = 227.8
    pol_scale = 52.0
    for tok_id, aa in _TOKEN_TO_AA.items():
        table[tok_id] = [
            _KD.get(aa, 0.0) / kd_scale,
            _CHARGE.get(aa, 0.0),
            _VOLUME.get(aa, 0.0) / vol_scale,
            _POLARITY.get(aa, 0.0) / pol_scale,
        ]
    return table


_BIO_TABLE = torch.tensor(_build_bio_table(), dtype=torch.float32)  # [32, 4]


def bio_features(token_ids: Tensor) -> Tensor:
    """Per-residue biophysical features from token IDs.

    Args:
        token_ids: (B, L) long tensor.

    Returns:
        Float tensor of shape (B, L, 4): [hydrophobicity, charge, volume, polarity].
        Values are 0 for special tokens (CLS, PAD, EOS, etc.).
    """
    table = _BIO_TABLE.to(token_ids.device)
    return table[token_ids]  # (B, L, 4)


# ---------------------------------------------------------------------------
# BLOSUM62 embeddings
# ---------------------------------------------------------------------------

def _build_blosum_table() -> np.ndarray:
    """Return a [PAD_VOCAB_SIZE, 20] table of BLOSUM62 rows, normalised to [-1, 1]."""
    try:
        from Bio.Align import substitution_matrices
        bl62 = substitution_matrices.load("BLOSUM62")
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        # Extract 20×20 submatrix
        mat = np.array([[bl62[a][b] for b in aa_order] for a in aa_order], dtype=np.float32)
        # Normalise to [-1, 1] by dividing by max absolute value
        mat = mat / np.abs(mat).max()

        table = np.zeros((_PAD_VOCAB_SIZE, 20), dtype=np.float32)
        for i, aa in enumerate(aa_order):
            tok_id = _AA_TO_TOKEN.get(aa)
            if tok_id is not None:
                table[tok_id] = mat[i]
        return table
    except Exception:
        return np.zeros((_PAD_VOCAB_SIZE, 20), dtype=np.float32)


_BLOSUM_TABLE = torch.tensor(_build_blosum_table(), dtype=torch.float32)  # [32, 20]


def blosum_features(token_ids: Tensor) -> Tensor:
    """Per-residue BLOSUM62 embeddings from token IDs.

    Args:
        token_ids: (B, L) long tensor.

    Returns:
        Float tensor of shape (B, L, 20). Zero for special tokens.
    """
    table = _BLOSUM_TABLE.to(token_ids.device)
    return table[token_ids]  # (B, L, 20)


# ---------------------------------------------------------------------------
# Unified extra-feature builder
# ---------------------------------------------------------------------------

#: Dimension of the combined extra feature vector for each feature flag combination.
def extra_feature_dim(rsa: bool, bio: bool, blosum: bool) -> int:
    return int(rsa) + (4 if bio else 0) + (20 if blosum else 0)


def compute_extra_features(
    batch: dict,
    device: str | torch.device,
    rsa_as_feature: bool = False,
    bio: bool = False,
    blosum: bool = False,
) -> Tensor | None:
    """Assemble per-position extra features from a collated batch.

    Args:
        batch:          Collated batch dict (must contain ``input_ids`` and,
                        when ``rsa_as_feature=True``, ``rsa``).
        device:         Target device.
        rsa_as_feature: Include per-residue RSA (1 dim, NaN → 0).
        bio:            Include biophysical properties (4 dims).
        blosum:         Include BLOSUM62 embedding (20 dims).

    Returns:
        Float tensor of shape ``(B, L, D)`` or ``None`` if no features requested.
    """
    parts: list[Tensor] = []

    input_ids = batch["input_ids"].to(device)  # (B, L)

    if rsa_as_feature:
        rsa_vals = batch["rsa"].to(device)  # (B, L) — NaN where unavailable
        rsa_clean = torch.nan_to_num(rsa_vals, nan=0.0).unsqueeze(-1)  # (B, L, 1)
        parts.append(rsa_clean)

    if bio:
        parts.append(bio_features(input_ids).to(device))  # (B, L, 4)

    if blosum:
        parts.append(blosum_features(input_ids).to(device))  # (B, L, 20)

    if not parts:
        return None
    return torch.cat(parts, dim=-1)  # (B, L, D)


# ---------------------------------------------------------------------------
# Interface subset filtering (for pdb_contacts.fasta pretraining)
# ---------------------------------------------------------------------------

def filter_contacts_by_size(
    samples: list,
    min_interface: int = 5,
    max_interface: int = 35,
) -> list:
    """Filter combined-FASTA samples by number of interface residues (uppercase chars).

    Args:
        samples:       List of ``(header, (token_ids, labels))`` HeaderedSample tuples.
        min_interface: Minimum number of interface residues (inclusive).
        max_interface: Maximum number of interface residues (inclusive).

    Returns:
        Filtered list.
    """
    out = []
    for header, (token_ids, labels) in samples:
        # Count interface positions: labels[i] == 1 (excl. BOS=-100 and EOS=-100)
        n_interface = sum(1 for l in labels if l == 1)
        if min_interface <= n_interface <= max_interface:
            out.append((header, (token_ids, labels)))
    return out
