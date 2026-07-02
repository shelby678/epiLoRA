"""Extract ESM-IF1 per-residue embeddings for all BEPIPRED sequences.

Run with the discotope3_web Python environment:
    /home/sferrier/epitope_mapping/discotope3_web/env/bin/python extract_esmif1_embeddings.py

Outputs one .npy file per sequence to data/esmif1_embed_cache/<header_key>.npy
Shape: (L, 512) float32, BOS/EOS stripped.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Paths (relative to autoprot dir)
AUTOPROT_DIR   = Path(__file__).parent
BEPIPRED_FASTA = AUTOPROT_DIR / "data/BEPIPRED.fasta"
STRUCTURES_DIR = AUTOPROT_DIR / "data/structures2/sabdab_dataset"
CACHE_DIR      = AUTOPROT_DIR / "data/esmif1_embed_cache"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GPU_LEN    = 1000  # use CPU for longer chains


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return list of (header, sequence) from a FASTA file."""
    entries, header, seq = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq)))
            header = line[1:].strip()
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        entries.append((header, "".join(seq)))
    return entries


def parse_seq_id(seq_id: str) -> tuple[str, str] | None:
    """Return (pdb_id_lower, antigen_chain) from header. Returns None if unparseable."""
    if " " in seq_id:
        parts = seq_id.split()
        pdb_id = parts[0].split("_")[0].lower()
        antigen = parts[1] if len(parts) >= 2 else None
    else:
        parts = seq_id.split("_")
        if len(parts) < 4:
            return None
        pdb_id = parts[0].lower()
        antigen = parts[3]
    return (pdb_id, antigen) if antigen else None


def extract_coords(pdb_path: Path, chain_id: str, seq_len: int):
    """Extract N, CA, C backbone coords for chain, shape (L, 3, 3) or None."""
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure as struc

    try:
        f = pdb_io.PDBFile.read(str(pdb_path))
        struct = pdb_io.get_structure(f, model=1)
    except Exception:
        return None

    # Filter to target chain + amino acids
    struct = struct[(struct.chain_id == chain_id) & struc.filter_amino_acids(struct)]

    residue_ids = struc.get_residues(struct)[0]  # residue IDs array
    n_residues  = len(residue_ids)
    if n_residues != seq_len:
        return None

    coords = np.full((n_residues, 3, 3), np.nan, dtype=np.float32)
    atom_names = ["N", "CA", "C"]

    for res_i, res_id in enumerate(residue_ids):
        res_atoms = struct[struct.res_id == res_id]
        for a_i, atom_name in enumerate(atom_names):
            mask = res_atoms.atom_name == atom_name
            if mask.any():
                coords[res_i, a_i] = res_atoms.coord[mask][0]

    return coords


def embed_chain(model, alphabet, coords_np: np.ndarray, aa_seq: str, device: str) -> np.ndarray | None:
    """Run ESM-IF1 encoder on one chain. Returns (L, 512) float32 or None."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "discotope3_web/src"))
    from esm_util_custom import get_encoder_output

    L = len(aa_seq)
    dev = device if L <= MAX_GPU_LEN else "cpu"

    try:
        with torch.no_grad():
            rep = get_encoder_output(model, alphabet, coords_np, aa_seq, device=dev)
            # rep: (L, 512) CPU tensor
        emb = rep.numpy().astype(np.float32)
        return emb if emb.shape == (L, 512) else None
    except Exception as e:
        logger.warning(f"  Embedding failed: {e}")
        return None


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ESM-IF1 ...")
    import esm
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    entries = parse_fasta(BEPIPRED_FASTA)
    # Filter out EVAL partition
    entries = [(h, s) for h, s in entries if not h.endswith(" EVAL") and " EVAL" not in h]
    # Also include holdout FASTAs
    for holdout in [AUTOPROT_DIR / "data/holdout1.fasta", AUTOPROT_DIR / "data/holdout2.fasta"]:
        if holdout.exists():
            entries += parse_fasta(holdout)
    logger.info(f"Total sequences: {len(entries)}")

    done, skipped, errors = 0, 0, 0

    for i, (header, seq) in enumerate(entries):
        # Skip EVAL
        parts = header.split()
        if len(parts) >= 3 and parts[2] == "EVAL":
            skipped += 1
            continue

        key = header.replace(" ", "_").replace("/", "-")
        cache_path = CACHE_DIR / f"{key}.npy"
        if cache_path.exists():
            done += 1
            continue

        aa_seq = seq.upper()  # uppercase = canonical AA
        L = len(aa_seq)

        parsed = parse_seq_id(header)
        if parsed is None:
            skipped += 1
            continue
        pdb_id, chain = parsed
        pdb_path = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            skipped += 1
            continue

        coords = extract_coords(pdb_path, chain, L)
        if coords is None:
            skipped += 1
            continue

        emb = embed_chain(model, alphabet, coords, aa_seq, DEVICE)
        if emb is None or emb.shape != (L, 512):
            errors += 1
            continue

        np.save(cache_path, emb)
        done += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(entries)}  done={done} skipped={skipped} errors={errors}")

    logger.info(f"Done. Cached: {done}  skipped: {skipped}  errors: {errors}")
    logger.info(f"Cache dir: {CACHE_DIR}")


if __name__ == "__main__":
    main()
