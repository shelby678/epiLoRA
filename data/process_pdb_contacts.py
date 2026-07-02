#!/usr/bin/env python3
"""Extract protein-protein contact labels from Cryo-EM PDB structures.

Iterates over data/pdb_db/??/????.cif.gz:
  - Skips structures whose experimental method is not ELECTRON MICROSCOPY.
  - Skips structures with > MAX_CHAINS polymer chains.
  - Computes interface residues: any heavy atom of residue A within CONTACT_DIST Å
    of any heavy atom of residue B, where A and B are on different chains.
  - Outputs data/pdb_contacts.fasta in pdb_chains.fasta format:
      Header: >PDBID CHAIN pretrain
      Sequence: UPPERCASE = interface residue, lowercase = non-interface

Usage:
    python data/process_pdb_contacts.py [--workers N] [--out data/pdb_contacts.fasta]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHAINS = 100          # skip structures with more polymer chains than this
MAX_RES_PER_CHAIN = 3000  # skip chains longer than this (memory safety)
MAX_ATOMS_PER_PAIR = 500_000_000  # skip chain pairs whose atom product exceeds this
CONTACT_DIST = 4.0        # Å — any heavy atom within this distance = contact

# Standard 3-letter to 1-letter amino acid mapping (uppercase keys)
_AA_3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standard residues
    "MSE": "M",  # selenomethionine
    "SEC": "C",  # selenocysteine
    "PYL": "K",  # pyrrolysine
    "HYP": "P",  # hydroxyproline
    "MLY": "K",  # methyl-lysine
    "CSE": "C",  # selenocysteine variant
    "CME": "C",  # S,S-(2-hydroxyethyl)thiocysteine
}


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------


def _process_one(cif_path: Path) -> list[tuple[str, str]] | None:
    """Process a single .cif.gz file.

    Returns:
        List of (fasta_header, labeled_sequence) pairs, or None if the
        structure is filtered out.  Returns [] if it passes filters but has
        no interface-containing chains.
    """
    try:
        with gzip.open(cif_path, "rt") as fh:
            d = MMCIF2Dict(fh)
    except Exception:
        return None

    # ── Experimental method filter ────────────────────────────────────────
    methods = d.get("_exptl.method", [])
    if isinstance(methods, str):
        methods = [methods]
    if not any("ELECTRON MICROSCOPY" in m.upper() for m in methods):
        return None

    # ── Read atom-site columns ────────────────────────────────────────────
    group_pdb  = d.get("_atom_site.group_PDB", [])
    if not group_pdb:
        return None

    auth_chain = d.get("_atom_site.auth_asym_id", [])
    comp_id    = d.get("_atom_site.label_comp_id", [])
    label_seq  = d.get("_atom_site.label_seq_id", [])
    elem       = d.get("_atom_site.type_symbol", [])
    xs         = d.get("_atom_site.Cartn_x", [])
    ys         = d.get("_atom_site.Cartn_y", [])
    zs         = d.get("_atom_site.Cartn_z", [])

    n_atoms = len(group_pdb)
    if n_atoms == 0:
        return None

    # ── First pass: count polymer chains & collect per-residue atoms ──────
    # chain → seq_id → { name, coords: list[np.array(3)] }
    chain_data: dict[str, dict[str, dict[str, Any]]] = {}
    poly_chains: set[str] = set()

    for i in range(n_atoms):
        if group_pdb[i] != "ATOM":
            continue
        e = elem[i] if i < len(elem) else "?"
        if e in ("H", "D"):       # skip H / deuterium
            continue
        sid = label_seq[i] if i < len(label_seq) else "."
        if sid in (".", "?"):     # non-polymer atom
            continue
        comp = comp_id[i].upper() if i < len(comp_id) else "?"
        if comp not in _AA_3TO1:  # not a standard amino acid
            continue

        ch = auth_chain[i]
        try:
            xi = float(xs[i]); yi = float(ys[i]); zi = float(zs[i])
        except (ValueError, IndexError):
            continue

        if ch not in chain_data:
            chain_data[ch] = {}
        if sid not in chain_data[ch]:
            chain_data[ch][sid] = {"name": comp, "coords": []}
        chain_data[ch][sid]["coords"].append(np.array([xi, yi, zi], dtype=np.float32))
        poly_chains.add(ch)

    n_poly = len(poly_chains)
    if n_poly > MAX_CHAINS or n_poly < 2:
        return None

    # ── Build per-chain sorted residue lists + atom arrays ───────────────
    chain_seqs: dict[str, list[str]] = {}
    chain_atoms: dict[str, np.ndarray] = {}   # (N_atoms, 3)
    chain_atom_res: dict[str, np.ndarray] = {}  # (N_atoms,) residue index

    for ch in poly_chains:
        if ch not in chain_data:
            continue
        residues = chain_data[ch]

        # Sort residues by numeric label_seq_id
        try:
            sorted_keys = sorted(residues.keys(), key=lambda k: int(k))
        except ValueError:
            sorted_keys = sorted(residues.keys())

        seq_chars: list[str] = []
        all_coords: list[np.ndarray] = []
        res_indices: list[int] = []

        for res_idx, key in enumerate(sorted_keys):
            res = residues[key]
            aa = _AA_3TO1.get(res["name"], "X")
            seq_chars.append(aa)
            for coord in res["coords"]:
                all_coords.append(coord)
                res_indices.append(res_idx)

        if not seq_chars or len(seq_chars) > MAX_RES_PER_CHAIN:
            continue

        chain_seqs[ch] = seq_chars
        chain_atoms[ch] = np.stack(all_coords)          # (N_atoms, 3)
        chain_atom_res[ch] = np.array(res_indices, dtype=np.int32)

    chains = [c for c in poly_chains if c in chain_seqs]
    if len(chains) < 2:
        return None

    # ── Contact detection: 4 Å any heavy atom ────────────────────────────
    chain_iface: dict[str, np.ndarray] = {
        ch: np.zeros(len(chain_seqs[ch]), dtype=bool) for ch in chains
    }

    from scipy.spatial import cKDTree  # local import to avoid top-level dep

    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            c1, c2 = chains[i], chains[j]
            a1 = chain_atoms[c1]
            a2 = chain_atoms[c2]

            if len(a1) * len(a2) > MAX_ATOMS_PER_PAIR:
                continue  # too expensive — skip this pair

            # query_ball_tree: for each atom in tree2 (c2), find c1 atoms within 4Å
            tree1 = cKDTree(a1)
            pairs = cKDTree(a2).query_ball_tree(tree1, r=CONTACT_DIST)

            # c2 interface atoms: those with at least one match
            c2_contact = np.array([k for k, p in enumerate(pairs) if p], dtype=np.int32)
            # c1 interface atoms: union of all matched indices
            c1_contact_flat = [idx for p in pairs for idx in p]
            c1_contact = np.unique(c1_contact_flat).astype(np.int32) if c1_contact_flat else np.array([], dtype=np.int32)

            if len(c2_contact) > 0:
                chain_iface[c2][np.unique(chain_atom_res[c2][c2_contact])] = True
            if len(c1_contact) > 0:
                chain_iface[c1][np.unique(chain_atom_res[c1][c1_contact])] = True

    # ── Build output entries ──────────────────────────────────────────────
    pdb_id = cif_path.stem.upper()  # stem of "8abc.cif.gz" → "8ABC"
    # (cif_path.stem strips ".cif.gz" doesn't work — .stem only strips one suffix)
    name = cif_path.name
    if name.endswith(".cif.gz"):
        pdb_id = name[: -len(".cif.gz")].upper()

    results: list[tuple[str, str]] = []
    for ch in chains:
        seq = chain_seqs[ch]
        iface = chain_iface[ch]

        # Skip chains with no interface residues (pure background)
        if not iface.any():
            continue

        labeled = "".join(
            aa.upper() if is_iface else aa.lower()
            for aa, is_iface in zip(seq, iface)
        )
        header = f"{pdb_id} {ch} pretrain"
        results.append((header, labeled))

    return results


def _worker(args: tuple[Path, int]) -> list[tuple[str, str]] | None:
    """Multiprocessing entry point — wraps _process_one and catches all errors."""
    path, _ = args
    try:
        return _process_one(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="data/pdb_db",
        help="Root directory of the PDB mirror (default: data/pdb_db)",
    )
    parser.add_argument(
        "--out", default="data/pdb_contacts.fasta",
        help="Output FASTA path (default: data/pdb_contacts.fasta)",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help="Number of parallel worker processes (default: all CPUs)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most this many files (0 = no limit, for testing)",
    )
    args = parser.parse_args()

    db_root = Path(args.db)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Collect all .cif.gz files
    cif_files = sorted(db_root.glob("??/*.cif.gz"))
    if args.limit > 0:
        cif_files = cif_files[: args.limit]

    logger.info(f"Found {len(cif_files):,} CIF files under {db_root}")
    logger.info(f"Workers: {args.workers}  Output: {out_path}")

    work = [(p, i) for i, p in enumerate(cif_files)]

    n_total = len(work)
    n_done = 0
    n_pass = 0  # structures passing filters
    n_entries = 0  # FASTA entries written

    with open(out_path, "w") as out_fh:
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_worker, work, chunksize=32):
                n_done += 1
                if result is not None and len(result) > 0:
                    n_pass += 1
                    for header, seq in result:
                        out_fh.write(f">{header}\n{seq}\n")
                        n_entries += 1

                if n_done % 10_000 == 0:
                    logger.info(
                        f"  {n_done:>7,}/{n_total:,}  passed={n_pass:,}  "
                        f"entries={n_entries:,}"
                    )

    logger.info(
        f"Done. {n_done:,} files processed → {n_pass:,} Cryo-EM structures "
        f"→ {n_entries:,} FASTA entries written to {out_path}"
    )


if __name__ == "__main__":
    main()
