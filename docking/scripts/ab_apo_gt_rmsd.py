#!/usr/bin/env python3
"""
CA RMSD between each apo/predicted antibody model (ab_1..ab_5) and the
ground-truth (crystal-bound) antibody Fv, per dname -- checks whether
antibody-prediction error explains which antigens dock successfully.

Ground truth chains are identified with capri_analysis.py's own CrystalRef
(robust sequence-alignment chain ID against the raw crystal PDB) -- NOT
runs/{dname}/reference.pdb, whose prepare_reference.py naively matches
crystal chain letters to ab_1.pdb's own chain letters and silently produces
an empty antibody chain whenever the crystal uses different letters (H/L vs
A/B), which happened for 8 dnames including several docking-success ones.

VH and VL are matched to the crystal's antibody chain(s) *independently* by
sequence-alignment score (not by concatenation order): the docked ab_i's
merged chain B has VH/VL concatenated in whatever order the source file
happened to use (checked: this varies per structure, e.g. 8oxw/7so5 are
VH-first, 7wsl is VL-first), so a single global alignment of the two full
concatenated sequences can silently register VH against VL for structures
where the two orders disagree, producing a ~15-20 Angstrom "RMSD" that is
purely a domain-registration artifact, not real prediction error -- caught
by checking 7wsl by hand (mob = VL+VH, crystal merges as H+L => misaligned).

Both RMSDs below are computed from ONE whole-Fv (VH+VL together) Kabsch
superposition, applied to two point subsets:
  fv_rmsd  -- every matched Fv residue
  cdr_rmsd -- only the matched CDR residues (chothia via abnumber, on the
              docked ab's own numbering -- same logic generate_restraints.py
              uses for active restraints), without a separate re-fit, so it
              reflects true loop displacement in the whole-Fv-superposed
              frame rather than a self-fit that would flatter CDR accuracy.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capri_analysis import (  # noqa: E402
    CrystalRef, load_atoms, get_chain_seq_and_ca, align_score, align_seqs, kabsch, AA3,
)
from generate_restraints import read_chain_residues, split_vh_vl, get_cdr_resseqs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HADDOCKING2 = Path.home() / "work/ab_MD/haddocking2"
CRYSTAL_DIR = HADDOCKING2 / "input_pdbs" / "selected_32_pdbs"
AB_DIR = HADDOCKING2 / "input_pdbs"
AG_DIR = HADDOCKING2 / "input_pdbs"


def domain_seq_and_ca(residues, mob_atoms=None):
    """residues: [(resseq, resname), ...] -> (rns, seq) using capri_analysis's AA3."""
    rns = [r for r, _ in residues]
    seq = "".join(AA3.get(rn, "X") for _, rn in residues)
    return rns, seq


def assign_crystal_chains_to_domains(ref, mob_vh_seq, mob_vl_seq):
    """Return (vh_rns, vh_seq, vh_ca, vl_rns, vl_seq, vl_ca) in crystal
    coordinates, matching each raw crystal ab_chain to VH or VL by
    alignment score against the docked ab's own VH/VL sequences (not by
    concatenation order). VH and VL coordinate dicts are kept SEPARATE --
    both chains' residue numbering typically starts near 1, so merging them
    into one dict keyed by raw resseq would silently let VL entries
    overwrite VH's wherever the numbers collide."""
    vh_parts, vl_parts = [], []
    for ch in ref.ab_chains:
        rns, seq, ca = get_chain_seq_and_ca(ref.crystal_atoms, ch)
        if not seq:
            continue
        s_vh, s_vl = align_score(seq, mob_vh_seq), align_score(seq, mob_vl_seq)
        (vh_parts if s_vh >= s_vl else vl_parts).append((rns, seq, ca))

    def merge(parts):
        rns, seq, ca = [], "", {}
        for r, s, c in parts:
            rns += r
            seq += s
            ca.update(c)
        return rns, seq, ca

    vh_rns, vh_seq, vh_ca = merge(vh_parts)
    vl_rns, vl_seq, vl_ca = merge(vl_parts)
    return vh_rns, vh_seq, vh_ca, vl_rns, vl_seq, vl_ca


def fv_and_cdr_rmsd(ref, mob_pdb):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mob_residues = read_chain_residues(mob_pdb, "B")
        mob_vh_res, mob_vl_res = split_vh_vl(mob_residues)
        cdr_resseqs = set(get_cdr_resseqs(mob_vh_res) + get_cdr_resseqs(mob_vl_res))

    mob_vh_rns, mob_vh_seq = domain_seq_and_ca(mob_vh_res)
    mob_vl_rns, mob_vl_seq = domain_seq_and_ca(mob_vl_res)
    mob_atoms = load_atoms(mob_pdb)
    _, _, mob_ca = get_chain_seq_and_ca(mob_atoms, "B")

    ref_vh_rns, ref_vh_seq, ref_vh_ca, ref_vl_rns, ref_vl_seq, ref_vl_ca = \
        assign_crystal_chains_to_domains(ref, mob_vh_seq, mob_vl_seq)

    valid = []  # list of (ref_xyz, mob_xyz, mob_resseq) -- ref_ca kept per-domain, never merged
    for ref_rns, ref_seq, ref_ca_dom, mob_rns, mob_seq in [
        (ref_vh_rns, ref_vh_seq, ref_vh_ca, mob_vh_rns, mob_vh_seq),
        (ref_vl_rns, ref_vl_seq, ref_vl_ca, mob_vl_rns, mob_vl_seq),
    ]:
        if not ref_seq or not mob_seq:
            continue
        for i, j in align_seqs(ref_seq, mob_seq):
            r, m = ref_rns[i], mob_rns[j]
            if r in ref_ca_dom and m in mob_ca:
                valid.append((ref_ca_dom[r], mob_ca[m], m))

    if len(valid) < 3:
        return None, None, len(valid), 0

    ref_pts = np.array([v[0] for v in valid])
    mob_pts = np.array([v[1] for v in valid])
    ref_c, mob_c = ref_pts.mean(0), mob_pts.mean(0)
    R = kabsch(mob_pts - mob_c, ref_pts - ref_c)
    diff = (mob_pts - mob_c) @ R.T - (ref_pts - ref_c)
    fv_rmsd = float(np.sqrt((diff ** 2).sum(1).mean()))

    cdr_mask = [v[2] in cdr_resseqs for v in valid]
    n_cdr = sum(cdr_mask)
    cdr_rmsd = float(np.sqrt((diff[cdr_mask] ** 2).sum(1).mean())) if n_cdr >= 3 else None
    return fv_rmsd, cdr_rmsd, len(valid), n_cdr


def main():
    dnames = sorted(p.stem for p in CRYSTAL_DIR.glob("*.pdb"))

    rows = []
    for dname in dnames:
        crystal_path = CRYSTAL_DIR / f"{dname}.pdb"
        ab_path = AB_DIR / dname / "ab_1.pdb"
        ag_path = AG_DIR / dname / "ag.pdb"
        if not (ab_path.exists() and ag_path.exists()):
            print(f"  [skip {dname}] missing ab_1.pdb/ag.pdb", file=sys.stderr)
            continue
        try:
            ref = CrystalRef(crystal_path, ab_path, ag_path)
        except Exception as e:
            print(f"  [skip {dname}] CrystalRef failed: {e}", file=sys.stderr)
            continue
        if not ref.ref_ab_seq:
            print(f"  [skip {dname}] CrystalRef found no antibody chains", file=sys.stderr)
            continue

        for ab in ["ab_1", "ab_2", "ab_3", "ab_4", "ab_5"]:
            pair_dirs = sorted((HADDOCKING2 / "runs" / dname).glob(f"{ab}_vs_*"))
            if not pair_dirs:
                continue
            ab_pdb = pair_dirs[0] / "input_ab.pdb"
            if not ab_pdb.exists():
                continue
            fv_rmsd, cdr_rmsd, n_matched, n_cdr = fv_and_cdr_rmsd(ref, ab_pdb)
            rows.append({
                "pdb_id": dname, "ab": ab,
                "fv_rmsd": fv_rmsd, "cdr_rmsd": cdr_rmsd,
                "n_matched": n_matched, "n_cdr_matched": n_cdr,
            })
            print(f"  {dname} {ab}: fv_rmsd={fv_rmsd} cdr_rmsd={cdr_rmsd} "
                  f"(n_matched={n_matched}, n_cdr={n_cdr})", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "scratch" / "ab_apo_gt_rmsd.csv", index=False)
    print(f"\nWrote scratch/ab_apo_gt_rmsd.csv ({len(df)} rows)")

    per_dname = df.groupby("pdb_id")[["fv_rmsd", "cdr_rmsd"]].mean()
    per_dname.to_csv(ROOT / "scratch" / "ab_apo_gt_rmsd_by_dname.csv")
    print("\nPer-dname mean RMSD:")
    print(per_dname.sort_values("fv_rmsd").to_string())


if __name__ == "__main__":
    main()
