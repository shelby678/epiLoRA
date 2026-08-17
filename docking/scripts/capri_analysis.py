#!/usr/bin/env python3
"""
Compute CAPRI docking quality metrics and plot score vs lRMSD.

For each completed run (runs/{pdb_id}/{run_name}/done), evaluates every
cluster model in haddock_out/10_seletopclusts/ against the crystal structure
complex in input_pdbs/selected_32_pdbs/{pdb_id}.pdb.

Crystal structure antibody/antigen chains are identified automatically by
sequence-aligning each crystal chain against the docked chain A (antigen) and
chain B (antibody). The crystal chains with the best match to chain B are
treated as the antibody reference; the rest as antigen.

CAPRI metrics:
  lRMSD  – CA RMSD of antigen after superimposing on antibody CA atoms
  iRMSD  – CA RMSD of interface residues after aligning on those residues
  fnat   – fraction of native heavy-atom contacts (< 5 Å) preserved
  DockQ  – combined score (Basu & Wallner 2016)
  quality – High / Medium / Acceptable / Incorrect

Output:
  capri_results.csv  – one row per cluster model
  capri_plot.png     – score vs lRMSD scatter, coloured by CAPRI class
"""
import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Bio.Align import PairwiseAligner, substitution_matrices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
CAPRI_COLORS = {
    "High":       "#2196F3",
    "Medium":     "#4CAF50",
    "Acceptable": "#FF9800",
    "Incorrect":  "#9E9E9E",
}
CAPRI_ORDER = ["High", "Medium", "Acceptable", "Incorrect"]

_ALIGNER = PairwiseAligner()
_ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")
_ALIGNER.open_gap_score    = -11
_ALIGNER.extend_gap_score  = -1
_ALIGNER.mode              = "global"


# ---------------------------------------------------------------------------
# PDB loading
# ---------------------------------------------------------------------------

def _open(path):
    p = str(path)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_atoms(path, chains=None):
    """
    Return list of (chain, resnum, resname, atomname, xyz).
    chains: set of chain IDs to keep, or None for all.
    Only standard amino acid residues.
    """
    atoms = []
    with _open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            ch      = line[21]
            if chains and ch not in chains:
                continue
            resname = line[17:20].strip()
            if resname not in AA3:
                continue
            resnum  = int(line[22:26])
            atomname = line[12:16].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            atoms.append((ch, resnum, resname, atomname, np.array([x, y, z])))
    return atoms


def ca_by_resnum(atoms, chain):
    """Return {resnum: xyz} for CA atoms in chain."""
    d = {}
    for ch, rn, rname, aname, xyz in atoms:
        if ch == chain and aname == "CA" and rn not in d:
            d[rn] = xyz
    return d


def chain_seq(ca_dict):
    """Return (resnums_sorted, seq_str) for a CA dict."""
    resnums = sorted(ca_dict.keys())
    # need resname; re-extract from atoms for sequence
    return resnums


def get_chain_seq_and_ca(atoms, chain):
    """Return (resnums, seq_str, ca_dict) for one chain."""
    ca   = {}
    seq_parts = {}
    for ch, rn, rname, aname, xyz in atoms:
        if ch != chain:
            continue
        if rn not in seq_parts:
            seq_parts[rn] = AA3.get(rname, "X")
        if aname == "CA" and rn not in ca:
            ca[rn] = xyz
    resnums = sorted(ca.keys())
    seq = "".join(seq_parts.get(r, "X") for r in resnums)
    return resnums, seq, ca


def get_heavy(atoms, chain):
    """Return {resnum: [xyz, ...]} for all heavy atoms in chain."""
    d = {}
    for ch, rn, rname, aname, xyz in atoms:
        if ch == chain and not aname.startswith("H"):
            d.setdefault(rn, []).append(xyz)
    return d


# ---------------------------------------------------------------------------
# Sequence alignment
# ---------------------------------------------------------------------------

def align_seqs(seq1, seq2):
    """Return list of (idx_in_seq1, idx_in_seq2) matched pairs (no gaps)."""
    if not seq1 or not seq2:
        return []
    try:
        aln = next(_ALIGNER.align(seq1, seq2))
    except (StopIteration, ValueError):
        return []
    pairs = []
    for (s1, e1), (s2, e2) in zip(aln.aligned[0], aln.aligned[1]):
        for i, j in zip(range(s1, e1), range(s2, e2)):
            pairs.append((i, j))
    return pairs


def align_score(seq1, seq2):
    if not seq1 or not seq2:
        return -1e9
    try:
        return next(_ALIGNER.align(seq1, seq2)).score
    except (StopIteration, ValueError):
        return -1e9


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def kabsch(P, Q):
    """Rotation matrix R that minimises RMSD(P @ R - Q)."""
    H    = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d    = np.linalg.det(Vt.T @ U.T)
    R    = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R


def aligned_rmsd(mob, ref):
    """Kabsch-align mob onto ref, return RMSD."""
    mc   = mob - mob.mean(0)
    rc   = ref - ref.mean(0)
    R    = kabsch(mc, rc)
    diff = mc @ R.T - rc
    return float(np.sqrt((diff**2).sum(1).mean()))


# ---------------------------------------------------------------------------
# Crystal reference builder
# ---------------------------------------------------------------------------

def _atoms_to_seq(atoms):
    """Return merged CA sequence from all chains in atoms list."""
    seq_parts = {}
    ca_seen = {}
    for ch, rn, rname, aname, xyz in atoms:
        key = (ch, rn)
        if key not in seq_parts:
            seq_parts[key] = AA3.get(rname, "X")
        if aname == "CA":
            ca_seen[key] = True
    keys = sorted(k for k in ca_seen)
    return "".join(seq_parts.get(k, "X") for k in keys)

class CrystalRef:
    """
    Pre-processes a crystal structure for one PDB ID.
    Identifies antibody chains (best global-alignment match to chain B of
    the prepared antibody) and antigen chains (remainder).
    """

    def __init__(self, crystal_path, ab_ref_path, ag_ref_path):
        """
        crystal_path: raw crystal PDB (input_pdbs/selected_32_pdbs/{id}.pdb)
        ab_ref_path:  prepared antibody PDB (e.g. input_pdbs/{id}/ab_1.pdb)
        ag_ref_path:  prepared antigen PDB  (e.g. input_pdbs/{id}/ag.pdb)
        """
        self.crystal_atoms = load_atoms(crystal_path)
        ab_atoms = load_atoms(ab_ref_path)
        ag_atoms = load_atoms(ag_ref_path)

        # Build combined sequences regardless of chain naming in the ref files
        ab_seq = _atoms_to_seq(ab_atoms)
        ag_seq = _atoms_to_seq(ag_atoms)

        # Identify crystal chains
        crystal_chains = sorted({a[0] for a in self.crystal_atoms})
        ab_chains, ag_chains = [], []
        for ch in crystal_chains:
            rns, seq, ca = get_chain_seq_and_ca(self.crystal_atoms, ch)
            if not seq:
                continue
            score_ab = align_score(seq, ab_seq)
            score_ag = align_score(seq, ag_seq)
            if score_ab >= score_ag:
                ab_chains.append(ch)
            else:
                ag_chains.append(ch)

        self.ab_chains = ab_chains
        self.ag_chains = ag_chains

        # Build merged antibody CA (sequential: chains in order)
        self.ref_ab_rns, self.ref_ab_seq, self.ref_ab_ca = \
            self._merge_chains(ab_chains)
        self.ref_ag_rns, self.ref_ag_seq, self.ref_ag_ca = \
            self._merge_chains(ag_chains)

        # Heavy atoms for contact computation
        self.ref_ab_heavy = self._merge_heavy(ab_chains)
        self.ref_ag_heavy = self._merge_heavy(ag_chains)

        # Native contacts and interface residues (computed once)
        self.native_contacts     = _contacts(self.ref_ab_heavy, self.ref_ag_heavy, 5.0)
        self.iface_ab, self.iface_ag = _interface_residues(
            self.ref_ab_heavy, self.ref_ag_heavy, 10.0)

        # Interface CA coords for iRMSD reference
        self.ref_iface_ab_ca = {rn: self.ref_ab_ca[rn]
                                 for rn in self.iface_ab if rn in self.ref_ab_ca}
        self.ref_iface_ag_ca = {rn: self.ref_ag_ca[rn]
                                 for rn in self.iface_ag if rn in self.ref_ag_ca}

    def _merge_chains(self, chains):
        """Merge multiple crystal chains into one sequential CA dict."""
        all_rns, all_seq, all_ca = [], [], {}
        offset = 0
        for ch in chains:
            rns, seq, ca = get_chain_seq_and_ca(self.crystal_atoms, ch)
            for r, s in zip(rns, seq):
                new_r = r + offset
                all_rns.append(new_r)
                all_seq.append(s)
                all_ca[new_r] = ca[r]
            if rns:
                offset = max(all_rns) + 10
        return all_rns, "".join(all_seq), all_ca

    def _merge_heavy(self, chains):
        """Merge heavy atoms from chains into one dict keyed by sequential resnum."""
        merged = {}
        offset = 0
        for ch in chains:
            heavy = get_heavy(self.crystal_atoms, ch)
            ca    = {a[1]: None for a in self.crystal_atoms
                     if a[0] == ch and a[3] == "CA"}
            rns   = sorted(ca.keys())
            for r in sorted(heavy):
                new_r = r + offset
                merged[new_r] = heavy[r]
            if rns:
                offset += max(rns) + 10
        return merged


# ---------------------------------------------------------------------------
# Contact / interface helpers
# ---------------------------------------------------------------------------

def _build_heavy_array(heavy_dict):
    """Stack all heavy atoms into (N,3) array with per-atom residue labels."""
    pts, labels = [], []
    for rn, xyzs in heavy_dict.items():
        for xyz in xyzs:
            pts.append(xyz)
            labels.append(rn)
    if not pts:
        return np.empty((0, 3)), np.array([], dtype=int)
    return np.array(pts), np.array(labels)


def _contacts(ab_heavy, ag_heavy, cutoff):
    """Return set of (ab_resnum, ag_resnum) pairs with any heavy-atom < cutoff."""
    ag_pts, ag_labels = _build_heavy_array(ag_heavy)
    if len(ag_pts) == 0:
        return set()
    tree = cKDTree(ag_pts)
    contacts = set()
    for ab_rn, ab_xyzs in ab_heavy.items():
        ab_arr = np.vstack(ab_xyzs)
        hits = tree.query_ball_point(ab_arr, r=cutoff)
        for idx_list in hits:
            for idx in idx_list:
                contacts.add((ab_rn, int(ag_labels[idx])))
    return contacts


def _interface_residues(ab_heavy, ag_heavy, cutoff):
    """Return (ab_resnums, ag_resnums) within cutoff of the other chain."""
    ag_pts, ag_labels = _build_heavy_array(ag_heavy)
    ab_pts, ab_labels = _build_heavy_array(ab_heavy)
    if len(ag_pts) == 0 or len(ab_pts) == 0:
        return set(), set()

    ag_tree = cKDTree(ag_pts)
    ab_tree = cKDTree(ab_pts)

    iface_ab = set()
    for ab_rn, ab_xyzs in ab_heavy.items():
        hits = ag_tree.query_ball_point(np.vstack(ab_xyzs), r=cutoff)
        if any(hits):
            iface_ab.add(ab_rn)

    iface_ag = set()
    for ag_rn, ag_xyzs in ag_heavy.items():
        hits = ab_tree.query_ball_point(np.vstack(ag_xyzs), r=cutoff)
        if any(hits):
            iface_ag.add(ag_rn)

    return iface_ab, iface_ag


# ---------------------------------------------------------------------------
# Per-model CAPRI computation
# ---------------------------------------------------------------------------

def _matched_coords(ref_rns, ref_seq, ref_ca, dock_rns, dock_seq, dock_ca):
    """Sequence-align then return matched (ref_pts, dock_pts) numpy arrays."""
    pairs = align_seqs(ref_seq, dock_seq)
    if len(pairs) < 3:
        return None, None
    ref_pts  = np.array([ref_ca[ref_rns[i]]  for i, _ in pairs if ref_rns[i]  in ref_ca])
    dock_pts = np.array([dock_ca[dock_rns[j]] for _, j in pairs if dock_rns[j] in dock_ca])
    # keep only pairs where both exist
    valid = [(i, j) for i, j in pairs
             if ref_rns[i] in ref_ca and dock_rns[j] in dock_ca]
    if len(valid) < 3:
        return None, None
    ref_pts  = np.array([ref_ca[ref_rns[i]]  for i, j in valid])
    dock_pts = np.array([dock_ca[dock_rns[j]] for i, j in valid])
    return ref_pts, dock_pts


def compute_capri(model_path, ref: CrystalRef):
    """Return (lrmsd, irmsd, fnat, dockq, quality) for one docked model."""
    atoms = load_atoms(model_path)

    # Docked chains
    dock_ab_rns, dock_ab_seq, dock_ab_ca = get_chain_seq_and_ca(atoms, "B")
    dock_ag_rns, dock_ag_seq, dock_ag_ca = get_chain_seq_and_ca(atoms, "A")
    dock_ab_heavy = get_heavy(atoms, "B")
    dock_ag_heavy = get_heavy(atoms, "A")

    if not dock_ab_ca or not dock_ag_ca:
        return None, None, None, None, None

    # ---- lRMSD ----
    ref_ab_pts, dock_ab_pts = _matched_coords(
        ref.ref_ab_rns, ref.ref_ab_seq, ref.ref_ab_ca,
        dock_ab_rns,    dock_ab_seq,    dock_ab_ca)
    ref_ag_pts, dock_ag_pts_raw = _matched_coords(
        ref.ref_ag_rns, ref.ref_ag_seq, ref.ref_ag_ca,
        dock_ag_rns,    dock_ag_seq,    dock_ag_ca)

    if ref_ab_pts is None or ref_ag_pts is None:
        lrmsd = None
    else:
        # Align docked antibody onto reference antibody
        ab_mob_c = dock_ab_pts - dock_ab_pts.mean(0)
        ab_ref_c = ref_ab_pts  - ref_ab_pts.mean(0)
        R  = kabsch(ab_mob_c, ab_ref_c)
        t_mob = dock_ab_pts.mean(0)
        t_ref = ref_ab_pts.mean(0)

        # Apply same transform to docked antigen
        dock_ag_pts_aligned = (dock_ag_pts_raw - t_mob) @ R.T + t_ref
        diff = dock_ag_pts_aligned - ref_ag_pts
        lrmsd = float(np.sqrt((diff**2).sum(1).mean()))

    # ---- iRMSD ----
    if ref_ab_pts is None:
        irmsd = None
    else:
        # Build matched coords for interface residues only
        # Map crystal iface residues → sequence indices → docked residues
        ab_iface_pairs = align_seqs(ref.ref_ab_seq, dock_ab_seq)
        ag_iface_pairs = align_seqs(ref.ref_ag_seq, dock_ag_seq)

        ab_ref_iface, ab_dock_iface = [], []
        for i, j in ab_iface_pairs:
            if i < len(ref.ref_ab_rns) and j < len(dock_ab_rns):
                r_ref = ref.ref_ab_rns[i]
                r_doc = dock_ab_rns[j]
                if r_ref in ref.iface_ab and r_ref in ref.ref_ab_ca and r_doc in dock_ab_ca:
                    ab_ref_iface.append(ref.ref_ab_ca[r_ref])
                    ab_dock_iface.append(dock_ab_ca[r_doc])

        ag_ref_iface, ag_dock_iface = [], []
        for i, j in ag_iface_pairs:
            if i < len(ref.ref_ag_rns) and j < len(dock_ag_rns):
                r_ref = ref.ref_ag_rns[i]
                r_doc = dock_ag_rns[j]
                if r_ref in ref.iface_ag and r_ref in ref.ref_ag_ca and r_doc in dock_ag_ca:
                    ag_ref_iface.append(ref.ref_ag_ca[r_ref])
                    ag_dock_iface.append(dock_ag_ca[r_doc])

        ref_iface  = np.array(ab_ref_iface  + ag_ref_iface)
        dock_iface = np.array(ab_dock_iface + ag_dock_iface)

        if len(ref_iface) < 3:
            irmsd = None
        else:
            irmsd = aligned_rmsd(dock_iface, ref_iface)

    # ---- fnat ----
    # Re-map native contacts to sequence-aligned docked residue numbers
    ab_pairs = align_seqs(ref.ref_ab_seq, dock_ab_seq)
    ag_pairs = align_seqs(ref.ref_ag_seq, dock_ag_seq)
    ref_ab_rn_to_dock = {ref.ref_ab_rns[i]: dock_ab_rns[j] for i, j in ab_pairs
                         if i < len(ref.ref_ab_rns) and j < len(dock_ab_rns)}
    ref_ag_rn_to_dock = {ref.ref_ag_rns[i]: dock_ag_rns[j] for i, j in ag_pairs
                         if i < len(ref.ref_ag_rns) and j < len(dock_ag_rns)}

    if not ref.native_contacts:
        fnat = None
    else:
        dock_contacts = _contacts(dock_ab_heavy, dock_ag_heavy, 5.0)
        native_found = 0
        for (ab_rn, ag_rn) in ref.native_contacts:
            d_ab = ref_ab_rn_to_dock.get(ab_rn)
            d_ag = ref_ag_rn_to_dock.get(ag_rn)
            if d_ab is not None and d_ag is not None:
                if (d_ab, d_ag) in dock_contacts:
                    native_found += 1
        fnat = native_found / len(ref.native_contacts)

    # ---- DockQ ----
    if lrmsd is None or irmsd is None or fnat is None:
        dq = None
        quality = None
    else:
        dq = _dockq(lrmsd, irmsd, fnat)
        quality = _capri_class(dq)

    return lrmsd, irmsd, fnat, dq, quality


def _dockq(lrmsd, irmsd, fnat):
    lrms_s = 1.0 / (1.0 + (lrmsd / 8.5) ** 2)
    irms_s = 1.0 / (1.0 + (irmsd / 1.5) ** 2)
    return (fnat + lrms_s + irms_s) / 3.0


def _capri_class(dq):
    if dq >= 0.8:  return "High"
    if dq >= 0.49: return "Medium"
    if dq >= 0.23: return "Acceptable"
    return "Incorrect"


# ---------------------------------------------------------------------------
# Score reader
# ---------------------------------------------------------------------------

def read_scores(capri_ss_path):
    """Return {model_stem: score} from capri_ss.tsv."""
    scores = {}
    with open(capri_ss_path) as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if header is None:
                header = line.split("\t")
                continue
            row  = dict(zip(header, line.split("\t")))
            stem = Path(row["model"]).stem.replace(".pdb", "")
            try:
                scores[stem] = float(row["score"])
            except (KeyError, ValueError):
                pass
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir",    default="runs")
    p.add_argument("--crystal-dir", default="input_pdbs/selected_32_pdbs")
    p.add_argument("--ab-dir",      default="input_pdbs",
                   help="Parent dir containing {pdb_id}/ab_1.pdb")
    p.add_argument("--ag-dir",      default="input_pdbs",
                   help="Parent dir containing {pdb_id}/ag.pdb")
    p.add_argument("--out-csv",  default="capri_results.csv")
    p.add_argument("--out-plot", default="capri_plot.png")
    args = p.parse_args()

    runs_root   = Path(args.runs_dir)
    crystal_dir = Path(args.crystal_dir)
    ab_dir      = Path(args.ab_dir)
    ag_dir      = Path(args.ag_dir)

    done_files  = sorted(runs_root.glob("*/*/done"))
    print(f"Found {len(done_files)} completed runs", file=sys.stderr)

    # Pre-build crystal refs (one per PDB ID)
    pdb_ids = sorted({d.parent.parent.name for d in done_files})
    refs = {}
    for pid in pdb_ids:
        crystal_path = crystal_dir / f"{pid}.pdb"
        ab_path      = ab_dir / pid / "ab_1.pdb"
        ag_path      = ag_dir / pid / "ag.pdb"
        if not crystal_path.exists():
            print(f"  [skip {pid}] no crystal PDB", file=sys.stderr)
            continue
        if not ab_path.exists() or not ag_path.exists():
            print(f"  [skip {pid}] missing ab_1.pdb or ag.pdb", file=sys.stderr)
            continue
        print(f"  Building reference for {pid}...", file=sys.stderr)
        try:
            refs[pid] = CrystalRef(crystal_path, ab_path, ag_path)
            print(f"    ab chains: {refs[pid].ab_chains}  "
                  f"ag chains: {refs[pid].ag_chains}  "
                  f"native contacts: {len(refs[pid].native_contacts)}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [error {pid}] {e}", file=sys.stderr)

    # Process each run
    rows = []
    for done in done_files:
        run_dir  = done.parent
        pdb_id   = run_dir.parent.name
        run_name = run_dir.name
        sel_dir  = run_dir / "haddock_out" / "10_seletopclusts"
        ss_path  = run_dir / "haddock_out" / "11_caprieval" / "capri_ss.tsv"

        if pdb_id not in refs:
            continue
        if not sel_dir.exists():
            continue

        ref   = refs[pdb_id]
        score_map = read_scores(ss_path) if ss_path.exists() else {}

        for model_gz in sorted(sel_dir.glob("cluster_*.pdb.gz")):
            stem  = model_gz.stem.replace(".pdb", "")
            score = score_map.get(stem)

            print(f"  {pdb_id}/{run_name}/{stem}", file=sys.stderr)
            try:
                lrmsd, irmsd, fnat, dq, quality = compute_capri(model_gz, ref)
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr)
                lrmsd = irmsd = fnat = dq = quality = None

            rows.append({
                "pdb_id":   pdb_id,
                "run":      run_name,
                "model":    stem,
                "score":    score,
                "lrmsd":    round(lrmsd, 3) if lrmsd is not None else None,
                "irmsd":    round(irmsd, 3) if irmsd is not None else None,
                "fnat":     round(fnat,  3) if fnat  is not None else None,
                "dockq":    round(dq,    3) if dq    is not None else None,
                "quality":  quality,
            })

    # Write CSV
    if not rows:
        sys.exit("No results collected.")

    import csv
    fieldnames = ["pdb_id","run","model","score","lrmsd","irmsd","fnat","dockq","quality"]
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {args.out_csv}", file=sys.stderr)

    # ---- Plot ----
    # Filter to rows with scores and lrmsd
    plot_rows = [r for r in rows if r["score"] is not None and r["lrmsd"] is not None]
    if not plot_rows:
        print("No plottable data (missing scores or lrmsd).", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for quality in CAPRI_ORDER:
        subset = [r for r in plot_rows if r["quality"] == quality]
        if not subset:
            continue
        ax.scatter(
            [r["lrmsd"] for r in subset],
            [r["score"] for r in subset],
            c=CAPRI_COLORS[quality],
            label=quality,
            alpha=0.7,
            edgecolors="none",
            s=30,
        )

    # Reference lines at CAPRI lRMSD thresholds
    for x, ls in [(5.0, "--"), (10.0, ":")]:
        ax.axvline(x, color="black", linewidth=0.8, linestyle=ls, alpha=0.5)

    ax.set_xlabel("Ligand RMSD / Å  (lower = better)", fontsize=13)
    ax.set_ylabel("HADDOCK score  (lower = better)", fontsize=13)
    ax.set_title("HADDOCK score vs ligand RMSD\n(coloured by CAPRI quality class)", fontsize=13)
    ax.legend(title="CAPRI class", fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"Plot saved → {args.out_plot}", file=sys.stderr)

    # Summary table
    print("\nCAPRI quality distribution across all cluster models:")
    for q in CAPRI_ORDER:
        n = sum(1 for r in rows if r["quality"] == q)
        print(f"  {q:12s}: {n}")


if __name__ == "__main__":
    main()
