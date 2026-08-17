#!/usr/bin/env python3
"""
Score crystal-complex ground truth structures with HADDOCK3 (topoaa + emscoring)
and regenerate the per-protein stratified docking plot with crystal reference scores.

For each PDB ID with completed docking runs, this script:
  1. Extracts the antibody Fv (VH as chain A, VL as chain B) and antigen from the
     crystal structure in their crystal-frame coordinates.
  2. Runs HADDOCK3 topoaa + emscoring on the crystal complex.
  3. Parses the HADDOCK score from the emscoring output PDB REMARK header.
  4. Draws a horizontal dashed line on each protein's subplot for that score.

Usage:
  python scripts/score_reference.py \\
    [--capri-csv capri_results.csv] \\
    [--crystal-dir input_pdbs/selected_32_pdbs] \\
    [--ab-dir input_pdbs] \\
    [--work-dir /tmp/crystal_scoring] \\
    [--out-scores crystal_scores.csv] \\
    [--out-plot capri_plot_by_protein.png]
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from capri_analysis import (
    load_atoms, get_chain_seq_and_ca, align_seqs, align_score,
    AA3, CAPRI_COLORS, CAPRI_ORDER,
)

HADDOCK3 = "/opt/conda/bin/haddock3"


# ---------------------------------------------------------------------------
# Crystal complex preparation
# ---------------------------------------------------------------------------

def _chain_resnums_set(crystal_atoms, chain):
    """Return set of residue numbers present in chain (standard AAs only)."""
    return {rn for ch, rn, rname, aname, xyz in crystal_atoms if ch == chain}


def _extract_chain_lines(crystal_path, keep):
    """
    Read raw PDB ATOM lines from crystal_path, filtering to residues in keep.
    keep: dict {chain_id: set(resnums)}
    Returns list of (orig_chain, orig_rn, raw_line).
    """
    lines = []
    with open(crystal_path) as f:
        for raw in f:
            if not raw.startswith("ATOM"):
                continue
            ch = raw[21]
            if ch not in keep:
                continue
            rname = raw[17:20].strip()
            if rname not in AA3:
                continue
            rn = int(raw[22:26])
            if rn not in keep[ch]:
                continue
            lines.append((ch, rn, raw))
    return lines


def _write_pdb(out_path, chain_blocks):
    """
    Write a multi-chain PDB file.
    chain_blocks: list of (new_chain, [(orig_ch, orig_rn, raw_line), ...])
    Residues are renumbered sequentially (resetting to 1 per new_chain).
    Atom serial numbers are reset globally.
    """
    atom_serial = 1
    out_lines = []
    for new_chain, block in chain_blocks:
        resnum_map = {}
        counter = 1
        for orig_ch, orig_rn, raw in block:
            key = (orig_ch, orig_rn)
            if key not in resnum_map:
                resnum_map[key] = counter
                counter += 1
        for orig_ch, orig_rn, raw in block:
            new_rn = resnum_map[(orig_ch, orig_rn)]
            new_raw = (f"ATOM  {atom_serial:5d}" + raw[11:21]
                       + new_chain + f"{new_rn:4d}" + raw[26:])
            out_lines.append(new_raw)
            atom_serial += 1
        out_lines.append("TER\n")
    out_lines.append("END\n")
    with open(out_path, "w") as f:
        f.writelines(out_lines)


def prepare_crystal_complex(crystal_path, ab1_path, ag_gt_path, work_dir):
    """
    Prepare a single crystal_complex.pdb containing the antibody Fv (chain B,
    sequential numbering matching docked models) and antigen (chain A, original
    crystal numbering), using ab_1.pdb and ag_gt.pdb as sequence guides.

    The combined PDB is fed to HADDOCK3 as a single molecule so topoaa + emscoring
    score the interface rather than each chain in isolation.

    Returns path to crystal_complex.pdb or raises on failure.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    crystal_atoms = load_atoms(crystal_path)
    ab1_atoms = load_atoms(ab1_path)
    ag_gt_atoms = load_atoms(ag_gt_path)

    # ---- Identify crystal chain roles by alignment ----
    ab1_chains = sorted({a[0] for a in ab1_atoms})
    ag_gt_chains = sorted({a[0] for a in ag_gt_atoms})

    # Sequences for each ab1 chain (VH and VL)
    ab1_seqs = {}
    for ch in ab1_chains:
        rns, seq, _ = get_chain_seq_and_ca(ab1_atoms, ch)
        ab1_seqs[ch] = (rns, seq)

    # Sequence for antigen (merged if multi-chain)
    ag_seq_cat = ""
    for ch in ag_gt_chains:
        _, seq, _ = get_chain_seq_and_ca(ag_gt_atoms, ch)
        ag_seq_cat += seq

    # Identify crystal chains as VH, VL, or antigen
    crystal_chains = sorted({a[0] for a in crystal_atoms})
    assignments = {}  # crystal_chain → ab1_chain_id or 'ag'
    for crys_ch in crystal_chains:
        rns, seq, _ = get_chain_seq_and_ca(crystal_atoms, crys_ch)
        if not seq:
            continue
        best_label = None
        best_score = -1e9
        for ab1_ch, (_, ab1_seq) in ab1_seqs.items():
            s = align_score(seq, ab1_seq)
            if s > best_score:
                best_score = s
                best_label = ab1_ch
        ag_s = align_score(seq, ag_seq_cat)
        if ag_s > best_score:
            best_score = ag_s
            best_label = "ag"
        assignments[crys_ch] = best_label

    ab_map = {}   # ab1_ch → [crystal_ch, ...]
    ag_crys_chains = []
    for crys_ch, label in assignments.items():
        if label == "ag":
            ag_crys_chains.append(crys_ch)
        else:
            ab_map.setdefault(label, []).append(crys_ch)

    if not ab_map:
        raise ValueError(f"No antibody chains identified in {crystal_path}")
    if not ag_crys_chains:
        raise ValueError(f"No antigen chains identified in {crystal_path}")

    # ---- Extract Fv residues for each crystal ab chain ----
    ab_keep = {}   # crystal_ch → set(resnums)
    for ab1_ch, crys_chains in ab_map.items():
        _, ab1_seq = ab1_seqs[ab1_ch]
        for crys_ch in crys_chains:
            crys_rns, crys_seq, _ = get_chain_seq_and_ca(crystal_atoms, crys_ch)
            pairs = align_seqs(crys_seq, ab1_seq)
            matched = {crys_rns[i] for i, j in pairs}
            ab_keep[crys_ch] = matched

    # ---- Build combined complex PDB ----
    # Antibody: merge all Fv chains → chain B, sequential numbering from 1
    ab_raw = _extract_chain_lines(crystal_path,
                                  {ch: ab_keep[ch] for ch in ab_keep})
    ab_blocks = [("B", ab_raw)]

    # Antigen: keep original chain ID(s) and residue numbers from the crystal
    # so they match what HADDOCK uses in the docked models
    ag_keep = {ch: _chain_resnums_set(crystal_atoms, ch) for ch in ag_crys_chains}
    ag_raw = _extract_chain_lines(crystal_path, ag_keep)
    ag_chain_groups = {}
    for orig_ch, orig_rn, raw in ag_raw:
        ag_chain_groups.setdefault(orig_ch, []).append((orig_ch, orig_rn, raw))
    ag_blocks = [(ch, lines) for ch, lines in sorted(ag_chain_groups.items())]

    if not ab_blocks[0][1]:
        raise ValueError(f"No Fv atoms extracted for {crystal_path}")
    if not ag_blocks:
        raise ValueError(f"No antigen atoms extracted for {crystal_path}")

    complex_path = work_dir / "crystal_complex.pdb"
    _write_pdb(complex_path, ab_blocks + ag_blocks)
    return complex_path


# ---------------------------------------------------------------------------
# HADDOCK3 scoring run
# ---------------------------------------------------------------------------

def run_haddock3_scoring(complex_pdb, run_dir):
    """
    Run HADDOCK3 topoaa + emscoring on the crystal complex.
    complex_pdb: single PDB with all chains (antibody chain B + antigen chain X).
    Returns the HADDOCK score (float), or None on failure.
    """
    import gzip as _gzip

    run_dir = Path(run_dir)
    # Clean any previous run so HADDOCK doesn't refuse to start
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    config_path = run_dir.parent / f"{run_dir.name}_config.toml"
    config = f"""\
run_dir = "{run_dir.resolve()}"

molecules = [
    "{Path(complex_pdb).resolve()}",
]

[topoaa]

[emscoring]
"""
    with open(config_path, "w") as f:
        f.write(config)

    log_path = run_dir.parent / f"{run_dir.name}.log"
    try:
        result = subprocess.run(
            [HADDOCK3, str(config_path)],
            capture_output=True, text=True, timeout=600
        )
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        if result.returncode != 0:
            print(f"    HADDOCK3 failed (rc={result.returncode}), see {log_path}",
                  file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"    HADDOCK3 timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    HADDOCK3 error: {e}", file=sys.stderr)
        return None

    # Find emscoring output — HADDOCK3 names the dir '1_emscoring'
    em_dir = run_dir / "1_emscoring"
    if not em_dir.exists():
        print(f"    emscoring directory not found: {em_dir}", file=sys.stderr)
        return None

    # Prefer .pdb.gz, fall back to .pdb
    pdbs = sorted(em_dir.glob("emscoring_*.pdb.gz")) or sorted(em_dir.glob("emscoring_*.pdb"))
    if not pdbs:
        print(f"    No emscoring output PDB in {em_dir}", file=sys.stderr)
        return None

    # Extract score from each output PDB; return the score of the complex
    # (the one with BSA > 0, or just the first one if only one)
    scores = []
    for pdb in pdbs:
        score = None
        bsa = None
        opener = _gzip.open if str(pdb).endswith(".gz") else open
        with opener(pdb, "rt") as f:
            for line in f:
                m = re.match(r"REMARK HADDOCK score:\s+([-\d.]+)", line)
                if m:
                    score = float(m.group(1))
                m2 = re.match(r"REMARK buried surface area:\s+([\d.]+)", line)
                if m2:
                    bsa = float(m2.group(1))
        if score is not None:
            scores.append((bsa or 0.0, score))

    if not scores:
        return None

    # The complex will have BSA > 0; if multiple models, pick highest BSA
    scores.sort(key=lambda x: -x[0])  # descending BSA
    return scores[0][1]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def read_capri_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for k in ("score", "lrmsd", "irmsd", "fnat", "dockq"):
                try:
                    row[k] = float(row[k]) if row[k] not in ("", "None") else None
                except ValueError:
                    row[k] = None
            rows.append(row)
    return rows


def make_stratified_plot(rows, crystal_scores, out_path):
    """
    5-column grid of per-protein subplots.
    Each panel: lRMSD (x) vs HADDOCK score (y), coloured by CAPRI quality.
    _gt runs use diamond markers; non-gt use circles.
    Crystal score shown as horizontal dashed line when available.
    """
    pdb_ids = sorted({r["pdb_id"] for r in rows})
    n = len(pdb_ids)
    ncols = 5
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
                             squeeze=False)

    for idx, pid in enumerate(pdb_ids):
        ax = axes[idx // ncols][idx % ncols]
        pid_rows = [r for r in rows
                    if r["pdb_id"] == pid
                    and r["score"] is not None
                    and r["lrmsd"] is not None]

        for q in CAPRI_ORDER:
            for is_gt in (False, True):
                subset = [r for r in pid_rows
                          if r["quality"] == q
                          and (("_gt" in r["run"]) == is_gt)]
                if not subset:
                    continue
                ax.scatter(
                    [r["lrmsd"] for r in subset],
                    [r["score"]  for r in subset],
                    c=CAPRI_COLORS[q],
                    marker="D" if is_gt else "o",
                    s=20 if is_gt else 15,
                    alpha=0.8,
                    edgecolors="none",
                    zorder=3 if is_gt else 2,
                )

        # Crystal reference score
        cs = crystal_scores.get(pid)
        if cs is not None:
            ax.axhline(cs, color="black", linewidth=1.2, linestyle="--",
                       zorder=4, label=f"crystal: {cs:.1f}")
            ax.legend(fontsize=6, loc="upper right", handlelength=1)

        ax.set_title(pid, fontsize=9, pad=3)
        ax.set_xlabel("lRMSD / Å", fontsize=7)
        ax.set_ylabel("HADDOCK score", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=CAPRI_COLORS[q], label=q) for q in CAPRI_ORDER
    ] + [
        plt.Line2D([0], [0], marker="D", color="gray",
                   linestyle="None", markersize=5, label="_gt run"),
        plt.Line2D([0], [0], marker="o", color="gray",
                   linestyle="None", markersize=5, label="MD run"),
        plt.Line2D([0], [0], color="black", linestyle="--",
                   linewidth=1.2, label="crystal complex"),
    ]
    fig.legend(handles=legend_elements, loc="lower right",
               bbox_to_anchor=(1.0, 0.0), fontsize=8, ncol=2)

    plt.suptitle(
        "HADDOCK score vs ligand RMSD  (◆ = ground-truth antigen run)",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capri-csv",    default="capri_results.csv")
    p.add_argument("--crystal-dir",  default="input_pdbs/selected_32_pdbs")
    p.add_argument("--ab-dir",       default="input_pdbs")
    p.add_argument("--work-dir",     default="/tmp/crystal_scoring")
    p.add_argument("--out-scores",   default="crystal_scores.csv")
    p.add_argument("--out-plot",     default="capri_plot_by_protein.png")
    p.add_argument("--skip-scoring", action="store_true",
                   help="Skip HADDOCK3 runs, just regenerate plot from existing scores")
    args = p.parse_args()

    crystal_dir = Path(args.crystal_dir)
    ab_dir      = Path(args.ab_dir)
    work_dir    = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load existing CAPRI results
    rows = read_capri_csv(args.capri_csv)
    pdb_ids = sorted({r["pdb_id"] for r in rows})
    print(f"Loaded {len(rows)} rows for {len(pdb_ids)} PDB IDs from {args.capri_csv}",
          file=sys.stderr)

    # Load existing crystal scores if file exists
    crystal_scores = {}
    if Path(args.out_scores).exists():
        with open(args.out_scores, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    crystal_scores[row["pdb_id"]] = float(row["haddock_score"])
                except (KeyError, ValueError):
                    pass
        print(f"Loaded {len(crystal_scores)} existing crystal scores from {args.out_scores}",
              file=sys.stderr)

    if not args.skip_scoring:
        for pid in pdb_ids:
            if pid in crystal_scores:
                print(f"  [{pid}] already scored ({crystal_scores[pid]:.2f}), skipping",
                      file=sys.stderr)
                continue

            crystal_path = crystal_dir / f"{pid}.pdb"
            ab1_path     = ab_dir / pid / "ab_1.pdb"
            ag_gt_path   = ab_dir / pid / "ag_gt.pdb"

            if not crystal_path.exists():
                print(f"  [{pid}] missing crystal PDB, skipping", file=sys.stderr)
                continue
            if not ab1_path.exists():
                print(f"  [{pid}] missing ab_1.pdb, skipping", file=sys.stderr)
                continue
            if not ag_gt_path.exists():
                print(f"  [{pid}] missing ag_gt.pdb, skipping", file=sys.stderr)
                continue

            print(f"  [{pid}] preparing crystal complex...", file=sys.stderr)
            pid_dir = work_dir / pid
            try:
                complex_pdb = prepare_crystal_complex(
                    crystal_path, ab1_path, ag_gt_path, pid_dir
                )
            except Exception as e:
                print(f"  [{pid}] prep failed: {e}", file=sys.stderr)
                continue

            print(f"  [{pid}] running HADDOCK3 emscoring...", file=sys.stderr)
            score = run_haddock3_scoring(complex_pdb, pid_dir / "haddock_run")
            if score is None:
                continue

            print(f"  [{pid}] crystal HADDOCK score = {score:.2f}", file=sys.stderr)
            crystal_scores[pid] = score

        # Save scores
        with open(args.out_scores, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pdb_id", "haddock_score"])
            w.writeheader()
            for pid, sc in sorted(crystal_scores.items()):
                w.writerow({"pdb_id": pid, "haddock_score": sc})
        print(f"Saved {len(crystal_scores)} crystal scores → {args.out_scores}",
              file=sys.stderr)

    # Generate stratified plot
    make_stratified_plot(rows, crystal_scores, args.out_plot)


if __name__ == "__main__":
    main()
