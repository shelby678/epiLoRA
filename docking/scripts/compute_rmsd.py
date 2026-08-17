#!/usr/bin/env python3
"""
Collect RMSD results for completed HADDOCK docking runs.

Scans for runs/{pdb_id}/{run_name}/done files, then reads the RMSD values
from haddock_out/11_caprieval/capri_ss.tsv (per model) and capri_clt.tsv
(per cluster). These metrics were computed by HADDOCK against the reference
structure at runs/{pdb_id}/reference.pdb.

Output columns:
  pdb_id, run, model, cluster_id, cluster_rank, model_rank_in_cluster,
  score, rmsd, irmsd, fnat, lrmsd, ilrmsd, dockq
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


METRIC_COLS = ["score", "rmsd", "irmsd", "fnat", "lrmsd", "ilrmsd", "dockq"]


def read_capri_ss(tsv_path):
    """Read per-model caprieval output; return a cleaned DataFrame."""
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    df["model"] = df["model"].apply(lambda p: Path(p).stem.replace(".pdb", ""))
    keep = ["model", "caprieval_rank", "cluster_id", "cluster_ranking",
            "model-cluster_ranking"] + METRIC_COLS
    keep = [c for c in keep if c in df.columns]
    return df[keep].rename(columns={
        "caprieval_rank":       "overall_rank",
        "cluster_ranking":      "cluster_rank",
        "model-cluster_ranking": "model_rank_in_cluster",
    })


def process_run(done_file):
    """Return a DataFrame for one completed run, or None if data is missing."""
    run_dir   = done_file.parent
    pdb_id    = run_dir.parent.name
    run_name  = run_dir.name
    capri_dir = run_dir / "haddock_out" / "11_caprieval"
    ss_path   = capri_dir / "capri_ss.tsv"
    sel_dir   = run_dir / "haddock_out" / "10_seletopclusts"

    if not sel_dir.exists():
        print(f"  [skip] 10_seletopclusts missing: {run_dir}", file=sys.stderr)
        return None
    if not ss_path.exists():
        print(f"  [skip] capri_ss.tsv missing: {capri_dir}", file=sys.stderr)
        return None

    df = read_capri_ss(ss_path)
    df.insert(0, "pdb_id", pdb_id)
    df.insert(1, "run",    run_name)
    return df


def best_per_run(df):
    """Return one row per run: the model with the lowest rmsd."""
    if "rmsd" not in df.columns:
        return df
    return (
        df.dropna(subset=["rmsd"])
          .sort_values("rmsd")
          .groupby(["pdb_id", "run"], sort=False)
          .first()
          .reset_index()
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir", default="runs",
        help="Path to the runs/ directory (default: %(default)s)",
    )
    p.add_argument(
        "--out", default="rmsd_results.csv",
        help="Per-model results CSV (default: %(default)s)",
    )
    p.add_argument(
        "--summary-out", default="rmsd_summary.csv",
        help="Best-model-per-run summary CSV (default: %(default)s)",
    )
    args = p.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.exists():
        sys.exit(f"Error: runs directory not found: {runs_root}")

    done_files = sorted(runs_root.glob("*/*/done"))
    print(f"Found {len(done_files)} completed runs", file=sys.stderr)

    frames = []
    for done in done_files:
        df = process_run(done)
        if df is not None:
            frames.append(df)

    if not frames:
        sys.exit("No results collected — check that 11_caprieval directories exist.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.out, index=False)
    print(f"Per-model results ({len(all_df)} rows) → {args.out}", file=sys.stderr)

    summary = best_per_run(all_df)
    summary.to_csv(args.summary_out, index=False)
    print(f"Best-per-run summary ({len(summary)} rows) → {args.summary_out}", file=sys.stderr)

    display_cols = ["pdb_id", "run", "model", "cluster_id", "rmsd", "score"]
    display_cols = [c for c in display_cols if c in summary.columns]
    print("\nBest model (lowest RMSD to reference) per completed run:")
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
