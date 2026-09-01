#!/usr/bin/env python3
"""
Recompute CAPRI quality (capri_analysis.compute_capri, independent of
HADDOCK's own caprieval) for every cluster model of every completed run,
for epiLoRA-constrained (runs/) vs surface-only vanilla (runs_vanilla/)
docking.

Only antigens with all 5 ab runs clustered in BOTH conditions are
included. Writes per-antigen counts and per-model details CSVs for
downstream plotting (plot_capri_figures.py, plot_runtime_boxplot.py).
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

from capri_analysis import CrystalRef, compute_capri, read_scores

GOOD = {"High", "Medium"}
ACCEPTABLE_ONLY = {"Acceptable"}


def complete_antigens(runs_dir, vanilla_dir, crystal_dir):
    """PDB ids with all 5 ab runs clustered in both conditions and a
    crystal reference available."""
    def clustered(root, pdb):
        return all(
            any((root / pdb / f"ab_{i}_vs_ag" / "haddock_out" /
                 "10_seletopclusts").glob("cluster_*.pdb.gz"))
            for i in range(1, 6))

    return sorted(
        p.name for p in runs_dir.iterdir()
        if p.is_dir()
        and (crystal_dir / f"{p.name}.pdb").exists()
        and clustered(runs_dir, p.name)
        and clustered(vanilla_dir, p.name))


def collect_models(pdb, constraint, runs_dir, crystal_dir, ab_dir):
    """(summary_row, model_rows) for one antigen and condition."""
    ref = CrystalRef(crystal_dir / f"{pdb}.pdb",
                     ab_dir / pdb / "ab_1.pdb", ab_dir / pdb / "ag.pdb")
    n_good = n_acc = n_tot = 0
    details = []
    for i in range(1, 6):
        run_dir = runs_dir / pdb / f"ab_{i}_vs_ag"
        sel_dir = run_dir / "haddock_out" / "10_seletopclusts"
        ss_path = run_dir / "haddock_out" / "11_caprieval" / "capri_ss.tsv"
        scores = read_scores(ss_path) if ss_path.exists() else {}
        for model_gz in sorted(sel_dir.glob("cluster_*.pdb.gz")):
            stem = model_gz.stem.replace(".pdb", "")
            lrmsd, irmsd, fnat, dq, quality = compute_capri(model_gz, ref)
            n_tot += 1
            if quality in GOOD:
                n_good += 1
            elif quality in ACCEPTABLE_ONLY:
                n_acc += 1
            details.append({
                "pdb_id": pdb,
                "constraint": constraint,
                "ab_run": i,
                "model": stem,
                "haddock_score": scores.get(stem),
                "lrmsd": lrmsd,
                "irmsd": irmsd,
                "fnat": fnat,
                "dockq": dq,
                "capri_quality": quality if quality else "Unknown",
            })
    summary = {
        "pdb_id": pdb,
        "constraint": constraint,
        "n_good": n_good,
        "n_acceptable": n_acc,
        "n_total": n_tot,
    }
    print(f"  {pdb:6s} {constraint:8s}: good={n_good:4d}  "
          f"acceptable={n_acc:4d}  total={n_tot:4d}", file=sys.stderr)
    return summary, details


def _worker(args):
    return collect_models(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--vanilla-dir", default="runs_vanilla")
    ap.add_argument("--crystal-dir", default="input_pdbs/selected_32_pdbs")
    ap.add_argument("--ab-dir", default="input_pdbs")
    ap.add_argument("--out-counts", default="results/capri_acceptable_counts.csv")
    ap.add_argument("--out-details", default="results/capri_model_details.csv")
    ap.add_argument("--jobs", "-j", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    runs_dir, vanilla_dir = Path(args.runs_dir), Path(args.vanilla_dir)
    crystal_dir, ab_dir = Path(args.crystal_dir), Path(args.ab_dir)

    antigens = complete_antigens(runs_dir, vanilla_dir, crystal_dir)
    print(f"Detected {len(antigens)} antigens with 5/5 runs in both "
          f"conditions: {antigens}", file=sys.stderr)

    conditions = [("vanilla", vanilla_dir), ("epilora", runs_dir)]
    tasks = [(pdb, cond, rdir, crystal_dir, ab_dir)
             for pdb in antigens for cond, rdir in conditions]

    with mp.Pool(min(args.jobs, len(tasks))) as pool:
        results = pool.map(_worker, tasks)

    summaries = [r[0] for r in results]
    details = [d for r in results for d in r[1]]

    for out in (args.out_counts, args.out_details):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out_counts, index=False)
    pd.DataFrame(details).to_csv(args.out_details, index=False)
    print(f"Wrote {args.out_counts} and {args.out_details} "
          f"({len(details)} models)", file=sys.stderr)


if __name__ == "__main__":
    main()
