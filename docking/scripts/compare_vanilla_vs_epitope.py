#!/usr/bin/env python3
"""
Compare docking quality between vanilla HADDOCK (CDR-active / full-surface
passive) and epiLoRA-constrained HADDOCK (CDR-active / surface∩epitope>20%
passive), joined on (pdb_id, run).

For each run, summarizes two ways:
  best  -- max DockQ among the seletopclusts top models actually produced
           ("could the docking find a good pose at all")
  top1  -- the model with the best (lowest) HADDOCK score
           ("would score-based ranking have picked a good pose")

Only runs present in both CSVs are compared (fair, matched set).
"""
import argparse
import sys
import pandas as pd

CAPRI_ORDER = ["High", "Medium", "Acceptable", "Incorrect"]


def per_run_best(df):
    idx = df.groupby(["pdb_id", "run"])["dockq"].idxmax()
    return df.loc[idx].set_index(["pdb_id", "run"])


def per_run_top1(df):
    d = df.dropna(subset=["score"])
    idx = d.groupby(["pdb_id", "run"])["score"].idxmin()
    return d.loc[idx].set_index(["pdb_id", "run"])


def quality_dist(df):
    return df["quality"].value_counts().reindex(CAPRI_ORDER, fill_value=0)


def summarize(label, vanilla, epitope, key):
    common = vanilla.index.intersection(epitope.index)
    v = vanilla.loc[common]
    e = epitope.loc[common]
    print(f"\n=== {label} (n={len(common)} matched runs) ===")
    print(f"{'':12s} {'vanilla':>10s} {'epitope':>10s}")
    for metric in ["dockq", "irmsd", "lrmsd", "fnat"]:
        print(f"{metric:12s} {v[metric].mean():10.3f} {e[metric].mean():10.3f}   (mean)")
        print(f"{'':12s} {v[metric].median():10.3f} {e[metric].median():10.3f}   (median)")

    print(f"\nCAPRI quality distribution ({label}):")
    vq, eq = quality_dist(v), quality_dist(e)
    print(f"{'':12s} {'vanilla':>10s} {'epitope':>10s}")
    for q in CAPRI_ORDER:
        print(f"{q:12s} {vq[q]:10d} {eq[q]:10d}")

    delta = e["dockq"] - v["dockq"]
    print(f"\nPer-run DockQ delta (epitope - vanilla), {label}:")
    print(f"  epitope better (>+0.01): {(delta > 0.01).sum()}")
    print(f"  vanilla better (<-0.01): {(delta < -0.01).sum()}")
    print(f"  ~tied:                   {(delta.abs() <= 0.01).sum()}")
    print(f"  mean delta: {delta.mean():+.4f}   median delta: {delta.median():+.4f}")

    return common, v, e, delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla-csv", default="scratch/vanilla_capri_results.csv")
    p.add_argument("--epitope-csv", default="capri_results.csv")
    p.add_argument("--out-csv", default="scratch/comparison_per_run.csv")
    args = p.parse_args()

    vanilla = pd.read_csv(args.vanilla_csv)
    epitope = pd.read_csv(args.epitope_csv)

    v_best, e_best = per_run_best(vanilla), per_run_best(epitope)
    v_top1, e_top1 = per_run_top1(vanilla), per_run_top1(epitope)

    common_best, vb, eb, delta_best = summarize("best-of-selection", v_best, e_best, "best")
    common_top1, vt, et, delta_top1 = summarize("top1-by-score", v_top1, e_top1, "top1")

    out = pd.DataFrame({
        "pdb_id":            [k[0] for k in common_best],
        "run":               [k[1] for k in common_best],
        "vanilla_best_dockq": vb["dockq"].values,
        "epitope_best_dockq": eb["dockq"].values,
        "delta_best_dockq":   delta_best.values,
        "vanilla_best_quality": vb["quality"].values,
        "epitope_best_quality": eb["quality"].values,
    })
    out.to_csv(args.out_csv, index=False)
    print(f"\nWrote per-run comparison -> {args.out_csv}")

    missing_v = set(epitope.groupby(["pdb_id", "run"]).groups) - set(vanilla.groupby(["pdb_id", "run"]).groups)
    missing_e = set(vanilla.groupby(["pdb_id", "run"]).groups) - set(epitope.groupby(["pdb_id", "run"]).groups)
    if missing_v or missing_e:
        print(f"\n(unmatched: {len(missing_v)} epitope-only runs, {len(missing_e)} vanilla-only runs -- excluded from comparison)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
