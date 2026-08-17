#!/usr/bin/env python3
"""
Per-antigen dumbbell plot: mean best-of-selection DockQ (vanilla vs
epiLoRA-constrained), one row per antigen (dname), sorted by mean DockQ so
easier/harder antigens are visually grouped.

Reads scripts/compare_vanilla_vs_epitope.py's output (scratch/comparison_per_run.csv).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"    # categorical slot 1 -- vanilla
ORANGE = "#eb6834"  # categorical slot 2 -- epiLoRA-constrained
GRAY = "#9a9990"    # recessive connecting line / gridlines
TEXT_SECONDARY = "#52514e"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", default="scratch/comparison_per_run.csv")
    p.add_argument("--out-png", default="scratch/dockq_by_antigen.png")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)

    agg = (
        df.groupby("pdb_id")
        .agg(
            vanilla=("vanilla_best_dockq", "mean"),
            epitope=("epitope_best_dockq", "mean"),
            n=("run", "count"),
        )
        .reset_index()
    )
    agg["mean_dockq"] = (agg["vanilla"] + agg["epitope"]) / 2
    agg = agg.sort_values("mean_dockq", ascending=True).reset_index(drop=True)

    n_ag = len(agg)
    fig, ax = plt.subplots(figsize=(8, 0.34 * n_ag + 1.5))
    y = np.arange(n_ag)

    ax.hlines(y, agg["vanilla"], agg["epitope"], color=GRAY, linewidth=1.5, zorder=1)
    ax.scatter(agg["vanilla"], y, s=70, color=BLUE, label="Vanilla", zorder=2,
               edgecolors="white", linewidths=0.8)
    ax.scatter(agg["epitope"], y, s=70, color=ORANGE, label="epiLoRA-constrained", zorder=2,
               edgecolors="white", linewidths=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{pid}  (n={n})" for pid, n in zip(agg["pdb_id"], agg["n"])],
                        fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("Mean best-of-selection DockQ per antigen (higher = better, easier target)",
                   fontsize=11)
    ax.set_title("Docking difficulty by antigen: vanilla vs epiLoRA-constrained HADDOCK",
                  fontsize=13, pad=12)

    ax.grid(axis="x", color=GRAY, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)

    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"Wrote {args.out_png}")

    print("\nAntigens sorted easiest -> hardest (mean of both conditions):")
    for _, r in agg[::-1].iterrows():
        print(f"  {r['pdb_id']:8s} n={int(r['n']):2d}  vanilla={r['vanilla']:.3f}  "
              f"epitope={r['epitope']:.3f}  delta={r['epitope']-r['vanilla']:+.3f}")


if __name__ == "__main__":
    main()
