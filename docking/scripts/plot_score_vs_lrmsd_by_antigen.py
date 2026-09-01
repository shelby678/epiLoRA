#!/usr/bin/env python3
"""
HADDOCK score vs ligand RMSD, faceted one panel per antigen (epiLoRA-
constrained runs only), CAPRI-Acceptable-or-better models highlighted in
yellow against muted gray for everything else.
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

YELLOW = "#eda100"      # highlight: CAPRI Acceptable+
YELLOW_EDGE = "#6b4f00"  # dark edge for contrast (yellow sits <3:1 on light surface)
GRAY = "#c9c8c0"        # muted: everything else
GRAY_EDGE = "#9a9990"
LINE_GRAY = "#9a9990"
TEXT_SECONDARY = "#52514e"

ACCEPTABLE_OR_BETTER = {"High", "Medium", "Acceptable"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", default="results/capri_results.csv")
    p.add_argument("--out-png", default="scratch/score_vs_lrmsd_by_antigen.png")
    p.add_argument("--ncols", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.in_csv).dropna(subset=["score", "lrmsd"])
    df = df.assign(is_gt=df["run"].str.endswith("_gt"))
    antigens = sorted(df["pdb_id"].unique())
    n = len(antigens)
    ncols = args.ncols
    nrows = math.ceil(n / ncols)

    x_max = df["lrmsd"].quantile(0.99) * 1.05
    y_min, y_max = df["score"].quantile([0.01, 0.99])
    y_pad = (y_max - y_min) * 0.08

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows),
                              sharex=True, sharey=True)
    axes = axes.flatten()

    for i, pid in enumerate(antigens):
        ax = axes[i]
        sub = df[df["pdb_id"] == pid]
        is_acc = sub["quality"].isin(ACCEPTABLE_OR_BETTER)

        for is_gt, marker, acc_size, other_size in [(False, "o", 34, 16), (True, "D", 30, 14)]:
            grp = sub[sub["is_gt"] == is_gt]
            grp_acc = is_acc.loc[grp.index]
            ax.scatter(grp.loc[~grp_acc, "lrmsd"], grp.loc[~grp_acc, "score"],
                       s=other_size, marker=marker, color=GRAY, edgecolors=GRAY_EDGE,
                       linewidths=0.4, zorder=1)
            ax.scatter(grp.loc[grp_acc, "lrmsd"], grp.loc[grp_acc, "score"],
                       s=acc_size, marker=marker, color=YELLOW, edgecolors=YELLOW_EDGE,
                       linewidths=0.6, zorder=2)

        for x, ls in [(5.0, "--"), (10.0, ":")]:
            ax.axvline(x, color=LINE_GRAY, linewidth=0.8, linestyle=ls, alpha=0.6, zorder=0)

        n_acc = int(is_acc.sum())
        ax.set_title(f"{pid}  ({n_acc}/{len(sub)} acc+)", fontsize=9.5, color=TEXT_SECONDARY)
        ax.grid(color=LINE_GRAY, alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=TEXT_SECONDARY)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    ax0 = axes[0]
    ax0.set_xlim(0, x_max)
    ax0.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.supxlabel("Ligand RMSD / Å  (lower = better)", fontsize=12)
    fig.supylabel("HADDOCK score  (lower = better)", fontsize=12)
    fig.suptitle("HADDOCK score vs ligand RMSD by antigen (epiLoRA-constrained)\n"
                 "CAPRI Acceptable-or-better highlighted in yellow  ·  "
                 "◆ = ground-truth (bound) antigen, ● = apo antigen",
                 fontsize=14, y=0.998)

    color_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=YELLOW,
                   markeredgecolor=YELLOW_EDGE, markersize=8, label="CAPRI Acceptable+"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY,
                   markeredgecolor=GRAY_EDGE, markersize=8, label="Incorrect"),
    ]
    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=TEXT_SECONDARY,
                   markeredgecolor="white", markersize=8, label="Apo antigen (ag)"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT_SECONDARY,
                   markeredgecolor="white", markersize=7, label="Ground-truth antigen (ag_gt)"),
    ]
    leg1 = fig.legend(handles=color_handles, loc="upper right", frameon=False, fontsize=10,
                       bbox_to_anchor=(0.995, 0.995), title="Quality")
    fig.add_artist(leg1)
    fig.legend(handles=shape_handles, loc="upper right", frameon=False, fontsize=10,
               bbox_to_anchor=(0.995, 0.935), title="Antigen")

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.90])
    plt.savefig(args.out_png, dpi=150)
    print(f"Wrote {args.out_png}  ({n} antigen panels)")


if __name__ == "__main__":
    main()
