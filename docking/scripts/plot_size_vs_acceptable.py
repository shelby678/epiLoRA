#!/usr/bin/env python3
"""Antigen size (mean residue count) vs fraction of CAPRI-Acceptable+ models,
one point per antigen -- checks whether bigger antigens simply fail more."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2a78d6"
TEXT_SECONDARY = "#52514e"
GRAY = "#9a9990"


def main():
    sz = pd.read_csv("scratch/surface_epitope_overlap.csv")
    sz["pdb_id"] = sz["pdb_stem"].str.replace("_ag_gt", "", regex=False).str.replace("_ag", "", regex=False)
    size_by_dname = sz.groupby("pdb_id")["n_residues"].mean().rename("mean_n_residues")

    df = pd.read_csv("capri_results.csv")
    acc = df["quality"].isin(["High", "Medium", "Acceptable"])
    df = df.assign(acc=acc)
    g = df.groupby("pdb_id").agg(n_models=("acc", "size"), n_acc=("acc", "sum"))
    g["frac_acc"] = g["n_acc"] / g["n_models"]
    out = g.join(size_by_dname)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    has_acc = out["n_acc"] > 0
    ax.scatter(out.loc[~has_acc, "mean_n_residues"], out.loc[~has_acc, "frac_acc"],
               s=60, color=GRAY, edgecolors="white", linewidths=0.6, zorder=2,
               label="No acceptable+ models")
    ax.scatter(out.loc[has_acc, "mean_n_residues"], out.loc[has_acc, "frac_acc"],
               s=60, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3,
               label="Has acceptable+ models")

    for pid, row in out.iterrows():
        if row["n_acc"] > 0:
            ax.annotate(pid, (row["mean_n_residues"], row["frac_acc"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9,
                        color=TEXT_SECONDARY)

    corr = out["mean_n_residues"].corr(out["frac_acc"])
    ax.set_xlabel("Mean antigen size (residues)", fontsize=11)
    ax.set_ylabel("Fraction of models CAPRI Acceptable-or-better", fontsize=11)
    ax.set_title(f"Antigen size vs docking success (epiLoRA-constrained)\nPearson r = {corr:.2f}",
                 fontsize=13)
    ax.grid(color=GRAY, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig("scratch/size_vs_acceptable.png", dpi=150)
    print("Wrote scratch/size_vs_acceptable.png")
    print(f"Pearson r (size vs frac acceptable) = {corr:.3f}")


if __name__ == "__main__":
    main()
