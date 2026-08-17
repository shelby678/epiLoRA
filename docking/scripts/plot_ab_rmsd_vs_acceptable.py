#!/usr/bin/env python3
"""Antibody apo-vs-ground-truth RMSD (whole-Fv and CDR-only) vs fraction of
CAPRI-Acceptable+ models, one point per antigen (dname)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2a78d6"
TEXT_SECONDARY = "#52514e"
GRAY = "#9a9990"


def main():
    ab = pd.read_csv("scratch/ab_apo_gt_rmsd_by_dname.csv").set_index("pdb_id")

    df = pd.read_csv("capri_results.csv")
    acc = df["quality"].isin(["High", "Medium", "Acceptable"])
    df = df.assign(acc=acc)
    g = df.groupby("pdb_id").agg(n_models=("acc", "size"), n_acc=("acc", "sum"))
    g["frac_acc"] = g["n_acc"] / g["n_models"]
    out = g.join(ab)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, col, label in [(axes[0], "fv_rmsd", "Whole-Fv RMSD (Å)"),
                           (axes[1], "cdr_rmsd", "CDR-only RMSD (Å)")]:
        has_acc = out["n_acc"] > 0
        ax.scatter(out.loc[~has_acc, col], out.loc[~has_acc, "frac_acc"],
                   s=60, color=GRAY, edgecolors="white", linewidths=0.6, zorder=2,
                   label="No acceptable+ models")
        ax.scatter(out.loc[has_acc, col], out.loc[has_acc, "frac_acc"],
                   s=60, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3,
                   label="Has acceptable+ models")
        for pid, row in out.iterrows():
            if row["n_acc"] > 0:
                ax.annotate(pid, (row[col], row["frac_acc"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=9,
                            color=TEXT_SECONDARY)
        corr = out[col].corr(out["frac_acc"])
        ax.set_xlabel(f"Mean apo-vs-ground-truth {label}", fontsize=11)
        ax.set_ylabel("Fraction of models CAPRI Acceptable-or-better", fontsize=11)
        ax.set_title(f"{label}\nPearson r = {corr:.2f}", fontsize=12)
        ax.grid(color=GRAY, alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        print(f"{label}: Pearson r = {corr:.3f}")

    fig.suptitle("Antibody prediction error vs docking success (epiLoRA-constrained)", fontsize=14)
    plt.tight_layout()
    plt.savefig("scratch/ab_rmsd_vs_acceptable.png", dpi=150)
    print("Wrote scratch/ab_rmsd_vs_acceptable.png")

    print("\n" + out.sort_values("fv_rmsd").to_string())


if __name__ == "__main__":
    main()
