#!/usr/bin/env python3
"""
Comparison figures for epiLoRA-constrained vs vanilla (surface-only)
docking, from the CSVs written by plot_capri_acceptable.py:

  capri_acceptable_plot.png  bar chart of CAPRI acceptable-or-better
                             cluster models per antigen
  haddock_score_violin.png   per-antigen HADDOCK score violins with
                             quartile boxes; dots = acceptable-or-better
  capri_combined_plot.pdf    both panels stacked on a shared x-axis
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE = "#3232cd"    # vanilla (surface constrained)
ORANGE = "#AF088F"  # epiLoRA constrained
BLACK = "#000000"
BAR_W = 0.38
ACCEPTABLE_OR_BETTER = {"High", "Medium", "Acceptable"}
CONDS = [("vanilla", BLUE, -BAR_W / 2), ("epilora", ORANGE, BAR_W / 2)]


def order_antigens(counts):
    """Antigens with any acceptable-or-better model first."""
    acc = counts.groupby("pdb_id")["n_acc"].sum()
    has = acc[acc > 0].index.tolist()
    return has + [p for p in acc.index if p not in set(has)]


def draw_bars(ax, counts, antigens, title):
    """Grouped bar chart of acceptable-or-better model counts."""
    x = np.arange(len(antigens))
    for cond, color, xoff in CONDS:
        sub = counts[counts.constraint == cond].set_index("pdb_id")
        vals = sub["n_acc"].reindex(antigens).fillna(0)
        label = "EpiLoRA" if cond == "epilora" else "vanilla"
        bars = ax.bar(x + xoff, vals, BAR_W, color=color, edgecolor="white",
                      linewidth=0.6, label=label, alpha=0.5)
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                        str(int(b.get_height())), ha="center", va="bottom",
                        fontsize=30, color=BLACK)
    ax.set_title(title, fontsize=36, pad=12, color=BLACK)
    ax.set_ylabel("# acceptable", fontsize=28, color=BLACK)
    ax.set_xlim(-0.6, len(antigens) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=24)
    _despine(ax)


def draw_violins(ax, details, antigens, rng):
    """Per-antigen HADDOCK score violins + boxes; dots for
    acceptable-or-better models."""
    x = np.arange(len(antigens))
    for idx, pdb in enumerate(antigens):
        for cond, color, xoff in CONDS:
            sub = details[(details.pdb_id == pdb) & (details.constraint == cond)]
            if sub.empty:
                continue
            scores = sub["haddock_score"].values
            cx = x[idx] + xoff

            parts = ax.violinplot(scores, positions=[cx], widths=BAR_W * 0.9,
                                  showextrema=False, showmedians=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("white")
                pc.set_linewidth(0.6)
                pc.set_alpha(0.5)

            bp = ax.boxplot(scores, positions=[cx], widths=BAR_W * 0.5,
                            patch_artist=False, showfliers=False, zorder=4)
            for element in ("boxes", "whiskers", "caps", "medians"):
                for el in bp[element]:
                    el.set_color(BLACK)
                    el.set_linewidth(1.5)

            acc = sub[sub.capri_quality.isin(ACCEPTABLE_OR_BETTER)]
            if not acc.empty:
                jitter = rng.uniform(-BAR_W * 0.18, BAR_W * 0.18, size=len(acc))
                ax.scatter(cx + jitter, acc["haddock_score"], s=80, c=color,
                           zorder=6)

    ax.set_ylabel("HADDOCK score (lower = better)", fontsize=28, color=BLACK)
    ax.set_xlim(-0.6, len(antigens) - 0.4)
    ax.tick_params(axis="y", labelsize=20)
    _despine(ax)
    ax.legend(handles=violin_legend(), loc="upper right", frameon=False,
              fontsize=22)


def violin_legend():
    return [
        Patch(facecolor=BLUE, edgecolor=BLACK, alpha=0.5, label="vanilla"),
        Patch(facecolor=ORANGE, edgecolor=BLACK, alpha=0.5, label="EpiLoRA"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLACK,
               markeredgecolor="white", markersize=12,
               label="CAPRI acceptable-or-better"),
    ]


def label_x(ax, antigens):
    ax.set_xticks(np.arange(len(antigens)))
    ax.set_xticklabels(antigens, fontsize=45, fontweight="bold")
    ax.tick_params(axis="x", length=0)


def _despine(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_color(BLACK)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default="results/capri_acceptable_counts.csv")
    ap.add_argument("--details", default="results/capri_model_details.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    counts = pd.read_csv(args.counts)
    counts["n_acc"] = counts.n_good + counts.n_acceptable
    details = pd.read_csv(args.details)
    details["haddock_score"] = pd.to_numeric(details["haddock_score"],
                                             errors="coerce")
    details = details.dropna(subset=["haddock_score"])

    antigens = order_antigens(counts)
    width = max(20, 2.8 * len(antigens))
    bar_title = ("CAPRI acceptable-or-better models: vanilla vs "
                 "EpiLoRA-constrained\n"
                 "(antigens with 5/5 runs complete in both conditions)")

    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "text.color": BLACK, "axes.labelcolor": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "axes.edgecolor": BLACK,
    })

    # Bar chart
    fig, ax = plt.subplots(figsize=(width, 7.5))
    draw_bars(ax, counts, antigens, bar_title)
    label_x(ax, antigens)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    fig.savefig(outdir / "capri_acceptable_plot.png", dpi=150)
    print(f"Wrote {outdir / 'capri_acceptable_plot.png'}", file=sys.stderr)

    # Violin plot
    fig, ax = plt.subplots(figsize=(width, 14))
    draw_violins(ax, details, antigens, np.random.default_rng(42))
    ax.set_title("HADDOCK score distribution: vanilla vs "
                 "EpiLoRA-constrained\n(dots = CAPRI acceptable-or-better)",
                 fontsize=36, pad=12, color=BLACK)
    label_x(ax, antigens)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(outdir / "haddock_score_violin.png", dpi=150)
    print(f"Wrote {outdir / 'haddock_score_violin.png'}", file=sys.stderr)

    # Combined: bars on top, violins below, shared x-axis
    fig, (ax_bar, ax_vio) = plt.subplots(
        2, 1, sharex=True, figsize=(width, 16),
        gridspec_kw={"height_ratios": [1, 1.4]})
    draw_bars(ax_bar, counts, antigens, bar_title)
    draw_violins(ax_vio, details, antigens, np.random.default_rng(42))
    label_x(ax_vio, antigens)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    fig.savefig(outdir / "capri_combined_plot.pdf")
    print(f"Wrote {outdir / 'capri_combined_plot.pdf'}", file=sys.stderr)


if __name__ == "__main__":
    main()
