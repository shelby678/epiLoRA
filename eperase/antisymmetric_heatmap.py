"""Antisymmetric substitution heatmap from an existing delta_matrix.npy (the
per-(sub,orig) mean Δ matrix written by substitution_scan.py).

Collapses the directional (sub, orig) matrix into an unordered {X, Y} matrix
via the antisymmetric part: A[X, Y] = (M[X<-Y] - M[Y<-X]) / 2. A = -Aᵀ, and
the diagonal is 0. Sign reads as: A[X,Y] > 0 means X is more epitope-favoring
than Y.

Displayed matrix is Aᵀ: rows = original, columns = substitution.

Output:
  - antisymmetric_matrix.csv         20x20 in BLOSUM order
  - antisymmetric_heatmap_upper.png   upper-triangular heatmap
  - antisymmetric_heatmap_full.png    full antisymmetric matrix
  - substitution_delta_boxplot.png    box plot of per-substitution Δ

BLOSUM order: ARNDCQEGHILKMFPSTWYV
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

plt.rcParams.update({"font.size": 17.5})

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# delta_matrix.npy is alphabetical (ACDEFGHIKLMNPQRSTVWY); we render BLOSUM62 order.
ALPHABETICAL = "ACDEFGHIKLMNPQRSTVWY"
BLOSUM = "ARNDCQEGHILKMFPSTWYV"
REORDER = [BLOSUM.index(a) for a in ALPHABETICAL]

# Diverging colormap: magenta = increase, white = 0, blue = decrease.
CMAP = LinearSegmentedColormap.from_list(
    "epi_delta", [to_rgb("#3232CD"), (1.0, 1.0, 1.0), to_rgb("#AF088F")], N=256
)


def style_axes(ax) -> None:
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="both", which="both", length=0)
    for s in ("bottom", "left"):
        ax.spines[s].set_visible(False)


def main() -> None:
    M = np.load(RESULTS / "delta_matrix.npy")  # rows=sub, cols=orig, alphabetical
    if M.shape != (20, 20):
        raise SystemExit(f"expected 20x20 delta_matrix.npy, got {M.shape}")

    A = (M - M.T) / 2.0
    A_blosum = A[np.ix_(REORDER, REORDER)]
    D = A_blosum.T  # rows=orig, cols=sub, BLOSUM order

    # ---- CSV ----------------------------------------------------------
    csv_path = RESULTS / "antisymmetric_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orig\\sub"] + list(BLOSUM))
        for i, a in enumerate(BLOSUM):
            row = [a] + ["" if i == j else f"{D[i, j]:+.6f}" for j in range(20)]
            w.writerow(row)
    print(f"wrote {csv_path}")

    # ---- heatmaps: upper-triangular + full -----------------------------
    abs_max = np.nanmax(np.abs(D))
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 1.0

    def draw(mask, out_name: Path) -> None:
        arr = np.ma.masked_invalid(D).copy()
        arr.mask = arr.mask | mask
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        ax.imshow(arr, cmap=CMAP, vmin=-abs_max, vmax=abs_max, interpolation="nearest")
        ax.set_xticks(range(20)); ax.set_xticklabels(list(BLOSUM))
        ax.set_yticks(range(20)); ax.set_yticklabels(list(BLOSUM))
        ax.set_xlabel("Substitution")
        ax.set_ylabel("Original")
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(out_name, dpi=200)
        plt.close(fig)
        print(f"wrote {out_name}")

    draw(np.eye(20, dtype=bool), RESULTS / "antisymmetric_heatmap_full.png")
    draw(np.tril(np.ones((20, 20), dtype=bool)), RESULTS / "antisymmetric_heatmap_upper.png")

    # ---- box plot: per-substitution Δ distribution, boxes coloured by mean
    per_sub = [M[i, :][~np.isnan(M[i, :])] for i in REORDER]  # BLOSUM order
    means = np.array([np.nanmean(p) for p in per_sub])
    abs_max_box = np.nanmax(np.abs(means)) or 1.0
    box_colors = [CMAP((m + abs_max_box) / (2 * abs_max_box)) for m in means]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(
        per_sub, positions=range(20), widths=0.65, patch_artist=True,
        showfliers=False, medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 0.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
    ax.set_xticks(range(20)); ax.set_xticklabels(list(BLOSUM))
    ax.set_xlabel("Substitution")
    ax.set_ylabel("Δ epitope probability")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="both", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_box = RESULTS / "substitution_delta_boxplot.png"
    fig.savefig(out_box, dpi=200)
    plt.close(fig)
    print(f"wrote {out_box}")

    print(f"antisymmetric matrix range: [{np.nanmin(D):+.4f}, "
          f"{np.nanmax(D):+.4f}], max |Δ| = {abs_max:.4f}")


if __name__ == "__main__":
    main()
