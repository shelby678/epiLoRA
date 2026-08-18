"""Antisymmetric substitution heatmap from an existing delta_matrix.npy (the
per-(sub,orig) mean Δ matrix written by substitution_scan.py).

"Symmetric, pretend we don't know the original amino acid" -- collapse the
directional (sub, orig) matrix into an unordered {X, Y} matrix by taking the
antisymmetric part:

    A[X, Y] = (M[X<-Y] - M[Y<-X]) / 2
            = (mean d(X substituted at Y-sites) - mean d(Y substituted at X-sites)) / 2

By construction A = -Aᵀ  (upper triangle = -lower triangle), matching the
physical intuition that "putting X where Y was" is the opposite of "putting Y
where X was". Sign reads as: A[X,Y] > 0  ⇔  residue X is more epitope-favoring
than Y  (substituting X at Y-sites boosts epitope prob more than the reverse).
The diagonal is exactly 0 (X<->X is a no-op).

The matrix is *mostly* antisymmetric (132/190 pairs opposite-sign, antisymmetric
part 1.83x larger than the symmetric part); the residual symmetric component
(M+Mᵀ)/2 comes from ESM2 being a context-dependent transformer (the effect of
a substitution depends on its neighbourhood, so the two directions don't
perfectly cancel). This script drops that residual and keeps only the
antisymmetric part -- the directional signal the user actually wants.

The displayed matrix is Aᵀ, so rows = original, columns = substitution ->
x-axis = Substitution, y-axis = Original. (A is indexed [sub, orig]; Aᵀ
puts orig on rows and sub on columns for the requested axis layout.)

Output:
  - antisymmetric_matrix.csv         20x20 in BLOSUM order
  - antisymmetric_heatmap_upper.png   upper-triangular heatmap, no cell
                                      numbers, no grid lines,
                                      x=Substitution, y=Original,
                                      #FF00BD=increase, white=0, #0035FF=decrease)
  - antisymmetric_heatmap_full.png    full antisymmetric matrix (diagonal=0,
                                      upper = -lower)

BLOSUM order: ARNDCQEGHILKMFPSTWYV (the canonical BLOSUM62 axis order).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"  # where substitution_scan.py writes delta_matrix.npy

# Source matrix is in alphabetical order (ACDEFGHIKLMNPQRSTVWY); BLOSUM62
# uses the canonical ARNDCQEGHILKMFPSTWYV order, which is what we render.
ALPHABETICAL = "ACDEFGHIKLMNPQRSTVWY"
BLOSUM = "ARNDCQEGHILKMFPSTWYV"
# index in BLOSUM-order matrix for each alphabetical-position residue
REORDER = [BLOSUM.index(a) for a in ALPHABETICAL]


def main() -> None:
    M = np.load(RESULTS / "delta_matrix.npy")  # rows=sub, cols=orig, alphabetical
    if M.shape != (20, 20):
        raise SystemExit(f"expected 20x20 delta_matrix.npy, got {M.shape}")

    # Antisymmetric part: A[i=sub, j=orig] = (M[i,j] - M[j,i]) / 2.
    # By construction A = -Aᵀ (upper triangle = -lower triangle), and the
    # diagonal is exactly 0. This is the directional signal -- "X is more
    # epitope-favoring than Y" -- with the (smaller) symmetric residual
    # (M+Mᵀ)/2 dropped.
    A = (M - M.T) / 2.0

    # Reorder rows and cols from alphabetical into BLOSUM order.
    A_blosum = A[np.ix_(REORDER, REORDER)]
    # Transpose for display so rows=original, cols=substitution ->
    # x=Substitution, y=Original (as requested).
    D = A_blosum.T  # rows=orig, cols=sub

    # ---- save CSV (full 20x20, BLOSUM order, diagonal blank) ---------------
    csv_path = RESULTS / "antisymmetric_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orig\\sub"] + list(BLOSUM))
        for i, a in enumerate(BLOSUM):
            row = [a]
            for j in range(20):
                if i == j:
                    row.append("")  # diagonal: no-op
                else:
                    row.append(f"{D[i, j]:+.6f}")
            w.writerow(row)
    print(f"wrote {csv_path}")

    # ---- heatmaps: upper-triangular + full ---------------------------------
    # Custom diverging colormap centred at 0: #FF00BD = increase, white = 0,
    # #0035FF = decrease. Symmetric limits so 0 sits at the colour midpoint.
    # No cell-number annotations, no grid lines. D is antisymmetric, so
    # upper triangle = -lower triangle (the sign of one half is the negation
    # of the other); the upper-triangular plot shows one half, the full plot
    # shows both.
    # ---- custom diverging colormap -----------------------------------------
    # Positive (increase) -> #FF00DC (hot pink/magenta), 0 -> white,
    # negative (decrease) -> #0043AA (deep blue). Symmetric limits so 0 sits
    # at the colour midpoint. Built from the two anchor colours with white in
    # the middle via LinearSegmentedColormap.
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    pos = to_rgb("#FF00BD")
    mid = (1.0, 1.0, 1.0)
    neg = to_rgb("#0035FF")
    cmap = LinearSegmentedColormap.from_list(
        "epi_delta", [neg, mid, pos], N=256
    )

    abs_max = np.nanmax(np.abs(D))
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 1.0

    def draw(mask_fn, out_name: Path, subtitle: str) -> None:
        arr = np.ma.masked_invalid(D).copy()
        arr.mask = arr.mask | mask_fn()
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        ax.imshow(arr, cmap=cmap, vmin=-abs_max, vmax=abs_max,
                  interpolation="nearest")
        ax.set_xticks(range(20)); ax.set_xticklabels(list(BLOSUM))
        ax.set_yticks(range(20)); ax.set_yticklabels(list(BLOSUM))
        ax.set_xlabel("Substitution")
        ax.set_ylabel("Original")
        ax.set_title(
            "Mean Δ epitope probability on eval-set epitope sites (antisymmetric)\n"
            f"{subtitle}  (BLOSUM order, ESM2 ensemble, "
            f"n = {20*19} directed residue pairs)"
        )
        fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04,
                     label="Δ epitope probability  A[X,Y] = (M[X←Y] − M[Y←X]) / 2")
        fig.tight_layout()
        fig.savefig(out_name, dpi=200)
        plt.close(fig)
        print(f"wrote {out_name}")

    # Upper triangular: keep i < j (strict). Mask lower + diagonal.
    draw(lambda: np.tril(np.ones((20, 20), dtype=bool), k=0),
        RESULTS / "antisymmetric_heatmap_upper.png",
        "upper triangle only")
    # Full antisymmetric matrix: diagonal is 0 (no-op) -> masked as blank.
    draw(lambda: np.eye(20, dtype=bool),
        RESULTS / "antisymmetric_heatmap_full.png",
        "full matrix")

    print(f"antisymmetric matrix range: [{np.nanmin(D):+.4f}, "
          f"{np.nanmax(D):+.4f}], max |Δ| = {abs_max:.4f}")


if __name__ == "__main__":
    main()
