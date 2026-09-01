#!/usr/bin/env python3
"""
Box plot of per-run HADDOCK3 wall-clock runtime (hours) for vanilla vs
epiLoRA-constrained docking, over the same antigen set as
plot_capri_acceptable.py (those with 5/5 runs complete in both
conditions).

Runtimes are parsed from "This HADDOCK3 run took: XhYmZs" log lines.
Only (pdb_id, run) pairs with a runtime in BOTH conditions are kept, so
the comparison is matched.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#3232cd"    # vanilla
ORANGE = "#AF088F"  # epiLoRA
BLACK = "#000000"

TOOK = re.compile(r"This HADDOCK3 run took:\s*(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")
LOG_DIRS = {"vanilla": "logs_vanilla", "epilora": "logs"}


def runtime_hours(log_path):
    """Hours from a 'This HADDOCK3 run took: XhYmZs' line, or None."""
    m = TOOK.search(Path(log_path).read_text(errors="ignore"))
    if not m:
        return None
    h, mi, s = (int(g or 0) for g in m.groups())
    return h + mi / 60 + s / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default="results/capri_acceptable_counts.csv",
                    help="counts CSV defining the antigen set")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--logs-vanilla-dir", default="logs_vanilla")
    ap.add_argument("--out-csv", default="results/runtime_by_run.csv")
    ap.add_argument("--out-plot", default="results/runtime_boxplot.png")
    args = ap.parse_args()

    log_dirs = {"vanilla": args.logs_vanilla_dir, "epilora": args.logs_dir}
    antigens = sorted(pd.read_csv(args.counts)["pdb_id"].unique())

    rows = []
    for pdb in antigens:
        for cond, ldir in log_dirs.items():
            for i in range(1, 6):
                log = Path(ldir) / pdb / f"ab_{i}_vs_ag.log"
                rows.append({
                    "pdb_id": pdb,
                    "constraint": cond,
                    "run": f"ab_{i}_vs_ag",
                    "hours": runtime_hours(log) if log.exists() else None,
                })
    df = pd.DataFrame(rows).dropna(subset=["hours"])

    # Keep only (pdb_id, run) pairs present in BOTH conditions (matched set)
    ncond = df.groupby(["pdb_id", "run"])["constraint"].nunique()
    df = (df.set_index(["pdb_id", "run"])
            .loc[ncond[ncond == 2].index]
            .reset_index())

    for out in (args.out_csv, args.out_plot):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(df)} rows, {len(df) // 2} matched pairs)",
          file=sys.stderr)

    van = df[df.constraint == "vanilla"]["hours"].values
    epi = df[df.constraint == "epilora"]["hours"].values

    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "text.color": BLACK, "axes.labelcolor": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "axes.edgecolor": BLACK,
    })

    fig, ax = plt.subplots(figsize=(6, 7))
    bp = ax.boxplot(
        [van, epi],
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color=BLACK, linewidth=1.5),
        whiskerprops=dict(color=BLACK, linewidth=1),
        capprops=dict(color=BLACK, linewidth=1),
        flierprops=dict(marker="o", markerfacecolor=BLACK, markersize=4,
                        markeredgecolor="none", alpha=0.6),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor=BLACK, markersize=6),
    )
    for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_edgecolor(BLACK)
        patch.set_linewidth(1)
        patch.set_alpha(0.85)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["vanilla", "EpiLoRA"], fontsize=15, fontweight="bold")
    ax.tick_params(length=0)
    ax.set_yticks([])
    ax.set_ylabel("Runtime per run / hours", fontsize=15, color=BLACK)
    ax.set_title("HADDOCK3 runtime per run: vanilla vs EpiLoRA-constrained\n"
                 "(matched runs; antigens with 5/5 complete in both)",
                 fontsize=16, pad=12, color=BLACK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_color(BLACK)

    for pos, vals in [(1, van), (2, epi)]:
        ax.text(pos, vals.mean(), f"  mean={vals.mean():.2f}h",
                fontsize=11, color=BLACK, va="center", ha="left")

    plt.tight_layout()
    fig.savefig(args.out_plot, dpi=150)
    print(f"Wrote {args.out_plot}", file=sys.stderr)

    print(f"vanilla: n={len(van)}  mean={van.mean():.3f}h  "
          f"median={np.median(van):.3f}h", file=sys.stderr)
    print(f"epilora: n={len(epi)}  mean={epi.mean():.3f}h  "
          f"median={np.median(epi):.3f}h", file=sys.stderr)


if __name__ == "__main__":
    main()
