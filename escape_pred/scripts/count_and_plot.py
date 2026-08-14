#!/usr/bin/env python3
"""Count descendants of each mutation in the UShER MAT and plot.

Reads the JSON exported by ``matUtils extract -j``, traverses the tree, and for
every mutation on every branch records the number of descendant leaves below
that branch.  If a mutation appears on multiple branches (homoplasy), its
total descendant count is the sum across all occurrences.

Outputs
-------
results/descendants.tsv        mutation  occurrences  total_descendants  max_descendants
results/descendants_plot.png   bar chart (top-200, most→least descendants)
results/top_mutations.tsv      top-N mutations with aa annotation

Usage
-----
    python count_and_plot.py --json work/tree.json --subsample work/subsample.tsv
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    KNOWN_ESCAPE_AAs,
    build_aa_map,
    nt_pos_of_mut,
)


def _compute_leaf_counts(root: dict) -> dict[int, int]:
    """Bottom-up (post-order) iterative computation of leaf counts."""
    leaf_counts: dict[int, int] = {}
    stack: list[tuple[dict, bool]] = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            children = node.get("children")
            if not children:
                leaf_counts[id(node)] = 1
            else:
                leaf_counts[id(node)] = sum(leaf_counts[id(c)]
                                            for c in children)
        else:
            stack.append((node, True))
            for c in node.get("children", []):
                stack.append((c, False))
    return leaf_counts


def traverse(root: dict, mut_desc: dict) -> None:
    """Iterative DFS; for each mutation on a branch, record descendant count."""
    leaf_counts = _compute_leaf_counts(root)
    stack: list[dict] = [root]
    while stack:
        node = stack.pop()
        n_desc = leaf_counts[id(node)]
        muts = node.get("branch_attrs", {}).get("mutations", {}).get("nuc", [])
        for m in muts:
            mut_desc[m]["occurrences"] += 1
            mut_desc[m]["total_desc"] += n_desc
            mut_desc[m]["max_desc"] = max(mut_desc[m]["max_desc"], n_desc)
        for c in node.get("children", []):
            stack.append(c)


def load_tree(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)["tree"]


def lookup_aa(nt_mut: str, pos_to_aa: dict[int, str]) -> str | None:
    pos = nt_pos_of_mut(nt_mut)
    if pos is not None:
        return pos_to_aa.get(pos)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=Path("work/tree.json"))
    p.add_argument("--subsample", type=Path,
                   default=Path("work/subsample.tsv"))
    p.add_argument("--ref", type=Path,
                   default=Path("data/sars-cov2/NC_045512.2.fasta"),
                   help="reference genome (for accurate nt→aa mapping)")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--top", type=int, default=50, help="label top-N bars")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[analyze] loading JSON tree …", flush=True)
    tree = load_tree(args.json)

    print("[analyze] counting descendants …", flush=True)
    mut_desc: dict[str, dict] = defaultdict(
        lambda: {"occurrences": 0, "total_desc": 0, "max_desc": 0})
    traverse(tree, mut_desc)

    rows = []
    for m, d in mut_desc.items():
        rows.append({"mutation": m, "occurrences": d["occurrences"],
                     "total_descendants": d["total_desc"],
                     "max_descendants": d["max_desc"]})
    df = pd.DataFrame(rows).sort_values("total_descendants", ascending=False) \
                            .reset_index(drop=True)
    df.to_csv(args.outdir / "descendants.tsv", sep="\t", index=False)
    print(f"[analyze] {len(df)} mutations, "
          f"top: {df.iloc[0]['mutation']} ({int(df.iloc[0]['total_descendants'])})",
          flush=True)

    # AA annotation
    ref_seq = None
    if args.ref.exists():
        from utils import load_reference
        ref_seq = load_reference(args.ref)
    pos_to_aa = build_aa_map(args.subsample, ref_seq)
    df["aa_mutation"] = df["mutation"].apply(lambda m: lookup_aa(m, pos_to_aa))
    df["is_spike"] = df["aa_mutation"].str.startswith("S:", na=False)
    df["is_escape"] = df["aa_mutation"].isin(KNOWN_ESCAPE_AAs)
    df.head(args.top * 3).to_csv(
        args.outdir / "top_mutations.tsv", sep="\t", index=False)

    # ---- Plot: top-200, most descendants on left → least on right ----
    n_show = min(len(df), 200)
    top = df.head(n_show).reset_index(drop=True)
    colors = [
        "#d62728" if r["is_escape"] else
        "#ff7f0e" if r["is_spike"] else
        "#1f77b4"
        for _, r in top.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(max(14, n_show * 0.06), 8))
    ax.bar(range(n_show), top["total_descendants"], color=colors,
           width=0.9, edgecolor="none")
    ax.set_xlabel("Mutation rank (most descendants ← → least)", fontsize=12)
    ax.set_ylabel("Number of descendant samples (total, all occurrences)",
                  fontsize=12)
    ax.set_title("SARS-CoV-2 whole-genome mutations ranked by tree descendants\n"
                 "(more descendants ⇒ more widely inherited ⇒ likely selected)",
                 fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, n_show - 0.5)

    n_label = min(args.top, n_show)
    for i in range(n_label):
        label = top.loc[i, "mutation"]
        aa = top.loc[i, "aa_mutation"]
        if isinstance(aa, str) and aa == aa:
            label = f"{label}→{aa}"
        ax.annotate(label, (i, top.loc[i, "total_descendants"]),
                    textcoords="offset points", xytext=(0, 4),
                    fontsize=6, rotation=90, ha="center",
                    color="#d62728" if top.loc[i, "is_escape"] else "black")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#d62728", label="Known immune-escape / adaptation"),
        Patch(color="#ff7f0e", label="Spike (S) mutation"),
        Patch(color="#1f77b4", label="Other genomic mutation"),
    ], loc="upper right")

    fig.tight_layout()
    out_png = args.outdir / "descendants_plot.png"
    fig.savefig(out_png, dpi=180)
    print(f"[analyze] plot → {out_png}  (top {n_show} of {len(df)} mutations)",
          flush=True)

    print("\nTop 20 mutations by descendants:")
    print(df.head(20)[["mutation", "aa_mutation", "occurrences",
                       "total_descendants", "is_escape"]].to_string(index=False))


if __name__ == "__main__":
    main()
