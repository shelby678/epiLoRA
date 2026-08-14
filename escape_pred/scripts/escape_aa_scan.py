#!/usr/bin/env python3
"""For each post-April-2021 escape mutation, test all 20 amino acids at the
site and plot epitope probability per amino acid.

Targets are derived **dynamically** from the literature escape-mutation list
(``KNOWN_ESCAPE_AAs`` in utils.py), filtered to those that first appeared in
the subsample after April 2021.  This ensures the script adapts when the
underlying data or parameters change.

WT is colored green, the escape mutation is colored red, and the
other 18 amino acids are black.

Usage
-----
    python escape_aa_scan.py --results-dir results --subsample work/subsample.tsv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from utils import (
    ALL_AAS,
    DEFAULT_WEIGHTS,
    GENE_COORDS,
    KNOWN_ESCAPE_AAs,
    AA_MUT_RE,
    extract_gene_protein,
    load_model,
    parse_fasta,
    predict_probs,
    window_seq,
)


def find_post_apr2021_targets(
    fasta_dir: Path,
    subsample_tsv: Path,
    min_date: str = "2021-04-01",
) -> list[tuple[str, str]]:
    """Find known escape mutations that first appeared after ``min_date``.

    Returns a list of (aa_mutation, nt_mutation) tuples, deduplicated.
    """
    # Collect all FASTA entries that are known escape mutations
    seen: set[tuple[str, str]] = set()
    for fname in ["selected_mutants.fasta", "median_mutants.fasta",
                  "least_mutants.fasta"]:
        path = fasta_dir / fname
        if not path.exists():
            continue
        for header, _ in parse_fasta(path):
            aa_mut = header.get("aa", "")
            nt_mut = header.get("mutation", "")
            if aa_mut in KNOWN_ESCAPE_AAs and nt_mut:
                seen.add((aa_mut, nt_mut))

    if not seen:
        return []

    # Find first-seen date of each nt mutation in the subsample
    sub = pd.read_csv(subsample_tsv, sep="\t",
                      usecols=["strain", "date", "substitutions"])
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")

    first_seen: dict[str, pd.Timestamp] = {}
    for aa_mut, nt_mut in seen:
        carriers = sub[sub["substitutions"].str.contains(
            str(nt_mut), na=False, regex=False)]
        if len(carriers):
            first_seen[nt_mut] = carriers["date"].min()

    cutoff = pd.Timestamp(min_date)
    targets = [(aa, nt) for aa, nt in seen
               if nt in first_seen and first_seen[nt] >= cutoff]
    targets.sort()
    return targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--subsample", type=Path,
                   default=Path("work/subsample.tsv"))
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out-plot", type=Path,
                   default=Path("results/escape_aa_scan.png"))
    p.add_argument("--min-date", type=str, default="2021-04-01",
                   help="minimum first-appearance date (YYYY-MM-DD)")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[scan] loading model (device={device}) …", flush=True)
    model = load_model(args.weights, device)

    # ---- Dynamically find post-April-2021 escape mutations ----
    print("[scan] finding post-April-2021 escape mutations …", flush=True)
    targets = find_post_apr2021_targets(
        args.results_dir, args.subsample, args.min_date)
    print(f"[scan] found {len(targets)} targets:", flush=True)
    for aa, nt in targets:
        print(f"  {aa} ({nt})")

    if not targets:
        print("[scan] no targets found, exiting", flush=True)
        return

    # ---- Load ancestor genomes from FASTA files ----
    fasta_map: dict[tuple[str, str], str] = {}
    for fname in ["selected_mutants.fasta", "median_mutants.fasta",
                  "least_mutants.fasta"]:
        path = args.results_dir / fname
        if not path.exists():
            continue
        for header, seq in parse_fasta(path):
            key = (header.get("mutation", ""), header.get("aa", ""))
            if key[0] and key[1]:
                fasta_map[key] = seq

    # ---- Scan all 20 amino acids at each target site ----
    aa_re = re.compile(r"^S:([A-Za-z*])(\d+)([A-Za-z*]+)$")
    results = []

    for aa_mut, nt_mut in targets:
        m = aa_re.match(aa_mut)
        if not m:
            continue
        ref_aa, pos1, alt_aa = m.group(1), int(m.group(2)), m.group(3)
        pos0 = pos1 - 1

        genome = fasta_map.get((nt_mut, aa_mut))
        if genome is None:
            print(f"[scan] {aa_mut} ({nt_mut}) not found in FASTA, skipping",
                  flush=True)
            continue

        spike = extract_gene_protein(genome, "S")
        if spike is None or pos0 >= len(spike) or spike[pos0] != ref_aa:
            print(f"[scan] {aa_mut}: WT mismatch, skipping", flush=True)
            continue

        print(f"[scan] {aa_mut} ({nt_mut}): scanning all 20 AA at pos {pos1}…",
              flush=True)

        for aa in ALL_AAS:
            mut_seq = spike[:pos0] + aa + spike[pos0 + 1:]
            if len(spike) > 2001:
                wt_win, p = window_seq(spike, pos0)
                mut_win = wt_win[:p] + aa + wt_win[p + 1:]
            else:
                mut_win, p = mut_seq, pos0

            probs = predict_probs(model, mut_win)
            prob = float(probs[p])
            results.append({
                "aa_mutation": aa_mut,
                "nt_mutation": nt_mut,
                "pos": pos1,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "test_aa": aa,
                "epitope_prob": prob,
                "is_wt": aa == ref_aa,
                "is_escape": aa == alt_aa,
            })

    if not results:
        print("[scan] no results, exiting", flush=True)
        return

    df = pd.DataFrame(results)
    df.to_csv(args.results_dir / "escape_aa_scan.tsv", sep="\t", index=False)
    print(f"[scan] wrote {len(df)} rows → "
          f"{args.results_dir / 'escape_aa_scan.tsv'}", flush=True)

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    panels = df.groupby(["aa_mutation", "nt_mutation"]).size().reset_index()
    n_panels = len(panels)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 6),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    for i, (_, row) in enumerate(panels.iterrows()):
        ax = axes[i]
        aa_mut = row["aa_mutation"]
        nt_mut = row["nt_mutation"]
        sub = df[(df["aa_mutation"] == aa_mut)
                 & (df["nt_mutation"] == nt_mut)]
        sub = sub.sort_values("epitope_prob", ascending=False)

        colors = [
            "#2ca02c" if r["is_wt"] else
            "#d62728" if r["is_escape"] else
            "black"
            for _, r in sub.iterrows()
        ]

        xlabels = sub["test_aa"].tolist()
        probs = sub["epitope_prob"].tolist()

        ax.bar(range(len(xlabels)), probs, color=colors, edgecolor="none",
               width=0.85)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=8, fontweight="bold")
        ax.set_title(f"{aa_mut}\n({nt_mut})", fontsize=10)
        if i == 0:
            ax.set_ylabel("Epitope probability", fontsize=12)
        ax.set_ylim(0, max(0.7, df["epitope_prob"].max() * 1.1))

        for j, r in sub.reset_index().iterrows():
            if r["is_wt"] or r["is_escape"]:
                ax.text(j, r["epitope_prob"] + 0.01,
                        f"{r['epitope_prob']:.3f}",
                        ha="center", fontsize=7,
                        color=colors[j], fontweight="bold")

    fig.legend(handles=[
        Patch(color="#2ca02c", label="WT residue"),
        Patch(color="#d62728", label="Escape mutation"),
        Patch(color="black", label="Other 18 amino acids"),
    ], loc="upper center", ncol=3, fontsize=10,
       bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f"Epitope probability for all 20 amino acids at {n_panels} "
        f"post-April-2021 escape sites\n"
        f"SARS-CoV-2 spike — epiLoRA (ESM2) prediction",
        fontsize=13, y=1.06)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=180, bbox_inches="tight")
    print(f"[scan] plot → {args.out_plot}", flush=True)


if __name__ == "__main__":
    main()
