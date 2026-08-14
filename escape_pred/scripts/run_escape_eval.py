#!/usr/bin/env python3
"""Run epiLoRA escape evaluation on all three datasets (selected / median / least).

For each ancestor genome in the three FASTAs, this script:

1. Parses the FASTA header to get the aa mutation (e.g. ``S:D614G``).
2. Extracts the relevant gene from the ancestor genome and translates it to
   the **WT protein** (the protein just *before* the selected mutation arose).
3. Applies the aa mutation to get the **mutant protein**.
4. Runs epiLoRA (ESM2, sequence-only) on both WT and mutant.
5. Records the epitope-probability difference at the mutation site::

       delta = WT_prob − mutant_prob

   Positive delta → the mutation lowers epitope probability → potential escape.

Usage
-----
    python run_escape_eval.py
    python run_escape_eval.py --weights /path/to/esm2.pt --results-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    AA_MUT_RE,
    DEFAULT_WEIGHTS,
    MAX_CTX,
    extract_gene_protein,
    load_model,
    parse_fasta,
    predict_probs,
    window_seq,
)


def score_mutation(model, wt_seq: str, pos0: int, alt_aa: str):
    """Run epiLoRA on WT and mutant, return (wt_prob, mut_prob) at pos0."""
    mut_seq = wt_seq[:pos0] + alt_aa + wt_seq[pos0 + 1:]

    if len(wt_seq) > MAX_CTX * 2 + 1:
        wt_win, p = window_seq(wt_seq, pos0)
        mut_win = wt_win[:p] + alt_aa + wt_win[p + 1:]
    else:
        wt_win, p = wt_seq, pos0
        mut_win = mut_seq

    wt_probs = predict_probs(model, wt_win)
    mut_probs = predict_probs(model, mut_win)
    return float(wt_probs[p]), float(mut_probs[p])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=Path, default=Path("results/escape_eval.tsv"))
    p.add_argument("--out-plot", type=Path,
                   default=Path("results/escape_eval_plot.png"))
    p.add_argument("--limit", type=int, default=0,
                   help="limit entries per dataset (0 = all)")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] loading model (device={device}) …", flush=True)
    model = load_model(args.weights, device)

    datasets = {
        "selected": args.results_dir / "selected_mutants.fasta",
        "median":   args.results_dir / "median_mutants.fasta",
        "least":    args.results_dir / "least_mutants.fasta",
    }

    all_rows = []
    for set_name, fasta_path in datasets.items():
        if not fasta_path.exists():
            print(f"[eval] {fasta_path} not found, skipping", flush=True)
            continue
        print(f"\n[eval] === {set_name} ({fasta_path.name}) ===", flush=True)

        n_total = n_skipped = n_done = 0
        for header, genome in parse_fasta(fasta_path):
            n_total += 1
            if args.limit and n_done >= args.limit:
                break

            aa_mut = header.get("aa", "")
            if not aa_mut or aa_mut == "NA":
                n_skipped += 1
                continue

            m = AA_MUT_RE.match(aa_mut)
            if not m:
                n_skipped += 1
                continue

            gene, ref_aa, pos1, alt_aa = (
                m.group(1), m.group(2), int(m.group(3)), m.group(4))

            if alt_aa == "*":
                n_skipped += 1
                continue

            wt_protein = extract_gene_protein(genome, gene)
            if wt_protein is None or len(wt_protein) < pos1:
                n_skipped += 1
                continue

            pos0 = pos1 - 1
            if pos0 >= len(wt_protein) or wt_protein[pos0] != ref_aa:
                n_skipped += 1
                continue

            try:
                wt_prob, mut_prob = score_mutation(
                    model, wt_protein, pos0, alt_aa)
            except Exception:
                n_skipped += 1
                continue

            all_rows.append({
                "set": set_name,
                "mutation": header.get("mutation", ""),
                "aa_mutation": aa_mut,
                "gene": gene,
                "pos": pos1,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "wt_prob": round(wt_prob, 6),
                "mutant_prob": round(mut_prob, 6),
                "delta": round(wt_prob - mut_prob, 6),
                "descendants": int(header.get("descendants", 0)),
                "occurrences": int(header.get("occurrences", 0)),
            })
            n_done += 1
            if n_done % 200 == 0:
                print(f"[eval]   {n_done} scored, {n_skipped} skipped",
                      flush=True)

        print(f"[eval] {set_name}: {n_done} scored, {n_skipped} skipped, "
              f"{n_total} total", flush=True)

    if not all_rows:
        print("[eval] no results, exiting", flush=True)
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"\n[eval] wrote {len(df)} results → {args.out}", flush=True)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY: epitope probability change (WT − mutant) by selection set")
    print("=" * 70)
    for s in ["selected", "median", "least"]:
        sub = df[df["set"] == s]
        if sub.empty:
            continue
        d = sub["delta"]
        pos = (d > 0).sum()
        print(f"\n  {s:10s}  n={len(sub):4d}  "
              f"mean delta={d.mean():+.5f}  median={d.median():+.5f}  "
              f"reduced epitope: {pos}/{len(sub)} ({100*pos/len(sub):.1f}%)")

    print("\n  By gene (mean delta, selected set):")
    for gene in ["S", "N", "ORF1a", "ORF1b", "ORF3a", "M", "ORF8"]:
        g = df[(df["gene"] == gene) & (df["set"] == "selected")]
        if not g.empty:
            print(f"    {gene:8s} n={len(g):4d}  mean={g['delta'].mean():+.5f}")

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sets = [s for s in ["selected", "median", "least"]
            if not df[df["set"] == s].empty]
    data = [df[df["set"] == s]["delta"].values for s in sets]
    axes[0].violinplot(data, positions=range(len(sets)), showmedians=True)
    axes[0].set_xticks(range(len(sets)))
    axes[0].set_xticklabels(sets)
    axes[0].axhline(0, color="grey", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Δ epitope prob (WT − mutant)", fontsize=12)
    axes[0].set_title("Does selection reduce epitope probability?\n"
                      "(positive = mutation hides from immune system)",
                      fontsize=13)

    colors = {"selected": "#d62728", "median": "#1f77b4", "least": "#2ca02c"}
    for s in sets:
        sub = df[df["set"] == s]
        axes[1].scatter(sub["descendants"], sub["delta"],
                        label=s, alpha=0.4, s=15, c=colors.get(s, "gray"))
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].axhline(0, color="grey", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Tree descendants (selection strength)", fontsize=12)
    axes[1].set_ylabel("Δ epitope prob (WT − mutant)", fontsize=12)
    axes[1].set_title("Epitope prob change vs selection strength", fontsize=13)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=180)
    print(f"\n[eval] plot → {args.out_plot}", flush=True)


if __name__ == "__main__":
    main()
