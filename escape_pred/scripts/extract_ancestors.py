#!/usr/bin/env python3
"""Extract ancestor genomes for selected, median, and least-selected mutations.

For each mutation in the tree, the **direct ancestor genome** = the reference
(Wuhan-Hu-1 / NC_045512.2) with all mutations on the path from the root to the
*parent* node applied.  This is the genome state just *before* the mutation
arose — the "background" on which selection acted.

Three output FASTAs
-------------------
1. ``selected_mutants.fasta``  — mutations with the most descendants
   (> ``--min-desc``).  These are the mutations most likely selected for.
2. ``median_mutants.fasta``    — N mutations around the median descendant
   count.  Neutral / drift.
3. ``least_mutants.fasta``     — N mutations with the fewest descendants
   (singletons or near-singletons).  Likely deleterious or neutral dead-ends.

Each FASTA header:
    >sel_00000 mutation=A23403G aa=S:D614G descendants=140066 occurrences=6 ...

Usage
-----
    python extract_ancestors.py --json work/tree.json --subsample work/subsample.tsv \
        --ref data/sars-cov2/NC_045512.2.fasta --outdir results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from utils import (
    KNOWN_ESCAPE_AAs,
    NT_POS_RE,
    apply_mutation,
    build_aa_map,
    load_reference,
    reconstruct_ancestor,
)


def iter_branches(node: dict, ancestor_path: list[str] | None = None):
    """Yield (node, ancestor_path, branch_muts) for every node.

    ``ancestor_path`` = mutations from root to the PARENT of this node
    (= the genome state just before this node's branch mutations arose).
    """
    if ancestor_path is None:
        ancestor_path = []
    muts = node.get("branch_attrs", {}).get("mutations", {}).get("nuc", [])
    yield (node, ancestor_path, muts)
    my_path = ancestor_path + muts
    for c in node.get("children", []):
        yield from iter_branches(c, my_path)


def count_descendants_iter(root: dict) -> dict[int, int]:
    """Bottom-up iterative leaf-count computation."""
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=Path("work/tree.json"))
    p.add_argument("--subsample", type=Path,
                   default=Path("work/subsample.tsv"))
    p.add_argument("--ref", type=Path,
                   default=Path("data/sars-cov2/NC_045512.2.fasta"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--n-selected", type=int, default=100,
                   help="number of top mutations for 'selected' set")
    p.add_argument("--n-median", type=int, default=100)
    p.add_argument("--n-least", type=int, default=100)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[ancestors] loading reference …", flush=True)
    ref = load_reference(args.ref)
    print(f"[ancestors] reference: {len(ref)} nt", flush=True)

    print("[ancestors] loading tree JSON …", flush=True)
    with open(args.json) as f:
        tree = json.load(f)["tree"]

    print("[ancestors] computing leaf counts …", flush=True)
    leaf_counts = count_descendants_iter(tree)

    print("[ancestors] traversing tree …", flush=True)
    aa_map = build_aa_map(args.subsample, ref)
    mut_entries: dict[str, list[tuple[list[str], int]]] = defaultdict(list)
    for node, ancestor_path, branch_muts in iter_branches(tree):
        if not branch_muts:
            continue
        n_desc = leaf_counts[id(node)]
        for m in branch_muts:
            mut_entries[m].append((list(ancestor_path), n_desc))

    rows = []
    for m, entries in mut_entries.items():
        total_desc = sum(d for _, d in entries)
        max_desc = max(d for _, d in entries)
        occurrences = len(entries)
        best_path, _ = max(entries, key=lambda x: x[1])
        # Map nt mutation → aa mutation (safely handles non-SNP mutations)
        pos_match = NT_POS_RE.match(m)
        aa_mut = aa_map.get(int(pos_match.group(1)), "") if pos_match else ""
        rows.append({
            "mutation": m,
            "aa_mutation": aa_mut,
            "occurrences": occurrences,
            "total_descendants": total_desc,
            "max_descendants": max_desc,
            "ancestor_path_len": len(best_path),
            "ancestor_path": best_path,
        })
    df = pd.DataFrame(rows).sort_values("total_descendants", ascending=False) \
                            .reset_index(drop=True)
    print(f"[ancestors] {len(df)} unique mutations in tree", flush=True)

    # ---- Define the three sets (top-N / median-N / bottom-N) ----
    selected = df.head(args.n_selected).copy()
    mid = len(df) // 2
    half_med = args.n_median // 2
    median = df.iloc[mid - half_med: mid + half_med].copy()
    least = df.tail(args.n_least).copy()

    print(f"[ancestors] selected: {len(selected)} (top {args.n_selected})")
    print(f"[ancestors] median:  {len(median)} (around rank {mid})")
    print(f"[ancestors] least:   {len(least)} (bottom {len(least)})")

    # ---- Write FASTAs ----
    def write_fasta(subdf: pd.DataFrame, out_path: Path, label: str) -> None:
        n_written = 0
        with open(out_path, "w") as f:
            for _, r in subdf.iterrows():
                seq = reconstruct_ancestor(ref, r["ancestor_path"])
                aa = r["aa_mutation"]
                is_escape = aa in KNOWN_ESCAPE_AAs
                header = (f">{label}_{n_written:05d} "
                          f"mutation={r['mutation']} "
                          f"aa={aa if isinstance(aa, str) and aa else 'NA'} "
                          f"descendants={int(r['total_descendants'])} "
                          f"occurrences={int(r['occurrences'])} "
                          f"max_descendants={int(r['max_descendants'])} "
                          f"ancestor_path_len={int(r['ancestor_path_len'])} "
                          f"escape={'1' if is_escape else '0'}")
                f.write(header + "\n")
                for i in range(0, len(seq), 70):
                    f.write(seq[i:i + 70] + "\n")
                n_written += 1
        print(f"[ancestors] wrote {n_written} genomes → {out_path}", flush=True)

    write_fasta(selected, args.outdir / "selected_mutants.fasta", "sel")
    write_fasta(median, args.outdir / "median_mutants.fasta", "med")
    write_fasta(least, args.outdir / "least_mutants.fasta", "least")

    summary = pd.concat([
        selected.assign(set="selected"),
        median.assign(set="median"),
        least.assign(set="least"),
    ])
    summary.drop(columns=["ancestor_path"]).to_csv(
        args.outdir / "ancestor_mutations.tsv", sep="\t", index=False)
    print(f"[ancestors] summary → {args.outdir / 'ancestor_mutations.tsv'}",
          flush=True)


if __name__ == "__main__":
    main()
