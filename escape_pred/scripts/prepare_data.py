#!/usr/bin/env python3
"""Prepare UShER inputs (VCF + starting tree) from Nextclade metadata.

The metadata.tsv already carries per-sample nucleotide substitutions (called by
Nextclade against Wuhan-Hu-1 / NC_045512.2), so we build the VCF directly from
those columns — no need to re-parse the 282 GB alignment FASTA.

Steps
-----
1. Load metadata (strain, date, pango_lineage, substitutions, deletions, …).
2. Filter: valid date, coverage > 0.9, QC == "good", non-empty substitutions.
3. Stratified subsample by (pango_lineage, year) to capture diversity.
4. Build a VCF (SNPs + deletions) using the reference genome to validate ref
   alleles and represent indels in standard VCF format.
5. Build a UPGMA starting tree from the Hamming-distance matrix (vectorised).

Usage
-----
    python prepare_data.py --n-samples 30000
    python prepare_data.py --n-samples 30000 --time-slice 2022  # only 2022+
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

REF_NAME = "Wuhan-Hu-1"

# Resolve default paths relative to this script's location (scripts/ → parent)
_BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# 1. Load & filter metadata
# --------------------------------------------------------------------------- #

COLS = [
    "strain", "date", "pango_lineage", "Nextstrain_clade",
    "substitutions", "deletions", "divergence", "coverage",
    "QC_overall_status", "aaSubstitutions",
]


def load_reference(path: Path) -> str:
    """Load the reference genome (NC_045512.2) as a string."""
    print(f"[prepare] loading reference → {path} …", flush=True)
    with open(path) as f:
        f.readline()  # header
        seq = "".join(line.strip() for line in f)
    print(f"[prepare]   reference: {len(seq)} nt", flush=True)
    return seq


def load_metadata(path: Path, time_slice: int | None) -> pd.DataFrame:
    print(f"[prepare] loading metadata (cols: {len(COLS)}) …", flush=True)
    df = pd.read_csv(path, sep="\t", usecols=COLS, dtype=str, na_values=["?"])
    print(f"[prepare]   {len(df):,} rows loaded", flush=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["year"] = df["date"].dt.year

    n0 = len(df)
    df = df[df["QC_overall_status"] == "good"]
    df = df[df["coverage"].astype(float) > 0.9]
    df = df[df["date"].notna()]
    df = df[df["substitutions"].notna() & (df["substitutions"] != "")]
    df = df.dropna(subset=["pango_lineage"])
    if time_slice is not None:
        df = df[df["year"] >= time_slice]
    print(f"[prepare]   {len(df):,} / {n0:,} rows passed QC", flush=True)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Stratified subsample
# --------------------------------------------------------------------------- #

def subsample(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Stratify by (pango_lineage, year) so every lineage × year is represented.
    Samples are shuffled *within* each stratum before taking the first N, so the
    subsample is not biased by row order in the metadata file."""
    print(f"[prepare] subsampling to ~{n_samples:,} …", flush=True)
    df = df.copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df["stratum"] = df["pango_lineage"] + "@" + df["year"].astype(str)
    n_strata = df["stratum"].nunique()
    base = max(1, n_samples // n_strata)

    # vectorised: rank within each stratum, keep first `base` per stratum
    df["_rank"] = df.groupby("stratum").cumcount()
    sub = df[df["_rank"] < base].drop(columns="_rank")

    # top up with random samples if under target
    if len(sub) < n_samples:
        remaining = df[~df["strain"].isin(sub["strain"])]
        extra = min(len(remaining), n_samples - len(sub))
        if extra > 0:
            sub = pd.concat([sub, remaining.sample(extra, random_state=42)],
                            ignore_index=True)
    if len(sub) > n_samples:
        sub = sub.sample(n_samples, random_state=42)
    sub = sub.drop(columns=["stratum", "_rank"], errors="ignore").reset_index(drop=True)
    print(f"[prepare]   {len(sub):,} samples  "
          f"({sub['pango_lineage'].nunique()} lineages, "
          f"{sub['year'].min():.0f}-{sub['year'].max():.0f})", flush=True)
    return sub


# --------------------------------------------------------------------------- #
# 3. Parse substitutions + deletions → binary GT matrix + VCF
# --------------------------------------------------------------------------- #

def parse_subs(subs_str: str) -> list[tuple[int, str, str]]:
    """'C241T,A405G' → [(241,'C','T'), (405,'A','G')]"""
    out = []
    for m in subs_str.split(","):
        m = m.strip()
        if len(m) < 3:
            continue
        ref, alt = m[0], m[-1]
        try:
            pos = int(m[1:-1])
        except ValueError:
            continue
        out.append((pos, ref, alt))
    return out


def parse_deletions(del_str: str) -> list[tuple[int, int]]:
    """'11288-11296,28271' → [(11288, 11296), (28271, 28271)]

    Returns (start, end) 1-based, inclusive.  A single number is a 1-bp deletion.
    """
    out = []
    for part in del_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                out.append((int(lo), int(hi)))
            except ValueError:
                continue
        else:
            try:
                p = int(part)
                out.append((p, p))
            except ValueError:
                continue
    return out


def build_gt_and_vcf(df: pd.DataFrame, ref: str, out_vcf: Path):
    """Return (sample_names, positions, gt_matrix) and write a VCF.

    SNPs are validated against the reference genome: the consensus ref allele
    must match the actual base in NC_045512.2, otherwise the variant is
    discarded.  Deletions are represented in standard VCF indel format
    (anchored on the preceding base) and included as binary GT columns.
    """
    print("[prepare] building VCF from substitutions + deletions …", flush=True)
    sample_names = [REF_NAME] + df["strain"].tolist()
    n_samp = len(sample_names)

    # ---- SNPs ----
    sample_alts: list[dict[int, str]] = [{}]
    ref_votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in df["substitutions"]:
        alts: dict[int, str] = {}
        for pos, ref_allele, alt in parse_subs(s):
            alts[pos] = alt
            ref_votes[pos][ref_allele] += 1
        sample_alts.append(alts)

    # Determine consensus ref per position and validate against reference
    consensus_ref: dict[int, str] = {}
    n_ref_mismatch = 0
    for pos, votes in ref_votes.items():
        cons = max(votes, key=votes.get)
        # Validate against actual reference (1-based)
        if 1 <= pos <= len(ref) and ref[pos - 1] != cons:
            n_ref_mismatch += 1
            continue  # discard — consensus doesn't match reference
        consensus_ref[pos] = cons

    if n_ref_mismatch:
        print(f"[prepare]   discarded {n_ref_mismatch} positions where "
              f"consensus ref ≠ reference", flush=True)

    # Collect unique ALTs per position
    pos_alts: dict[int, list[str]] = {}
    for alts in sample_alts:
        for pos, alt in alts.items():
            if pos not in consensus_ref:
                continue
            ref_base = consensus_ref[pos]
            if pos not in pos_alts:
                pos_alts[pos] = []
            if alt not in pos_alts[pos] and alt != ref_base:
                pos_alts[pos].append(alt)

    # ---- Deletions ----
    # Each unique (start, end) is a variant.  VCF representation:
    #   POS = start - 1 (anchor base, 1-based)
    #   REF = ref[start-1 : end+1]  (anchor + deleted bases)
    #   ALT = ref[start-1]         (just the anchor)
    sample_dels: list[set[int]] = [set()]  # reference = no deletions
    del_variants: dict[int, tuple[int, int, str, str]] = {}  # id → (pos, end, ref, alt)
    del_id_counter = 0
    n_del_events = 0

    for del_str in df.get("deletions", pd.Series(dtype=str)):
        if pd.isna(del_str) or not del_str.strip():
            sample_dels.append(set())
            continue
        dels = set()
        for start, end in parse_deletions(del_str):
            if start < 2 or end >= len(ref):
                continue  # can't anchor at position 0 or past the end
            key = (start, end)
            if key not in del_variants:
                anchor = start - 1  # 1-based position of anchor base
                ref_allele = ref[anchor - 1:end]  # anchor + deleted bases
                alt_allele = ref[anchor - 1]       # just the anchor
                del_id_counter += 1
                del_variants[key] = (anchor, end, ref_allele, alt_allele)
            dels.add(key[0] * 100000 + key[1])  # unique hash for (start, end)
            n_del_events += 1
        sample_dels.append(dels)

    # ---- Build unified GT matrix ----
    # SNP rows: one per position, binary (0=ref, 1=any alt)
    # Deletion rows: one per (start,end), binary (0=no del, 1=del)
    snp_positions = sorted(consensus_ref)
    del_keys = sorted(del_variants.keys())
    n_snp = len(snp_positions)
    n_del = len(del_keys)
    n_var = n_snp + n_del

    pos_to_row = {p: i for i, p in enumerate(snp_positions)}
    del_to_row = {k: i + n_snp for i, k in enumerate(del_keys)}
    gt = np.zeros((n_var, n_samp), dtype=np.int8)

    # Fill SNPs
    for j, alts_map in enumerate(sample_alts):
        for pos, alt in alts_map.items():
            if pos not in pos_to_row:
                continue
            ref_base = consensus_ref[pos]
            if alt == ref_base:
                continue
            pa = pos_alts.get(pos)
            if not pa or alt not in pa:
                continue
            gt[pos_to_row[pos], j] = 1

    # Fill deletions
    del_lookup = {k[0] * 100000 + k[1]: k for k in del_keys}
    for j, dels in enumerate(sample_dels):
        for h in dels:
            key = del_lookup[h]
            gt[del_to_row[key], j] = 1

    n_snp_variants = sum(len(v) for v in pos_alts.values())
    print(f"[prepare]   {n_snp:,} SNP positions ({n_snp_variants:,} unique "
          f"ref→alt), {n_del:,} deletion events, {n_samp:,} samples",
          flush=True)

    # ---- Write VCF ----
    print(f"[prepare] writing VCF → {out_vcf} …", flush=True)
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##reference={REF_NAME}\n")
        f.write("##contig=<ID=NC_045512.2,length=29903>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        f.write("\t".join(sample_names) + "\n")

        # SNPs
        for pos in snp_positions:
            alts = pos_alts.get(pos, [])
            if not alts:
                continue
            ref_base = consensus_ref[pos]
            alt_str = ",".join(alts)
            ac = int((gt[pos_to_row[pos]] > 0).sum())
            f.write(f"NC_045512.2\t{pos}\t.\t{ref_base}\t{alt_str}\t.\t.\t"
                    f"AC={ac};AN={n_samp}\tGT\t")
            f.write("\t".join(str(x) for x in gt[pos_to_row[pos]]))
            f.write("\n")

        # Deletions
        for key in del_keys:
            start, end = key
            anchor, _, ref_allele, alt_allele = del_variants[key]
            row = del_to_row[key]
            ac = int((gt[row] > 0).sum())
            del_id = f"del_{start}_{end}"
            f.write(f"NC_045512.2\t{anchor}\t{del_id}\t{ref_allele}\t"
                    f"{alt_allele}\t.\t.\tAC={ac};AN={n_samp}\tGT\t")
            f.write("\t".join(str(x) for x in gt[row]))
            f.write("\n")

    print(f"[prepare]   VCF written ({out_vcf.stat().st_size / 1e9:.2f} GB)",
          flush=True)

    return sample_names, snp_positions + del_keys, gt


# --------------------------------------------------------------------------- #
# 4. UPGMA starting tree from Hamming distance
# --------------------------------------------------------------------------- #

def build_starting_tree(sample_names: list[str], gt: np.ndarray,
                        out_nwk: Path) -> None:
    print("[prepare] building UPGMA starting tree …", flush=True)
    n_samp = gt.shape[1]
    if gt.shape[0] == 0:
        print("[prepare]   no variants; writing star tree", flush=True)
        with open(out_nwk, "w") as f:
            f.write("(" + ",".join(sample_names) + ");\n")
        return

    # Hamming distance:  D = s + sᵀ - 2·XᵀX   (X = gt as float, shape n_var×n_samp)
    print("[prepare]   computing distance matrix …", flush=True)
    X = gt.astype(np.float32)
    col_sums = X.sum(axis=0)                      # (n_samp,)
    dots = X.T @ X                                 # (n_samp, n_samp) via BLAS
    D = col_sums[:, None] + col_sums[None, :] - 2.0 * dots
    np.maximum(D, 0, out=D)
    np.fill_diagonal(D, 0)

    print("[prepare]   UPGMA linkage …", flush=True)
    dists = squareform(D, checks=False)
    Z = linkage(dists, method="average")  # UPGMA

    tree, _ = to_tree(Z, rd=True)

    def _newick(node) -> str:
        if node.is_leaf():
            return sample_names[node.id]
        parts = [f"{_newick(c)}:{c.dist:.6f}"
                 for c in (node.left, node.right) if c is not None]
        return "(" + ",".join(parts) + ")"

    out_nwk.parent.mkdir(parents=True, exist_ok=True)
    with open(out_nwk, "w") as f:
        f.write(_newick(tree) + ";\n")
    print(f"[prepare]   starting tree → {out_nwk}", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path,
                   default=_BASE_DIR / "data/sars-cov2/metadata.tsv")
    p.add_argument("--reference", type=Path,
                   default=_BASE_DIR / "data/sars-cov2/NC_045512.2.fasta",
                   help="reference genome FASTA (for ref-allele validation + indel representation)")
    p.add_argument("--outdir", type=Path, default=_BASE_DIR / "work")
    p.add_argument("--n-samples", type=int, default=30000)
    p.add_argument("--time-slice", type=int, default=None,
                   help="only keep samples from this year onward")
    args = p.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_reference(args.reference)
    df = load_metadata(args.metadata, args.time_slice)
    df = subsample(df, args.n_samples)

    df[["strain", "date", "pango_lineage", "Nextstrain_clade",
        "substitutions", "deletions", "aaSubstitutions"]].to_csv(
        outdir / "subsample.tsv", sep="\t", index=False)

    sample_names, variant_list, gt = build_gt_and_vcf(df, ref, outdir / "samples.vcf")
    build_starting_tree(sample_names, gt, outdir / "start_tree.nwk")
    print("[prepare] done.", flush=True)


if __name__ == "__main__":
    main()
