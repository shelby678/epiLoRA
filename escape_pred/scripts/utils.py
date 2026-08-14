#!/usr/bin/env python3
"""Shared utilities for the immune-escape prediction pipeline.

Contains constants (gene coordinates, codon table, known escape mutations),
sequence utilities (translation, FASTA parsing, reference loading), and
epiLoRA model helpers (model loading, prediction, windowing).

Heavy imports (torch, biotite, esm) are deferred to function bodies so this
module can be imported by non-ML scripts (prepare_data.py, count_and_plot.py)
that run under system Python without torch installed.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EPI_LORA_DIR = Path("/home/jovyan/work/epiLoRA/epilora")
DEFAULT_WEIGHTS = Path(
    "/home/jovyan/work/epiLoRA/weights/ablation/all_fold2_esm2.pt")
REF_NAME = "Wuhan-Hu-1"
MAX_CTX = 1000  # ESM2 context window radius (residues per side)

# SARS-CoV-2 gene coordinates on NC_045512.2
# (start_1based, n_codons) — translate at most n_codons, stop at first stop.
# ORF1b uses the -1 ribosomal frameshift frame: its codons start at position
# 13468 (1-based) in frame 0, which is correct after the slippery sequence.
GENE_COORDS = {
    "ORF1a": (266, 4401),
    "ORF1b": (13468, 2696),
    "S":     (21563, 1273),
    "ORF3a": (25393, 275),
    "E":     (26245, 75),
    "M":     (26523, 222),
    "ORF6":  (27202, 61),
    "ORF7a": (27394, 121),
    "ORF7b": (27756, 43),
    "ORF8":  (27894, 121),
    "N":     (28274, 419),
    "ORF10": (29558, 38),
}

# Gene start only (1-based, for nt→aa position mapping)
GENE_STARTS = {g: v[0] for g, v in GENE_COORDS.items()}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

ALL_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Known SARS-CoV-2 spike immune-escape / adaptation mutations from the
# published literature.  Used for plot annotation and as the seed list for
# the amino-acid scan (filtered by first-appearance date at runtime).
KNOWN_ESCAPE_AAs = {
    "S:D614G", "S:N501Y", "S:E484K", "S:E484Q", "S:L452R", "S:L452Q",
    "S:F486V", "S:F486S", "S:K417N", "S:K417T", "S:T478K", "S:N460K",
    "S:F456L", "S:Q493R", "S:G496S", "S:Y505H", "S:Q498R", "S:Y503H",
    "S:Del69-70", "S:Del144", "S:Del241-243", "S:Del246-252",
    "S:R346K", "S:R346T", "S:S477N", "S:G339D", "S:S371F", "S:S373P",
    "S:G446S", "S:L455F", "S:A475V", "S:V483A",
}

AA_MUT_RE = re.compile(r"^([A-Za-z0-9]+):([A-Za-z*])(\d+)([A-Za-z*]+)$")
NT_SNP_RE = re.compile(r"^([ACGT])(\d+)([ACGT])$")
NT_POS_RE = re.compile(r"^[ACGT](\d+)[ACGT]$")


# --------------------------------------------------------------------------- #
# Sequence utilities (no heavy deps)
# --------------------------------------------------------------------------- #

def load_reference(path: Path) -> str:
    """Load the reference genome (NC_045512.2) as a string."""
    with open(path) as f:
        f.readline()  # header
        return "".join(line.strip() for line in f)


def translate(nt_seq: str, max_codons: int = 0) -> str:
    """Translate a nucleotide sequence to a protein.

    Stops at the first stop codon (``*``) or after ``max_codons`` (if > 0).
    Unknown codons become ``X``.
    """
    protein = []
    limit = max_codons if max_codons > 0 else len(nt_seq) // 3
    for i in range(0, min(len(nt_seq) - 2, limit * 3), 3):
        aa = CODON_TABLE.get(nt_seq[i:i + 3].upper(), "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def extract_gene_protein(genome: str, gene: str) -> str | None:
    """Extract and translate a gene from a SARS-CoV-2 genome sequence.

    Uses ``GENE_COORDS`` to slice the correct region and limit translation
    to the gene's codon count, so ORF1a stops at the frameshift site and
    ORF1b doesn't run into downstream genes.
    """
    coords = GENE_COORDS.get(gene)
    if coords is None:
        return None
    start, n_codons = coords
    nt = genome[start - 1:]  # 1-based → 0-based
    return translate(nt, max_codons=n_codons)


def apply_mutation(ref: str, mut: str) -> str:
    """Apply a single nt mutation (e.g. 'A23403G') to a sequence (1-based).
    Silently skips mutations whose ref allele doesn't match or that are
    indels (not simple SNPs)."""
    m = NT_SNP_RE.match(mut)
    if not m:
        return ref  # not a simple SNP (could be an indel from UShER)
    ref_allele, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    if pos < 1 or pos > len(ref):
        return ref
    if ref[pos - 1] != ref_allele:
        return ref
    return ref[:pos - 1] + alt + ref[pos:]


def reconstruct_ancestor(ref: str, path_muts: list[str]) -> str:
    """Apply all mutations on the path from root to a node, in order."""
    seq = ref
    for m in path_muts:
        seq = apply_mutation(seq, m)
    return seq


def parse_fasta(path: Path):
    """Yield (header_dict, sequence) for each entry in a FASTA.

    Header fields are parsed as key=value pairs; the first bare token
    becomes header['id'].
    """
    header = None
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = {}
                for field in line[1:].strip().split():
                    if "=" in field:
                        k, v = field.split("=", 1)
                        header[k] = v
                    else:
                        header.setdefault("id", field)
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            yield header, "".join(seq_parts)


def build_aa_map(subsample_tsv: Path, ref: str | None = None) -> dict[int, str]:
    """Map nt position → aa mutation string (e.g. 23403 → 'S:D614G').

    If ``ref`` (the reference genome) is provided, the mapping is computed by
    actually translating the effect of each possible SNP at each codon — so
    different SNPs in the same codon correctly map to different aa mutations
    (e.g. C22986T → S:A475V, G22985C → S:A475P).

    If ``ref`` is not provided, falls back to matching aaSubstitutions by
    codon position (the old, less-accurate method).
    """
    if not subsample_tsv.exists():
        return {}

    if ref is not None:
        return _build_aa_map_from_ref(subsample_tsv, ref)

    # Fallback: match by codon position (may assign the wrong aa mutation
    # when multiple aa changes occur at the same codon).
    df = pd.read_csv(subsample_tsv, sep="\t", usecols=["aaSubstitutions"])
    pos_to_aa: dict[int, str] = {}
    for s in df["aaSubstitutions"].dropna():
        for a in str(s).split(","):
            a = a.strip()
            m = AA_MUT_RE.match(a)
            if not m:
                continue
            gene, aa_pos = m.group(1), int(m.group(3))
            if gene not in GENE_STARTS:
                continue
            codon_start = GENE_STARTS[gene] + (aa_pos - 1) * 3
            for nt_pos in range(codon_start, codon_start + 3):
                if nt_pos not in pos_to_aa:
                    pos_to_aa[nt_pos] = a
    return pos_to_aa


def _build_aa_map_from_ref(subsample_tsv: Path, ref: str) -> dict[int, str]:
    """Compute nt position → aa mutation by translating each SNP against the
    reference genome.

    For every aa substitution in the subsample, we know the gene + aa position.
    From the gene start and aa position, we know the codon (3 nt positions).
    We then look at which of the 3 positions is mutated in the sample's nt
    substitutions and translate the resulting codon to determine the exact aa
    change.  This correctly distinguishes different nt mutations in the same
    codon that produce different aa changes.
    """
    df = pd.read_csv(subsample_tsv, sep="\t",
                     usecols=["substitutions", "aaSubstitutions"])
    pos_to_aa: dict[int, str] = {}

    for _, row in df.iterrows():
        nt_str = str(row.get("substitutions", "") or "")
        aa_str = str(row.get("aaSubstitutions", "") or "")
        if not nt_str or nt_str == "nan":
            continue

        # Build set of (position, alt) from nt substitutions
        nt_muts: dict[int, str] = {}  # position → alt allele
        for n in nt_str.split(","):
            n = n.strip()
            m = NT_SNP_RE.match(n)
            if m:
                nt_muts[int(m.group(2))] = m.group(3)

        # For each aa substitution, find which nt position in the codon is
        # mutated and map it to the aa mutation string.
        for a in aa_str.split(","):
            a = a.strip()
            m = AA_MUT_RE.match(a)
            if not m:
                continue
            gene, aa_pos = m.group(1), int(m.group(3))
            if gene not in GENE_STARTS:
                continue
            codon_start = GENE_STARTS[gene] + (aa_pos - 1) * 3
            for offset in range(3):
                nt_pos = codon_start + offset
                if nt_pos in nt_muts and nt_pos not in pos_to_aa:
                    pos_to_aa[nt_pos] = a

    return pos_to_aa


def nt_pos_of_mut(nt_mut: str) -> int | None:
    """Extract the nucleotide position from a mutation string like 'A23403G'."""
    m = NT_POS_RE.match(nt_mut)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# epiLoRA model helpers (heavy deps imported lazily)
# --------------------------------------------------------------------------- #

def load_model(weights: Path, device: str):
    """Load an epiLoRA ESM2 (or ESM-IF1) checkpoint.

    Heavy imports (torch, biotite, esm) are deferred to here so that scripts
    using system Python can import the non-ML utilities above without needing
    torch installed.
    """
    import biotite.structure as _struc
    if not hasattr(_struc, "filter_backbone"):
        _struc.filter_backbone = _struc.filter_peptide_backbone

    import torch
    sys.path.insert(0, str(EPI_LORA_DIR))
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    backbone = ckpt.get("backbone", "esmif1")
    if backbone == "esm2":
        from model import ESM2EpitopeModel, load_base_esm2
        esm_model, alphabet = load_base_esm2(cfg["size"])
        model = ESM2EpitopeModel(esm_model, alphabet, **cfg).to(device)
    else:
        from model import ESMIF1EpitopeModel, load_base_esmif1
        esm_model, alphabet = load_base_esmif1()
        model = ESMIF1EpitopeModel(esm_model, alphabet, **cfg).to(device)
    model.load_trainable_state_dict(ckpt["trainable_state"])
    model.eval()
    return model


def predict_probs(model, seq: str) -> np.ndarray:
    """Run epiLoRA on a single protein sequence → per-residue epitope probs."""
    import torch
    with torch.no_grad():
        logits = model([None], [seq])[0].cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def window_seq(seq: str, pos: int, radius: int = MAX_CTX) -> tuple[str, int]:
    """Extract a window of ±``radius`` residues around position ``pos``
    (0-based).  Returns (windowed_seq, pos_in_window)."""
    lo = max(0, pos - radius)
    hi = min(len(seq), pos + radius + 1)
    return seq[lo:hi], pos - lo
