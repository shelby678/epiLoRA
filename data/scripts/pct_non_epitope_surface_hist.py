#!/usr/bin/env python3
"""Histogram of percent of SURFACE residues annotated as non-epitope, for
data/train_test_eval/all_epitopes.fasta.

Surface accessibility is computed with freesasa (default Shrake-Rupley
parameters, default atomic-radii classifier -- i.e. no custom config) on the
antigen chain(s) alone (antibody chains excluded, matching the antigen-only
convention used elsewhere in this repo, e.g. docking/scripts/
surface_epitope_overlap.py). A residue counts as "surface" if its relative
side-chain-inclusive total SASA >= 0.20, the same cutoff already used in
this repo (benchmarking/discotope3, benchmarking/webtools/ispred4 default).

Sequence casing: lowercase = epitope, UPPERCASE = non-epitope
(see data/README.md).
"""
import re
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import freesasa
from Bio.PDB import MMCIFParser, PDBIO, Select

from structures import chain_residues

DATA_DIR = Path(__file__).resolve().parents[1]
FASTA = DATA_DIR / "train_test_eval" / "all_epitopes.fasta"
STRUCT_DIR = DATA_DIR / "raw" / "all-structures-extracted"
OUT_PNG = DATA_DIR / "pct_non_epitope_surface_hist.png"

RSA_SURFACE_CUTOFF = 0.20
INSTANCE_RE = re.compile(r"^(.+)-[^-]+-[^-]+$")

BLUE = "#2a78d6"
TEXT_SECONDARY = "#52514e"
GRAY = "#9a9990"

_parser = MMCIFParser(QUIET=True)


def parse_fasta(path):
    header, seq = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq)
            header, seq = line[1:], []
        else:
            seq.append(line)
    if header is not None:
        yield header, "".join(seq)


class ChainResidueSelect(Select):
    """Keep only the exact residues we already validated against the sequence."""

    def __init__(self, wanted):
        self.wanted = wanted  # {(chain_id, res.id): True}

    def accept_chain(self, chain):
        return any(cid == chain.id for cid, _ in self.wanted)

    def accept_residue(self, residue):
        return (residue.get_parent().id, residue.id) in self.wanted


def surface_mask_for_entry(instance, antigen_chains, seq):
    m = INSTANCE_RE.match(instance)
    if not m:
        return None
    pdb_id = m.group(1)
    cif_path = Path(STRUCT_DIR) / pdb_id / f"{pdb_id}_sabdab.cif"
    if not cif_path.exists():
        return None

    structure = _parser.get_structure(pdb_id, str(cif_path))
    model = next(iter(structure))

    res_list = []
    for c in antigen_chains:
        res_list.extend(chain_residues(model, c))
    if len(res_list) != len(seq):
        return None

    wanted = {(c, res.id) for res in res_list for c in [res.get_parent().id]}

    io = PDBIO()
    io.set_structure(structure)
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        io.save(tmp_path, ChainResidueSelect(wanted))
        fs_struct = freesasa.Structure(tmp_path)
        result = freesasa.calc(fs_struct)
        residue_areas = result.residueAreas()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    mask = []
    for res in res_list:
        chain_id = res.get_parent().id
        key = f"{res.id[1]}{res.id[2].strip()}"
        area = residue_areas.get(chain_id, {}).get(key)
        if area is None or not area.hasRelativeAreas:
            mask.append(False)
        else:
            mask.append(area.relativeTotal >= RSA_SURFACE_CUTOFF)
    return mask


def main():
    pct_non_epitope_surface = []
    n_ok = n_skipped_align = n_skipped_no_surface = n_skipped_error = 0

    t0 = time.time()
    entries = list(parse_fasta(FASTA))
    for i, (header, seq) in enumerate(entries):
        fields = header.split()
        instance, antigen_chain_field = fields[0], fields[3]
        antigen_chains = antigen_chain_field.split("|")
        try:
            mask = surface_mask_for_entry(instance, antigen_chains, seq)
        except Exception:
            n_skipped_error += 1
            continue
        if mask is None:
            n_skipped_align += 1
            continue

        surface_seq = "".join(c for c, is_surf in zip(seq, mask) if is_surf)
        if not surface_seq:
            n_skipped_no_surface += 1
            continue

        n_upper = sum(1 for c in surface_seq if c.isupper())
        pct_non_epitope_surface.append(100.0 * n_upper / len(surface_seq))
        n_ok += 1

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(entries)} ({time.time() - t0:.0f}s elapsed)",
                  file=sys.stderr)

    print(f"n_ok={n_ok} n_skipped_align={n_skipped_align} "
          f"n_skipped_no_surface={n_skipped_no_surface} n_skipped_error={n_skipped_error}",
          file=sys.stderr)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.hist(pct_non_epitope_surface, bins=30, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("% of surface residues annotated as non-epitope", fontsize=11)
    ax.set_ylabel("Number of sequences", fontsize=11)
    ax.set_title(
        f"Non-epitope fraction among surface residues (RSA ≥ {RSA_SURFACE_CUTOFF:.0%})\n"
        f"(all_epitopes.fasta, n={len(pct_non_epitope_surface)})",
        fontsize=13,
    )
    ax.grid(color=GRAY, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    mean_pct = sum(pct_non_epitope_surface) / len(pct_non_epitope_surface)
    ax.axvline(mean_pct, color=TEXT_SECONDARY, linestyle="--", linewidth=1.2)
    ax.annotate(f"mean = {mean_pct:.1f}%", (mean_pct, ax.get_ylim()[1] * 0.95),
                textcoords="offset points", xytext=(6, 0), fontsize=9,
                color=TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")
    print(f"n={len(pct_non_epitope_surface)} mean={mean_pct:.2f}% "
          f"min={min(pct_non_epitope_surface):.2f}% max={max(pct_non_epitope_surface):.2f}%")


if __name__ == "__main__":
    main()
