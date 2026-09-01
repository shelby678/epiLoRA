"""Build the companion surface-annotation FASTA that loss_mask="surface"
training runs read (train.py + configs/loss_surface.yaml).

    python make_surface_fasta.py \
        --fasta data/train_test_eval/min_resolution_10_epitopes.fasta \
        --structures data/raw/all-structures-extracted

Writes ``<fasta stem>_surface.fasta`` next to ``--fasta`` (override with
``--out``): every record of ``--fasta``, with identical headers (fold labels
included) and the same sequence, but cased per residue -- lowercase = surface
(relative solvent accessibility >= --cutoff; freesasa's relativeTotal on the
antigen chain(s) with the antibody excluded, see data.load_rsa), uppercase =
buried. That mirrors the epitope FASTA's lowercase=epitope convention, so
train.py can read the file back with the same parse_fasta and get a 0/1
surface mask per residue (data.load_surface_masks).

RSA comes from the same per-structure cache training reads, so this is a
cheap one-off: generate it once and every epoch of every fold reuses it.
Entries whose RSA can't be had (missing/failed structure -- the same ones
training skips for missing coords) are left out and reported.

The default cutoff is the repo's standing "surface" convention
(data.RSA_SURFACE_CUTOFF, from data/scripts/pct_non_epitope_surface_hist.py).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from data import (RSA_SURFACE_CUTOFF, default_cache_dir, load_rsa, parse_fasta,
                  parse_seq_id)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent  # so defaults don't depend on cwd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, required=True,
                   help="ablation FASTA to annotate (headers are copied verbatim)")
    p.add_argument("--structures", type=Path,
                   default=REPO_ROOT / "data/raw/all-structures-extracted")
    p.add_argument("--cutoff", type=float, default=RSA_SURFACE_CUTOFF,
                   help="a residue is surface iff its RSA >= this (default: "
                        f"{RSA_SURFACE_CUTOFF}, the repo convention)")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: <fasta stem>_surface.fasta "
                        "next to --fasta)")
    args = p.parse_args()

    out = args.out or args.fasta.with_name(f"{args.fasta.stem}_surface.fasta")
    cache_dir = default_cache_dir(args.structures)

    by_part = parse_fasta(args.fasta)
    entries = [e for v in by_part.values() for e in v]
    logger.info(f"{args.fasta}: {len(entries)} records; cutoff={args.cutoff}; "
                f"cache={cache_dir}")

    n_written = n_missing = 0
    n_res = n_surf = n_epi = n_epi_surf = 0
    missing = []
    with open(out, "w") as f:
        for header, seq, labels in entries:
            pdb_id, chains = parse_seq_id(header)
            cp = args.structures / pdb_id / f"{pdb_id}_sabdab.cif"
            if not cp.exists():
                raise FileNotFoundError(f"no structure file for {header!r}: {cp}")
            rsa = load_rsa(cp, chains, len(seq), cache_dir)
            if rsa is None:
                missing.append(header)
                continue
            surface = rsa >= args.cutoff
            f.write(f">{header}\n")
            f.write("".join(c.lower() if s else c.upper()
                            for c, s in zip(seq, surface)) + "\n")
            n_written += 1
            n_res += len(seq)
            n_surf += int(surface.sum())
            n_epi += sum(labels)
            n_epi_surf += int((surface & np.asarray(labels, dtype=bool)).sum())

    for header in missing:
        logger.warning(f"no RSA for {header} -- left out (training skips it "
                       f"for missing coords anyway)")
    pct_surf = 100.0 * n_surf / n_res if n_res else float("nan")
    pct_epi_surf = 100.0 * n_epi_surf / n_epi if n_epi else float("nan")
    logger.info(f"Wrote {n_written}/{len(entries)} records -> {out}")
    logger.info(f"Surface residues: {n_surf}/{n_res} ({pct_surf:.1f}%); "
                f"epitope residues on the surface: {n_epi_surf}/{n_epi} ({pct_epi_surf:.1f}%)")


if __name__ == "__main__":
    main()
