#!/usr/bin/env python3
"""Run epiLoRA on WT vs mutant protein sequences to predict immune escape.

Given a protein sequence and a point mutation (e.g., ``D614G``), this script:

1. Runs epiLoRA (ESM2 backbone, sequence-only) on the **WT** sequence.
2. Creates the **mutant** sequence by substituting the residue at the site.
3. Runs epiLoRA on the mutant sequence.
4. Returns the epitope-probability difference at the mutation site::

       delta = WT_prob - mutant_prob

   *Positive* delta → the mutation **lowers** epitope probability at that site
   ⇒ the variant may help hide from the immune system (escape).
   *Negative* delta → the mutation **raises** epitope probability ⇒ not escape.

For sequences longer than ESM2's context window (~2000 residues) the model is
run on a centred window around the mutation site (±1000 residues), so every
gene — even ORF1a (4401 aa) — can be scored.

Usage
-----
    # Single mutation
    python run_epilora_escape.py --seq "MFVFLVLLPLVSSQCV..." --mutation D614G

    # From a FASTA file
    python run_epilora_escape.py --fasta protein.fasta --mutation D614G

    # Specify weights
    python run_epilora_escape.py --seq SEQ --mutation N501Y \
        --weights /path/to/esm2.pt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from utils import (
    DEFAULT_WEIGHTS,
    MAX_CTX,
    load_model,
    predict_probs,
    window_seq,
)


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """'D614G' → ('D', 613, 'G')  (position converted to 0-based)."""
    m = re.match(r"^([A-Za-z*]+)(\d+)([A-Za-z*]+)$", mutation)
    if not m:
        raise ValueError(f"Cannot parse mutation: {mutation!r}")
    return m.group(1), int(m.group(2)) - 1, m.group(3)  # 0-based


def run_escape_prediction(model, seq: str, mutation: str) -> dict:
    """Compare WT vs mutant epitope probability at the mutation site."""
    ref_aa, pos0, alt_aa = parse_mutation(mutation)

    if pos0 >= len(seq):
        raise ValueError(f"Position {pos0 + 1} exceeds sequence length {len(seq)}")
    if seq[pos0] != ref_aa:
        raise ValueError(
            f"WT residue mismatch: expected {ref_aa} at pos {pos0 + 1}, "
            f"got {seq[pos0]}")

    mutant_seq = seq[:pos0] + alt_aa + seq[pos0 + 1:]

    if len(seq) > MAX_CTX * 2 + 1:
        wt_win, pos_in_win = window_seq(seq, pos0)
        mut_win = wt_win[:pos_in_win] + alt_aa + wt_win[pos_in_win + 1:]
    else:
        wt_win, pos_in_win = seq, pos0
        mut_win = mutant_seq

    wt_probs = predict_probs(model, wt_win)
    mut_probs = predict_probs(model, mut_win)

    wt_prob = float(wt_probs[pos_in_win])
    mut_prob = float(mut_probs[pos_in_win])

    return {
        "ref_aa": ref_aa,
        "alt_aa": alt_aa,
        "pos": pos0 + 1,
        "wt_prob": round(wt_prob, 6),
        "mutant_prob": round(mut_prob, 6),
        "delta": round(wt_prob - mut_prob, 6),
        "seq_len": len(seq),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", type=str, default=None,
                   help="WT protein sequence (single-letter)")
    p.add_argument("--fasta", type=Path, default=None,
                   help="FASTA file with the WT protein sequence")
    p.add_argument("--mutation", type=str, required=True,
                   help="point mutation, e.g. D614G")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = p.parse_args()

    if args.seq:
        seq = args.seq.strip()
    elif args.fasta:
        from utils import parse_fasta
        _, seq = next(parse_fasta(args.fasta))
    else:
        p.error("provide --seq or --fasta")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[escape] loading model (device={device}) …", flush=True)
    model = load_model(args.weights, device)

    print(f"[escape] sequence length: {len(seq)}", flush=True)
    print(f"[escape] mutation: {args.mutation}", flush=True)

    result = run_escape_prediction(model, seq, args.mutation)
    print(f"\n  WT epitope prob:      {result['wt_prob']:.6f}")
    print(f"  Mutant epitope prob:  {result['mutant_prob']:.6f}")
    print(f"  Delta (WT - mut):     {result['delta']:+.6f}")
    if result["delta"] > 0:
        print(f"  → Mutation REDUCES epitope probability (potential escape)")
    else:
        print(f"  → Mutation does not reduce epitope probability")


if __name__ == "__main__":
    main()
