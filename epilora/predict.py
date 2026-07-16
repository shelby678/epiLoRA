"""Run the trained epiLoRA model on an antigen structure.

    python predict.py --pdb antigen.pdb --chain A --weights weights/epilora_if1.pt

Prints per-residue epitope probabilities (and writes a CSV with ``--out``).
ESM-IF1 is an inverse-folding model, so the input is a PDB structure + chain;
the sequence is read from the structure itself.

Must run in the fair-esm (py3.9) environment — see README / requirements.txt.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from model import ESMIF1EpitopeModel, load_base_esmif1


def load_model(weights: Path, device: str) -> ESMIF1EpitopeModel:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    esm_model, alphabet = load_base_esmif1()
    model = ESMIF1EpitopeModel(esm_model, alphabet, **cfg).to(device)
    model.load_trainable_state_dict(ckpt["trainable_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: ESMIF1EpitopeModel, coords, seq) -> np.ndarray:
    logits = model([coords], [seq])[0].cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> per-residue probability


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", type=Path, required=True, help="antigen PDB file")
    p.add_argument("--chain", default=None, help="chain id (default: first chain)")
    p.add_argument("--weights", type=Path, default=Path("weights/epilora_if1.pt"))
    p.add_argument("--out", type=Path, default=None, help="optional CSV output path")
    p.add_argument("--threshold", type=float, default=0.5, help="epitope call cutoff")
    args = p.parse_args()

    if not args.weights.exists():
        p.error(f"weights not found: {args.weights}\n"
                f"Download the checkpoint and place it there (see README), "
                f"or train one with train.py.")

    from esm.inverse_folding.util import load_coords

    chain = args.chain
    if chain is None:
        import esm.inverse_folding.util as ifu
        chains = ifu.get_chains(ifu.load_structure(str(args.pdb)))
        chain = chains[0]
        print(f"[predict] no --chain given; using first chain '{chain}'", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coords, seq = load_coords(str(args.pdb), chain)
    model = load_model(args.weights, device)
    probs = predict(model, coords, seq)

    print(f"# {args.pdb} chain {chain}: {len(seq)} residues  (val_auc-trained model)")
    print("pos\taa\tprob\tepitope")
    for i, (aa, pr) in enumerate(zip(seq, probs), start=1):
        print(f"{i}\t{aa}\t{pr:.4f}\t{'1' if pr >= args.threshold else '0'}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pos", "aa", "prob", "epitope"])
            for i, (aa, pr) in enumerate(zip(seq, probs), start=1):
                w.writerow([i, aa, f"{pr:.4f}", int(pr >= args.threshold)])
        print(f"[predict] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
