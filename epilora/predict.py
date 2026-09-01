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
import torch.nn as nn

from data import build_extra_feats, rsa_for_structure_file
from model import ESMIF1EpitopeModel, load_base_esmif1


def load_model(weights: Path, device: str) -> nn.Module:
    """Load a checkpoint saved by train.py, dispatching on its ``backbone`` field
    (older checkpoints predate that field and default to esmif1)."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    backbone = ckpt.get("backbone", "esmif1")
    if backbone == "esmif1":
        esm_model, alphabet = load_base_esmif1()
        model = ESMIF1EpitopeModel(esm_model, alphabet, **cfg).to(device)
    elif backbone == "esm2":
        from model import ESM2EpitopeModel, load_base_esm2
        esm_model, alphabet = load_base_esm2(cfg["size"])
        model = ESM2EpitopeModel(esm_model, alphabet, **cfg).to(device)
    elif backbone == "esm3":
        from model import ESM3EpitopeModel, load_base_esm3
        esm_model = load_base_esm3()
        model = ESM3EpitopeModel(esm_model, **cfg).to(device)
    elif backbone in ("prostt5", "prott5"):
        from model import ProstT5EpitopeModel, load_base_prostt5
        t5_model, tokenizer = load_base_prostt5(cfg["name"])
        model = ProstT5EpitopeModel(t5_model, tokenizer, **cfg).to(device)
    else:
        from model import ESMCEpitopeModel, load_base_esmc
        esm_model = load_base_esmc(cfg["size"])
        model = ESMCEpitopeModel(esm_model, **cfg).to(device)
    model.load_trainable_state_dict(ckpt["trainable_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: ESMIF1EpitopeModel, coords, seq, feats=None) -> np.ndarray:
    """``feats`` is the (L, n) extra head-feature matrix (see extra_feats_for),
    required only for a checkpoint whose head reads extra features."""
    logits = model([coords], [seq], None if feats is None else [feats])[0].cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> per-residue probability


def extra_feats_for(model, structure_path: Path, chain: str, seq: str):
    """The extra per-residue head features ``model`` needs for this chain, or
    None if its head reads the embedding alone. RSA is computed on the given
    chain alone, matching training -- so pass an antigen structure."""
    if not model.n_extra_feats:
        return None
    rsa = (rsa_for_structure_file(structure_path, [chain], len(seq))
           if "rsa" in model.extra_feats else None)
    return build_extra_feats(model.extra_feats, seq, rsa)


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
    feats = extra_feats_for(model, args.pdb, chain, seq)
    if feats is not None:
        print(f"[predict] head reads extra features: {', '.join(model.extra_feats)}",
              file=sys.stderr)
    probs = predict(model, coords, seq, feats)

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
