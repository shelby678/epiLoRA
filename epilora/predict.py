"""Run the trained epiLoRA model on an antigen structure.

    python predict.py --pdb antigen.pdb --chain A --weights weights/epilora_if1.pt

Prints per-residue epitope probabilities (and writes a CSV with ``--out``).
ESM-IF1 (the champion) is an inverse-folding model, so its input is a PDB
structure + chain and the sequence is read from the structure itself; the
sequence-only backbones (ESM2/ESM3/ESMc/ProstT5) read just the chain's residue
sequence; the OpenDDE backbone reads the sequence plus the chain's CA geometry
(its confidence lane, see models/opendde/). Structure-reading needs fair-esm's
util (fair-esm env); the other backbones run in any env that can load their
checkpoint -- esm3/esmc under epilora/env_esm3, opendde under
machine_config.yaml's env_opendde.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import (backbone_coords_for_structure_file, build_extra_feats,
                  parse_structure_model, rsa_for_structure_file, select_residues)
from model import ESMIF1EpitopeModel, load_base_esmif1


def load_model(weights: Path, device: str, emb_cache=None) -> nn.Module:
    """Load a checkpoint saved by train.py, dispatching on its ``backbone`` field
    (older checkpoints predate that field and default to esmif1).

    ``emb_cache`` (optional) is read only by the opendde backbone: the
    directory its frozen trunk features are cached in (train.py points it at
    the coords cache next to --structures); every other backbone ignores it."""
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
    elif backbone == "opendde":
        from models.opendde import OpenDDEPairformerEpitopeModel, load_base_opendde
        trunk = load_base_opendde(cfg["checkpoint"], device=device)
        model = OpenDDEPairformerEpitopeModel(trunk, emb_cache=emb_cache,
                                              **cfg).to(device)
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


def chain_sequence(structure_path: Path, chain: str) -> str:
    """Residue sequence of ``chain`` (standard amino acids, file order) -- the
    input of the sequence-only backbones, matching training's FASTA convention
    (data.select_residues, so extra feats line up with the sequence)."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    model = parse_structure_model(structure_path)
    if model is None:
        raise ValueError(f"could not parse structure {structure_path}")
    residues = select_residues(model, [chain], structure_path)
    if not residues:
        raise ValueError(f"chain {chain!r} has no standard amino-acid residues "
                         f"in {structure_path}")
    return "".join(protein_letters_3to1.get(r.resname, "X") for r in residues)


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

    # Which backbone this checkpoint trained -- esmif1 reads the structure's
    # coordinates, every other backbone reads the chain sequence alone.
    backbone = torch.load(args.weights, map_location="cpu",
                          weights_only=False).get("backbone", "esmif1")

    chain = args.chain
    if chain is None:
        if backbone == "esmif1":
            import esm.inverse_folding.util as ifu
            chains = ifu.get_chains(ifu.load_structure(str(args.pdb)))
        else:
            structure = parse_structure_model(args.pdb)
            chains = [] if structure is None else [c.id for c in structure.get_chains()]
        if not chains:
            p.error(f"no chains found in {args.pdb}")
        chain = chains[0]
        print(f"[predict] no --chain given; using first chain '{chain}'", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if backbone == "esmif1":
        from esm.inverse_folding.util import load_coords
        coords, seq = load_coords(str(args.pdb), chain)
    elif backbone == "opendde":
        # The trunk reads sequence; its confidence lane additionally reads the
        # CA geometry (see models/opendde/) -- both without fair-esm.
        seq = chain_sequence(args.pdb, chain)
        coords = backbone_coords_for_structure_file(args.pdb, [chain], len(seq))
    else:
        coords, seq = None, chain_sequence(args.pdb, chain)
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
