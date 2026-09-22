"""Run the trained epiLoRA model on an antigen.

    python predict.py --pdb antigen.pdb --chain A --weights weights/epilora_if1.pt
    python predict.py --sequence MKTAYIAKQRQISFVKSHFSRQ --weights weights/opendde_fold1.pt

Prints per-residue epitope probabilities (and writes a CSV with ``--out``).
ESM-IF1 (the champion) is an inverse-folding model, so its input is a PDB
structure + chain and the sequence is read from the structure itself; the
sequence-only backbones (ESM2/ESM3/ESMc/ProstT5/ProtT5/OpenDDE) read just the
residue sequence -- either from a --pdb chain or straight from ``--sequence``
(a bare one-letter-code string or a single-record FASTA file), so no
structure is needed for them. Structure-reading needs fair-esm's util
(fair-esm env); the other backbones run in any env that can load their
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

from data import (build_extra_feats, parse_structure_model,
                  rsa_for_structure_file, select_residues)
from model import ESMIF1EpitopeModel, load_base_esmif1

# One-letter codes the sequence-only backbones accept from --sequence: the
# 20 standard amino acids plus the ambiguity/unknown codes PDB tools emit.
AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWYBXZUO")


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


def extra_feats_for(model, structure_path: Path | None, chain, seq: str):
    """The extra per-residue head features ``model`` needs, or None if its
    head reads the embedding alone. RSA is computed on the given chain alone,
    matching training -- so pass an antigen structure (None is fine for heads
    that don't read rsa)."""
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


def sequence_from_arg(value: str) -> str:
    """The antigen sequence from predict.py's ``--sequence``: a FASTA file
    with exactly one record, or a bare one-letter-code string (whitespace
    ignored, case-insensitive)."""
    path = Path(value)
    if path.exists():
        records, seq, in_record = [], [], False
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith(">"):
                if in_record:
                    records.append("".join(seq))
                seq, in_record = [], True
            elif in_record and line:
                seq.append(line)
        if in_record:
            records.append("".join(seq))
        if len(records) != 1:
            raise ValueError(f"{path}: expected exactly one FASTA record, "
                             f"found {len(records)}")
        return records[0]
    s = "".join(value.split()).upper()
    if not s or not set(s) <= AA_LETTERS:
        raise ValueError("--sequence must be one-letter amino-acid codes or "
                         "the path of an existing FASTA file")
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", type=Path, default=None,
                   help="antigen structure file (required for the esmif1 backbone; "
                        "the sequence-only backbones read the sequence of --chain from it)")
    p.add_argument("--sequence", type=str, default=None,
                   help="antigen residue sequence (one-letter codes) or a single-record "
                        "FASTA file -- the sequence-only backbones' structure-free input")
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
    # coordinates, every other backbone reads the residue sequence alone
    # (from --sequence or from --pdb's chain).
    backbone = torch.load(args.weights, map_location="cpu",
                          weights_only=False).get("backbone", "esmif1")

    if (args.pdb is None) == (args.sequence is None):
        p.error("give exactly one of --pdb or --sequence")
    if args.sequence is not None and backbone == "esmif1":
        p.error("--sequence is only for the sequence-only backbones; esmif1 "
                "reads a structure (--pdb)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chain = args.chain
    if backbone == "esmif1":
        if chain is None:
            import esm.inverse_folding.util as ifu
            chains = ifu.get_chains(ifu.load_structure(str(args.pdb)))
            if not chains:
                p.error(f"no chains found in {args.pdb}")
            chain = chains[0]
            print(f"[predict] no --chain given; using first chain '{chain}'", file=sys.stderr)
        from esm.inverse_folding.util import load_coords
        coords, seq = load_coords(str(args.pdb), chain)
    else:
        coords = None
        if args.sequence is not None:
            try:
                seq = sequence_from_arg(args.sequence)
            except ValueError as e:
                p.error(str(e))
        else:
            if chain is None:
                structure = parse_structure_model(args.pdb)
                chains = [] if structure is None else [c.id for c in structure.get_chains()]
                if not chains:
                    p.error(f"no chains found in {args.pdb}")
                chain = chains[0]
                print(f"[predict] no --chain given; using first chain '{chain}'", file=sys.stderr)
            seq = chain_sequence(args.pdb, chain)

    model = load_model(args.weights, device)
    if model.n_extra_feats and "rsa" in model.extra_feats and args.pdb is None:
        p.error("this checkpoint's head reads the rsa feature, which is computed "
                "from a structure -- rerun with --pdb instead of --sequence")
    feats = extra_feats_for(model, args.pdb, chain, seq)
    if feats is not None:
        print(f"[predict] head reads extra features: {', '.join(model.extra_feats)}",
              file=sys.stderr)
    probs = predict(model, coords, seq, feats)

    src = (f"{args.pdb} chain {chain}" if args.pdb is not None and chain is not None
           else str(args.pdb) if args.pdb is not None else "--sequence input")
    print(f"# {src}: {len(seq)} residues  (val_auc-trained model)")
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
