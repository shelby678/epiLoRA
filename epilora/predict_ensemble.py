"""Run the champion epiLoRA model -- ESM-IF1 + LoRA + RYS, 5-fold ensemble -- on
one or more antigen PDB structures.

    python predict_ensemble.py --pdb ../data/raw/5IBV.pdb
    python predict_ensemble.py --pdb a.pdb b.pdb --chain A --out-dir preds/

Unlike predict.py (one checkpoint, one chain), this averages the sigmoid
probabilities of several checkpoints -- the same ensembling scheme
eval_final.py uses -- and sweeps every chain of every PDB by default. The
default --weights are the five CV folds of the champion config
(allowed_species_homo_sapiens: rank=4, alpha=8, n_lora_layers=8, RYS 4-8),
whose per-fold held-out ROC-AUC is 0.793 +/- 0.009.

Must run in the fair-esm environment (epilora/env/bin/python).
"""

from __future__ import annotations

# fair-esm 2.0 predates biotite 1.0, which renamed filter_backbone ->
# filter_peptide_backbone. Alias it back before esm.inverse_folding is imported,
# otherwise `from biotite.structure import filter_backbone` raises ImportError.
import biotite.structure as _bs

if not hasattr(_bs, "filter_backbone"):
    _bs.filter_backbone = _bs.filter_peptide_backbone

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from predict import load_model, predict as predict_proba

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAMPION_WEIGHTS = sorted(
    (REPO_ROOT / "weights/ablation").glob("allowed_species_homo_sapiens_fold?.pt")
)


@torch.no_grad()
def ensemble_probs(models, coords, seq) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue epitope probability averaged over ``models``.

    Returns (mean, std) across the ensemble members; the std is a cheap
    fold-to-fold agreement signal, not a calibrated uncertainty.
    """
    stack = np.stack([predict_proba(m, coords, seq) for m in models])
    return stack.mean(axis=0), stack.std(axis=0)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pdb", type=Path, nargs="+", required=True, help="antigen PDB file(s)")
    p.add_argument("--chain", nargs="+", default=None,
                   help="chain id(s) to score (default: every chain in each PDB)")
    p.add_argument("--weights", type=Path, nargs="+", default=CHAMPION_WEIGHTS,
                   help="checkpoints to average (default: the 5 champion CV folds)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write one <pdb>_<chain>.csv per chain here")
    p.add_argument("--threshold", type=float, default=0.5, help="epitope call cutoff")
    args = p.parse_args()

    if not args.weights:
        p.error("no checkpoints given and no champion folds found under weights/ablation/")
    missing = [w for w in args.weights if not w.exists()]
    if missing:
        p.error("weights not found: " + ", ".join(str(m) for m in missing))

    import esm.inverse_folding.util as ifu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ensemble] device={device}  n_models={len(args.weights)}", file=sys.stderr)
    for w in args.weights:
        print(f"[ensemble]   {w}", file=sys.stderr)
    models = [load_model(w, device) for w in args.weights]

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for pdb in args.pdb:
        # Load+backbone-filter the whole file once, then slice chains from memory --
        # ifu.load_coords(path, chain) would otherwise re-parse the same file from
        # disk for every chain.
        structure = ifu.load_structure(str(pdb))
        all_chains = list(ifu.get_chains(structure))
        chains = args.chain or all_chains
        for chain in chains:
            if chain not in all_chains:
                p.error(f"chain {chain!r} not found in {pdb.name} (chains present: {', '.join(all_chains)})")
            coords, seq = ifu.extract_coords_from_structure(structure[structure.chain_id == chain])
            mean, std = ensemble_probs(models, coords, seq)
            called = int((mean >= args.threshold).sum())
            print(f"\n# {pdb.name} chain {chain}: {len(seq)} residues, "
                  f"{called} called epitope at p>={args.threshold} "
                  f"(mean p={mean.mean():.4f}, max p={mean.max():.4f})")
            print("pos\taa\tprob\tstd\tepitope")
            for i, (aa, pr, sd) in enumerate(zip(seq, mean, std), start=1):
                print(f"{i}\t{aa}\t{pr:.4f}\t{sd:.4f}\t{'1' if pr >= args.threshold else '0'}")

            if args.out_dir is not None:
                out = args.out_dir / f"{pdb.stem}_{chain}.csv"
                with open(out, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["pos", "aa", "prob", "prob_std", "epitope"])
                    for i, (aa, pr, sd) in enumerate(zip(seq, mean, std), start=1):
                        w.writerow([i, aa, f"{pr:.4f}", f"{sd:.4f}", int(pr >= args.threshold)])
                print(f"[ensemble] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
