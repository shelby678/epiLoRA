"""Run epiLoRA on the query structure using whichever checkpoint(s)
3_pick_fold.py selected, and average.

    python 4_predict_epitope.py --pdb query.pdb --chain A \
        --fold_choice fold_choice.json --out_csv prediction.csv --log log

Must run in the fair-esm environment (epilora/env/bin/python3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "epilora"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import load_model, predict as predict_probs  # noqa: E402
from epitope_pipeline_common import default_chain, load_query, parse_chains, write_value_csv  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pdb", required=True)
p.add_argument("--chain", default=None, help="single chain id, or '|'-separated for a multi-chain antigen")
p.add_argument("--fold_choice", required=True)
p.add_argument("--out_csv", required=True)
p.add_argument("--log", required=True)
p.add_argument("--threshold", type=float, default=0.5)
args = p.parse_args()

chains = parse_chains(args.chain) if args.chain else [default_chain(args.pdb)]
coords, seq = load_query(args.pdb, chains)

fold_choice = json.loads(Path(args.fold_choice).read_text())
weights = [Path(w) for w in fold_choice["weights"]]

device = "cuda" if torch.cuda.is_available() else "cpu"
models = [load_model(w, device) for w in weights]
probs_stack = np.stack([predict_probs(m, coords, seq) for m in models])
mean_probs = probs_stack.mean(axis=0)

rows = [
    {"pos": i, "aa": aa, "value": 100.0 * float(p_), "prob": float(p_),
     "epitope": int(p_ >= args.threshold)}
    for i, (aa, p_) in enumerate(zip(seq, mean_probs), start=1)
]
write_value_csv(Path(args.out_csv), rows, ["pos", "aa", "value", "prob", "epitope"])

with open(args.log, "w") as log:
    log.write(f"pdb: {args.pdb}  chains: {chains}\n")
    log.write(f"mode: {fold_choice['mode']}\n")
    log.write(f"reason: {fold_choice['reason']}\n")
    log.write(f"checkpoints used: {[str(w) for w in weights]}\n")
    log.write(f"residues: {len(seq)}  predicted epitope (p>={args.threshold}): "
              f"{sum(r['epitope'] for r in rows)}\n")
