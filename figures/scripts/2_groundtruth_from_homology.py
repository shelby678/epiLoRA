"""Transfer epitope calls onto the query structure's residues by homology,
unioned across *every* hit 1_homology_search.py found against the training
corpus: a residue is called an epitope if it lines up with an epitope call
(lowercase) in ANY matching training sequence.

    python 2_groundtruth_from_homology.py --pdb query.pdb --chain A \
        --all_hits_json all_hits_train.json --out_csv groundtruth.csv --log log

Each query residue gets value=100.0/0.0, plus a homology_covered flag so "not
an epitope by homology" and "no hit ever covered this residue" stay
distinguishable in the CSV. Must run in the fair-esm environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epitope_pipeline_common import default_chain, load_query, parse_chains, write_value_csv  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pdb", required=True)
p.add_argument("--chain", default=None, help="single chain id, or '|'-separated for a multi-chain antigen")
p.add_argument("--all_hits_json", required=True)
p.add_argument("--out_csv", required=True)
p.add_argument("--log", required=True)
args = p.parse_args()

chains = parse_chains(args.chain) if args.chain else [default_chain(args.pdb)]
_, seq = load_query(args.pdb, chains)
hits = json.loads(Path(args.all_hits_json).read_text())

epitope_by_pos: dict[int, bool] = {}
for hit in hits:
    target_seq = hit["seq"]
    mapping = {int(k): v for k, v in hit["query_to_target_pos"].items()}
    for pos, tpos in mapping.items():
        is_epitope = target_seq[tpos - 1].islower()
        epitope_by_pos[pos] = epitope_by_pos.get(pos, False) or is_epitope

rows = []
n_covered = n_epitope = 0
for pos, aa in enumerate(seq, start=1):
    covered = pos in epitope_by_pos
    is_epitope = epitope_by_pos.get(pos, False)
    n_covered += int(covered)
    n_epitope += int(is_epitope)
    rows.append({
        "pos": pos, "aa": aa,
        "value": 100.0 if is_epitope else 0.0,
        "epitope": int(is_epitope),
        "homology_covered": int(covered),
    })

write_value_csv(Path(args.out_csv), rows, ["pos", "aa", "value", "epitope", "homology_covered"])

with open(args.log, "w") as log:
    log.write(f"pdb: {args.pdb}  chains: {chains}\n")
    log.write(f"hits found: {len(hits)}\n")
    log.write(f"residues: {len(seq)}  homology_covered: {n_covered}  called epitope (union): {n_epitope}\n")
    if n_covered == 0:
        log.write("WARNING: no homology coverage at all -- every residue defaults to non-epitope\n")
