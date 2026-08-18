"""Transfer epitope calls onto the query structure's residues by homology,
unioned across *every* hit homology_search.py found (--out_all_hits) against
the training corpus -- no identity/similarity cutoff decides "the" match;
a residue is called an epitope if it lines up with an epitope call in ANY
matching training sequence, same as combine_epitopes.py unions a cluster's
member calls onto its representative (data/README.md).

    python groundtruth_from_homology.py --pdb query.pdb --chain A \
        --all_hits_json all_hits_train.json --out_csv groundtruth.csv --log log

For each query residue: value=100.0 if ANY hit's aligned training residue at
that position was called an epitope (lowercase) else 0.0; homology_covered=1
if ANY hit aligns to that position at all (regardless of its call) -- so
"not an epitope by homology" and "no hit ever covered this residue" stay
distinguishable in the CSV even though they render the same B-factor.

Must run in the fair-esm environment (epilora/env/bin/python3).
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

# pos -> True/False as soon as ANY hit calls it an epitope; pos -> False if
# only ever seen as non-epitope; absent if no hit ever covers pos at all.
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
    for hit in sorted(hits, key=lambda h: -h["pident"])[:10]:
        log.write(f"  {hit.get('header', hit.get('instance'))}  pident={hit['pident']:.3f}  "
                  f"aligned_cols={len(hit['query_to_target_pos'])}\n")
    log.write(f"residues: {len(seq)}  homology_covered: {n_covered}  called epitope (union): {n_epitope}\n")
    if n_covered == 0:
        log.write("WARNING: no homology coverage at all -- every residue defaults to non-epitope; "
                   "this is an absence-of-evidence B-factor, not a negative epitope call\n")
