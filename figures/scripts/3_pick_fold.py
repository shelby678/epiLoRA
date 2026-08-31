"""Pick which champion checkpoint(s) to run on the query, from
1_homology_search.py's best hit against the champion's own training FASTA.

    python 3_pick_fold.py --homology_json homology_champion.json \
        --weights_glob "../weights/ablation/allowed_species_homo_sapiens_fold*.pt" \
        --out_json fold_choice.json --log log

If the query overlaps a training cluster above the (lenient) fold-avoidance
threshold, use the single fold whose training split excluded that cluster
(fold `i` = the cluster's fold-group label -- the only checkpoint guaranteed
not to have leaked this antigen or its homologs). Otherwise fall back to the
full 5-fold ensemble mean.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--homology_json", required=True)
p.add_argument("--weights_glob", required=True)
p.add_argument("--out_json", required=True)
p.add_argument("--log", required=True)
args = p.parse_args()

homology = json.loads(Path(args.homology_json).read_text())
all_weights = sorted(glob.glob(args.weights_glob))
if not all_weights:
    raise SystemExit(f"no checkpoints matched --weights_glob {args.weights_glob!r}")

FOLD_RE = re.compile(r"fold(\d+)\.pt$")


def fold_of(w: str) -> int:
    m = FOLD_RE.search(w)
    if not m:
        raise SystemExit(f"checkpoint filename doesn't end in 'fold<N>.pt': {w}")
    return int(m.group(1))

if homology["hit"]:
    fold = homology["fold_group"]
    weights = [w for w in all_weights if fold_of(w) == fold]
    if not weights:
        raise SystemExit(f"homology hit picked fold {fold}, but no checkpoint in "
                         f"--weights_glob matches fold{fold} -- {all_weights}")
    reason = (f"query overlaps training cluster {homology['header']!r} "
              f"(pident={homology['pident']:.3f}) at/above the fold-avoidance threshold, "
              f"which is held out of fold {fold}'s training set (every other fold trained on "
              f"it) -- using that single checkpoint to avoid leakage")
else:
    weights = all_weights
    reason = "no training cluster overlap found above the fold-avoidance threshold -- " \
             "no fold is specially unleaked, so using the full 5-fold ensemble mean"

out = {"mode": "single_fold" if homology["hit"] else "ensemble", "fold": homology.get("fold_group"),
       "weights": weights, "reason": reason}

Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump(out, f)

with open(args.log, "w") as log:
    log.write(reason + "\n")
    log.write(f"weights: {weights}\n")
