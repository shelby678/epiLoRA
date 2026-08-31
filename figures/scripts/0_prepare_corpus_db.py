"""Stage a training-data FASTA for mmseqs search: uppercase for search (these
FASTAs use lowercase non-standardly to mark epitope residues, which would
otherwise get read as mmseqs low-complexity soft-masking), with a side JSON
mapping each staged record's throwaway id back to its real metadata for
1_homology_search.py to look up after the fact.

    python 0_prepare_corpus_db.py --in_fasta epitopes.fasta \
        --out_fasta db/train.search.fasta --out_meta db/train.meta.json --log log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epitope_pipeline_common import fold_group_of, parse_training_fasta  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--in_fasta", required=True)
p.add_argument("--out_fasta", required=True)
p.add_argument("--out_meta", required=True)
p.add_argument("--log", required=True)
args = p.parse_args()

meta = {}
n_records = 0
with open(args.out_fasta, "w") as out:
    for header, seq in parse_training_fasta(args.in_fasta):
        rec_id = f"rec{n_records}"
        out.write(f">{rec_id}\n{seq.upper()}\n")
        try:
            fold_group = fold_group_of(header)
        except ValueError:
            fold_group = None  # no trailing 'i.j' fold label
        meta[rec_id] = {"header": header, "seq": seq, "fold_group": fold_group}
        n_records += 1

Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_meta, "w") as f:
    json.dump(meta, f)

with open(args.log, "w") as log:
    log.write(f"in_fasta: {args.in_fasta}\n")
    log.write(f"records staged: {n_records}\n")
