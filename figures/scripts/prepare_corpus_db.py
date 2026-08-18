"""Stage a training-data FASTA for mmseqs search: uppercase for search (these
FASTAs use lowercase non-standardly, to mark epitope residues -- see
data/README.md -- which would otherwise get read as mmseqs low-complexity
soft-masking), with a side JSON mapping each staged record's throwaway id
back to its real metadata, for homology_search.py to look up after the fact.

Auto-detects which of two formats --in_fasta is (no flag needed):

  flat   data/results/epitopes.fasta or data/train_test_eval/*_epitopes.fasta:
         one record per line-pair, header ``{instance} {date} {resolution}
         {chains} {heavy} {light} [fold_label]`` -- fold_label present only
         in the train_test_eval/*_epitopes.fasta files (one row per 95%-
         identity cluster); absent in data/results/epitopes.fasta (one row
         per raw, pre-clustering antigen instance -- the right corpus for
         "does this exact/near-exact structure appear in the training
         data," since the cluster-collapsed files only keep whichever
         member combine_epitopes.py elected as each cluster's representative).
  block  data/train_test_eval/eval/train_clusters.fasta: ``>>CLUSTER {rep}
         {fold_label}`` blocks, each holding every individual raw member
         structure (gapped/aligned to the cluster frame) -- lets
         matches_members.csv list actual SAbDab structures, not just one
         representative per cluster.

    python prepare_corpus_db.py --in_fasta all_epitopes.fasta \
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


def parse_clusters(path):
    """Yield (rep_instance, fold_label, [(instance, date, resolution, chains,
    heavy_species, light_species, aligned_seq), ...]) for the block format
    -- see cluster_fasta.py's own writer."""
    cluster_rep = fold_label = None
    members = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">>CLUSTER"):
            if cluster_rep is not None:
                yield cluster_rep, fold_label, members
            _, cluster_rep, fold_label = line.split()
            members = []
        elif line.startswith(">"):
            instance, date, resolution, chains, heavy, light = line[1:].split()[:6]
            members.append([instance, date, resolution, chains, heavy, light, ""])
        else:
            members[-1][6] += line
    if cluster_rep is not None:
        yield cluster_rep, fold_label, members


is_block_format = any(line.startswith(">>CLUSTER") for line in Path(args.in_fasta).open())

meta = {}
n_records = 0
with open(args.out_fasta, "w") as out:
    if is_block_format:
        n_clusters = 0
        for cluster_rep, fold_label, members in parse_clusters(args.in_fasta):
            n_clusters += 1
            for instance, date, resolution, chains, heavy, light, aligned_seq in members:
                rec_id = f"rec{n_records}"
                out.write(f">{rec_id}\n{aligned_seq.replace('-', '').upper()}\n")
                meta[rec_id] = {
                    "instance": instance, "pdb_id": instance.split("-")[0],
                    "date": date, "resolution": resolution, "antigen_chains": chains,
                    "heavy_species": heavy, "light_species": light,
                    "cluster_rep": cluster_rep, "fold_label": fold_label,
                }
                n_records += 1
    else:
        for header, seq in parse_training_fasta(args.in_fasta):
            rec_id = f"rec{n_records}"
            out.write(f">{rec_id}\n{seq.upper()}\n")
            try:
                fold_group = fold_group_of(header)
            except ValueError:
                fold_group = None  # no trailing 'i.j' label -- e.g. data/results/epitopes.fasta
            meta[rec_id] = {"header": header, "seq": seq, "fold_group": fold_group}
            n_records += 1

Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_meta, "w") as f:
    json.dump(meta, f)

with open(args.log, "w") as log:
    log.write(f"in_fasta: {args.in_fasta}\n")
    log.write(f"format: {'block (cluster_fasta.py)' if is_block_format else 'flat'}\n")
    log.write(f"records staged: {n_records}\n")
