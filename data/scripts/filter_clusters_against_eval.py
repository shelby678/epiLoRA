"""Drop clusters.fasta members too similar to eval.fasta (rule filter_clusters_against_eval).

Mirrors get_eval_set.py's own leakage guard from the other direction: get_eval_set drops
eval candidates that are similar to older (train-side) sequences, but nothing stopped an
eval sequence's near-duplicate from staying in the train/ablation fastas built from
clusters.fasta. This script removes any cluster member with >=MIN_SIMILARITY identity to
an eval.fasta sequence, keeping the rest of that member's cluster intact (combine_epitopes
already knows how to re-elect a backbone if the original representative is the one
dropped). A cluster left with zero members is omitted entirely.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

in_clusters, in_eval, out_fasta, log_path = sys.argv[1:5]
MIN_SIMILARITY = 0.40
MMSEQS = os.path.expanduser("~/work/env/bin/mmseqs")


def parse_fasta(path):
    header, seq = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq)
            header, seq = line[1:], []
        else:
            seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def parse_clusters(path):
    """Yield (raw ">>CLUSTER ..." header line, [[member_header, aligned_seq], ...])."""
    cluster_header, members = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">>CLUSTER"):
            if cluster_header is not None:
                yield cluster_header, members
            cluster_header = line
            members = []
        elif line.startswith(">"):
            members.append([line[1:], ""])
        else:
            members[-1][1] += line
    if cluster_header is not None:
        yield cluster_header, members


clusters = list(parse_clusters(in_clusters))
eval_records = list(parse_fasta(in_eval))
n_members = sum(len(members) for _, members in clusters)

dropped = set()
if eval_records and n_members:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        members_fasta, eval_fasta = tmp / "members.fasta", tmp / "eval.fasta"

        with open(members_fasta, "w") as f:
            i = 0
            for _, members in clusters:
                for _, aligned_seq in members:
                    f.write(f">m{i}\n{aligned_seq.upper().replace('-', '')}\n")
                    i += 1

        with open(eval_fasta, "w") as f:
            for j, (_, seq) in enumerate(eval_records):
                f.write(f">e{j}\n{seq.upper()}\n")

        result_m8 = tmp / "result.m8"
        subprocess.run(
            [MMSEQS, "easy-search", str(members_fasta), str(eval_fasta), str(result_m8), str(tmp / "work"),
             "--min-seq-id", str(MIN_SIMILARITY)],
            check=True, capture_output=True, text=True,
        )
        with open(result_m8) as f:
            for line in f:
                query_id = line.split("\t", 1)[0]
                dropped.add(int(query_id[1:]))

n_clusters_in = len(clusters)
n_clusters_out = n_members_dropped = 0
with open(out_fasta, "w") as out:
    i = 0
    for cluster_header, members in clusters:
        kept = []
        for header, seq in members:
            if i in dropped:
                n_members_dropped += 1
            else:
                kept.append((header, seq))
            i += 1
        if not kept:
            continue
        out.write(f"{cluster_header}\n")
        for header, seq in kept:
            out.write(f">{header}\n{seq}\n")
        n_clusters_out += 1

with open(log_path, "w") as log:
    log.write(f"clusters in: {n_clusters_in}\n")
    log.write(f"clusters out (>=1 member remaining): {n_clusters_out}\n")
    log.write(f"members in: {n_members}\n")
    log.write(f"members dropped (>={int(MIN_SIMILARITY * 100)}% similar to an eval seq): {n_members_dropped}\n")
