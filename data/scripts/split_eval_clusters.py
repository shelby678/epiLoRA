"""Split clusters.fasta into an eval set and a train set by deposit date, then make sure
neither leaks into the other.

A cluster is an eval candidate if its oldest member's deposit date is newer than the cutoff
date; every other cluster starts out in the train set. Any eval candidate with a member
>=MIN_SIMILARITY identical to a train-cluster member is moved into the train set instead of
being discarded -- reclaiming its data rather than wasting it. Since growing the train set
can turn a previously-safe eval cluster newly unsafe, this repeats (mmseqs easy-search,
reclaim, repeat) until a full pass finds nothing left to reclaim.
"""
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

in_clusters, eval_out, train_out, log_path, cutoff_date_str = sys.argv[1:6]
CUTOFF = date.fromisoformat(cutoff_date_str)
MIN_SIMILARITY = 0.40
MMSEQS = os.path.expanduser("~/work/env/bin/mmseqs")


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


def parse_date(member_header):
    y, m, d = member_header.split()[1].split("/")
    return date(int(y), int(m), int(d))


def oldest_date(members):
    return min(parse_date(header) for header, _ in members)


def write_members_fasta(path, clusters, prefix):
    """Flatten each cluster's members to plain ungapped/uppercased records; return a
    parallel list mapping each written record back to its index in `clusters`."""
    member_cluster = []
    with open(path, "w") as f:
        i = 0
        for cluster_idx, (_, members) in enumerate(clusters):
            for _, aligned_seq in members:
                f.write(f">{prefix}{i}\n{aligned_seq.upper().replace('-', '')}\n")
                member_cluster.append(cluster_idx)
                i += 1
    return member_cluster


def flagged_eval_cluster_indices(eval_clusters, train_clusters, tmp):
    if not eval_clusters or not train_clusters:
        return set()
    eval_fasta, train_fasta = tmp / "eval.fasta", tmp / "train.fasta"
    member_cluster = write_members_fasta(eval_fasta, eval_clusters, "e")
    write_members_fasta(train_fasta, train_clusters, "t")
    result_m8 = tmp / "result.m8"
    subprocess.run(
        [MMSEQS, "easy-search", str(eval_fasta), str(train_fasta), str(result_m8), str(tmp / "work"),
         "--min-seq-id", str(MIN_SIMILARITY)],
        check=True, capture_output=True, text=True,
    )
    flagged = set()
    with open(result_m8) as f:
        for line in f:
            query_id = line.split("\t", 1)[0]
            flagged.add(member_cluster[int(query_id[1:])])
    return flagged


def write_clusters(path, clusters):
    with open(path, "w") as out:
        for cluster_header, members in clusters:
            out.write(f"{cluster_header}\n")
            for header, seq in members:
                out.write(f">{header}\n{seq}\n")


all_clusters = list(parse_clusters(in_clusters))
eval_clusters, train_clusters = [], []
for cluster_header, members in all_clusters:
    (eval_clusters if oldest_date(members) > CUTOFF else train_clusters).append((cluster_header, members))

n_candidates = len(eval_clusters)
n_iterations = n_reclaimed = 0
with tempfile.TemporaryDirectory() as tmp_root:
    while True:
        n_iterations += 1
        iter_tmp = Path(tmp_root) / f"iter{n_iterations}"
        iter_tmp.mkdir()
        flagged = flagged_eval_cluster_indices(eval_clusters, train_clusters, iter_tmp)
        if not flagged:
            break
        train_clusters.extend(c for i, c in enumerate(eval_clusters) if i in flagged)
        eval_clusters = [c for i, c in enumerate(eval_clusters) if i not in flagged]
        n_reclaimed += len(flagged)

write_clusters(eval_out, eval_clusters)
write_clusters(train_out, train_clusters)

with open(log_path, "w") as log:
    log.write(f"clusters in: {len(all_clusters)}\n")
    log.write(f"eval candidates (oldest member > {CUTOFF.isoformat()}): {n_candidates}\n")
    log.write(f"convergence iterations: {n_iterations}\n")
    log.write(
        f"eval candidates reclaimed into train "
        f"(>={int(MIN_SIMILARITY * 100)}% similar to a train-cluster member): {n_reclaimed}\n"
    )
    log.write(f"eval clusters (final): {len(eval_clusters)}\n")
    log.write(f"train clusters (final): {len(train_clusters)}\n")
