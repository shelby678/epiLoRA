"""Cluster antigen seqs at 95% identity with mmseqs2, assign each cluster a random fold label.

This rule no longer merges epitope calls -- it just clusters and records every member's
epitope-cased sequence aligned to its cluster representative's coordinate frame (using
mmseqs2's own clustering alignments, via result2msa/A3M), so `combine_epitopes` can later
decide, per ablation, which members' epitope calls to merge.

Output format (fasta-like, one block per cluster)::

    >>CLUSTER {rep_instance} {antigen_chains} {fold_label}
    >{instance} {date} {resolution} {heavy_species} {light_species}
    {seq aligned to rep, '-' where this member has no residue at that column}
    >{instance2} ...
    {seq2...}
    >>CLUSTER ...
"""
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

in_fasta, out_fasta, log_path = sys.argv[1:4]
MIN_SEQ_ID = 0.95
MMSEQS = os.path.expanduser("~/work/env/bin/mmseqs")

random.seed(0)
FOLD_LABELS = [f"{i}.{j}" for i in range(1, 6) for j in range(2)]


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


def align_to_rep(rep_len, aligned_row, member_seq):
    """Place member_seq's (case-preserved) residues into rep's column frame; '-' for gaps."""
    out = ["-"] * rep_len
    rep_col = mem_pos = 0
    for ch in aligned_row:
        if ch == "-":
            rep_col += 1
        elif ch.islower():
            mem_pos += 1
        else:
            out[rep_col] = member_seq[mem_pos]
            rep_col += 1
            mem_pos += 1
    return "".join(out)


records = list(parse_fasta(in_fasta))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    query_fasta = tmp / "query.fasta"
    with open(query_fasta, "w") as f:
        for i, (_, seq) in enumerate(records):
            f.write(f">seq{i}\n{seq.upper()}\n")

    query_db = tmp / "queryDB"
    clu_db = tmp / "queryDB_clu"
    msa_db = tmp / "queryDB_clu_msa"
    unpacked = tmp / "unpacked"
    unpacked.mkdir()

    subprocess.run([MMSEQS, "createdb", str(query_fasta), str(query_db)], check=True, capture_output=True, text=True)
    subprocess.run(
        [MMSEQS, "cluster", str(query_db), str(clu_db), str(tmp / "work"), "--min-seq-id", str(MIN_SEQ_ID)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [MMSEQS, "result2msa", str(query_db), str(query_db), str(clu_db), str(msa_db), "--msa-format-mode", "5"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [MMSEQS, "unpackdb", str(msa_db), str(unpacked), "--unpack-name-mode", "0"],
        check=True, capture_output=True, text=True,
    )

    # unpacked.iterdir() order is filesystem/readdir order, not a semantic order --
    # sort by rep_index so fold-label assignment (below) is reproducible across
    # runs/machines given the same random.seed, instead of depending on directory order.
    parsed_clusters = []
    for cluster_file in unpacked.iterdir():
        entries = list(parse_fasta(cluster_file))
        rep_id, _ = entries[0]
        rep_index = int(rep_id[3:])
        parsed_clusters.append((rep_index, entries))
    parsed_clusters.sort(key=lambda x: x[0])

    with open(out_fasta, "w") as out:
        n_clusters = 0
        for rep_index, entries in parsed_clusters:
            rep_header, rep_seq = records[rep_index]
            rep_fields = rep_header.split()
            rep_instance, antigen_chains = rep_fields[0], rep_fields[3]
            fold_label = random.choice(FOLD_LABELS)

            out.write(f">>CLUSTER {rep_instance} {antigen_chains} {fold_label}\n")
            for member_id, aligned_row in entries:
                member_index = int(member_id[3:])
                member_header, member_seq = records[member_index]
                m_instance, m_date, m_resolution, _, m_heavy, m_light = member_header.split()[:6]
                aligned_seq = align_to_rep(len(rep_seq), aligned_row, member_seq)
                out.write(f">{m_instance} {m_date} {m_resolution} {m_heavy} {m_light}\n")
                out.write(aligned_seq + "\n")
            n_clusters += 1

with open(log_path, "w") as log:
    log.write(f"input records: {len(records)}\n")
    log.write(f"clusters: {n_clusters}\n")
