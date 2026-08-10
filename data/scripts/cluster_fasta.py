"""Cluster antigen seqs at 95% identity with mmseqs2, assign each cluster a random fold label.

Output format (fasta-like, one block per cluster)::

    >>CLUSTER {rep_instance} {fold_label}
    >{instance} {date} {resolution} {antigen_chains} {heavy_species} {light_species}
    {seq aligned to the cluster's shared frame, '-' where this member has no residue at that column}
    >{instance2} ...
    {seq2...}
    >>CLUSTER ...

Each member carries its own antigen_chains (rather than only the cluster header
carrying the representative's) so that combine_epitopes.py can elect a different
member as backbone without stranding it with the wrong PDB entry's chain IDs.
"""
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# usage: cluster_fasta.py <in_fasta> <out_fasta> <log_path>
in_fasta, out_fasta, log_path = sys.argv[1:4]
MIN_SEQ_ID = 0.95  # mmseqs2 clustering identity threshold
MMSEQS = os.path.expanduser("~/work/env/bin/mmseqs")

random.seed(0)
# 5-fold CV labels, each fold split into two halves -> one label assigned per cluster
FOLD_LABELS = [f"{i}.{j}" for i in range(1, 6) for j in range(2)]


def parse_fasta(path):
    """Yield (header, sequence) pairs from a fasta file, header with the leading '>' stripped."""
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


def parse_member_row(aligned_row, member_seq):
    """Split an mmseqs A3M row into per-rep-column chars (None = deletion) and insertion
    runs (case-preserved), keyed by the rep column each insertion precedes (rep_len itself
    keys a trailing insertion after the last column). mmseqs2's alignment is local, so it can
    stop short of member_seq's own end; whatever it doesn't cover is kept as part of that
    trailing insertion instead of silently dropped."""
    chars_by_col = []
    insertions = {}
    rep_col = mem_pos = 0
    run_start = None  # mem_pos where the current insertion run started, if any
    for ch in aligned_row:
        if ch.islower():
            # insertion: consumes a member residue but no rep column
            if run_start is None:
                run_start = mem_pos
            mem_pos += 1
            continue
        if run_start is not None:
            # run just ended -- file it under the rep column it precedes
            insertions[rep_col] = member_seq[run_start:mem_pos]
            run_start = None
        if ch == "-":
            # deletion: rep has this column, member doesn't
            chars_by_col.append(None)
            rep_col += 1
        else:
            # match/mismatch: both have this column
            chars_by_col.append(member_seq[mem_pos])
            rep_col += 1
            mem_pos += 1
    tail_start = run_start if run_start is not None else mem_pos
    if tail_start < len(member_seq):
        insertions[rep_col] = member_seq[tail_start:]
    return chars_by_col, insertions


def render_row(rep_len, chars_by_col, insertions, max_len):
    """Lay out one member across the cluster's shared frame: `rep_len` rep-derived columns
    plus `max_len[point]` extra columns wherever any cluster member has an insertion at
    that point. Members without a residue in a given slot get '-' there."""
    parts = []
    for point in range(rep_len + 1):  # rep_len+1 insertion points: before/between/after columns
        width = max_len.get(point, 0)
        if width:
            # pad this member's run (possibly empty) out to the cluster-wide width for this point
            ins = insertions.get(point, "")
            parts.append(ins + "-" * (width - len(ins)))
        if point < rep_len:
            char = chars_by_col[point]
            parts.append(char if char is not None else "-")
    return "".join(parts)


records = list(parse_fasta(in_fasta))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    query_fasta = tmp / "query.fasta"
    with open(query_fasta, "w") as f:
        # mmseqs needs uppercase sequences; headers are just the record's index into `records`
        for i, (_, seq) in enumerate(records):
            f.write(f">seq{i}\n{seq.upper()}\n")

    query_db = tmp / "queryDB"
    clu_db = tmp / "queryDB_clu"
    msa_db = tmp / "queryDB_clu_msa"
    unpacked = tmp / "unpacked"
    unpacked.mkdir()

    # build an mmseqs sequence database from the input fasta
    subprocess.run([MMSEQS, "createdb", str(query_fasta), str(query_db)], check=True, capture_output=True, text=True)
    # cluster sequences at MIN_SEQ_ID identity
    subprocess.run(
        [MMSEQS, "cluster", str(query_db), str(clu_db), str(tmp / "work"), "--min-seq-id", str(MIN_SEQ_ID)],
        check=True, capture_output=True, text=True,
    )
    # align every cluster member to its representative, producing one A3M MSA per cluster
    subprocess.run(
        [MMSEQS, "result2msa", str(query_db), str(query_db), str(clu_db), str(msa_db), "--msa-format-mode", "5"],
        check=True, capture_output=True, text=True,
    )
    # split the packed MSA database into individual per-cluster A3M files
    subprocess.run(
        [MMSEQS, "unpackdb", str(msa_db), str(unpacked), "--unpack-name-mode", "0"],
        check=True, capture_output=True, text=True,
    )

    # sort by rep_index so fold-label assignment (below) is reproducible across
    # runs/machines given the same random.seed, instead of depending on directory order.
    parsed_clusters = []
    for cluster_file in unpacked.iterdir():
        entries = list(parse_fasta(cluster_file))
        rep_id, _ = entries[0]
        rep_index = int(rep_id[3:])  # strip the "seq" prefix added above
        parsed_clusters.append((rep_index, entries))
    parsed_clusters.sort(key=lambda x: x[0])

    with open(out_fasta, "w") as out:
        n_clusters = 0
        for rep_index, entries in parsed_clusters:
            # the rep's original header carries the metadata written on the >>CLUSTER line
            rep_header, rep_seq = records[rep_index]
            rep_instance = rep_header.split()[0]
            fold_label = random.choice(FOLD_LABELS)
            rep_len = len(rep_seq)

            # align each member against the rep and track the widest insertion at each point,
            # so every member can later be padded out to one shared cluster-wide frame
            parsed_members = []
            max_len = {}
            for member_id, aligned_row in entries:
                member_index = int(member_id[3:])
                member_header, member_seq = records[member_index]
                chars_by_col, insertions = parse_member_row(aligned_row, member_seq)
                parsed_members.append((member_header, chars_by_col, insertions))
                for point, ins in insertions.items():
                    if len(ins) > max_len.get(point, 0):
                        max_len[point] = len(ins)

            # write the cluster header followed by each member's header + aligned sequence
            out.write(f">>CLUSTER {rep_instance} {fold_label}\n")
            for member_header, chars_by_col, insertions in parsed_members:
                m_instance, m_date, m_resolution, m_chains, m_heavy, m_light = member_header.split()[:6]
                aligned_seq = render_row(rep_len, chars_by_col, insertions, max_len)
                out.write(f">{m_instance} {m_date} {m_resolution} {m_chains} {m_heavy} {m_light}\n")
                out.write(aligned_seq + "\n")
            n_clusters += 1

with open(log_path, "w") as log:
    # basic run stats for sanity-checking the pipeline
    log.write(f"input records: {len(records)}\n")
    log.write(f"clusters: {n_clusters}\n")
