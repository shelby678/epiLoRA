"""Real multiple-sequence-alignment of the Ebola antigen sequences via mmseqs2.

Adapted from data/scripts/cluster_fasta.py's mmseqs2 invocation (createdb ->
cluster -> result2msa -> unpackdb) and its A3M-row-to-cluster-frame renderer
(parse_member_row/render_row, copied verbatim below since that script's
top-level argparse/execution isn't importable as a library).

One important deviation from cluster_fasta.py's defaults: this uses
``--cov-mode 0`` (both sequences must cover >= --cov of their own length)
instead of cluster_fasta.py's hardcoded ``--cov-mode 5`` (only the SHORTER
sequence must be covered). cov-mode 5 is right for cluster_fasta.py's own
job -- collapsing a fragment into its full-length parent -- but wrong here:
many records in this dataset are several antigen chains concatenated in one
sequence (e.g. GP1+GP2 together), and under cov-mode 5 a concatenated
"hub" sequence would satisfy the coverage requirement against a GP1-only
fragment AND separately a GP2-only fragment, chaining otherwise-unrelated
domains into one cluster through the hub. cov-mode 0 requires the hub
itself to be mostly covered too, which a single-domain fragment can't
satisfy, so real homology groups (e.g. all GP1/GP2/sGP entries, which truly
are the same gene product) still merge, while genuinely distinct antigens
(e.g. Nucleoprotein) do not.

    python msa_ebola.py in_fasta out_fasta log_path [--min_seq_id 0.3] [--cov 0.5]
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("in_fasta")
p.add_argument("out_fasta")
p.add_argument("log_path")
p.add_argument("--min_seq_id", type=float, default=0.3,
                help="mmseqs2 identity threshold for homology clustering (range 0.0-1.0)")
p.add_argument("--cov", type=float, default=0.5,
                help="mmseqs2 -c coverage threshold, --cov-mode 0 (both sequences)")
p.add_argument("--cluster_mode", type=int, default=1,
                help="mmseqs2 --cluster-mode: 1 connected component (default, guarantees "
                     "transitive closure for true cross-strain homology groups), 0 Set-Cover")
args = p.parse_args()
MMSEQS = "mmseqs"  # assumed on PATH


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


def parse_member_row(aligned_row, member_seq):
    """See data/scripts/cluster_fasta.py:parse_member_row (copied verbatim)."""
    chars_by_col = []
    insertions = {}
    rep_col = mem_pos = 0
    run_start = None
    for ch in aligned_row:
        if ch.islower():
            if run_start is None:
                run_start = mem_pos
            mem_pos += 1
            continue
        if run_start is not None:
            insertions[rep_col] = member_seq[run_start:mem_pos]
            run_start = None
        if ch == "-":
            chars_by_col.append(None)
            rep_col += 1
        else:
            chars_by_col.append(member_seq[mem_pos])
            rep_col += 1
            mem_pos += 1
    tail_start = run_start if run_start is not None else mem_pos
    if tail_start < len(member_seq):
        insertions[rep_col] = member_seq[tail_start:]
    return chars_by_col, insertions


def render_row(rep_len, chars_by_col, insertions, max_len):
    """See data/scripts/cluster_fasta.py:render_row (copied verbatim)."""
    parts = []
    for point in range(rep_len + 1):
        width = max_len.get(point, 0)
        if width:
            ins = insertions.get(point, "")
            parts.append(ins + "-" * (width - len(ins)))
        if point < rep_len:
            char = chars_by_col[point]
            parts.append(char if char is not None else "-")
    return "".join(parts)


records = list(parse_fasta(args.in_fasta))

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
        [MMSEQS, "cluster", str(query_db), str(clu_db), str(tmp / "work"),
         "--min-seq-id", str(args.min_seq_id), "-c", str(args.cov), "--cov-mode", "0",
         "--cluster-mode", str(args.cluster_mode)],
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

    parsed_clusters = []
    for cluster_file in unpacked.iterdir():
        entries = list(parse_fasta(cluster_file))
        rep_id, _ = entries[0]
        rep_index = int(rep_id[3:])
        parsed_clusters.append((rep_index, entries))
    parsed_clusters.sort(key=lambda x: x[0])

    with open(args.out_fasta, "w") as out:
        for rep_index, entries in parsed_clusters:
            rep_header, rep_seq = records[rep_index]
            rep_instance = rep_header.split()[0]
            rep_len = len(rep_seq)

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

            out.write(f">>CLUSTER {rep_instance} n={len(entries)}\n")
            for member_header, chars_by_col, insertions in parsed_members:
                aligned_seq = render_row(rep_len, chars_by_col, insertions, max_len)
                out.write(f">{member_header}\n{aligned_seq}\n")

with open(args.log_path, "w") as log:
    log.write(f"min_seq_id: {args.min_seq_id}\n")
    log.write(f"cov: {args.cov} (cov-mode 0, both sequences)\n")
    log.write(f"cluster_mode: {args.cluster_mode}\n")
    log.write(f"input records: {len(records)}\n")
    log.write(f"clusters: {len(parsed_clusters)}\n")
    for rep_index, entries in parsed_clusters:
        log.write(f"  cluster rep={records[rep_index][0].split()[0]} n={len(entries)}\n")
