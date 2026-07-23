"""Build the held-out eval fasta: seqs newer than the cutoff, dropped if >40% similar to an older seq."""
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

in_fasta, out_fasta, log_path = sys.argv[1:4]
CUTOFF = date(2024, 12, 13)
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


def parse_date(header):
    y, m, d = header.split()[1].split("/")
    return date(int(y), int(m), int(d))


records = list(parse_fasta(in_fasta))
older = [(h, s) for h, s in records if parse_date(h) <= CUTOFF]
newer = [(h, s) for h, s in records if parse_date(h) > CUTOFF]

dropped_ids = set()
if older and newer:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        newer_fasta, older_fasta = tmp / "newer.fasta", tmp / "older.fasta"
        with open(newer_fasta, "w") as f:
            for i, (_, seq) in enumerate(newer):
                f.write(f">q{i}\n{seq.upper()}\n")
        with open(older_fasta, "w") as f:
            for i, (_, seq) in enumerate(older):
                f.write(f">t{i}\n{seq.upper()}\n")

        result_m8 = tmp / "result.m8"
        subprocess.run(
            [MMSEQS, "easy-search", str(newer_fasta), str(older_fasta), str(result_m8), str(tmp / "work"),
             "--min-seq-id", str(MIN_SIMILARITY)],
            check=True, capture_output=True, text=True,
        )
        with open(result_m8) as f:
            for line in f:
                query_id = line.split("\t", 1)[0]
                dropped_ids.add(int(query_id[1:]))

n_kept = n_dropped = 0
with open(out_fasta, "w") as out:
    for i, (header, seq) in enumerate(newer):
        if i in dropped_ids:
            n_dropped += 1
            continue
        out.write(f">{header}\n{seq}\n")
        n_kept += 1

with open(log_path, "w") as log:
    log.write(f"candidates (newer than cutoff): {len(newer)}\n")
    log.write(f"kept: {n_kept}\n")
    log.write(f"dropped (too similar to pre-cutoff seq): {n_dropped}\n")
