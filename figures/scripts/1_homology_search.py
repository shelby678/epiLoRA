"""Search the query antigen's sequence against a pre-built mmseqs corpus DB
(see 0_prepare_corpus_db.py + mmseqs createdb), reporting every above-threshold
hit mapped back to its original corpus metadata.

    python 1_homology_search.py --pdb query.pdb --chain A|D --target_db db/train.db \
        --target_meta db/train.meta.json --out_json out.json --out_all_hits all_hits.json \
        --log log [--min_seq_id 0.85] [--min_aln_len 20]

A hit is >= --min_seq_id identity over >= --min_aln_len aligned residues.
Deliberately no mmseqs -c/--cov-mode coverage filter: a query that's a short
subsequence of a much longer training antigen (or vice versa) should still
count; --min_aln_len is just a floor against coincidental few-residue
high-identity stretches.

--chain accepts a single chain id or '|'-several for a multi-chain antigen
(concatenated in order); defaults to the first chain in the file. Must run in
the fair-esm environment (epilora/env/bin/python3).

--out_json is the single best hit, or {"hit": false} if none. --out_all_hits
is every hit best-first, each shaped like the best-hit object plus its corpus
metadata (including 'seq', needed for epitope-call transfer).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epitope_pipeline_common import default_chain, load_query, parse_chains  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pdb", required=True)
p.add_argument("--chain", default=None, help="single chain id, or '|'-separated for a multi-chain antigen")
p.add_argument("--target_db", required=True,
                help="mmseqs DB prefix built by 0_prepare_corpus_db.py + mmseqs createdb")
p.add_argument("--target_meta", required=True)
p.add_argument("--out_json", required=True)
p.add_argument("--out_all_hits", required=True)
p.add_argument("--log", required=True)
p.add_argument("--min_seq_id", type=float, default=0.85)
p.add_argument("--min_aln_len", type=int, default=20,
                help="minimum aligned residues, regardless of either sequence's full length")
args = p.parse_args()

chains = parse_chains(args.chain) if args.chain else [default_chain(args.pdb)]
_, seq = load_query(args.pdb, chains)

FIELDS = ["target", "pident", "alnlen", "qstart", "qend", "tstart", "tend", "qaln", "taln", "evalue", "bits"]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    query_fasta = tmp / "query.fasta"
    query_fasta.write_text(f">query\n{seq}\n")
    result_m8 = tmp / "result.m8"
    subprocess.run(
        ["mmseqs", "easy-search", str(query_fasta), args.target_db, str(result_m8), str(tmp / "work"),
         "--min-seq-id", str(args.min_seq_id),
         "--format-output", ",".join(FIELDS)],
        check=True, capture_output=True, text=True,
    )
    rows = []
    with open(result_m8) as f:
        for line in f:
            row = dict(zip(FIELDS, line.rstrip("\n").split("\t")))
            if int(row["alnlen"]) >= args.min_aln_len:
                rows.append(row)

all_meta = json.loads(Path(args.target_meta).read_text())


def alignment_mapping(row: dict) -> dict[str, int]:
    """1-based query position (str, for JSON) -> 1-based target position,
    for this row's aligned, non-gap-on-both-sides columns only."""
    qpos, tpos = int(row["qstart"]), int(row["tstart"])
    mapping = {}
    for qc, tc in zip(row["qaln"], row["taln"]):
        if qc != "-" and tc != "-":
            mapping[str(qpos)] = tpos
        if qc != "-":
            qpos += 1
        if tc != "-":
            tpos += 1
    return mapping


hits = [{"pident": float(row["pident"]), "query_to_target_pos": alignment_mapping(row), **all_meta[row["target"]]}
        for row in rows]

out = {"hit": True, **hits[0]} if hits else {"hit": False}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump(out, f)

Path(args.out_all_hits).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_all_hits, "w") as f:
    json.dump(hits, f)

with open(args.log, "w") as log:
    log.write(f"pdb: {args.pdb}  chains: {chains}\n")
    log.write(f"target_db: {args.target_db}\n")
    log.write(f"min_seq_id: {args.min_seq_id}  min_aln_len: {args.min_aln_len}\n")
    log.write(f"candidate hits above threshold: {len(rows)}\n")
    if out["hit"]:
        log.write(f"best hit: {rows[0]['target']}  pident={out['pident']:.3f}  "
                  f"aligned_cols={len(out['query_to_target_pos'])}\n")
    else:
        log.write("no hit above threshold\n")
