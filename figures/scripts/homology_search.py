"""Extract one antigen's sequence from a structure (like
extract_query_fasta used to, folded in here since every other consumer in
this pipeline already takes --pdb/--chain directly) and search it against a
pre-built mmseqs target DB (see prepare_corpus_db.py), reporting the best
hit's alignment mapped back to its original metadata -- plus, optionally,
every above-threshold hit ranked best-first (--out_matches, e.g. to list
every actual SAbDab structure a query resembles against a block-format
corpus) and/or every hit's full alignment mapping (--out_all_hits, for
groundtruth_from_homology.py's union-across-all-hits epitope call).

    python homology_search.py --pdb query.pdb --chain A|D --target_db db/train.db \
        --target_meta db/train.meta.json --out_json out.json --log log \
        [--out_matches matches.csv] [--out_all_hits all_hits.json] \
        [--min_seq_id 0.85] [--min_aln_len 20]

--chain accepts a single chain id, or '|'-separated ids for a multi-chain
antigen (e.g. 'A|D'), concatenated in order -- same convention as everywhere
else in this pipeline. If --chain is omitted, uses the first chain in the
file. Must run in the fair-esm environment (epilora/env/bin/python3), for
epitope_pipeline_common's Bio.PDB-based structure loading.

"Pretty similar" is >= --min_seq_id identity (mmseqs2 --min-seq-id, identity
computed over the aligned region -- mmseqs2's default --seq-id-mode 0) over
at least --min_aln_len aligned residues. Deliberately no mmseqs -c/--cov-mode
length-ratio filter: those modes (including mode 5, "short seq. needs to be
at least x% of the other seq. length" in this mmseqs version) reject exactly
the case we want to allow -- a query that's a short subsequence of a much
longer training antigen, or vice versa. --min_aln_len is an absolute floor
instead, just to keep a coincidental few-residue high-identity stretch from
counting as a hit. Below --min_seq_id or --min_aln_len, this reports no hit
rather than guessing from a weak/partial alignment.

Output JSON (--out_json, for the single best hit):
    {"hit": false}
or:
    {"hit": true, "pident": <float>, "query_to_target_pos": {...}, **target_meta.json[best hit]}
i.e. every field prepare_corpus_db.py recorded for that target
(header/seq/fold_group, or instance/pdb_id/date/..., depending on which
corpus --target_db was built from) is copied straight through.
``query_to_target_pos`` maps 1-based query position -> 1-based target
position, covering only the aligned, non-gap-on-both-sides columns --
residues outside that range (or in an insertion/deletion) have no entry, so
downstream scripts can tell "not an epitope by homology" apart from "no
homology coverage here" instead of silently defaulting one to the other.

Output CSV (--out_matches, optional, every hit above threshold, best first):
one row per hit: every target_meta.json field (except any 'seq') plus
pident/alnlen.

Output JSON (--out_all_hits, optional): a list, one entry per hit above
threshold (best first), each shaped like --out_json's single-hit object
(minus the "hit" key) -- i.e. every hit's own pident/query_to_target_pos
plus its target_meta.json fields, including 'seq' this time (needed to look
up each hit's own epitope calls).
"""
from __future__ import annotations

import argparse
import csv
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
                help="mmseqs DB prefix built by prepare_corpus_db.py + mmseqs createdb")
p.add_argument("--target_meta", required=True)
p.add_argument("--out_json", required=True)
p.add_argument("--out_matches", default=None, help="optional: CSV of every above-threshold hit, best first")
p.add_argument("--out_all_hits", default=None, help="optional: JSON list of every above-threshold hit's full mapping")
p.add_argument("--log", required=True)
p.add_argument("--min_seq_id", type=float, default=0.85)
p.add_argument("--min_aln_len", type=int, default=20,
                help="minimum aligned residues, regardless of either sequence's full length")
args = p.parse_args()
MMSEQS = "mmseqs"  # assumed on PATH

chains = parse_chains(args.chain) if args.chain else [default_chain(args.pdb)]
_, seq = load_query(args.pdb, chains)

FIELDS = ["target", "pident", "alnlen", "qstart", "qend", "tstart", "tend", "qaln", "taln", "evalue", "bits"]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    query_fasta = tmp / "query.fasta"
    query_fasta.write_text(f">query\n{seq}\n")
    result_m8 = tmp / "result.m8"
    subprocess.run(
        [MMSEQS, "easy-search", str(query_fasta), args.target_db, str(result_m8), str(tmp / "work"),
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


# Splat whatever prepare_corpus_db.py recorded for a
# target straight through -- schema (header/seq/fold_group vs.
# instance/pdb_id/date/...) depends on which corpus --target_db is.
hits = [{"pident": float(row["pident"]), "query_to_target_pos": alignment_mapping(row), **all_meta[row["target"]]}
        for row in rows]

out = {"hit": True, **hits[0]} if hits else {"hit": False}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump(out, f)

if args.out_matches is not None:
    Path(args.out_matches).parent.mkdir(parents=True, exist_ok=True)
    match_rows = [{**{k: v for k, v in hit.items() if k not in ("seq", "query_to_target_pos")},
                   "alnlen": int(row["alnlen"])}
                  for hit, row in zip(hits, rows)]
    with open(args.out_matches, "w", newline="") as f:
        if match_rows:
            w = csv.DictWriter(f, fieldnames=list(match_rows[0].keys()))
            w.writeheader()
            w.writerows(match_rows)
        else:
            f.write("")

if args.out_all_hits is not None:
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
