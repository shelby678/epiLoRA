"""Filter the SAbDab summary tsv down to Ebola antigen rows.

Applies the same row filter as data/scripts/filter_tsv.py (non-empty Hchain/
Lchain/antigen_chain, antigen_type contains PROTEIN), plus a species filter:
keep the row if any '|'-separated token of antigen_species matches "ebola"
case-insensitively (the column mixes antigen species with co-crystallized
host/expression-system organisms, e.g. "sudan ebolavirus|homo sapiens").
"""
import csv
import re
import sys

in_tsv, out_tsv, log_path = sys.argv[1:4]

EBOLA_RE = re.compile(r"ebola", re.IGNORECASE)

with open(in_tsv, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    fieldnames = reader.fieldnames
    rows = list(reader)

kept = []
for row in rows:
    if row["Hchain"] in ("", "NA"):
        continue
    if row["Lchain"] in ("", "NA"):
        continue
    if row["antigen_chain"] in ("", "NA"):
        continue
    if "PROTEIN" not in (row["antigen_type"] or "").upper():
        continue
    tokens = (row["antigen_species"] or "").split("|")
    if not any(EBOLA_RE.search(t) for t in tokens):
        continue
    kept.append(row)

with open(out_tsv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(kept)

with open(log_path, "w") as log:
    log.write(f"input rows: {len(rows)}\n")
    log.write(f"kept rows: {len(kept)}\n")
    species_seen = sorted({row["antigen_species"] for row in kept})
    log.write(f"antigen_species values kept ({len(species_seen)}):\n")
    for s in species_seen:
        log.write(f"  {s}\n")
