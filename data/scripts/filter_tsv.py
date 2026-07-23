"""Filter the SAbDab summary tsv (rule_filter_tsv)."""
import csv
import sys

in_tsv, out_tsv, log_path = sys.argv[1:4]

MAX_RESOLUTION = 5.0


with open(in_tsv, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    fieldnames = reader.fieldnames
    rows = list(reader)

kept = []
species_seen = set()
for row in rows:
    if row["Hchain"] in ("", "NA"):
        continue
    if row["Lchain"] in ("", "NA"):
        continue
    if row["antigen_chain"] in ("", "NA"):
        continue
    if "PROTEIN" not in (row["antigen_type"] or "").upper():
        continue
    if row["resolution"] in ("", "NA") or float(row["resolution"]) > MAX_RESOLUTION:
        continue
    kept.append(row)
    species_seen.add(row["heavy_species"])
    species_seen.add(row["light_species"])

with open(out_tsv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(kept)

with open(log_path, "w") as log:
    log.write(f"input rows: {len(rows)}\n")
    log.write(f"kept rows: {len(kept)}\n")
    log.write(f"species used ({len(species_seen)}):\n")
    for s in sorted(species_seen):
        log.write(f"  {s}\n")
