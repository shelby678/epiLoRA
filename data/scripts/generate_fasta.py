"""Build the antigen fasta from the filtered tsv (rule generate_fasta)."""
import csv
import sys

from structures import chain_sequence, load_model

in_tsv, structures_dir, out_fasta, log_path = sys.argv[1:5]

MIN_LEN = 100
MAX_LEN = 1300

n_written = n_skipped_no_seq = n_skipped_len = 0
with open(in_tsv, newline="") as f, open(out_fasta, "w") as out:
    for row in csv.DictReader(f, delimiter="\t"):
        chains = row["antigen_chain"].split("|")
        try:
            model = load_model(structures_dir, row["PDB"])
            seq = "".join(chain_sequence(model, c) for c in chains)
        except Exception:
            seq = ""
        if not seq:
            n_skipped_no_seq += 1
            continue
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            n_skipped_len += 1
            continue
        heavy_species = row["heavy_species"].replace(" ", "_")
        light_species = row["light_species"].replace(" ", "_")
        out.write(
            f">{row['INSTANCE']} {row['date']} {row['resolution']} {row['antigen_chain']} "
            f"{heavy_species} {light_species}\n"
        )
        out.write(seq + "\n")
        n_written += 1

with open(log_path, "w") as log:
    log.write(f"records written: {n_written}\n")
    log.write(f"records skipped (no sequence): {n_skipped_no_seq}\n")
    log.write(f"records skipped (length outside {MIN_LEN}-{MAX_LEN}aa): {n_skipped_len}\n")
