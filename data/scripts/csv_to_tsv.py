"""Convert the raw SAbDab summary csv to tsv (rule_csv_to_tsv).
"""
import csv
import sys

in_csv, out_tsv, log_path = sys.argv[1:4]

with open(in_csv, newline="") as f:
    rows = list(csv.reader(f))

with open(out_tsv, "w", newline="") as f:
    csv.writer(f, delimiter="\t").writerows(rows)
