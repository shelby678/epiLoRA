"""Mark epitope residues (lowercase) by proximity to the antibody (rule get_epitopes)."""
import csv
import sys

import numpy as np
from scipy.spatial import cKDTree

from structures import chain_residues, heavy_atoms, load_model

in_fasta, in_tsv, structures_dir, out_fasta, log_path = sys.argv[1:6]
CONTACT_DIST = 4.0

with open(in_tsv, newline="") as f:
    rows_by_instance = {r["INSTANCE"]: r for r in csv.DictReader(f, delimiter="\t")}


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


n_ok = n_skipped = 0
with open(out_fasta, "w") as out:
    for header, seq in parse_fasta(in_fasta):
        instance = header.split()[0]
        row = rows_by_instance.get(instance)
        if row is None:
            n_skipped += 1
            continue
        try:
            model = load_model(structures_dir, row["PDB"])
            ab_residues = chain_residues(model, row["Hchain"]) + chain_residues(model, row["Lchain"])
            ab_coords = np.array([a.coord for res in ab_residues for a in heavy_atoms(res)])
            tree = cKDTree(ab_coords)

            marked = []
            for chain_id in row["antigen_chain"].split("|"):
                for res in chain_residues(model, chain_id):
                    atoms = heavy_atoms(res)
                    coords = np.array([a.coord for a in atoms])
                    dists, _ = tree.query(coords)
                    marked.append(dists.min() <= CONTACT_DIST)

            if len(marked) != len(seq):
                n_skipped += 1
                continue
            new_seq = "".join(c.lower() if m else c.upper() for c, m in zip(seq, marked))
        except Exception:
            n_skipped += 1
            continue

        out.write(f">{header}\n{new_seq}\n")
        n_ok += 1

with open(log_path, "w") as log:
    log.write(f"records written: {n_ok}\n")
    log.write(f"records skipped: {n_skipped}\n")
