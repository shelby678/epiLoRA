"""Run DiscoTope-3.0 on holdout1 and holdout2, evaluate ROC-AUC.

For each holdout FASTA:
  1. Collect the antigen-chain PDB file for each entry.
  2. Run discotope3_web predict_webserver.py (batch mode) on the PDB directory.
  3. Parse per-residue scores for the correct antigen chain.
  4. Compare against epitope labels (uppercase = 1, lowercase = 0) from the FASTA.
  5. Report ROC-AUC.

Run:
    /path/to/python run_discotope3_holdout.py > run_discotope3_holdout.log 2>&1
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DISCOTOPE_DIR  = Path("/home/sferrier/epitope_mapping/discotope3_web")
PREDICT_SCRIPT = DISCOTOPE_DIR / "src" / "predict_webserver.py"
DT_PYTHON      = DISCOTOPE_DIR / "env" / "bin" / "python"

STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
HOLDOUT1_FASTA = Path("data/holdout1.fasta")
HOLDOUT2_FASTA = Path("data/holdout2.fasta")
RESULTS_TSV    = Path("results.tsv")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:]
                parts = []
            elif header is not None:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def parse_header(header: str) -> tuple[str, str]:
    """Return (pdb_id_lower, antigen_chain) from header like '8dyx_HL I 1'."""
    fields = header.split()
    pdb_id = fields[0].split("_")[0].lower()
    antigen_chain = fields[1]
    return pdb_id, antigen_chain


def load_discotope_scores(csv_path: Path) -> dict[int, float]:
    """Return {res_id: score} from a DiscoTope-3.0 CSV output."""
    scores: dict[int, float] = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            scores[int(row["res_id"])] = float(row["DiscoTope-3.0_score"])
    return scores


def run_holdout(holdout_fasta: Path, holdout_name: str, out_base: Path) -> float:
    records = read_fasta(holdout_fasta)
    print(f"\n{'='*60}")
    print(f"HOLDOUT: {holdout_name}  ({len(records)} antigens)")
    print(f"{'='*60}")

    # Extract just the antigen chain from each PDB — one file per holdout entry
    pdb_dir = out_base / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str, str, str]] = []  # (header, pdb_id, chain, out_stem)
    for header, seq in records:
        pdb_id, antigen_chain = parse_header(header)
        pdb_src = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
        if not pdb_src.exists():
            print(f"  WARNING: PDB not found for {header}, skipping")
            continue
        # Write single-chain PDB with only the antigen chain
        out_stem = f"{pdb_id}_{antigen_chain}"
        out_pdb = pdb_dir / f"{out_stem}.pdb"
        if not out_pdb.exists():
            with open(pdb_src) as fin, open(out_pdb, "w") as fout:
                for line in fin:
                    rec = line[:6].strip()
                    if rec in ("ATOM", "HETATM", "TER"):
                        if line[21] == antigen_chain:
                            fout.write(line)
                    elif rec in ("HEADER", "TITLE", "REMARK", "SEQRES"):
                        fout.write(line)
                fout.write("END\n")
        entries.append((header, pdb_id, antigen_chain, out_stem))

    print(f"  PDBs collected: {len(list(pdb_dir.glob('*.pdb')))}")

    # Run DiscoTope-3.0
    dt_out = out_base / "discotope_output"
    dt_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(DT_PYTHON), str(PREDICT_SCRIPT),
        "--pdb_dir", str(pdb_dir.resolve()),
        "--struc_type", "solved",
        "--out_dir", str(dt_out.resolve()),
    ]
    print(f"  Running DiscoTope-3.0 ...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=DISCOTOPE_DIR)
    if result.returncode != 0:
        print(f"  DiscoTope-3.0 FAILED:\n{result.stderr[-2000:]}", file=sys.stderr)
        return float("nan")
    if result.stderr:
        print(result.stderr[-500:], file=sys.stderr)

    # Parse outputs and evaluate
    all_labels: list[int] = []
    all_scores: list[float] = []
    n_missing = 0

    for header, pdb_id, antigen_chain, out_stem in entries:
        # DiscoTope-3.0 names output as {stem}_{chain}_discotope3.csv
        # Since we pass single-chain PDBs named {pdb_id}_{chain}.pdb, the chain
        # inside the file is still `antigen_chain`, so output is {stem}_{chain}_discotope3.csv
        csv_path = dt_out / "output" / f"{out_stem}_{antigen_chain}_discotope3.csv"
        if not csv_path.exists():
            csv_path = dt_out / "output" / f"{out_stem.upper()}_{antigen_chain}_discotope3.csv"
        if not csv_path.exists():
            print(f"  WARNING: no output CSV for {header} (expected {csv_path.name})")
            n_missing += 1
            continue

        # Retrieve seq from records dict
        seq = next(s for h, s in records if h == header)
        scores = load_discotope_scores(csv_path)
        # Labels from sequence: uppercase=1, lowercase=0
        labels = [1 if aa.isupper() else 0 for aa in seq]

        score_vals = list(scores.values())
        if len(score_vals) != len(labels):
            # Align by position (use as many as match)
            min_len = min(len(score_vals), len(labels))
            score_vals = score_vals[:min_len]
            labels = labels[:min_len]

        all_labels.extend(labels)
        all_scores.extend(score_vals)

    if not all_labels or len(set(all_labels)) < 2:
        print("  ERROR: not enough labeled data for AUC")
        return float("nan")

    auc = roc_auc_score(all_labels, all_scores)
    print(f"  Residues evaluated: {len(all_labels)}")
    print(f"  Epitope residues:   {sum(all_labels)}")
    print(f"  Missing outputs:    {n_missing}")
    print(f"  ROC-AUC:            {auc:.4f}")

    return auc


if __name__ == "__main__":
    out_base1 = Path("discotope3_holdout1")
    out_base2 = Path("discotope3_holdout2")

    auc1 = run_holdout(HOLDOUT1_FASTA, "holdout1", out_base1)
    auc2 = run_holdout(HOLDOUT2_FASTA, "holdout2", out_base2)

    print(f"\n{'='*60}")
    print(f"SUMMARY — DiscoTope-3.0 on sabdab holdouts (≤30% sim to BEPIPRED)")
    print(f"  holdout1 ROC-AUC: {auc1:.4f}")
    print(f"  holdout2 ROC-AUC: {auc2:.4f}")
    print(f"  mean:             {np.mean([auc1, auc2]):.4f}")
    print(f"{'='*60}")

    with open(RESULTS_TSV, "a") as f:
        for idx, auc in [(1, auc1), (2, auc2)]:
            f.write(
                f"discotope3\tdiscotope3-holdout\t{idx}\t0\t"
                f"nan\tnan\t{auc:.6f}\tnan\tnan\tnan\t"
                f"DiscoTope-3.0 on sabdab holdout{idx} (≤30% sim)\n"
            )
