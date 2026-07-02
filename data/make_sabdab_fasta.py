"""Convert data/sabdab/{1..5}/ paired FASTAs into a single pdb_chains.fasta-style file.

Each sequence appears in exactly one validation set across the 5 folds, so we assign
partition numbers by iterating the valid sets:
  valid from fold i  →  partition str(i)

Header reformatting:
  PDB-style:   "7wvg_C_D_B"  →  ">7wvg_CD B 1"
               (pdb_code_HeavyLight  AntigenChain  Partition)
  IEDB-style:  all-digits / non-PDB pattern  →  left as-is (no partition info added;
               these entries will always land in training for all folds)

Sequence encoding:
  label 1  →  uppercase residue
  label 0  →  lowercase residue

Output: data/sabdab_chains.fasta
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

BASE = Path(__file__).parent
COMBINED_DIR = BASE / "sabdab"
OUTPUT = BASE / "sabdab_chains.fasta"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fasta_pairs(aa_path: Path, bce_path: Path) -> list[tuple[str, str, str]]:
    """Return list of (header, aa_sequence, label_string) from paired FASTA files."""
    def read_fasta(path: Path) -> dict[str, str]:
        seqs: dict[str, str] = {}
        cur_id: str | None = None
        cur_parts: list[str] = []
        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if cur_id is not None:
                        seqs[cur_id] = "".join(cur_parts)
                    cur_id = line[1:]
                    cur_parts = []
                elif cur_id is not None:
                    cur_parts.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(cur_parts)
        return seqs

    aa_seqs = read_fasta(aa_path)
    bce_seqs = read_fasta(bce_path)

    records: list[tuple[str, str, str]] = []
    for header, seq in aa_seqs.items():
        labels = bce_seqs.get(header, "")
        if len(seq) != len(labels):
            print(f"  WARNING: length mismatch for '{header}' ({len(seq)} vs {len(labels)}), skipping",
                  file=sys.stderr)
            continue
        records.append((header, seq, labels))
    return records


# A PDB-style header has the form:  PDBID_H_L_A
# PDBID can be 4+ alphanumerics (e.g. 7wvg, 9mer, 8diu); H, L, A are single chain chars.
_PDB_RE = re.compile(r'^([A-Za-z0-9]+)_([A-Za-z0-9])_([A-Za-z0-9])_([A-Za-z0-9])$')


def reformat_header(raw_header: str, partition: str) -> str:
    """Reformat a raw FASTA header into pdb_chains.fasta format.

    PDB-style:  "7wvg_C_D_B"  →  "7wvg_CD B 1"
    IEDB-style: "1597967"     →  "1597967"  (unchanged, no partition suffix)
    """
    m = _PDB_RE.match(raw_header)
    if m:
        pdb_code, heavy, light, antigen = m.groups()
        return f"{pdb_code}_{heavy}{light} {antigen} {partition}"
    # Non-PDB header (e.g. IEDB numeric ID) — keep ID, append partition with placeholder chain
    return f"{raw_header} - {partition}"


def case_encode(aa_seq: str, label_str: str) -> str:
    """Encode sequence: uppercase = epitope (1), lowercase = non-epitope (0)."""
    out: list[str] = []
    for aa, lbl in zip(aa_seq, label_str):
        if lbl == "1":
            out.append(aa.upper())
        else:
            out.append(aa.lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Collect all sequences across all valid sets.
    # Since this is a proper 5-fold partition, each sequence appears in exactly one valid set.
    seen_headers: set[str] = set()
    all_records: list[tuple[str, str]] = []   # (formatted_header, encoded_sequence)

    for fold in range(1, 6):
        aa_path  = COMBINED_DIR / str(fold) / "valid_aa.fasta"
        bce_path = COMBINED_DIR / str(fold) / "valid_bce.fasta"

        if not aa_path.exists() or not bce_path.exists():
            print(f"  Fold {fold}: files not found, skipping.", file=sys.stderr)
            continue

        records = load_fasta_pairs(aa_path, bce_path)
        n_dup = 0

        for raw_header, aa_seq, label_str in records:
            if raw_header in seen_headers:
                n_dup += 1
                continue
            seen_headers.add(raw_header)

            new_header = reformat_header(raw_header, str(fold))
            encoded_seq = case_encode(aa_seq, label_str)
            all_records.append((new_header, encoded_seq))

        print(f"  Fold {fold}: {len(records)} sequences (partition={fold})"
              + (f", {n_dup} duplicates skipped" if n_dup else ""))

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for header, seq in all_records:
        lines.append(f">{header}")
        lines.append(seq)
    OUTPUT.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {len(all_records)} sequences to {OUTPUT}")

    # Verification: sum of valid set sizes should equal total
    total_valid = sum(
        sum(1 for line in open(COMBINED_DIR / str(f) / "valid_aa.fasta") if line.startswith(">"))
        for f in range(1, 6)
        if (COMBINED_DIR / str(f) / "valid_aa.fasta").exists()
    )
    print(f"Sum of all valid set sizes: {total_valid}")
    if len(all_records) == total_valid:
        print("✓ Entry count matches — no duplicates across folds.")
    else:
        print(f"⚠ Mismatch: wrote {len(all_records)} but expected {total_valid} "
              f"({total_valid - len(all_records)} duplicates found across folds)")

    # Spot-check: show partition distribution
    from collections import Counter
    parts = Counter()
    for header, _ in all_records:
        fields = header.split()
        p = fields[2] if len(fields) >= 3 else "no-partition"
        parts[p] += 1
    print("\nPartition distribution:")
    for k in sorted(parts):
        print(f"  partition {k}: {parts[k]} sequences")


if __name__ == "__main__":
    main()
