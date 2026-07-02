"""Filter sabdab_chains.fasta to entries whose antigen sequence does not appear in BEPIPRED.fasta.

Comparison is case-insensitive (case encodes epitope labels, not identity).

Output: data/sabdab_novel.fasta
"""

from __future__ import annotations
from pathlib import Path

BASE = Path(__file__).parent
SABDAB  = BASE / "sabdab_chains.fasta"
BEPIPRED = BASE / "BEPIPRED.fasta"
OUTPUT  = BASE / "sabdab_novel.fasta"


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


def main() -> None:
    sabdab_records  = read_fasta(SABDAB)
    bepipred_records = read_fasta(BEPIPRED)

    # Build set of antigen sequences from BEPIPRED (uppercase for case-insensitive compare)
    bepipred_seqs: set[str] = {seq.upper() for _, seq in bepipred_records}

    novel: list[tuple[str, str]] = []
    n_overlap = 0
    for header, seq in sabdab_records:
        if seq.upper() in bepipred_seqs:
            n_overlap += 1
        else:
            novel.append((header, seq))

    lines: list[str] = []
    for header, seq in novel:
        lines.append(f">{header}")
        lines.append(seq)
    OUTPUT.write_text("\n".join(lines) + "\n")

    print(f"SAbDab total:     {len(sabdab_records)}")
    print(f"In BEPIPRED:      {n_overlap}")
    print(f"Novel (kept):     {len(novel)}")
    print(f"Output:           {OUTPUT}")


if __name__ == "__main__":
    main()
