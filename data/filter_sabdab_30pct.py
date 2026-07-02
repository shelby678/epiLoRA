"""Filter sabdab_chains.fasta to entries with ≤30% sequence identity to any BEPIPRED entry.

Uses mmseqs2 easy-search results (query, target, pident, ...).
A sabdab entry is excluded if any hit has pident > 30.

Output: data/sabdab_novel30.fasta
"""

from __future__ import annotations
from pathlib import Path

BASE     = Path(__file__).parent
SABDAB   = BASE / "sabdab_chains.fasta"
HITS     = Path("/tmp/mmseqs_filter/sabdab_vs_bepipred.tsv")
OUTPUT   = BASE / "sabdab_novel30.fasta"


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
    # Find sabdab query IDs with any BEPIPRED hit > 30% identity
    too_similar: set[str] = set()
    with open(HITS) as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            query, pident = fields[0], float(fields[2])
            if pident > 30.0:
                too_similar.add(query)

    sabdab_records = read_fasta(SABDAB)

    kept, excluded = [], []
    for header, seq in sabdab_records:
        # mmseqs uses everything before the first space as the query ID
        query_id = header.split()[0]
        if query_id in too_similar:
            excluded.append(header)
        else:
            kept.append((header, seq))

    lines: list[str] = []
    for header, seq in kept:
        lines.append(f">{header}")
        lines.append(seq)
    OUTPUT.write_text("\n".join(lines) + "\n")

    print(f"SAbDab total:          {len(sabdab_records)}")
    print(f"Excluded (>30% sim):   {len(excluded)}")
    print(f"Kept (≤30% sim):       {len(kept)}")
    print(f"Output:                {OUTPUT}")


if __name__ == "__main__":
    main()
