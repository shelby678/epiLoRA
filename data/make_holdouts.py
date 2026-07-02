"""Split sabdab_novel30.fasta into two balanced holdout sets of 120 antigens each.

Stratified by partition so both sets have similar partition distributions.
Remaining 18 entries are written to sabdab_novel30_remainder.fasta.

Output:
  data/holdout1.fasta  (120 entries)
  data/holdout2.fasta  (120 entries)
  data/holdout_remainder.fasta  (18 entries)
"""

from __future__ import annotations
import random
from collections import defaultdict
from pathlib import Path

BASE   = Path(__file__).parent
INPUT  = BASE / "sabdab_novel30.fasta"
OUT1   = BASE / "holdout1.fasta"
OUT2   = BASE / "holdout2.fasta"
OUTR   = BASE / "holdout_remainder.fasta"

SEED   = 42
GROUP_SIZE = 120


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


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    lines = []
    for header, seq in records:
        lines.append(f">{header}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    records = read_fasta(INPUT)

    # Group by partition (3rd space-separated field in header)
    by_partition: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for header, seq in records:
        fields = header.split()
        part = fields[2] if len(fields) >= 3 else "?"
        by_partition[part].append((header, seq))

    # Shuffle within each partition
    for part in by_partition:
        rng.shuffle(by_partition[part])

    # Stratified split: assign entries round-robin to group1, group2, remainder
    group1: list[tuple[str, str]] = []
    group2: list[tuple[str, str]] = []
    remainder: list[tuple[str, str]] = []

    pool = []
    for part in sorted(by_partition):
        pool.extend(by_partition[part])
    rng.shuffle(pool)

    for i, rec in enumerate(pool):
        if len(group1) < GROUP_SIZE:
            group1.append(rec)
        elif len(group2) < GROUP_SIZE:
            group2.append(rec)
        else:
            remainder.append(rec)

    write_fasta(group1, OUT1)
    write_fasta(group2, OUT2)
    write_fasta(remainder, OUTR)

    print(f"holdout1:   {len(group1)} entries → {OUT1}")
    print(f"holdout2:   {len(group2)} entries → {OUT2}")
    print(f"remainder:  {len(remainder)} entries → {OUTR}")

    # Show partition breakdown
    def part_dist(recs):
        d: dict[str, int] = defaultdict(int)
        for h, _ in recs:
            d[h.split()[2] if len(h.split()) >= 3 else "?"] += 1
        return dict(sorted(d.items()))

    print(f"\nPartition distribution:")
    print(f"  holdout1:  {part_dist(group1)}")
    print(f"  holdout2:  {part_dist(group2)}")
    print(f"  remainder: {part_dist(remainder)}")


if __name__ == "__main__":
    main()
