#!/usr/bin/env python3
"""
Create training FASTA file from sabdab_novel.fasta excluding sequences in holdout sets.

This script:
1. Extracts sequence identifiers from holdout1.fasta and holdout2.fasta
2. Creates a training FASTA file from sabdab_novel.fasta excluding holdout sequences
3. Reports statistics on the resulting splits
"""

from pathlib import Path
from typing import Set

def extract_holdout_identifiers() -> Set[str]:
    """Extract sequence identifiers from holdout files."""
    holdout_ids = set()

    for holdout_file in ["data/holdout1.fasta", "data/holdout2.fasta"]:
        with open(holdout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Extract identifier: ">8dyx_HL I 1" -> "8dyx_HL I 1"
                    identifier = line[1:]
                    holdout_ids.add(identifier)
                    print(f"Holdout: {identifier}")

    return holdout_ids

def create_training_fasta(holdout_ids: Set[str]) -> None:
    """Create training FASTA file excluding holdout sequences."""
    input_file = "data/sabdab_novel.fasta"
    output_file = "data/sabdab_training.fasta"

    print(f"\nProcessing {input_file}...")

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        current_header = None
        current_sequence = None

        sequences_total = 0
        sequences_excluded = 0
        sequences_included = 0

        def write_sequence():
            nonlocal sequences_total, sequences_excluded, sequences_included
            if current_header and current_sequence:
                sequences_total += 1
                identifier = current_header[1:]  # Remove '>'

                if identifier in holdout_ids:
                    sequences_excluded += 1
                    print(f"  Excluded: {identifier}")
                else:
                    sequences_included += 1
                    outfile.write(current_header + '\n')
                    outfile.write(current_sequence + '\n')

        for line in infile:
            line = line.strip()
            if line.startswith('>'):
                # Write previous sequence if it exists
                write_sequence()

                # Start new sequence
                current_header = line
                current_sequence = None
            elif line:
                # Sequence line
                if current_sequence is None:
                    current_sequence = line
                else:
                    current_sequence += line

        # Write final sequence
        write_sequence()

    print(f"\nResults:")
    print(f"  Total sequences in {input_file}: {sequences_total}")
    print(f"  Excluded (in holdouts): {sequences_excluded}")
    print(f"  Included in training: {sequences_included}")
    print(f"  Training file created: {output_file}")

def main():
    print("Creating training FASTA file excluding holdout sequences")
    print("=" * 60)

    # Extract holdout identifiers
    print("Extracting holdout identifiers...")
    holdout_ids = extract_holdout_identifiers()
    print(f"Found {len(holdout_ids)} holdout sequences")

    # Create training FASTA
    create_training_fasta(holdout_ids)

    print("\n✓ Training FASTA file created successfully!")

if __name__ == "__main__":
    main()