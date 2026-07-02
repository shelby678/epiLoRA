#!/usr/bin/env python3
"""
Simple antigen chain extraction using PDB text parsing.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

# Paths
AUTOPROT_DIR = Path(__file__).parent
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"

# Input and output directories
ORIGINAL_INPUT_DIR = SURF2SPOT_DIR / "test_sabdab_proper" / "input"
ANTIGEN_ONLY_INPUT_DIR = SURF2SPOT_DIR / "test_sabdab_antigen_only" / "input"

def parse_sabdab_fasta() -> Dict[str, Tuple[str, str]]:
    """Parse sabdab_novel30.fasta to get antigen chain info."""
    antigen_info = {}

    with open(SABDAB_FASTA) as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
                parts = header.split()
                pdb_chain_info = parts[0]  # e.g., "8dyx_HL"
                antigen_chain = parts[1]   # e.g., "I"

                pdb_code = pdb_chain_info.split("_")[0].lower()
                antigen_info[f"{pdb_code}_{antigen_chain}"] = (pdb_code, antigen_chain)

    return antigen_info

def extract_antigen_chain_simple(input_pdb: Path, output_pdb: Path, antigen_chain: str):
    """Extract only the antigen chain using simple text parsing."""

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    chain_found = False
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith(('HEADER', 'TITLE', 'COMPND', 'SOURCE', 'REMARK')):
                outfile.write(line)
            elif line.startswith(('ATOM', 'HETATM')):
                # Check if this atom belongs to our antigen chain
                if line[21] == antigen_chain:
                    outfile.write(line)
                    chain_found = True
            elif line.startswith('END'):
                outfile.write(line)

    return chain_found

def extract_all_antigen_chains():
    """Extract antigen chains from all PDB files."""

    print("🔬 Extracting Antigen Chains for Surf2Spot (Simple Method)")
    print("=" * 60)

    # Get antigen chain information
    antigen_info = parse_sabdab_fasta()
    print(f"📋 Found {len(antigen_info)} antigens to process")

    # Check input directory
    if not ORIGINAL_INPUT_DIR.exists():
        print(f"❌ Input directory not found: {ORIGINAL_INPUT_DIR}")
        return

    input_files = list(ORIGINAL_INPUT_DIR.glob("*.pdb"))
    print(f"📁 Found {len(input_files)} PDB files in input directory")

    # Process each file
    success_count = 0
    for input_file in input_files:
        filename = input_file.stem  # e.g., "8dyx_I"

        if filename not in antigen_info:
            print(f"⚠️  Skipping {filename} - not found in antigen list")
            continue

        pdb_code, antigen_chain = antigen_info[filename]
        output_file = ANTIGEN_ONLY_INPUT_DIR / f"{filename}.pdb"

        print(f"🧬 Processing {filename}: extracting chain {antigen_chain}")

        # Count chains in original file
        chains_in_file = set()
        atom_count_original = 0
        with open(input_file) as f:
            for line in f:
                if line.startswith("ATOM"):
                    chains_in_file.add(line[21])
                    atom_count_original += 1

        print(f"   📊 Original: {len(chains_in_file)} chains, {atom_count_original} atoms")

        # Extract antigen chain
        if extract_antigen_chain_simple(input_file, output_file, antigen_chain):
            # Count atoms in extracted file
            atom_count_extracted = 0
            with open(output_file) as f:
                for line in f:
                    if line.startswith("ATOM"):
                        atom_count_extracted += 1

            print(f"   ✅ Extracted: chain {antigen_chain}, {atom_count_extracted} atoms")
            reduction = atom_count_original / atom_count_extracted if atom_count_extracted > 0 else 0
            print(f"   📈 Size reduction: {reduction:.1f}x smaller")
            success_count += 1
        else:
            print(f"   ❌ Failed to extract chain {antigen_chain}")

    print(f"\n📊 SUMMARY:")
    print(f"   Successfully extracted: {success_count}/{len(input_files)} files")
    print(f"   Output directory: {ANTIGEN_ONLY_INPUT_DIR}")

    if success_count > 0:
        print(f"\n🎯 PROCESSING IMPROVEMENT:")
        print(f"   Before: Full antibody-antigen complexes (~10-20 chains each)")
        print(f"   After:  Antigen chain only (1 chain each)")
        print(f"   Expected: Much faster processing & better epitope predictions!")

if __name__ == "__main__":
    extract_all_antigen_chains()