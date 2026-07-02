#!/usr/bin/env python3
"""
Create PDB structures with B-factor epitope scoring for visualization.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional

# Paths
AUTOPROT_DIR = Path(__file__).parent
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
ANTIGEN_ONLY_DIR = SURF2SPOT_DIR / "test_sabdab_antigen_only"
PREDICT_DIR = ANTIGEN_ONLY_DIR / "predict"
PDB_SOURCE_DIR = AUTOPROT_DIR / "data" / "structures2" / "sabdab_dataset"
OUTPUT_DIR = AUTOPROT_DIR / "epitope_visualization"

def parse_sabdab_fasta() -> Dict[str, Tuple[str, List[int]]]:
    """Parse sabdab_novel30.fasta into ground truth labels."""
    data = {}

    with open(SABDAB_FASTA, 'r') as f:
        antigen_id = None
        sequence = None

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous entry if exists
                if antigen_id and sequence:
                    # Parse epitope labels from sequence (uppercase = epitope, lowercase = non-epitope)
                    epitope_labels = [1 if aa.isupper() else 0 for aa in sequence]
                    clean_sequence = sequence.upper()  # Convert all to uppercase

                    data[antigen_id] = (clean_sequence, epitope_labels)

                # Parse header: >8dyx_HL I 1 -> extract antigen chain ID
                header_parts = line[1:].split()
                if len(header_parts) >= 3:
                    complex_id = header_parts[0]  # e.g., "8dyx_HL"
                    chain_id = header_parts[1]    # e.g., "I"
                    antigen_id = f"{complex_id.split('_')[0]}_{chain_id}"  # e.g., "8dyx_I"
                else:
                    antigen_id = None

                sequence = ""

            elif antigen_id:  # Sequence line
                sequence += line

        # Handle last entry
        if antigen_id and sequence:
            epitope_labels = [1 if aa.isupper() else 0 for aa in sequence]
            clean_sequence = sequence.upper()
            data[antigen_id] = (clean_sequence, epitope_labels)

    return data

def parse_ply_predictions(ply_file: Path) -> Optional[List[float]]:
    """Parse PLY file to extract epitope scores."""
    try:
        # PLY files contain vertex data with x, y, z, score
        scores = []

        with open(ply_file, 'r') as f:
            in_vertex_section = False
            vertex_count = 0
            target_vertices = 0

            for line in f:
                line = line.strip()

                if line.startswith('element vertex'):
                    target_vertices = int(line.split()[-1])
                    continue

                if line == 'end_header':
                    in_vertex_section = True
                    continue

                if in_vertex_section and vertex_count < target_vertices:
                    try:
                        parts = line.split()
                        if len(parts) >= 4:
                            # Last column should be the score
                            score = float(parts[-1])
                            scores.append(score)
                            vertex_count += 1
                    except (ValueError, IndexError):
                        continue

        return scores if scores else None

    except Exception as e:
        print(f"Error parsing {ply_file}: {e}")
        return None

def align_scores_to_residues(scores: List[float], sequence_length: int) -> List[float]:
    """Align surface scores to residue positions."""
    if len(scores) == sequence_length:
        return scores

    # If we have more scores than residues, average them
    if len(scores) > sequence_length:
        # Group scores and average
        group_size = len(scores) / sequence_length
        aligned_scores = []

        for i in range(sequence_length):
            start_idx = int(i * group_size)
            end_idx = int((i + 1) * group_size)
            residue_scores = scores[start_idx:end_idx]
            aligned_scores.append(np.mean(residue_scores) if residue_scores else 0.0)

        return aligned_scores

    # If we have fewer scores, pad with zeros
    else:
        return scores + [0.0] * (sequence_length - len(scores))

def get_individual_aucs(gt_data: Dict[str, Tuple[str, List[int]]]) -> List[Tuple[str, float]]:
    """Calculate individual AUC for each antigen to find median performers."""
    aucs = []

    for antigen_id, (sequence, epitope_labels) in gt_data.items():
        # Look for prediction file
        pred_file = PREDICT_DIR / f"{antigen_id}_domain_0_pred.ply"

        if not pred_file.exists():
            continue

        # Parse prediction scores
        raw_scores = parse_ply_predictions(pred_file)
        if raw_scores is None:
            continue

        # Align scores to sequence length
        scores = align_scores_to_residues(raw_scores, len(epitope_labels))

        # Calculate AUC if we have both classes
        try:
            if len(set(epitope_labels)) > 1:
                auc = roc_auc_score(epitope_labels, scores)
                aucs.append((antigen_id, auc))
        except:
            continue

    return sorted(aucs, key=lambda x: x[1])

def create_pdb_with_bfactor(antigen_id: str, scores: List[float], output_file: Path, title_suffix: str):
    """Create PDB structure with B-factor encoding."""
    # Find source PDB file
    pdb_id = antigen_id.split('_')[0]
    chain_id = antigen_id.split('_')[1]

    source_pdb = None
    possible_files = [
        PDB_SOURCE_DIR / f"{pdb_id}.pdb",
        ANTIGEN_ONLY_DIR / "input" / f"{antigen_id}.pdb",
        ANTIGEN_ONLY_DIR / f"{antigen_id}.pdb"
    ]

    for pdb_file in possible_files:
        if pdb_file.exists():
            source_pdb = pdb_file
            break

    if not source_pdb:
        print(f"❌ Could not find source PDB for {antigen_id}")
        return False

    try:
        # Read source PDB and modify B-factors
        with open(source_pdb, 'r') as f:
            lines = f.readlines()

        output_lines = []
        residue_index = 0
        current_residue_num = None
        current_bfactor = 0.0

        for line in lines:
            if line.startswith('ATOM') and len(line) >= 80:
                # Extract chain ID (column 22)
                line_chain = line[21:22].strip()

                # Only process atoms from the target chain
                if line_chain == chain_id or (chain_id.islower() and line_chain.lower() == chain_id):
                    # Get residue number
                    try:
                        res_num = int(line[22:26].strip())

                        # Check if we've moved to a new residue
                        if res_num != current_residue_num:
                            current_residue_num = res_num

                            # Map to score (already scaled to 0-100 range)
                            if residue_index < len(scores):
                                current_bfactor = scores[residue_index]  # Already 0-100 range
                            else:
                                current_bfactor = 0.0

                            residue_index += 1

                        # Replace B-factor (columns 61-66) - all atoms in same residue get same B-factor
                        new_line = line[:60] + f"{current_bfactor:6.2f}" + line[66:]
                        output_lines.append(new_line)

                    except:
                        output_lines.append(line)
                else:
                    # Skip atoms from other chains
                    continue
            else:
                # Keep header lines
                if line.startswith(('HEADER', 'TITLE', 'REMARK', 'CRYST1', 'MODEL')):
                    output_lines.append(line)

        # Add custom header
        header_lines = [
            f"REMARK   1 EPITOPE VISUALIZATION FOR {antigen_id.upper()}\n",
            f"REMARK   1 {title_suffix.upper()}\n",
            f"REMARK   1 B-FACTOR ENCODES EPITOPE SCORES (0-100 SCALE)\n",
            f"REMARK   1 HIGH B-FACTOR = HIGH EPITOPE PROBABILITY\n"
        ]

        # Write output PDB
        with open(output_file, 'w') as f:
            # Write custom header first
            for header_line in header_lines:
                f.write(header_line)
            f.write('\n')

            # Write structure lines
            for line in output_lines:
                f.write(line)

        print(f"✅ Created: {output_file.name}")
        return True

    except Exception as e:
        print(f"❌ Error creating PDB for {antigen_id}: {e}")
        return False

def main():
    print("🎨 Creating Epitope Visualization PDB Structures")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Parse ground truth
    print("📖 Loading ground truth...")
    gt_data = parse_sabdab_fasta()
    print(f"   Loaded {len(gt_data)} antigens")

    # Calculate individual AUCs to find median performers
    print("📊 Calculating individual AUCs...")
    aucs = get_individual_aucs(gt_data)
    print(f"   Calculated AUC for {len(aucs)} antigens")

    if not aucs:
        print("❌ No AUC calculations available")
        return

    # Find median performers (around 50th percentile)
    median_idx = len(aucs) // 2
    selected_antigens = aucs[median_idx-2:median_idx+3]  # Get 5 around median

    print(f"\n🎯 Selected median performers:")
    for antigen_id, auc in selected_antigens:
        print(f"   {antigen_id}: AUC = {auc:.3f}")

    # Create visualizations for top 3 median performers
    created_count = 0
    for antigen_id, auc in selected_antigens[:3]:
        print(f"\n🔬 Processing {antigen_id} (AUC = {auc:.3f})")

        sequence, epitope_labels = gt_data[antigen_id]

        # Get prediction scores
        pred_file = PREDICT_DIR / f"{antigen_id}_domain_0_pred.ply"
        if not pred_file.exists():
            print(f"   ❌ Prediction file not found")
            continue

        raw_scores = parse_ply_predictions(pred_file)
        if raw_scores is None:
            print(f"   ❌ Could not parse prediction scores")
            continue

        pred_scores = align_scores_to_residues(raw_scores, len(epitope_labels))

        # Normalize prediction scores to 0-100 range for visualization
        # Surf2Spot scores are negative (closer to 0 = higher epitope probability)
        # Convert: -0.5 to 0.0 → 0.0 to 100.0
        if pred_scores:
            min_score = min(pred_scores)
            max_score = max(pred_scores)
            if min_score != max_score:  # Avoid division by zero
                # Flip and normalize: more negative becomes lower, closer to 0 becomes higher
                normalized_scores = [(score - min_score) / (max_score - min_score) * 100 for score in pred_scores]
            else:
                normalized_scores = [50.0] * len(pred_scores)  # Middle value if all same
        else:
            normalized_scores = []

        # Create ground truth PDB (B-factor = 0 or 100 for binary labels)
        gt_output = OUTPUT_DIR / f"{antigen_id}_ground_truth.pdb"
        gt_scores = [float(x) * 100 for x in epitope_labels]  # Convert 0/1 to 0/100
        if create_pdb_with_bfactor(antigen_id, gt_scores,
                                   gt_output, "Ground Truth Epitopes"):
            created_count += 1

        # Create prediction PDB (B-factor = normalized 0-100 prediction score)
        pred_output = OUTPUT_DIR / f"{antigen_id}_predicted.pdb"
        if create_pdb_with_bfactor(antigen_id, normalized_scores,
                                   pred_output, "Surf2Spot Predictions"):
            created_count += 1

        print(f"   📊 Sequence length: {len(sequence)}")
        print(f"   📊 Epitope residues: {sum(epitope_labels)}/{len(epitope_labels)} ({sum(epitope_labels)/len(epitope_labels)*100:.1f}%)")
        print(f"   📊 Prediction range: {min(pred_scores):.3f} - {max(pred_scores):.3f}")

    print(f"\n🎉 Created {created_count} visualization files in {OUTPUT_DIR}/")
    print("\n💡 Visualization Guide:")
    print("   • Load PDB files in PyMOL or ChimeraX")
    print("   • Color by B-factor to see epitope scores")
    print("   • Red/high B-factor = epitope regions")
    print("   • Blue/low B-factor = non-epitope regions")
    print("   • Compare ground_truth vs predicted structures")

if __name__ == "__main__":
    main()