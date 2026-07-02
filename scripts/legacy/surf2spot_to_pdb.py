#!/usr/bin/env python3
"""
Convert Surf2Spot epitope prediction scores to PDB B-factors for visualization.

This script takes median-performing antigens and creates PDB files where the
B-factor column contains the epitope prediction scores.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
import warnings
warnings.filterwarnings('ignore')

# Paths
AUTOPROT_DIR = Path(__file__).parent
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"
PDB_DIR = AUTOPROT_DIR / "data" / "structures2" / "sabdab_dataset"
OUTPUT_DIR = AUTOPROT_DIR / "surf2spot_pdb_predictions"

PREDICT_DIRS = [
    SURF2SPOT_DIR / "test_NB" / "predict",
    SURF2SPOT_DIR / "test_NB" / "esm_predict",
    SURF2SPOT_DIR / "test_NB_esmall" / "predict",
]

# Select median performers for visualization
SELECTED_ANTIGENS = [
    ("5d1z_DC I 1", "5d1z", "I", 0.276, 0.584),  # F1=0.276, AUC=0.584
    ("9naq_HL A 1", "9naq", "A", 0.261, 0.554),  # F1=0.261, AUC=0.554
    ("7pgb_RS T 1", "7pgb", "T", 0.182, 0.532),  # F1=0.182, AUC=0.532
    ("5dhv_HL M 1", "5dhv", "M", 0.250, 0.534),  # F1=0.250, AUC=0.534
    ("9f91_EF I 1", "9f91", "I", 0.211, 0.509),  # F1=0.211, AUC=0.509
]

def parse_sabdab_fasta() -> Dict[str, Tuple[str, List[int]]]:
    """Parse sabdab_novel30.fasta for ground truth."""
    gt_data = {}
    with open(SABDAB_FASTA) as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            elif header:
                sequence = line
                labels = [1 if c.isupper() else 0 for c in sequence]
                gt_data[header] = (sequence.upper(), labels)
    return gt_data

def find_prediction_file(pdb_code: str, chain: str) -> Optional[Path]:
    """Find prediction CSV file for antigen."""
    antigen_id = f"{pdb_code.lower()}_{chain}"

    for predict_dir in PREDICT_DIRS:
        if not predict_dir.exists():
            continue

        patterns = [
            f"{antigen_id}.csv",
            f"{pdb_code.lower()}_{chain}.csv",
            f"{pdb_code.upper()}_{chain}.csv",
        ]

        for pattern in patterns:
            filepath = predict_dir / pattern
            if filepath.exists():
                return filepath

        # Try domain files
        domain_files = list(predict_dir.glob(f"{antigen_id}*_domain_*_pred.csv"))
        if domain_files:
            return domain_files[0]  # Return first domain file for simplicity

    return None

def load_prediction_scores(csv_file: Path) -> List[float]:
    """Load prediction scores from CSV."""
    try:
        df = pd.read_csv(csv_file)
        if 'score' in df.columns:
            return df['score'].tolist()
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
    return []

def find_pdb_file(pdb_code: str) -> Optional[Path]:
    """Find PDB structure file."""
    pdb_patterns = [
        PDB_DIR / pdb_code.lower() / "structure" / f"{pdb_code.lower()}.pdb",
        PDB_DIR / pdb_code.upper() / "structure" / f"{pdb_code.upper()}.pdb",
        PDB_DIR / f"{pdb_code.lower()}.pdb",
        PDB_DIR / f"{pdb_code.upper()}.pdb",
    ]

    for pattern in pdb_patterns:
        if pattern.exists():
            return pattern
    return None

def get_chain_sequence(structure: Structure, chain_id: str) -> Tuple[str, List[int]]:
    """Extract sequence and residue numbers from PDB chain."""
    sequence = ""
    resnums = []

    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            for residue in chain:
                if residue.id[0] == ' ':  # Standard amino acid residue
                    resname = residue.resname
                    # Convert 3-letter to 1-letter amino acid code
                    aa_map = {
                        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                    }
                    if resname in aa_map:
                        sequence += aa_map[resname]
                        resnums.append(residue.id[1])
            break

    return sequence, resnums

def update_bfactors_with_scores(structure: Structure, chain_id: str,
                               scores: List[float], gt_labels: List[int]) -> Structure:
    """Update B-factors in structure with prediction scores."""

    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            residue_idx = 0

            for residue in chain:
                if residue.id[0] == ' ':  # Standard residue
                    # Set B-factor based on available data
                    if residue_idx < len(scores):
                        score = scores[residue_idx]
                        # Normalize score to reasonable B-factor range (0-100)
                        bfactor = score * 100
                    else:
                        bfactor = 0.0  # No prediction available

                    # Set B-factor for all atoms in residue
                    for atom in residue:
                        atom.bfactor = bfactor

                    residue_idx += 1
            break

    return structure

def create_prediction_pdb(header: str, pdb_code: str, chain: str,
                         f1_score: float, auc_score: float):
    """Create PDB file with prediction scores as B-factors."""

    print(f"\n🔬 Processing {header}")
    print(f"   PDB: {pdb_code}, Chain: {chain}, F1: {f1_score:.3f}, AUC: {auc_score:.3f}")

    # Find PDB structure file
    pdb_file = find_pdb_file(pdb_code)
    if not pdb_file:
        print(f"   ❌ PDB file not found for {pdb_code}")
        return

    # Find prediction file
    pred_file = find_prediction_file(pdb_code, chain)
    if not pred_file:
        print(f"   ❌ Prediction file not found for {pdb_code}_{chain}")
        return

    # Load prediction scores
    scores = load_prediction_scores(pred_file)
    if not scores:
        print(f"   ❌ Failed to load scores from {pred_file}")
        return

    # Load ground truth
    gt_data = parse_sabdab_fasta()
    gt_labels = None
    for gt_header, (sequence, labels) in gt_data.items():
        if header == gt_header:
            gt_labels = labels
            break

    if gt_labels is None:
        print(f"   ❌ Ground truth not found for {header}")
        return

    # Load PDB structure
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_code, pdb_file)
    except Exception as e:
        print(f"   ❌ Failed to parse PDB {pdb_file}: {e}")
        return

    # Get chain sequence
    pdb_sequence, resnums = get_chain_sequence(structure, chain)

    print(f"   📋 PDB sequence length: {len(pdb_sequence)}")
    print(f"   📋 GT labels length: {len(gt_labels)}")
    print(f"   📋 Prediction scores: {len(scores)}")

    # Align lengths (take minimum)
    min_len = min(len(pdb_sequence), len(gt_labels), len(scores))
    scores_use = scores[:min_len]
    gt_labels_use = gt_labels[:min_len]

    # Update B-factors with prediction scores
    updated_structure = update_bfactors_with_scores(
        structure, chain, scores_use, gt_labels_use
    )

    # Save updated PDB
    output_file = OUTPUT_DIR / f"{pdb_code}_{chain}_surf2spot_predictions.pdb"
    OUTPUT_DIR.mkdir(exist_ok=True)

    io = PDBIO()
    io.set_structure(updated_structure)
    io.save(str(output_file))

    # Also create a ground truth PDB for comparison
    gt_structure = parser.get_structure(f"{pdb_code}_gt", pdb_file)
    gt_scores = [float(label) for label in gt_labels_use]  # Convert 0/1 to float
    gt_updated = update_bfactors_with_scores(gt_structure, chain, gt_scores, gt_labels_use)

    gt_output_file = OUTPUT_DIR / f"{pdb_code}_{chain}_ground_truth.pdb"
    io.set_structure(gt_updated)
    io.save(str(gt_output_file))

    print(f"   ✅ Saved prediction PDB: {output_file}")
    print(f"   ✅ Saved ground truth PDB: {gt_output_file}")

    # Print some statistics
    mean_score = np.mean(scores_use)
    max_score = np.max(scores_use)
    min_score = np.min(scores_use)
    epitope_residues = sum(gt_labels_use)

    print(f"   📊 Score stats: mean={mean_score:.3f}, min={min_score:.3f}, max={max_score:.3f}")
    print(f"   📊 Epitope residues: {epitope_residues}/{len(gt_labels_use)} ({100*epitope_residues/len(gt_labels_use):.1f}%)")

def create_visualization_script():
    """Create PyMOL script for visualizing the results."""
    script_content = '''# PyMOL script for visualizing Surf2Spot epitope predictions
# B-factors represent prediction scores (0-100 scale)

# Load prediction PDB files
'''

    for header, pdb_code, chain, f1, auc in SELECTED_ANTIGENS:
        script_content += f'''
# {header} - F1: {f1:.3f}, AUC: {auc:.3f}
load {pdb_code}_{chain}_surf2spot_predictions.pdb, {pdb_code}_pred
load {pdb_code}_{chain}_ground_truth.pdb, {pdb_code}_gt

# Color by B-factor (prediction scores)
spectrum b, red_white_blue, {pdb_code}_pred and chain {chain}
spectrum b, yellow_red, {pdb_code}_gt and chain {chain}

# Show only the antigen chain
hide everything, not chain {chain}
show cartoon, chain {chain}
show surface, {pdb_code}_pred and chain {chain}

'''

    script_content += '''
# Color scheme:
# Predictions (red_white_blue): Blue = low scores, Red = high scores
# Ground truth (yellow_red): Yellow = non-epitope (0), Red = epitope (1)

# Basic viewing setup
bg_color white
set surface_transparency, 0.3
set cartoon_transparency, 0.5
'''

    script_file = OUTPUT_DIR / "visualize_predictions.pml"
    with open(script_file, 'w') as f:
        f.write(script_content)

    print(f"📝 Created PyMOL visualization script: {script_file}")

def main():
    """Main function to create prediction PDB files."""
    print("🧬 Converting Surf2Spot Predictions to PDB B-factors")
    print("=" * 60)

    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📋 Processing {len(SELECTED_ANTIGENS)} median-performing antigens:")

    for header, pdb_code, chain, f1, auc in SELECTED_ANTIGENS:
        print(f"   • {header} (F1: {f1:.3f}, AUC: {auc:.3f})")

    # Process each antigen
    success_count = 0
    for header, pdb_code, chain, f1, auc in SELECTED_ANTIGENS:
        try:
            create_prediction_pdb(header, pdb_code, chain, f1, auc)
            success_count += 1
        except Exception as e:
            print(f"   ❌ Error processing {header}: {e}")

    print(f"\n📊 Summary:")
    print(f"   Successfully processed: {success_count}/{len(SELECTED_ANTIGENS)} antigens")
    print(f"   Output files saved to: {OUTPUT_DIR}")

    # Create visualization script
    if success_count > 0:
        create_visualization_script()

        print(f"\n🎨 Visualization Instructions:")
        print(f"   1. Open PyMOL")
        print(f"   2. Change to directory: cd {OUTPUT_DIR}")
        print(f"   3. Run script: @visualize_predictions.pml")
        print(f"   4. Files with '_surf2spot_predictions.pdb' contain prediction scores")
        print(f"   5. Files with '_ground_truth.pdb' contain true epitope labels")
        print(f"   6. B-factors: 0-100 scale for predictions, 0/100 for ground truth")

if __name__ == "__main__":
    main()