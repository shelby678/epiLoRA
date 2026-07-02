#!/usr/bin/env python3
"""
Run Surf2Spot epitope prediction on sabdab_novel30.fasta validation set.

This script:
1. Converts sabdab_novel30.fasta to Surf2Spot input format
2. Copies necessary PDB files to Surf2Spot input directory
3. Runs the complete Surf2Spot NB pipeline
4. Evaluates results against ground truth epitope labels
"""

import os
import sys
import shutil
import subprocess
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional

# Paths
AUTOPROT_DIR = Path(__file__).parent
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"
PDB_DIR = AUTOPROT_DIR / "data" / "structures2" / "sabdab_dataset"

# Surf2Spot directories
INPUT_DIR = SURF2SPOT_DIR / "test_sabdab" / "input"
PREPROCESS_DIR = SURF2SPOT_DIR / "test_sabdab" / "preprocess"
PREDICT_DIR = SURF2SPOT_DIR / "test_sabdab" / "predict"

def parse_sabdab_fasta() -> Dict[str, Tuple[str, List[int]]]:
    """
    Parse sabdab_novel30.fasta into ground truth labels.

    Returns:
        Dict mapping header -> (sequence, epitope_labels)
        where epitope_labels[i] = 1 if uppercase, 0 if lowercase
    """
    gt_data = {}

    with open(SABDAB_FASTA) as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]  # Remove >
            elif header:
                sequence = line
                # Convert case to labels: uppercase = epitope (1), lowercase = non-epitope (0)
                labels = [1 if c.isupper() else 0 for c in sequence]
                gt_data[header] = (sequence.upper(), labels)

    return gt_data

def get_pdb_info(header: str) -> Tuple[str, str]:
    """Extract PDB code and chain from sabdab header."""
    parts = header.split()
    pdb_chain_info = parts[0]  # e.g., "8dyx_HL"
    antigen_chain = parts[1]   # e.g., "I"

    # Extract PDB code from pdb_chain_info
    pdb_code = pdb_chain_info.split("_")[0]  # e.g., "8dyx"

    return pdb_code.lower(), antigen_chain

def setup_surf2spot_input(gt_data: Dict[str, Tuple[str, List[int]]]) -> List[str]:
    """
    Set up Surf2Spot input directory with PDB files.

    Returns:
        List of successfully prepared antigen identifiers
    """
    # Create input directory
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing files
    for f in INPUT_DIR.glob("*.pdb"):
        f.unlink()

    prepared = []
    missing = []

    for header, (sequence, labels) in gt_data.items():
        pdb_code, antigen_chain = get_pdb_info(header)

        # Look for PDB file in structures directory
        # Try various naming patterns - files are in {pdb_code}/structure/{pdb_code}.pdb
        pdb_patterns = [
            PDB_DIR / f"{pdb_code}" / "structure" / f"{pdb_code}.pdb",
            PDB_DIR / f"{pdb_code.upper()}" / "structure" / f"{pdb_code.upper()}.pdb",
            PDB_DIR / f"{pdb_code}.pdb",
            PDB_DIR / f"{pdb_code.upper()}.pdb",
        ]

        pdb_file = None
        for pattern in pdb_patterns:
            if pattern.exists():
                pdb_file = pattern
                break

        if pdb_file:
            # Copy to Surf2Spot input with consistent naming
            dest_name = f"{pdb_code}_{antigen_chain}.pdb"
            shutil.copy2(pdb_file, INPUT_DIR / dest_name)
            prepared.append(f"{pdb_code}_{antigen_chain}")
            print(f"✓ Prepared {dest_name}")
        else:
            missing.append(f"{pdb_code}_{antigen_chain} (tried: {[p.name for p in pdb_patterns]})")

    print(f"\nPrepared: {len(prepared)} antigens")
    if missing:
        print(f"Missing PDB files for {len(missing)} antigens:")
        for m in missing[:10]:  # Show first 10
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    return prepared

def run_surf2spot_pipeline():
    """Run the complete Surf2Spot NB pipeline."""
    os.chdir(SURF2SPOT_DIR)

    print("\n" + "="*60)
    print("Running Surf2Spot NB Pipeline")
    print("="*60)

    # Step 1: NB-preprocess
    print("\n1. NB-preprocess...")
    cmd = [
        "conda", "run", "-n", "surf2spot",
        "Surf2Spot", "NB-preprocess",
        "-i", str(INPUT_DIR),
        "-o", str(PREPROCESS_DIR)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in NB-preprocess:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✓ NB-preprocess completed")

    # Step 2: NB-craft
    print("\n2. NB-craft...")
    cmd = [
        "conda", "run", "-n", "surf2spot_tools",  # Note: different environment
        "Surf2Spot", "NB-craft",
        "-i", str(PREPROCESS_DIR)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in NB-craft:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✓ NB-craft completed")

    # Step 3: NB-predict
    print("\n3. NB-predict...")
    model_path = SURF2SPOT_DIR / "model" / "NB" / "model.pt"
    cmd = [
        "conda", "run", "-n", "surf2spot",
        "Surf2Spot", "NB-predict",
        "-i", str(PREPROCESS_DIR),
        "-o", str(PREDICT_DIR),
        "--model", str(model_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in NB-predict:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✓ NB-predict completed")

    # Step 4: NB-draw
    print("\n4. NB-draw...")
    cmd = [
        "conda", "run", "-n", "surf2spot",
        "Surf2Spot", "NB-draw",
        "-i", str(PREPROCESS_DIR),
        "-o", str(PREDICT_DIR)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in NB-draw:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✓ NB-draw completed")

    return True

def load_surf2spot_predictions(antigen_id: str) -> Optional[List[float]]:
    """Load Surf2Spot prediction scores for an antigen."""
    # Try various CSV naming patterns
    patterns = [
        PREDICT_DIR / f"{antigen_id}.csv",
        PREDICT_DIR / f"{antigen_id}_pred.csv",
    ]

    # Also try domain split patterns
    domain_pattern = PREDICT_DIR / f"{antigen_id}_domain_*_pred.csv"
    domain_files = sorted(glob.glob(str(domain_pattern)))

    # Try simple CSV first
    for pattern in patterns:
        if pattern.exists():
            try:
                df = pd.read_csv(pattern)
                if 'score' in df.columns:
                    return df['score'].tolist()
            except Exception as e:
                print(f"Error reading {pattern}: {e}")

    # Try domain CSVs
    if domain_files:
        try:
            dfs = [pd.read_csv(f) for f in domain_files]
            if all('aa_id' in df.columns and 'score' in df.columns for df in dfs):
                # Merge domain predictions
                max_aa_id = max(df['aa_id'].max() for df in dfs)
                merged_scores = [0.0] * max_aa_id

                for df in dfs:
                    for _, row in df.iterrows():
                        idx = int(row['aa_id']) - 1
                        if 0 <= idx < len(merged_scores):
                            merged_scores[idx] = max(merged_scores[idx], float(row['score']))

                return merged_scores
        except Exception as e:
            print(f"Error merging domain predictions for {antigen_id}: {e}")

    return None

def evaluate_predictions(gt_data: Dict[str, Tuple[str, List[int]]], prepared: List[str]):
    """Evaluate Surf2Spot predictions against ground truth."""
    print("\n" + "="*60)
    print("Evaluating Predictions")
    print("="*60)

    results = []
    all_scores, all_labels = [], []
    skipped = []

    for header, (sequence, labels) in gt_data.items():
        pdb_code, antigen_chain = get_pdb_info(header)
        antigen_id = f"{pdb_code}_{antigen_chain}"

        if antigen_id not in prepared:
            skipped.append(f"{header} (not prepared)")
            continue

        scores = load_surf2spot_predictions(antigen_id)
        if scores is None:
            skipped.append(f"{header} (no prediction file)")
            continue

        # Align prediction and ground truth lengths
        n_gt, n_pred = len(labels), len(scores)
        if n_pred != n_gt:
            n = min(n_gt, n_pred)
            labels_use, scores_use = labels[:n], scores[:n]
            note = f" (len mismatch gt={n_gt} pred={n_pred}, using first {n})"
        else:
            labels_use, scores_use = labels, scores
            note = ""

        n_pos = sum(labels_use)
        if n_pos == 0 or n_pos == len(labels_use):
            skipped.append(f"{header} (all-same labels, n_pos={n_pos})")
            continue

        # Clean NaN scores
        scores_clean = [0.0 if pd.isna(s) else s for s in scores_use]

        try:
            auc = roc_auc_score(labels_use, scores_clean)
            results.append((header, auc, n_pos, len(labels_use), note))
            all_scores.extend(scores_clean)
            all_labels.extend(labels_use)
        except Exception as e:
            skipped.append(f"{header} (AUC calculation error: {e})")

    # Print results
    print(f"{'Antigen':<30} {'AUC':>6}  {'Pos':>5}  {'Len':>5}")
    print("-" * 60)
    for header, auc, n_pos, n_res, note in sorted(results, key=lambda x: -x[1]):
        print(f"{header:<30} {auc:>6.3f}  {n_pos:>5}  {n_res:>5}{note}")

    if results:
        overall_auc = roc_auc_score(all_labels, all_scores)
        mean_auc = np.mean([r[1] for r in results])
        median_auc = np.median([r[1] for r in results])

        print()
        print(f"Overall ROC AUC (all residues pooled): {overall_auc:.4f}")
        print(f"Mean per-antigen AUC:                  {mean_auc:.4f}")
        print(f"Median per-antigen AUC:                {median_auc:.4f}")
        print(f"Antigens evaluated: {len(results)}")

    print(f"Antigens skipped:   {len(skipped)}")
    if skipped:
        print("  Skipped:")
        for s in skipped[:10]:  # Show first 10
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

def main():
    print("Surf2Spot Epitope Prediction on sabdab_novel30.fasta")
    print("="*60)

    # Check prerequisites
    if not SABDAB_FASTA.exists():
        print(f"ERROR: {SABDAB_FASTA} not found")
        return 1

    if not SURF2SPOT_DIR.exists():
        print(f"ERROR: Surf2Spot directory not found at {SURF2SPOT_DIR}")
        return 1

    model_file = SURF2SPOT_DIR / "model" / "NB" / "model.pt"
    if not model_file.exists():
        print(f"ERROR: Surf2Spot NB model not found at {model_file}")
        return 1

    # Parse ground truth data
    print("Parsing ground truth data...")
    gt_data = parse_sabdab_fasta()
    print(f"Loaded {len(gt_data)} antigens from {SABDAB_FASTA}")

    # Setup Surf2Spot input
    print("\nSetting up Surf2Spot input...")
    prepared = setup_surf2spot_input(gt_data)

    if not prepared:
        print("ERROR: No antigens could be prepared for Surf2Spot")
        return 1

    # Run Surf2Spot pipeline
    success = run_surf2spot_pipeline()
    if not success:
        print("ERROR: Surf2Spot pipeline failed")
        return 1

    # Evaluate predictions
    evaluate_predictions(gt_data, prepared)

    print("\n✓ Surf2Spot evaluation completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())