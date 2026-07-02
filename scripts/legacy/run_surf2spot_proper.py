#!/usr/bin/env python3
"""
Run Surf2Spot epitope prediction on sabdab_novel30.fasta with proper environment setup.

This script:
1. Uses the correct micromamba environments for Surf2Spot
2. Runs the complete pipeline: preprocess -> craft -> predict -> draw
3. Evaluates results against ground truth epitope labels
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
INPUT_DIR = SURF2SPOT_DIR / "test_sabdab_proper" / "input"
PREPROCESS_DIR = SURF2SPOT_DIR / "test_sabdab_proper" / "preprocess"
PREDICT_DIR = SURF2SPOT_DIR / "test_sabdab_proper" / "predict"

# Environment setup
MICROMAMBA_PATH = "/home/sferrier/miniforge3/micromamba"
SURF2SPOT_ENV = "/home/sferrier/miniforge3/envs/surf2spot"
SURF2SPOT_TOOLS_ENV = "/home/sferrier/miniforge3/envs/surf2spot_tools"

def extract_antigen_chain_simple(input_pdb: Path, output_pdb: Path, antigen_chain: str) -> bool:
    """Extract only the antigen chain using simple text parsing."""
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    chain_found = False
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith(('HEADER', 'TITLE', 'COMPND', 'SOURCE', 'REMARK')):
                outfile.write(line)
            elif line.startswith(('ATOM', 'HETATM')):
                # Check if this atom belongs to our antigen chain
                if len(line) > 21 and line[21] == antigen_chain:
                    outfile.write(line)
                    chain_found = True
            elif line.startswith('END'):
                outfile.write(line)

    return chain_found

def run_with_micromamba(env_path: str, cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a command in a specific micromamba environment."""
    if cwd is None:
        cwd = SURF2SPOT_DIR

    # Create a bash command that properly activates micromamba and runs the command
    shell_cmd = f'''
    eval "$({MICROMAMBA_PATH} shell hook --shell=bash)" && \
    micromamba activate {env_path} && \
    {' '.join(cmd)}
    '''

    print(f"Running: {' '.join(cmd)} (in {Path(env_path).name})")
    return subprocess.run(['bash', '-c', shell_cmd], cwd=cwd, capture_output=True, text=True)

def parse_sabdab_fasta() -> Dict[str, Tuple[str, List[int]]]:
    """Parse sabdab_novel30.fasta into ground truth labels."""
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
    """Set up Surf2Spot input directory with PDB files."""
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
            # Extract only antigen chain (not full complex)
            dest_name = f"{pdb_code}_{antigen_chain}.pdb"
            output_path = INPUT_DIR / dest_name

            # Extract antigen chain using simple text parsing
            chain_found = extract_antigen_chain_simple(pdb_file, output_path, antigen_chain)

            if chain_found:
                prepared.append(f"{pdb_code}_{antigen_chain}")
                print(f"✓ Extracted antigen chain {antigen_chain} from {pdb_code} -> {dest_name}")
            else:
                missing.append(f"{pdb_code}_{antigen_chain}")
                print(f"✗ Chain {antigen_chain} not found in {pdb_code}")
        else:
            missing.append(f"{pdb_code}_{antigen_chain}")

    print(f"\nPrepared: {len(prepared)} antigens")
    if missing:
        print(f"Missing PDB files: {len(missing)} antigens")
        for m in missing[:5]:  # Show first 5
            print(f"  {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    return prepared

def run_surf2spot_pipeline():
    """Run the complete Surf2Spot NB pipeline with proper environment setup."""
    print("\n" + "="*60)
    print("Running Surf2Spot NB Pipeline (Proper Environment)")
    print("="*60)

    # Step 1: NB-preprocess (surf2spot environment)
    print("\n1. NB-preprocess...")
    domain_tsv = PREPROCESS_DIR / "chainsaw.tsv"

    result = run_with_micromamba(
        SURF2SPOT_ENV,
        ["python", "-m", "Surf2Spot.main", "NB-preprocess",
         "-i", str(INPUT_DIR), "-o", str(PREPROCESS_DIR),
         "-ds", str(domain_tsv)]
    )

    if result.returncode != 0:
        print(f"ERROR in NB-preprocess:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✓ NB-preprocess completed")

    # Step 2: NB-craft (surf2spot_tools environment for ESM)
    print("\n2. NB-craft...")
    seq_fasta = PREPROCESS_DIR / "seq.fasta"
    embeddings_h5 = PREPROCESS_DIR / "seq_prottrans.h5"
    domain_tsv = PREPROCESS_DIR / "chainsaw.tsv"

    result = run_with_micromamba(
        SURF2SPOT_TOOLS_ENV,
        ["python", "-m", "Surf2Spot.main", "NB-craft",
         "-i", str(PREPROCESS_DIR),
         "-s", str(seq_fasta),
         "-emb", str(embeddings_h5),
         "-ds", str(domain_tsv)]
    )

    if result.returncode != 0:
        print(f"ERROR in NB-craft:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✓ NB-craft completed")

    # Step 3: NB-predict (surf2spot environment)
    print("\n3. NB-predict...")
    model_path = SURF2SPOT_DIR / "model" / "NB" / "model.pt"
    result = run_with_micromamba(
        SURF2SPOT_ENV,
        ["python", "-m", "Surf2Spot.main", "NB-predict",
         "-i", str(PREPROCESS_DIR), "-o", str(PREDICT_DIR),
         "-emb", str(embeddings_h5),
         "--model", str(model_path)]
    )

    if result.returncode != 0:
        print(f"ERROR in NB-predict:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✓ NB-predict completed")

    # Step 4: NB-draw (surf2spot environment)
    print("\n4. NB-draw...")
    result = run_with_micromamba(
        SURF2SPOT_ENV,
        ["python", "-m", "Surf2Spot.main", "NB-draw",
         "-i", str(PREPROCESS_DIR), "-o", str(PREDICT_DIR)]
    )

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
    domain_files = list(PREDICT_DIR.glob(f"{antigen_id}*_domain_*_pred.csv"))

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

        # Compare with previous attempt
        print(f"\nComparison with previous attempt:")
        print(f"  Previous: 37 antigens, mean AUC = 0.507, overall AUC = 0.527")
        print(f"  Current:  {len(results)} antigens, mean AUC = {mean_auc:.3f}, overall AUC = {overall_auc:.3f}")

    print(f"Antigens skipped:   {len(skipped)}")
    if skipped:
        print("  Skipped:")
        for s in skipped[:10]:  # Show first 10
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

def main():
    print("Surf2Spot Epitope Prediction - Proper Environment Setup")
    print("="*60)

    # Check prerequisites
    if not SABDAB_FASTA.exists():
        print(f"ERROR: {SABDAB_FASTA} not found")
        return 1

    if not SURF2SPOT_DIR.exists():
        print(f"ERROR: Surf2Spot directory not found at {SURF2SPOT_DIR}")
        return 1

    if not Path(MICROMAMBA_PATH).exists():
        print(f"ERROR: Micromamba not found at {MICROMAMBA_PATH}")
        return 1

    for env_name, env_path in [("surf2spot", SURF2SPOT_ENV), ("surf2spot_tools", SURF2SPOT_TOOLS_ENV)]:
        if not Path(env_path).exists():
            print(f"ERROR: Environment {env_name} not found at {env_path}")
            return 1

    model_file = SURF2SPOT_DIR / "model" / "NB" / "model.pt"
    if not model_file.exists():
        print(f"ERROR: Surf2Spot NB model not found at {model_file}")
        return 1

    print("✓ All prerequisites satisfied")

    # Parse ground truth data
    print("\nParsing ground truth data...")
    gt_data = parse_sabdab_fasta()
    print(f"Loaded {len(gt_data)} antigens from {SABDAB_FASTA.name}")

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

    print("\n✓ Surf2Spot evaluation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())