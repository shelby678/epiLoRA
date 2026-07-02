#!/usr/bin/env python3
"""
Evaluate existing Surf2Spot predictions against sabdab_novel30.fasta ground truth.

This script checks what predictions are available and evaluates them properly
against the validation set to understand if previous bad results were due to
setup issues or prediction quality.
"""

import os
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

# Surf2Spot prediction directories to check
PREDICT_DIRS = [
    SURF2SPOT_DIR / "test_NB" / "predict",
    SURF2SPOT_DIR / "test_NB" / "esm_predict",
    SURF2SPOT_DIR / "test_NB_esmall" / "predict",
    SURF2SPOT_DIR / "test_sabdab" / "predict",  # If our new run worked
]

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

def get_antigen_id(header: str) -> str:
    """Extract antigen identifier from sabdab header."""
    parts = header.split()
    pdb_chain_info = parts[0]  # e.g., "8dyx_HL"
    antigen_chain = parts[1]   # e.g., "I"

    pdb_code = pdb_chain_info.split("_")[0].lower()  # e.g., "8dyx"
    return f"{pdb_code}_{antigen_chain}"

def find_prediction_files(antigen_id: str) -> List[Path]:
    """Find all available prediction files for an antigen across all directories."""
    found_files = []

    for predict_dir in PREDICT_DIRS:
        if not predict_dir.exists():
            continue

        # Try various naming patterns
        patterns = [
            f"{antigen_id}.csv",
            f"{antigen_id}_pred.csv",
            f"{antigen_id}_{antigen_id.split('_')[1]}.csv",  # e.g., 8dyx_I_I.csv
        ]

        for pattern in patterns:
            filepath = predict_dir / pattern
            if filepath.exists():
                found_files.append(filepath)

        # Try domain split patterns
        domain_pattern = predict_dir / f"{antigen_id}*_domain_*_pred.csv"
        domain_files = list(predict_dir.glob(f"{antigen_id}*_domain_*_pred.csv"))
        if domain_files:
            found_files.extend(domain_files)

    return found_files

def load_prediction_scores(files: List[Path]) -> Optional[List[float]]:
    """Load and merge prediction scores from multiple files."""
    if not files:
        return None

    # Try simple CSV first (single file)
    if len(files) == 1 and not "domain" in files[0].name:
        try:
            df = pd.read_csv(files[0])
            if 'score' in df.columns:
                return df['score'].tolist()
        except Exception as e:
            print(f"  Error reading {files[0]}: {e}")
            return None

    # Handle domain split files
    domain_files = [f for f in files if "domain" in f.name]
    if domain_files:
        try:
            dfs = [pd.read_csv(f) for f in domain_files]
            if all('aa_id' in df.columns and 'score' in df.columns for df in dfs):
                # Merge domain predictions
                max_aa_id = max(df['aa_id'].max() for df in dfs if len(df) > 0)
                merged_scores = [0.0] * max_aa_id

                for df in dfs:
                    for _, row in df.iterrows():
                        idx = int(row['aa_id']) - 1
                        if 0 <= idx < len(merged_scores):
                            merged_scores[idx] = max(merged_scores[idx], float(row['score']))

                return merged_scores
        except Exception as e:
            print(f"  Error merging domain predictions: {e}")

    # Try any remaining single file
    for file in files:
        try:
            df = pd.read_csv(file)
            if 'score' in df.columns:
                return df['score'].tolist()
        except Exception as e:
            continue

    return None

def evaluate_predictions():
    """Evaluate all available predictions against sabdab_novel30.fasta."""
    print("Evaluating Existing Surf2Spot Predictions")
    print("=" * 60)

    # Check which prediction directories exist
    existing_dirs = [d for d in PREDICT_DIRS if d.exists()]
    print(f"Found {len(existing_dirs)} prediction directories:")
    for d in existing_dirs:
        n_csvs = len(list(d.glob("*.csv")))
        print(f"  {d.relative_to(SURF2SPOT_DIR)}: {n_csvs} CSV files")
    print()

    if not existing_dirs:
        print("ERROR: No prediction directories found!")
        return

    # Load ground truth
    gt_data = parse_sabdab_fasta()
    print(f"Ground truth: {len(gt_data)} antigens from sabdab_novel30.fasta")

    # Evaluate predictions
    results = []
    all_scores, all_labels = [], []
    skipped = []
    found_predictions = []

    for header, (sequence, labels) in gt_data.items():
        antigen_id = get_antigen_id(header)
        pred_files = find_prediction_files(antigen_id)

        if not pred_files:
            skipped.append(f"{header} (no prediction files)")
            continue

        scores = load_prediction_scores(pred_files)
        if scores is None:
            skipped.append(f"{header} (failed to load predictions)")
            continue

        found_predictions.append((header, pred_files))

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
            results.append((header, auc, n_pos, len(labels_use), note, pred_files[0].parent.name))
            all_scores.extend(scores_clean)
            all_labels.extend(labels_use)
        except Exception as e:
            skipped.append(f"{header} (AUC calculation error: {e})")

    # Print results
    print(f"\nFound predictions for {len(found_predictions)} antigens:")
    print(f"{'Antigen':<30} {'AUC':>6}  {'Pos':>5}  {'Len':>5}  {'Source'}")
    print("-" * 80)

    for header, auc, n_pos, n_res, note, source in sorted(results, key=lambda x: -x[1]):
        print(f"{header:<30} {auc:>6.3f}  {n_pos:>5}  {n_res:>5}  {source}{note}")

    if results:
        overall_auc = roc_auc_score(all_labels, all_scores)
        mean_auc = np.mean([r[1] for r in results])
        median_auc = np.median([r[1] for r in results])

        print()
        print(f"Overall ROC AUC (all residues pooled): {overall_auc:.4f}")
        print(f"Mean per-antigen AUC:                  {mean_auc:.4f}")
        print(f"Median per-antigen AUC:                {median_auc:.4f}")
        print(f"Antigens evaluated: {len(results)}")

        # Analyze by source
        by_source = {}
        for header, auc, n_pos, n_res, note, source in results:
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(auc)

        print("\nResults by prediction source:")
        for source, aucs in by_source.items():
            print(f"  {source}: {len(aucs)} antigens, mean AUC = {np.mean(aucs):.4f}")

    print(f"\nAntigens without predictions: {len(skipped)}")
    if skipped:
        print("  Missing predictions:")
        for s in skipped[:10]:  # Show first 10
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    # Show some example predictions that were found
    if found_predictions:
        print(f"\nExample prediction files found:")
        for header, files in found_predictions[:5]:
            print(f"  {header}:")
            for f in files:
                rel_path = f.relative_to(SURF2SPOT_DIR)
                print(f"    {rel_path}")

if __name__ == "__main__":
    evaluate_predictions()