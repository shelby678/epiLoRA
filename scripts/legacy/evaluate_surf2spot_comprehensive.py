#!/usr/bin/env python3
"""
Comprehensive Surf2Spot evaluation comparing original vs corrected (antigen-only) approaches.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional

# Paths
AUTOPROT_DIR = Path(__file__).parent
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"

# Prediction directories for both approaches
ORIGINAL_PREDICT_DIRS = [
    SURF2SPOT_DIR / "test_NB" / "predict",
    SURF2SPOT_DIR / "test_NB" / "esm_predict",
    SURF2SPOT_DIR / "test_NB_esmall" / "predict",
    SURF2SPOT_DIR / "test_sabdab_proper" / "predict",
]

CORRECTED_PREDICT_DIRS = [
    SURF2SPOT_DIR / "test_sabdab_antigen_only" / "predict",
]

def parse_sabdab_fasta() -> Dict[str, Tuple[str, List[int]]]:
    """Parse sabdab_novel30.fasta into ground truth labels."""
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

def get_antigen_id(header: str) -> str:
    """Extract antigen identifier from sabdab header."""
    parts = header.split()
    pdb_chain_info = parts[0]
    antigen_chain = parts[1]
    pdb_code = pdb_chain_info.split("_")[0].lower()
    return f"{pdb_code}_{antigen_chain}"

def find_prediction_files(antigen_id: str, predict_dirs: List[Path]) -> List[Path]:
    """Find prediction files for an antigen in given directories."""
    found_files = []
    pdb_code, antigen_chain = antigen_id.split("_")

    for predict_dir in predict_dirs:
        if not predict_dir.exists():
            continue

        patterns = [
            f"{antigen_id}.csv",
            f"{antigen_id}_pred.csv",
            f"{pdb_code}_{antigen_chain}.csv",
            f"{pdb_code.upper()}_{antigen_chain}.csv",
        ]

        for pattern in patterns:
            filepath = predict_dir / pattern
            if filepath.exists():
                found_files.append(filepath)

        domain_files = list(predict_dir.glob(f"{antigen_id}*_domain_*_pred.csv"))
        if domain_files:
            found_files.extend(domain_files)

    return found_files

def load_prediction_scores(files: List[Path]) -> Optional[List[float]]:
    """Load and merge prediction scores from files."""
    if not files:
        return None

    if len(files) == 1 and "domain" not in files[0].name:
        try:
            df = pd.read_csv(files[0])
            if 'score' in df.columns:
                return df['score'].tolist()
        except Exception:
            return None

    domain_files = [f for f in files if "domain" in f.name]
    if domain_files:
        try:
            dfs = [pd.read_csv(f) for f in domain_files]
            if all('aa_id' in df.columns and 'score' in df.columns for df in dfs):
                max_aa_id = max(df['aa_id'].max() for df in dfs if len(df) > 0)
                merged_scores = [0.0] * max_aa_id

                for df in dfs:
                    for _, row in df.iterrows():
                        idx = int(row['aa_id']) - 1
                        if 0 <= idx < len(merged_scores):
                            merged_scores[idx] = max(merged_scores[idx], float(row['score']))
                return merged_scores
        except Exception:
            pass

    for file in files:
        try:
            df = pd.read_csv(file)
            if 'score' in df.columns:
                return df['score'].tolist()
        except Exception:
            continue

    return None

def find_optimal_threshold(y_true: List[int], y_scores: List[float]) -> float:
    """Find threshold that maximizes F1 score."""
    if not y_scores or not y_true:
        return 0.5

    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        if sum(y_pred) > 0 and sum(y_pred) < len(y_pred):
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    return best_threshold

def evaluate_approach(predict_dirs: List[Path], approach_name: str) -> Tuple[List[Dict], List[float], List[int]]:
    """Evaluate a specific approach (original or corrected)."""
    print(f"\n🔍 Evaluating {approach_name} Approach")
    print("-" * 50)

    # Check available directories
    existing_dirs = [d for d in predict_dirs if d.exists()]
    print(f"📁 Available directories: {len(existing_dirs)}/{len(predict_dirs)}")
    for d in existing_dirs:
        n_csvs = len(list(d.glob("*.csv")))
        print(f"   {d.relative_to(SURF2SPOT_DIR)}: {n_csvs} CSV files")

    if not existing_dirs:
        print(f"❌ No prediction directories found for {approach_name}")
        return [], [], []

    # Load ground truth
    gt_data = parse_sabdab_fasta()

    # Evaluate predictions
    results = []
    all_scores, all_labels = [], []

    for header, (sequence, labels) in gt_data.items():
        antigen_id = get_antigen_id(header)
        pred_files = find_prediction_files(antigen_id, existing_dirs)

        if not pred_files:
            continue

        scores = load_prediction_scores(pred_files)
        if scores is None:
            continue

        # Align lengths
        n_gt, n_pred = len(labels), len(scores)
        if n_pred != n_gt:
            n = min(n_gt, n_pred)
            labels_use, scores_use = labels[:n], scores[:n]
        else:
            labels_use, scores_use = labels, scores

        n_pos = sum(labels_use)
        if n_pos == 0 or n_pos == len(labels_use):
            continue

        # Clean scores
        scores_clean = [0.0 if pd.isna(s) else s for s in scores_use]

        try:
            auc = roc_auc_score(labels_use, scores_clean)
            optimal_threshold = find_optimal_threshold(labels_use, scores_clean)

            # F1 metrics
            y_pred_opt = [1 if score >= optimal_threshold else 0 for score in scores_clean]
            f1_opt = f1_score(labels_use, y_pred_opt, zero_division=0)
            precision_opt = precision_score(labels_use, y_pred_opt, zero_division=0)
            recall_opt = recall_score(labels_use, y_pred_opt, zero_division=0)

            results.append({
                'header': header,
                'antigen_id': antigen_id,
                'auc': auc,
                'f1_optimal': f1_opt,
                'precision_optimal': precision_opt,
                'recall_optimal': recall_opt,
                'optimal_threshold': optimal_threshold,
                'n_pos': n_pos,
                'n_total': len(labels_use),
                'approach': approach_name
            })

            all_scores.extend(scores_clean)
            all_labels.extend(labels_use)

        except Exception:
            continue

    return results, all_scores, all_labels

def comprehensive_evaluation():
    """Compare original vs corrected Surf2Spot approaches."""
    print("🔬 Comprehensive Surf2Spot Evaluation: Original vs Corrected")
    print("=" * 70)

    # Evaluate original approach (full complexes)
    original_results, original_scores, original_labels = evaluate_approach(
        ORIGINAL_PREDICT_DIRS, "Original (Full Complexes)"
    )

    # Evaluate corrected approach (antigen-only)
    corrected_results, corrected_scores, corrected_labels = evaluate_approach(
        CORRECTED_PREDICT_DIRS, "Corrected (Antigen-Only)"
    )

    # Print comparison
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 70)

    if original_results:
        orig_auc = np.mean([r['auc'] for r in original_results])
        orig_f1 = np.mean([r['f1_optimal'] for r in original_results])
        print(f"📈 Original Approach (Full Complexes):")
        print(f"   Antigens: {len(original_results)}")
        print(f"   Mean AUC: {orig_auc:.4f}")
        print(f"   Mean F1:  {orig_f1:.4f}")
    else:
        print(f"❌ Original Approach: No results")

    if corrected_results:
        corr_auc = np.mean([r['auc'] for r in corrected_results])
        corr_f1 = np.mean([r['f1_optimal'] for r in corrected_results])
        print(f"📈 Corrected Approach (Antigen-Only):")
        print(f"   Antigens: {len(corrected_results)}")
        print(f"   Mean AUC: {corr_auc:.4f}")
        print(f"   Mean F1:  {corr_f1:.4f}")

        if original_results:
            auc_improvement = corr_auc - orig_auc
            f1_improvement = corr_f1 - orig_f1
            print(f"📈 Improvement from Correction:")
            print(f"   ΔA-UIC: {auc_improvement:+.4f}")
            print(f"   ΔF1:   {f1_improvement:+.4f}")
    else:
        print(f"❌ Corrected Approach: No results (pipeline may have failed)")

    # Problem analysis
    print(f"\n🔍 PROBLEM ANALYSIS")
    print("=" * 70)

    if not corrected_results:
        print(f"⚠️  The corrected approach produced no prediction files!")
        print(f"   Possible issues:")
        print(f"   1. Pipeline failed after preprocessing")
        print(f"   2. Prediction step encountered errors")
        print(f"   3. File naming/structure incompatibility")
        print(f"   4. Need to debug prediction CSV generation")

    if original_results and not corrected_results:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Check prediction logs for errors")
        print(f"   2. Verify corrected preprocessing worked properly")
        print(f"   3. Run prediction step with debug output")
        print(f"   4. Compare file structures between approaches")

    # Show which antigens we can analyze with original approach
    if original_results:
        print(f"\n🎯 TOP PERFORMERS (Original Approach):")
        top_f1 = sorted(original_results, key=lambda x: x['f1_optimal'], reverse=True)[:5]
        for r in top_f1:
            print(f"   {r['header']:<30} F1: {r['f1_optimal']:.3f}, AUC: {r['auc']:.3f}")

    return original_results, corrected_results

if __name__ == "__main__":
    original_results, corrected_results = comprehensive_evaluation()

    if original_results or corrected_results:
        print(f"\n✅ Evaluation completed!")
        total_antigens = len(original_results) + len(corrected_results)
        print(f"   Total antigens evaluated: {total_antigens}")
    else:
        print(f"\n❌ No predictions found in either approach")