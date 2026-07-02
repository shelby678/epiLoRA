#!/usr/bin/env python3
"""
Surf2Spot evaluation with F1 scores on sabdab_novel30.fasta validation set.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional

# Paths
AUTOPROT_DIR = Path(__file__).parent
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"

PREDICT_DIRS = [
    SURF2SPOT_DIR / "test_NB" / "predict",
    SURF2SPOT_DIR / "test_NB" / "esm_predict",
    SURF2SPOT_DIR / "test_NB_esmall" / "predict",
    SURF2SPOT_DIR / "test_sabdab_proper" / "predict",
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

def find_prediction_files(antigen_id: str) -> List[Path]:
    """Find all available prediction files for an antigen."""
    found_files = []
    pdb_code, antigen_chain = antigen_id.split("_")

    for predict_dir in PREDICT_DIRS:
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
    thresholds = np.linspace(0.1, 0.9, 81)  # Test thresholds from 0.1 to 0.9
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        if sum(y_pred) > 0 and sum(y_pred) < len(y_pred):  # Avoid all-0 or all-1 predictions
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    return best_threshold

def calculate_metrics_with_threshold(y_true: List[int], y_scores: List[float], threshold: float) -> Dict[str, float]:
    """Calculate precision, recall, F1 with given threshold."""
    y_pred = [1 if score >= threshold else 0 for score in y_scores]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'n_pred_pos': sum(y_pred),
        'n_true_pos': sum(y_true)
    }

def evaluate_with_f1():
    """Evaluate Surf2Spot with both AUC and F1 metrics."""
    print("🔬 Surf2Spot Evaluation: AUC + F1 Scores")
    print("=" * 60)

    # Load data
    gt_data = parse_sabdab_fasta()
    print(f"📋 Ground truth: {len(gt_data)} antigens from sabdab_novel30.fasta")

    results = []
    all_scores, all_labels = [], []
    skipped = []

    # Evaluate each antigen
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

        # Align lengths
        n_gt, n_pred = len(labels), len(scores)
        if n_pred != n_gt:
            n = min(n_gt, n_pred)
            labels_use, scores_use = labels[:n], scores[:n]
        else:
            labels_use, scores_use = labels, scores

        n_pos = sum(labels_use)
        if n_pos == 0 or n_pos == len(labels_use):
            skipped.append(f"{header} (mono-class: n_pos={n_pos}/{len(labels_use)})")
            continue

        # Clean scores
        scores_clean = [0.0 if pd.isna(s) else s for s in scores_use]

        # Calculate AUC
        try:
            auc = roc_auc_score(labels_use, scores_clean)
        except Exception:
            skipped.append(f"{header} (AUC error)")
            continue

        # Calculate F1 with optimal threshold
        optimal_threshold = find_optimal_threshold(labels_use, scores_clean)
        metrics_optimal = calculate_metrics_with_threshold(labels_use, scores_clean, optimal_threshold)

        # Also calculate F1 with standard threshold (0.5)
        metrics_05 = calculate_metrics_with_threshold(labels_use, scores_clean, 0.5)

        results.append({
            'header': header,
            'antigen_id': antigen_id,
            'auc': auc,
            'n_pos': n_pos,
            'n_total': len(labels_use),
            'optimal_threshold': optimal_threshold,
            'f1_optimal': metrics_optimal['f1'],
            'precision_optimal': metrics_optimal['precision'],
            'recall_optimal': metrics_optimal['recall'],
            'f1_05': metrics_05['f1'],
            'precision_05': metrics_05['precision'],
            'recall_05': metrics_05['recall'],
        })

        all_scores.extend(scores_clean)
        all_labels.extend(labels_use)

    # Print detailed results
    print(f"\n🎯 RESULTS - {len(results)} antigens evaluated:")
    print(f"{'Antigen':<30} {'AUC':>6} {'F1(opt)':>8} {'F1(0.5)':>8} {'Thr':>5} {'Pos':>3}/{'Tot'}")
    print("-" * 75)

    for r in sorted(results, key=lambda x: -x['f1_optimal']):
        print(f"{r['header']:<30} {r['auc']:>6.3f} {r['f1_optimal']:>8.3f} {r['f1_05']:>8.3f} "
              f"{r['optimal_threshold']:>5.2f} {r['n_pos']:>3}/{r['n_total']}")

    # Overall metrics
    if results:
        print(f"\n📈 SUMMARY METRICS:")

        # AUC metrics
        overall_auc = roc_auc_score(all_labels, all_scores)
        mean_auc = np.mean([r['auc'] for r in results])

        # F1 metrics (per-antigen averages)
        mean_f1_optimal = np.mean([r['f1_optimal'] for r in results])
        mean_f1_05 = np.mean([r['f1_05'] for r in results])
        mean_precision_optimal = np.mean([r['precision_optimal'] for r in results])
        mean_recall_optimal = np.mean([r['recall_optimal'] for r in results])

        # Overall F1 (pooled predictions)
        optimal_threshold_global = find_optimal_threshold(all_labels, all_scores)
        overall_metrics_optimal = calculate_metrics_with_threshold(all_labels, all_scores, optimal_threshold_global)
        overall_metrics_05 = calculate_metrics_with_threshold(all_labels, all_scores, 0.5)

        print(f"   📊 AUC Metrics:")
        print(f"      Overall AUC (pooled):     {overall_auc:.4f}")
        print(f"      Mean per-antigen AUC:     {mean_auc:.4f}")

        print(f"   🎯 F1 Metrics (Per-antigen Average):")
        print(f"      Mean F1 (optimal thresh): {mean_f1_optimal:.4f}")
        print(f"      Mean F1 (0.5 threshold):  {mean_f1_05:.4f}")
        print(f"      Mean Precision (optimal): {mean_precision_optimal:.4f}")
        print(f"      Mean Recall (optimal):    {mean_recall_optimal:.4f}")

        print(f"   🌍 F1 Metrics (Overall/Pooled):")
        print(f"      Overall F1 (optimal):     {overall_metrics_optimal['f1']:.4f} (threshold={optimal_threshold_global:.2f})")
        print(f"      Overall F1 (0.5):         {overall_metrics_05['f1']:.4f}")
        print(f"      Overall Precision (opt):  {overall_metrics_optimal['precision']:.4f}")
        print(f"      Overall Recall (opt):     {overall_metrics_optimal['recall']:.4f}")

        print(f"   📋 Dataset Info:")
        print(f"      Antigens evaluated:       {len(results)}")
        print(f"      Total residues:           {len(all_labels):,}")
        print(f"      Epitope residues:         {sum(all_labels):,} ({100*sum(all_labels)/len(all_labels):.1f}%)")

        # Top performers by F1
        print(f"\n🏆 TOP F1 PERFORMERS:")
        for r in sorted(results, key=lambda x: -x['f1_optimal'])[:5]:
            print(f"   {r['header']:<25} F1={r['f1_optimal']:.3f}, AUC={r['auc']:.3f}")

    print(f"\n⚠️  Skipped: {len(skipped)} antigens")
    if skipped and len(skipped) <= 10:
        for s in skipped:
            print(f"   {s}")

    return results

if __name__ == "__main__":
    results = evaluate_with_f1()

    if results:
        print(f"\n✅ Evaluation completed!")
        mean_f1 = np.mean([r['f1_optimal'] for r in results])
        mean_auc = np.mean([r['auc'] for r in results])
        print(f"   Summary: {len(results)} antigens, F1={mean_f1:.3f}, AUC={mean_auc:.3f}")