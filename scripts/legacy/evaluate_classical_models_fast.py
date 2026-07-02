#!/usr/bin/env python3
"""
Fast evaluation of classical ML models with optimizations for large datasets.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import time
from typing import Dict, List, Tuple, Optional

# Paths
AUTOPROT_DIR = Path(__file__).parent
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"

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

def create_sequence_features(sequence: str, epitope_labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create simple sequence-based features for each residue."""
    features = []

    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    for i, aa in enumerate(sequence):
        feat = [0.0] * 30  # Feature vector

        # One-hot amino acid encoding (20 features)
        if aa in aa_to_idx:
            feat[aa_to_idx[aa]] = 1.0

        # Position features
        feat[20] = i / len(sequence)  # Relative position
        feat[21] = 1.0 if i < len(sequence) * 0.1 else 0.0  # N-terminal
        feat[22] = 1.0 if i > len(sequence) * 0.9 else 0.0  # C-terminal

        # Local context features (±2 window)
        for j in range(-2, 3):
            if j != 0 and 0 <= i + j < len(sequence):
                neighbor_aa = sequence[i + j]
                if neighbor_aa in aa_to_idx:
                    feat[23 + j + 2] = aa_to_idx[neighbor_aa] / 19.0  # Normalized

        # Hydrophobicity (simplified)
        hydrophobic = 'AILMFPWV'
        feat[28] = 1.0 if aa in hydrophobic else 0.0

        # Charge
        positive = 'RK'
        negative = 'DE'
        feat[29] = 1.0 if aa in positive else (-1.0 if aa in negative else 0.0)

        features.append(feat)

    return np.array(features), np.array(epitope_labels)

def prepare_dataset(gt_data: Dict[str, Tuple[str, List[int]]], sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare dataset for classical ML models."""
    all_features = []
    all_labels = []
    all_ids = []

    for antigen_id, (sequence, epitope_labels) in gt_data.items():
        features, labels = create_sequence_features(sequence, epitope_labels)

        for i in range(len(features)):
            all_features.append(features[i])
            all_labels.append(labels[i])
            all_ids.append(f"{antigen_id}_{i}")

    X = np.array(all_features)
    y = np.array(all_labels)

    # Sample for faster evaluation if requested
    if sample_size and len(X) > sample_size:
        print(f"   Sampling {sample_size:,} from {len(X):,} total samples for faster evaluation...")

        # Stratified sampling to maintain class balance
        positive_idx = np.where(y == 1)[0]
        negative_idx = np.where(y == 0)[0]

        # Calculate samples per class
        pos_ratio = len(positive_idx) / len(y)
        n_pos = int(sample_size * pos_ratio)
        n_neg = sample_size - n_pos

        # Sample
        sampled_pos = np.random.choice(positive_idx, min(n_pos, len(positive_idx)), replace=False)
        sampled_neg = np.random.choice(negative_idx, min(n_neg, len(negative_idx)), replace=False)

        sampled_idx = np.concatenate([sampled_pos, sampled_neg])

        X = X[sampled_idx]
        y = y[sampled_idx]
        all_ids = [all_ids[i] for i in sampled_idx]

    return X, y, all_ids

def evaluate_classical_models(X: np.ndarray, y: np.ndarray, fast_mode: bool = True) -> Dict[str, Dict[str, float]]:
    """Evaluate classical ML models using cross-validation."""

    if fast_mode:
        # Fast models for large datasets
        models = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
            ]),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
            'KNN (k=3)': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
            'SVM (Linear)': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='linear', probability=True, random_state=42, C=1.0))
            ])
        }
    else:
        # Full model suite (slower)
        models = {
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'SVM (RBF)': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ]),
            'SVM (Linear)': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='linear', probability=True, random_state=42))
            ]),
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
            ])
        }

    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed

    print("🤖 Evaluating Classical ML Models")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n📊 {name}:")
        start_time = time.time()

        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

            results[name] = {
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std(),
                'scores': cv_scores.tolist(),
                'time': time.time() - start_time
            }

            print(f"   ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"   CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
            print(f"   Time: {results[name]['time']:.1f}s")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = {'mean_auc': 0.0, 'std_auc': 0.0, 'error': str(e)}

    return results

def compare_with_surf2spot(classical_results: Dict[str, Dict[str, float]], surf2spot_auc: float = 0.3941):
    """Compare classical models with Surf2Spot results."""
    print(f"\n🏆 Model Comparison")
    print("=" * 80)
    print(f"{'Model':<20} {'ROC-AUC':<12} {'Std':<8} {'Time(s)':<8} {'vs Surf2Spot':<15}")
    print("-" * 80)

    # Sort by mean AUC
    sorted_results = sorted(classical_results.items(), key=lambda x: x[1].get('mean_auc', 0), reverse=True)

    for name, results in sorted_results:
        if 'error' not in results:
            auc = results['mean_auc']
            std = results['std_auc']
            time_s = results.get('time', 0)
            improvement = ((auc - surf2spot_auc) / surf2spot_auc * 100) if surf2spot_auc > 0 else 0

            print(f"{name:<20} {auc:.4f}       {std:.3f}    {time_s:6.1f}   {improvement:+.1f}%")
        else:
            print(f"{name:<20} ERROR        -        -        -")

    print(f"\nBaselines:")
    print(f"  Surf2Spot (Antigen-Only): {surf2spot_auc:.4f}")
    print(f"  Random Classifier:         0.5000")

def main():
    print("🔬 Classical ML Models for Epitope Prediction (Fast Mode)")
    print("=" * 70)

    # Load ground truth data
    print("📖 Loading ground truth data...")
    gt_data = parse_sabdab_fasta()
    print(f"   Loaded {len(gt_data)} antigen sequences")

    # Prepare dataset with sampling for speed
    print("🔧 Preparing dataset...")
    X, y, ids = prepare_dataset(gt_data, sample_size=10000)  # Sample 10k for speed

    # Print dataset statistics
    total_residues = len(y)
    epitope_residues = np.sum(y)
    epitope_ratio = epitope_residues / total_residues * 100

    print(f"   Total residues: {total_residues:,}")
    print(f"   Epitope residues: {epitope_residues:,} ({epitope_ratio:.1f}%)")
    print(f"   Feature dimensions: {X.shape[1]}")
    print(f"   Class balance: {epitope_ratio:.1f}% epitope, {100-epitope_ratio:.1f}% non-epitope")

    # Evaluate classical models
    classical_results = evaluate_classical_models(X, y, fast_mode=True)

    # Compare with Surf2Spot
    compare_with_surf2spot(classical_results)

    # Find best model
    valid_results = {k: v for k, v in classical_results.items() if 'error' not in v}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['mean_auc'])

        print(f"\n🥇 Best Classical Model: {best_model[0]}")
        print(f"   ROC-AUC: {best_model[1]['mean_auc']:.4f} ± {best_model[1]['std_auc']:.4f}")
        print(f"   Training Time: {best_model[1]['time']:.1f}s")

        if best_model[1]['mean_auc'] > 0.3941:
            improvement = (best_model[1]['mean_auc'] - 0.3941) / 0.3941 * 100
            print(f"   🎉 Improvement over Surf2Spot: +{improvement:.1f}%")
        else:
            decline = (0.3941 - best_model[1]['mean_auc']) / 0.3941 * 100
            print(f"   📉 Performance vs Surf2Spot: -{decline:.1f}%")

        print(f"\n💡 Key Insights:")
        print(f"   • Simple sequence features can {'outperform' if best_model[1]['mean_auc'] > 0.3941 else 'compete with'} complex structure-based methods")
        print(f"   • Class imbalance ({epitope_ratio:.1f}% epitope) is a major challenge")
        print(f"   • Fast classical models provide good baselines for epitope prediction")

if __name__ == "__main__":
    main()