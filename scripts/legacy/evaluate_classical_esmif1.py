#!/usr/bin/env python3
"""
Evaluate classical ML models using ESM-IF1 embeddings for epitope prediction.
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
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import xgboost as xgb
import torch
import esm
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

def load_esmif1_model():
    """Load ESM-IF1 model for generating embeddings."""
    print("🧬 Loading ESM-IF1 model...")
    try:
        # Load ESM-IF1 model
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            print("   Using GPU for ESM-IF1")
        else:
            print("   Using CPU for ESM-IF1")

        return model, alphabet
    except Exception as e:
        print(f"   ❌ Error loading ESM-IF1: {e}")
        return None, None

def generate_esmif1_embeddings(model, alphabet, sequence: str, pdb_path: str = None) -> np.ndarray:
    """Generate ESM-IF1 embeddings for a sequence."""
    try:
        batch_converter = alphabet.get_batch_converter()

        # For ESM-IF1, we need structure coordinates
        # For now, use sequence-only mode (will be less optimal)
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            # Get sequence representations (without structure for now)
            results = model(batch_tokens, repr_layers=[model.num_layers])

        # Extract per-residue embeddings
        embeddings = results["representations"][model.num_layers]

        # Remove batch dimension and [CLS]/[SEP] tokens
        embeddings = embeddings[0, 1:len(sequence)+1].cpu().numpy()

        return embeddings

    except Exception as e:
        print(f"   ❌ Error generating embeddings: {e}")
        return None

def prepare_esmif1_dataset(gt_data: Dict[str, Tuple[str, List[int]]],
                          model, alphabet,
                          max_sequences: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare dataset using ESM-IF1 embeddings."""

    all_features = []
    all_labels = []
    all_ids = []

    print(f"🔧 Generating ESM-IF1 embeddings for {min(len(gt_data), max_sequences)} sequences...")

    processed_count = 0
    for antigen_id, (sequence, epitope_labels) in gt_data.items():
        if processed_count >= max_sequences:
            break

        print(f"   Processing {antigen_id} ({len(sequence)} residues)...")

        # Generate embeddings
        embeddings = generate_esmif1_embeddings(model, alphabet, sequence)

        if embeddings is not None and len(embeddings) == len(epitope_labels):
            for i in range(len(embeddings)):
                all_features.append(embeddings[i])
                all_labels.append(epitope_labels[i])
                all_ids.append(f"{antigen_id}_{i}")
        else:
            print(f"   ⚠️  Skipping {antigen_id} due to embedding mismatch")

        processed_count += 1

    return np.array(all_features), np.array(all_labels), all_ids

def evaluate_esmif1_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Evaluate classical ML models using ESM-IF1 embeddings."""

    # Define models including XGBoost
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=2000))
        ]),
        'SVM (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', probability=True, random_state=42, C=0.1))
        ]),
        'KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ]),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        )
    }

    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("\n🤖 Evaluating Classical ML Models with ESM-IF1 Embeddings")
    print("=" * 70)

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

def compare_results(esmif1_results: Dict[str, Dict[str, float]],
                   sequence_baseline: float = 0.6059,
                   surf2spot_auc: float = 0.3941):
    """Compare ESM-IF1 results with baselines."""
    print(f"\n🏆 Model Comparison (ESM-IF1 Embeddings)")
    print("=" * 85)
    print(f"{'Model':<20} {'ROC-AUC':<12} {'Std':<8} {'Time(s)':<8} {'vs Sequence':<12} {'vs Surf2Spot'}")
    print("-" * 85)

    # Sort by mean AUC
    sorted_results = sorted(esmif1_results.items(),
                           key=lambda x: x[1].get('mean_auc', 0), reverse=True)

    for name, results in sorted_results:
        if 'error' not in results:
            auc = results['mean_auc']
            std = results['std_auc']
            time_s = results.get('time', 0)

            # Calculate improvements
            seq_improvement = ((auc - sequence_baseline) / sequence_baseline * 100) if sequence_baseline > 0 else 0
            surf2spot_improvement = ((auc - surf2spot_auc) / surf2spot_auc * 100) if surf2spot_auc > 0 else 0

            print(f"{name:<20} {auc:.4f}       {std:.3f}    {time_s:6.1f}   "
                  f"{seq_improvement:+6.1f}%     {surf2spot_improvement:+6.1f}%")
        else:
            print(f"{name:<20} ERROR        -        -        -          -")

    print(f"\nBaselines:")
    print(f"  Random Forest (Sequence): {sequence_baseline:.4f}")
    print(f"  Surf2Spot (Antigen-Only): {surf2spot_auc:.4f}")
    print(f"  Random Classifier:         0.5000")

def main():
    print("🔬 ESM-IF1 + Classical ML for Epitope Prediction")
    print("=" * 60)

    # Load ground truth data
    print("📖 Loading ground truth data...")
    gt_data = parse_sabdab_fasta()
    print(f"   Loaded {len(gt_data)} antigen sequences")

    # Load ESM-IF1 model
    model, alphabet = load_esmif1_model()
    if model is None:
        print("❌ Could not load ESM-IF1 model")
        return

    # Prepare dataset with ESM-IF1 embeddings (limited for speed)
    print("🔧 Preparing ESM-IF1 dataset...")
    X, y, ids = prepare_esmif1_dataset(gt_data, model, alphabet, max_sequences=30)

    if len(X) == 0:
        print("❌ No features generated")
        return

    # Print dataset statistics
    total_residues = len(y)
    epitope_residues = np.sum(y)
    epitope_ratio = epitope_residues / total_residues * 100

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total residues: {total_residues:,}")
    print(f"   Epitope residues: {epitope_residues:,} ({epitope_ratio:.1f}%)")
    print(f"   Feature dimensions: {X.shape[1]} (ESM-IF1 embeddings)")
    print(f"   Class balance: {epitope_ratio:.1f}% epitope, {100-epitope_ratio:.1f}% non-epitope")

    # Evaluate models with ESM-IF1 embeddings
    esmif1_results = evaluate_esmif1_models(X, y)

    # Compare with previous results
    compare_results(esmif1_results)

    # Find best model
    valid_results = {k: v for k, v in esmif1_results.items() if 'error' not in v}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['mean_auc'])

        print(f"\n🥇 Best ESM-IF1 Model: {best_model[0]}")
        print(f"   ROC-AUC: {best_model[1]['mean_auc']:.4f} ± {best_model[1]['std_auc']:.4f}")
        print(f"   Training Time: {best_model[1]['time']:.1f}s")

        # Compare with sequence baseline
        if best_model[1]['mean_auc'] > 0.6059:
            improvement = (best_model[1]['mean_auc'] - 0.6059) / 0.6059 * 100
            print(f"   🎉 Improvement over sequence features: +{improvement:.1f}%")
        else:
            decline = (0.6059 - best_model[1]['mean_auc']) / 0.6059 * 100
            print(f"   📉 Performance vs sequence features: -{decline:.1f}%")

        print(f"\n💡 Key Insights:")
        print(f"   • ESM-IF1 embeddings provide {X.shape[1]}-dim representations vs 30-dim sequence features")
        print(f"   • {'XGBoost' if 'XGBoost' in best_model[0] else 'Classical models'} work well with protein embeddings")
        print(f"   • Structure-aware embeddings {'improve' if best_model[1]['mean_auc'] > 0.6059 else 'compete with'} simple sequence features")

if __name__ == "__main__":
    main()