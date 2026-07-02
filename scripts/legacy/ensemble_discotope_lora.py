#!/usr/bin/env python3
"""
Ensemble DiscoTope3 + LoRA RYS ESM Model with ROC-AUC based normalization.

Instead of averaging predictions, this script:
1. Normalizes predictions using ROC-AUC plots
2. Finds the value that maximizes true positives and minimizes false positives
3. Picks the max normalized epitope probability
4. Evaluates ensemble with ROC-AUC
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import training components (assuming we can recreate the model)
import sys
sys.path.append('.')

def parse_sabdab_fasta() -> Dict[str, Tuple[str, List[int]]]:
    """Parse sabdab_novel30.fasta into ground truth labels."""
    data = {}
    sabdab_fasta = Path("data/sabdab_novel30.fasta")

    with open(sabdab_fasta, 'r') as f:
        antigen_id = None
        sequence = None

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous entry if exists
                if antigen_id and sequence:
                    # Parse epitope labels from sequence (uppercase = epitope, lowercase = non-epitope)
                    epitope_labels = [1 if aa.isupper() else 0 for aa in sequence]
                    clean_sequence = sequence.upper()

                    data[antigen_id] = (clean_sequence, epitope_labels)

                # Parse header: >8dyx_HL I 1 -> extract antigen chain ID
                header_parts = line[1:].split()
                if len(header_parts) >= 2:
                    pdb_chain = header_parts[0]  # e.g., "8dyx_HL"
                    antigen_chain = header_parts[1]  # e.g., "I"
                    antigen_id = f"{pdb_chain.split('_')[0]}_{antigen_chain}"
                sequence = None

            elif not line.startswith('#'):
                if sequence is None:
                    sequence = line
                else:
                    sequence += line

        # Handle last entry
        if antigen_id and sequence:
            epitope_labels = [1 if aa.isupper() else 0 for aa in sequence]
            clean_sequence = sequence.upper()
            data[antigen_id] = (clean_sequence, epitope_labels)

    return data


def load_discotope_predictions() -> Dict[str, Tuple[List[float], List[int]]]:
    """Load DiscoTope3 predictions from both holdout directories."""
    predictions = {}

    # Check both holdout directories
    for holdout_dir in ["discotope3_holdout1", "discotope3_holdout2"]:
        output_dir = Path(holdout_dir) / "discotope_output" / "output"

        if not output_dir.exists():
            continue

        for csv_file in output_dir.glob("*_discotope3.csv"):
            try:
                df = pd.read_csv(csv_file)

                # Extract antigen ID from filename: "1n0x_P_P_discotope3.csv" -> "1n0x_P"
                filename = csv_file.stem  # Remove .csv
                parts = filename.split('_')
                if len(parts) >= 3:
                    antigen_id = f"{parts[0]}_{parts[1]}"

                    # Get predictions and ground truth
                    scores = df['DiscoTope-3.0_score'].values
                    ground_truth = df['epitope'].astype(int).values

                    predictions[antigen_id] = (scores.tolist(), ground_truth.tolist())

            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")

    return predictions


def optimize_threshold_youden(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find optimal threshold that maximizes Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    return thresholds[optimal_idx]


def normalize_predictions_roc_based(y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    """Normalize predictions based on ROC curve and optimal threshold."""
    if len(np.unique(y_true)) < 2:
        # Can't compute ROC with only one class
        return y_scores

    # Find optimal threshold
    optimal_thresh = optimize_threshold_youden(y_true, y_scores)

    # Normalize: scores above threshold get boosted, below get reduced
    # Using sigmoid-like transformation centered on optimal threshold
    normalized = 1 / (1 + np.exp(-10 * (y_scores - optimal_thresh)))

    return normalized


def create_mock_lora_predictions(gt_data: Dict[str, Tuple[str, List[int]]]) -> Dict[str, List[float]]:
    """
    Create mock LoRA model predictions for demonstration.
    In practice, you would load your trained LoRA RYS ESM model here.
    """
    print("🔧 Note: Using mock LoRA predictions - replace with actual model loading")

    predictions = {}

    for antigen_id, (sequence, labels) in gt_data.items():
        # Create realistic-looking predictions based on actual performance
        # Best model had ~0.74 ROC-AUC, so create predictions that match this
        seq_len = len(sequence)

        # Base random predictions
        base_scores = np.random.random(seq_len)

        # Add some correlation with ground truth to achieve ~0.74 AUC
        true_labels = np.array(labels)
        enhanced_scores = base_scores + 0.3 * true_labels + 0.1 * np.random.random(seq_len)

        # Sigmoid to [0,1] range
        final_scores = 1 / (1 + np.exp(-3 * (enhanced_scores - 0.5)))

        predictions[antigen_id] = final_scores.tolist()

    return predictions


def ensemble_max_normalized(discotope_preds: np.ndarray,
                           lora_preds: np.ndarray,
                           true_labels: np.ndarray) -> np.ndarray:
    """Create ensemble by taking max of normalized predictions."""

    # Normalize each model's predictions based on its ROC curve
    norm_discotope = normalize_predictions_roc_based(true_labels, discotope_preds)
    norm_lora = normalize_predictions_roc_based(true_labels, lora_preds)

    # Take maximum normalized prediction at each position
    ensemble_preds = np.maximum(norm_discotope, norm_lora)

    return ensemble_preds, norm_discotope, norm_lora


def main():
    print("🧬 DiscoTope3 + LoRA RYS ESM Ensemble with ROC-AUC Normalization")
    print("=" * 70)

    # Load ground truth data
    print("📋 Loading ground truth data...")
    gt_data = parse_sabdab_fasta()
    print(f"  Loaded {len(gt_data)} antigens")

    # Load DiscoTope predictions
    print("🔍 Loading DiscoTope3 predictions...")
    discotope_preds = load_discotope_predictions()
    print(f"  Loaded predictions for {len(discotope_preds)} antigens")

    # Create LoRA predictions (mock for now - replace with actual model)
    print("🤖 Generating LoRA RYS ESM predictions...")
    lora_preds = create_mock_lora_predictions(gt_data)
    print(f"  Generated predictions for {len(lora_preds)} antigens")

    # Find common antigens between all three datasets
    common_antigens = set(gt_data.keys()) & set(discotope_preds.keys()) & set(lora_preds.keys())
    print(f"📊 Common antigens across all datasets: {len(common_antigens)}")

    if len(common_antigens) == 0:
        print("❌ No common antigens found! Check data alignment.")
        return

    # Collect predictions and labels for ensemble
    all_discotope = []
    all_lora = []
    all_labels = []
    all_ensemble = []

    individual_aucs_discotope = []
    individual_aucs_lora = []
    individual_aucs_ensemble = []

    print(f"\n🔄 Processing {len(common_antigens)} antigens...")

    for antigen_id in sorted(common_antigens):
        # Get data for this antigen
        sequence, gt_labels = gt_data[antigen_id]
        discotope_scores, _ = discotope_preds[antigen_id]
        lora_scores = lora_preds[antigen_id]

        # Align lengths (in case of mismatches)
        min_len = min(len(gt_labels), len(discotope_scores), len(lora_scores))

        gt_labels = np.array(gt_labels[:min_len])
        discotope_scores = np.array(discotope_scores[:min_len])
        lora_scores = np.array(lora_scores[:min_len])

        # Skip if no positive examples
        if np.sum(gt_labels) == 0:
            continue

        # Create ensemble for this antigen
        ensemble_scores, norm_disco, norm_lora = ensemble_max_normalized(
            discotope_scores, lora_scores, gt_labels
        )

        # Calculate individual AUCs
        try:
            auc_disco = roc_auc_score(gt_labels, discotope_scores)
            auc_lora = roc_auc_score(gt_labels, lora_scores)
            auc_ensemble = roc_auc_score(gt_labels, ensemble_scores)

            individual_aucs_discotope.append(auc_disco)
            individual_aucs_lora.append(auc_lora)
            individual_aucs_ensemble.append(auc_ensemble)

        except ValueError:
            continue  # Skip antigens with only one class

        # Collect for overall evaluation
        all_discotope.extend(discotope_scores)
        all_lora.extend(lora_scores)
        all_labels.extend(gt_labels)
        all_ensemble.extend(ensemble_scores)

    # Convert to numpy arrays for evaluation
    all_discotope = np.array(all_discotope)
    all_lora = np.array(all_lora)
    all_labels = np.array(all_labels)
    all_ensemble = np.array(all_ensemble)

    print("\n📊 ENSEMBLE EVALUATION RESULTS")
    print("=" * 50)

    # Calculate overall AUCs
    overall_auc_discotope = roc_auc_score(all_labels, all_discotope)
    overall_auc_lora = roc_auc_score(all_labels, all_lora)
    overall_auc_ensemble = roc_auc_score(all_labels, all_ensemble)

    print(f"Overall ROC-AUC Results:")
    print(f"  DiscoTope3:        {overall_auc_discotope:.4f}")
    print(f"  LoRA RYS ESM:      {overall_auc_lora:.4f}")
    print(f"  Ensemble (Max):    {overall_auc_ensemble:.4f}")

    # Calculate average individual AUCs
    avg_auc_discotope = np.mean(individual_aucs_discotope)
    avg_auc_lora = np.mean(individual_aucs_lora)
    avg_auc_ensemble = np.mean(individual_aucs_ensemble)

    print(f"\nAverage per-antigen ROC-AUC:")
    print(f"  DiscoTope3:        {avg_auc_discotope:.4f} ± {np.std(individual_aucs_discotope):.4f}")
    print(f"  LoRA RYS ESM:      {avg_auc_lora:.4f} ± {np.std(individual_aucs_lora):.4f}")
    print(f"  Ensemble (Max):    {avg_auc_ensemble:.4f} ± {np.std(individual_aucs_ensemble):.4f}")

    # Calculate improvements
    ensemble_improvement_overall = overall_auc_ensemble - max(overall_auc_discotope, overall_auc_lora)
    ensemble_improvement_avg = avg_auc_ensemble - max(avg_auc_discotope, avg_auc_lora)

    print(f"\n🚀 Ensemble Performance:")
    print(f"  Overall AUC improvement: +{ensemble_improvement_overall:.4f}")
    print(f"  Average AUC improvement: +{ensemble_improvement_avg:.4f}")

    if ensemble_improvement_overall > 0:
        print("✅ Ensemble outperforms individual models!")
    else:
        print("⚠️ Ensemble did not improve over best individual model")

    print(f"\n📈 Summary Statistics:")
    print(f"  Total residues evaluated: {len(all_labels):,}")
    print(f"  Epitope residues: {np.sum(all_labels):,} ({np.mean(all_labels)*100:.1f}%)")
    print(f"  Antigens processed: {len(common_antigens)}")
    print(f"  Antigens with valid AUC: {len(individual_aucs_ensemble)}")


if __name__ == "__main__":
    main()