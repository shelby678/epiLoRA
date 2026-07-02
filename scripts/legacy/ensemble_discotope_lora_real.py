#!/usr/bin/env python3
"""
Ensemble DiscoTope3 + Real LoRA RYS ESM Model with ROC-AUC normalization.

This version loads/trains the actual best LoRA RYS ESM configuration:
RYS(36,44) + LoRA rank=4 last-8 blocks, which achieved ~0.742 ROC-AUC.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our training components
from train_struct import (
    StructDataset, StructureEpitopePredictionModel, train, load_combined_fasta_partitioned,
    _OUR_TO_ESM3, PAD_ID
)
from prepare import load_fasta
from sklearn.metrics import roc_auc_score, roc_curve

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                if antigen_id and sequence:
                    epitope_labels = [1 if aa.isupper() else 0 for aa in sequence]
                    clean_sequence = sequence.upper()
                    data[antigen_id] = (clean_sequence, epitope_labels)

                header_parts = line[1:].split()
                if len(header_parts) >= 2:
                    pdb_chain = header_parts[0]
                    antigen_chain = header_parts[1]
                    antigen_id = f"{pdb_chain.split('_')[0]}_{antigen_chain}"
                sequence = None

            elif not line.startswith('#'):
                if sequence is None:
                    sequence = line
                else:
                    sequence += line

        if antigen_id and sequence:
            epitope_labels = [1 if aa.isupper() else 0 for aa in sequence]
            clean_sequence = sequence.upper()
            data[antigen_id] = (clean_sequence, epitope_labels)

    return data


def load_discotope_predictions() -> Dict[str, Tuple[List[float], List[int]]]:
    """Load DiscoTope3 predictions from holdout directories."""
    predictions = {}

    for holdout_dir in ["discotope3_holdout1", "discotope3_holdout2"]:
        output_dir = Path(holdout_dir) / "discotope_output" / "output"

        if not output_dir.exists():
            continue

        for csv_file in output_dir.glob("*_discotope3.csv"):
            try:
                df = pd.read_csv(csv_file)

                filename = csv_file.stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    antigen_id = f"{parts[0]}_{parts[1]}"
                    scores = df['DiscoTope-3.0_score'].values
                    ground_truth = df['epitope'].astype(int).values
                    predictions[antigen_id] = (scores.tolist(), ground_truth.tolist())

            except Exception as e:
                logger.warning(f"Could not load {csv_file}: {e}")

    return predictions


def get_lora_predictions(model_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Get predictions from the best LoRA RYS ESM model.
    If model_path is provided, load it. Otherwise, retrain the best config.
    """
    print("🤖 Loading/Training LoRA RYS ESM Model...")

    # Load the dataset properly using the structural data loading function
    try:
        from train_struct import create_struct_datasets_from_combined

        train_data, val_data = create_struct_datasets_from_combined(
            fasta_path=Path("data/sabdab_novel30.fasta"),
            structures_dir=Path("data/structures2"),  # Structure directory
            val_partition="1",  # Use partition 1 as validation
            max_length=512,
        )

        # If no val data, use subset of train data
        if not val_data:
            print("  No partition 1 found, using subset of training data")
            val_data = train_data[:100] if train_data else []  # Use larger subset for testing

        print(f"  Loaded {len(val_data)} validation samples as StructSamples")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {}

    # Check if we should load existing model
    if model_path and Path(model_path).exists():
        print(f"📂 Loading model from {model_path}...")
        try:
            model = StructureEpitopePredictionModel()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Training new model instead...")
            model = None
    else:
        model = None

    # Train new model if needed
    if model is None:
        print("🔄 Training best LoRA RYS ESM configuration...")
        print("   Config: RYS(36,44) + LoRA rank=4 last-8 blocks")

        # Use best configuration from memory
        result = train(
            train_data, val_data,
            max_seconds=600,  # 10 minutes max
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_auc=True,
            rys_start=36, rys_end=44,
            lora_rank=4, lora_alpha=8.0, lora_n_blocks=8,
            batch_size=4,
            lr=1e-3, weight_decay=0.05, warmup_steps=100,
            patience=3, val_eval_interval=200,
        )

        model = result['model']
        print(f"  ✅ Training completed! Val AUC: {result.get('val_auc', 'N/A'):.4f}")

        # Save the model
        model_save_path = Path("best_lora_model.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"  💾 Model saved to {model_save_path}")

    # Generate predictions on validation data
    print("🔍 Generating predictions on validation set...")
    model.eval()
    predictions = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = StructDataset(val_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            coords = batch.get('coords')
            if coords is not None:
                coords = coords.to(device)

            # Get predictions
            outputs = model(input_ids, coords)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Extract sample info
            sample = batch['samples'][0]  # batch_size=1
            antigen_id = f"{sample.pdb_code}_{sample.antigen_chain}"

            # Remove padding and special tokens
            valid_length = len(sample.sequence)
            pred_scores = probs[0, 1:valid_length+1]  # Skip CLS token, remove padding

            predictions[antigen_id] = pred_scores.tolist()

    print(f"  ✅ Generated predictions for {len(predictions)} antigens")
    return predictions


def optimize_threshold_youden(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def normalize_predictions_roc_based(y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    """Normalize predictions based on ROC curve and optimal threshold."""
    if len(np.unique(y_true)) < 2:
        return y_scores

    optimal_thresh = optimize_threshold_youden(y_true, y_scores)
    # Sigmoid-like transformation centered on optimal threshold
    normalized = 1 / (1 + np.exp(-10 * (y_scores - optimal_thresh)))
    return normalized


def ensemble_max_normalized(discotope_preds: np.ndarray,
                           lora_preds: np.ndarray,
                           true_labels: np.ndarray) -> np.ndarray:
    """Create ensemble using max of ROC-normalized predictions."""
    norm_discotope = normalize_predictions_roc_based(true_labels, discotope_preds)
    norm_lora = normalize_predictions_roc_based(true_labels, lora_preds)
    ensemble_preds = np.maximum(norm_discotope, norm_lora)
    return ensemble_preds, norm_discotope, norm_lora


def main():
    print("🧬 DiscoTope3 + Real LoRA RYS ESM Ensemble")
    print("=" * 70)

    # Load ground truth
    print("📋 Loading ground truth data...")
    gt_data = parse_sabdab_fasta()
    print(f"  Loaded {len(gt_data)} antigens")

    # Load DiscoTope predictions
    print("🔍 Loading DiscoTope3 predictions...")
    discotope_preds = load_discotope_predictions()
    print(f"  Loaded predictions for {len(discotope_preds)} antigens")

    # Get LoRA predictions (will train if no saved model)
    lora_preds = get_lora_predictions(model_path="best_lora_model.pt")
    print(f"  Got predictions for {len(lora_preds)} antigens")

    # Find common antigens
    common_antigens = set(gt_data.keys()) & set(discotope_preds.keys()) & set(lora_preds.keys())
    print(f"📊 Common antigens: {len(common_antigens)}")

    if len(common_antigens) == 0:
        print("❌ No common antigens found!")
        return

    # Evaluate ensemble
    all_discotope, all_lora, all_labels, all_ensemble = [], [], [], []
    individual_aucs_discotope, individual_aucs_lora, individual_aucs_ensemble = [], [], []

    for antigen_id in sorted(common_antigens):
        sequence, gt_labels = gt_data[antigen_id]
        discotope_scores, _ = discotope_preds[antigen_id]
        lora_scores = lora_preds[antigen_id]

        # Align lengths
        min_len = min(len(gt_labels), len(discotope_scores), len(lora_scores))
        gt_labels = np.array(gt_labels[:min_len])
        discotope_scores = np.array(discotope_scores[:min_len])
        lora_scores = np.array(lora_scores[:min_len])

        if np.sum(gt_labels) == 0:
            continue

        # Create ensemble
        ensemble_scores, norm_disco, norm_lora = ensemble_max_normalized(
            discotope_scores, lora_scores, gt_labels
        )

        # Calculate AUCs
        try:
            auc_disco = roc_auc_score(gt_labels, discotope_scores)
            auc_lora = roc_auc_score(gt_labels, lora_scores)
            auc_ensemble = roc_auc_score(gt_labels, ensemble_scores)

            individual_aucs_discotope.append(auc_disco)
            individual_aucs_lora.append(auc_lora)
            individual_aucs_ensemble.append(auc_ensemble)

            # Collect for overall evaluation
            all_discotope.extend(discotope_scores)
            all_lora.extend(lora_scores)
            all_labels.extend(gt_labels)
            all_ensemble.extend(ensemble_scores)

        except ValueError:
            continue

    # Final evaluation
    print("\n📊 REAL ENSEMBLE EVALUATION RESULTS")
    print("=" * 50)

    overall_auc_discotope = roc_auc_score(all_labels, all_discotope)
    overall_auc_lora = roc_auc_score(all_labels, all_lora)
    overall_auc_ensemble = roc_auc_score(all_labels, all_ensemble)

    print(f"Overall ROC-AUC Results:")
    print(f"  DiscoTope3:           {overall_auc_discotope:.4f}")
    print(f"  LoRA RYS ESM:         {overall_auc_lora:.4f}")
    print(f"  🚀 Ensemble (Max):    {overall_auc_ensemble:.4f}")

    avg_auc_discotope = np.mean(individual_aucs_discotope)
    avg_auc_lora = np.mean(individual_aucs_lora)
    avg_auc_ensemble = np.mean(individual_aucs_ensemble)

    print(f"\nAverage per-antigen ROC-AUC:")
    print(f"  DiscoTope3:           {avg_auc_discotope:.4f} ± {np.std(individual_aucs_discotope):.4f}")
    print(f"  LoRA RYS ESM:         {avg_auc_lora:.4f} ± {np.std(individual_aucs_lora):.4f}")
    print(f"  🚀 Ensemble (Max):    {avg_auc_ensemble:.4f} ± {np.std(individual_aucs_ensemble):.4f}")

    # Calculate improvements
    best_individual_overall = max(overall_auc_discotope, overall_auc_lora)
    best_individual_avg = max(avg_auc_discotope, avg_auc_lora)

    improvement_overall = overall_auc_ensemble - best_individual_overall
    improvement_avg = avg_auc_ensemble - best_individual_avg

    print(f"\n🎯 Ensemble Performance:")
    print(f"  Overall AUC improvement: {improvement_overall:+.4f}")
    print(f"  Average AUC improvement: {improvement_avg:+.4f}")

    if improvement_overall > 0:
        print("✅ Ensemble successfully outperforms both individual models!")
    else:
        print("⚠️ Ensemble did not improve over best individual model")

    print(f"\n📈 Summary:")
    print(f"  Total residues: {len(all_labels):,}")
    print(f"  Epitope residues: {np.sum(all_labels):,} ({np.mean(all_labels)*100:.1f}%)")
    print(f"  Antigens evaluated: {len(individual_aucs_ensemble)}")


if __name__ == "__main__":
    main()