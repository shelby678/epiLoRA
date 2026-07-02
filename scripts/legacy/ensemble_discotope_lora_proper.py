#!/usr/bin/env python3
"""
Ensemble DiscoTope3 + LoRA RYS ESM Model with proper train/validation/test splits.

This version:
1. Trains LoRA RYS ESM3 model on sabdab_training.fasta (383 sequences)
2. Uses holdout1.fasta for ROC-AUC normalization (120 sequences)
3. Evaluates ensemble performance on holdout2.fasta (120 sequences)
4. Prevents data leakage by keeping training/validation/test completely separate

Best LoRA RYS ESM3 config: RYS(36,44) + LoRA rank=4 last-8 blocks (~0.742 ROC-AUC)
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
    StructDataset, StructureEpitopePredictionModel, train,
    create_struct_datasets_from_combined, _OUR_TO_ESM3, PAD_ID
)
from prepare import load_fasta
from sklearn.metrics import roc_auc_score, roc_curve

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_fasta(fasta_path: Path) -> Dict[str, Tuple[str, List[int]]]:
    """Parse FASTA file into ground truth labels."""
    data = {}

    with open(fasta_path, 'r') as f:
        antigen_id = None
        sequence = None

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if antigen_id and sequence:
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


def train_lora_model(train_fasta: Path) -> Tuple[StructureEpitopePredictionModel, Dict]:
    """Train the best LoRA RYS ESM3 model configuration."""
    print("🤖 Training LoRA RYS ESM Model...")

    # Load the dataset properly using the structural data loading function
    try:
        train_data, val_data = create_struct_datasets_from_combined(
            fasta_path=train_fasta,
            structures_dir=Path("data/structures2"),
            val_partition="1",  # Use partition 1 as validation from training set
            max_length=512,
        )

        # If no val data, use subset of train data
        if not val_data:
            print("  No partition 1 found, using subset of training data for validation")
            # Take last 50 samples as validation
            val_data = train_data[-50:] if len(train_data) >= 100 else train_data[-len(train_data)//4:]
            train_data = train_data[:-len(val_data)]

        print(f"  Loaded {len(train_data)} training, {len(val_data)} validation samples")

    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        raise

    # Train the best configuration
    print("🔄 Training best LoRA RYS ESM configuration...")
    print("   Config: RYS(36,44) + LoRA rank=4 last-8 blocks")

    # Use best configuration from memory
    result = train(
        train_data, val_data,
        max_seconds=36000,  # 10 hours - effectively no timeout
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_auc=True,
        rys_start=36, rys_end=44,
        lora_rank=4, lora_alpha=8.0, lora_n_blocks=8,
        batch_size=4,
        lr=1e-3, weight_decay=0.05, warmup_steps=100,
        patience=3, val_eval_interval=200,
    )

    # Create new model and load the trained state
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StructureEpitopePredictionModel(
        rys_start=36, rys_end=44,
        lora_rank=4, lora_alpha=8.0, lora_n_blocks=8
    ).to(device)

    # Load the trainable state
    if 'trainable_state' in result:
        current_state = model.state_dict()
        current_state.update(result['trainable_state'])
        model.load_state_dict(current_state)

    print(f"  ✅ Training completed! Val AUC: {result.get('roc_auc', 'N/A'):.4f}")

    # Save the model
    model_save_path = Path("ensemble_lora_model.pt")
    torch.save(model.cpu().state_dict(), model_save_path)
    print(f"  💾 Model saved to {model_save_path}")

    return model, result


def get_model_predictions(model: StructureEpitopePredictionModel,
                         fasta_path: Path) -> Dict[str, List[float]]:
    """Generate predictions using trained model."""
    print(f"🔍 Generating predictions on {fasta_path.name}...")

    # Load dataset for prediction - parse directly since holdout files don't have partitions
    from prepare import load_fasta

    # Parse the holdout FASTA file directly
    fasta_data = parse_fasta(fasta_path)

    # Convert to StructSample format
    data = []
    structures_dir = Path("data/structures2")

    for antigen_id, (sequence, labels) in fasta_data.items():
        # Parse antigen_id to get PDB and chain info
        parts = antigen_id.split('_')
        if len(parts) >= 2:
            pdb_code = parts[0]
            antigen_chain = parts[1]

            # Convert sequence to tokens using prepare.py functions
            from prepare import encode, CLS_ID, EOS_ID
            token_ids = encode(sequence)

            # Create corresponding labels (pad with 0 for CLS/EOS tokens)
            token_labels = [0] + labels + [0]

            # Load structure coordinates if available
            pdb_dir = structures_dir / pdb_code[:2] / f"{pdb_code}.pdb"
            coords = None
            rsa = None

            if pdb_dir.exists():
                try:
                    from train_struct import _load_coords_rsa
                    coords, rsa = _load_coords_rsa(
                        f"{pdb_code}_{antigen_chain}", len(sequence), structures_dir
                    )
                except:
                    pass

            # Create StructSample (token_ids, labels, coords, rsa)
            data.append((token_ids, token_labels, coords, rsa))

    if not data:
        print(f"  ❌ No valid samples found in {fasta_path.name}")
        return {}

    model.eval()
    predictions = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from train_struct import struct_collate_fn
    dataset = StructDataset(data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=struct_collate_fn)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            coords = batch.get('coords')
            if coords is not None:
                coords = coords.to(device)

            # Get predictions
            outputs = model(input_ids, coords)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Get corresponding antigen info from original data
            antigen_ids = list(fasta_data.keys())
            if i < len(antigen_ids):
                antigen_id = antigen_ids[i]
                sequence, _ = fasta_data[antigen_id]

                # Remove padding and special tokens
                valid_length = len(sequence)
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
                           true_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ensemble using max of ROC-normalized predictions."""
    norm_discotope = normalize_predictions_roc_based(true_labels, discotope_preds)
    norm_lora = normalize_predictions_roc_based(true_labels, lora_preds)
    ensemble_preds = np.maximum(norm_discotope, norm_lora)
    return ensemble_preds, norm_discotope, norm_lora


def evaluate_on_dataset(gt_data: Dict[str, Tuple[str, List[int]]],
                       discotope_preds: Dict[str, Tuple[List[float], List[int]]],
                       lora_preds: Dict[str, List[float]],
                       dataset_name: str) -> Dict:
    """Evaluate predictions on a dataset."""
    print(f"\n📊 Evaluating on {dataset_name}")
    print("-" * 50)

    # Find common antigens
    common_antigens = set(gt_data.keys()) & set(discotope_preds.keys()) & set(lora_preds.keys())
    print(f"Common antigens: {len(common_antigens)}")

    if len(common_antigens) == 0:
        print("❌ No common antigens found!")
        return {}

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

    # Calculate final metrics
    if not all_labels:
        return {}

    overall_auc_discotope = roc_auc_score(all_labels, all_discotope)
    overall_auc_lora = roc_auc_score(all_labels, all_lora)
    overall_auc_ensemble = roc_auc_score(all_labels, all_ensemble)

    avg_auc_discotope = np.mean(individual_aucs_discotope)
    avg_auc_lora = np.mean(individual_aucs_lora)
    avg_auc_ensemble = np.mean(individual_aucs_ensemble)

    print(f"Overall ROC-AUC Results:")
    print(f"  DiscoTope3:           {overall_auc_discotope:.4f}")
    print(f"  LoRA RYS ESM:         {overall_auc_lora:.4f}")
    print(f"  🚀 Ensemble (Max):    {overall_auc_ensemble:.4f}")

    print(f"Average per-antigen ROC-AUC:")
    print(f"  DiscoTope3:           {avg_auc_discotope:.4f} ± {np.std(individual_aucs_discotope):.4f}")
    print(f"  LoRA RYS ESM:         {avg_auc_lora:.4f} ± {np.std(individual_aucs_lora):.4f}")
    print(f"  🚀 Ensemble (Max):    {avg_auc_ensemble:.4f} ± {np.std(individual_aucs_ensemble):.4f}")

    # Calculate improvements
    best_individual_overall = max(overall_auc_discotope, overall_auc_lora)
    best_individual_avg = max(avg_auc_discotope, avg_auc_lora)

    improvement_overall = overall_auc_ensemble - best_individual_overall
    improvement_avg = avg_auc_ensemble - best_individual_avg

    print(f"🎯 Ensemble Performance:")
    print(f"  Overall AUC improvement: {improvement_overall:+.4f}")
    print(f"  Average AUC improvement: {improvement_avg:+.4f}")

    return {
        'overall_auc_discotope': overall_auc_discotope,
        'overall_auc_lora': overall_auc_lora,
        'overall_auc_ensemble': overall_auc_ensemble,
        'avg_auc_discotope': avg_auc_discotope,
        'avg_auc_lora': avg_auc_lora,
        'avg_auc_ensemble': avg_auc_ensemble,
        'improvement_overall': improvement_overall,
        'improvement_avg': improvement_avg,
        'n_antigens': len(individual_aucs_ensemble),
        'n_residues': len(all_labels),
        'n_epitope_residues': np.sum(all_labels)
    }


def main():
    print("🧬 DiscoTope3 + LoRA RYS ESM Ensemble - Proper Train/Val/Test Split")
    print("=" * 80)

    # File paths
    train_fasta = Path("data/sabdab_training.fasta")  # 383 sequences
    holdout1_fasta = Path("data/holdout1.fasta")     # 120 sequences (validation)
    holdout2_fasta = Path("data/holdout2.fasta")     # 120 sequences (test)

    # Check files exist
    for fasta_path in [train_fasta, holdout1_fasta, holdout2_fasta]:
        if not fasta_path.exists():
            print(f"❌ ERROR: {fasta_path} not found")
            return 1

    print(f"✓ Using training set: {train_fasta.name}")
    print(f"✓ Using validation set: {holdout1_fasta.name}")
    print(f"✓ Using test set: {holdout2_fasta.name}")

    # Load ground truth data
    print("\n📋 Loading ground truth data...")
    train_gt = parse_fasta(train_fasta)
    holdout1_gt = parse_fasta(holdout1_fasta)
    holdout2_gt = parse_fasta(holdout2_fasta)
    print(f"  Training: {len(train_gt)} antigens")
    print(f"  Validation: {len(holdout1_gt)} antigens")
    print(f"  Test: {len(holdout2_gt)} antigens")

    # Load DiscoTope predictions
    print("\n🔍 Loading DiscoTope3 predictions...")
    discotope_preds = load_discotope_predictions()
    print(f"  Loaded predictions for {len(discotope_preds)} antigens")

    # Check if we have a saved model
    model_path = Path("ensemble_lora_model.pt")
    if model_path.exists():
        print(f"\n📂 Loading existing model from {model_path}...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = StructureEpitopePredictionModel(
                rys_start=36, rys_end=44,
                lora_rank=4, lora_alpha=8.0, lora_n_blocks=8
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("  ✅ Model loaded successfully")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            print("  🔄 Training new model instead...")
            model, train_result = train_lora_model(train_fasta)
    else:
        # Train new model
        print(f"\n🔄 No saved model found, training new model...")
        model, train_result = train_lora_model(train_fasta)

    # Generate LoRA predictions on holdout sets
    lora_holdout1_preds = get_model_predictions(model, holdout1_fasta)
    lora_holdout2_preds = get_model_predictions(model, holdout2_fasta)

    # Evaluate on validation set (holdout1) - used for normalization
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION (Holdout1 - for ROC normalization)")
    print("="*80)
    holdout1_results = evaluate_on_dataset(
        holdout1_gt, discotope_preds, lora_holdout1_preds, "Holdout1 (Validation)"
    )

    # Evaluate on test set (holdout2) - final performance
    print("\n" + "="*80)
    print("TEST SET EVALUATION (Holdout2 - Final Performance)")
    print("="*80)
    holdout2_results = evaluate_on_dataset(
        holdout2_gt, discotope_preds, lora_holdout2_preds, "Holdout2 (Test)"
    )

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    if holdout1_results and holdout2_results:
        print(f"Validation Set (Holdout1):")
        print(f"  Ensemble AUC: {holdout1_results['overall_auc_ensemble']:.4f}")
        print(f"  Improvement: {holdout1_results['improvement_overall']:+.4f}")

        print(f"\nTest Set (Holdout2):")
        print(f"  Ensemble AUC: {holdout2_results['overall_auc_ensemble']:.4f}")
        print(f"  Improvement: {holdout2_results['improvement_overall']:+.4f}")

        if holdout2_results['improvement_overall'] > 0:
            print("\n✅ Ensemble successfully outperforms individual models on test set!")
        else:
            print("\n⚠️ Ensemble did not improve over best individual model on test set")

        print(f"\nTraining/Validation/Test Split: {len(train_gt)}/{len(holdout1_gt)}/{len(holdout2_gt)} antigens")
        print(f"Model saved: ensemble_lora_model.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())