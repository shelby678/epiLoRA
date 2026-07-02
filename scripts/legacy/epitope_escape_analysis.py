#!/usr/bin/env python3
"""
Epitope Escape Mutation Analysis

This script finds the minimum number of mutations needed to make an antigen non-immunogenic
by eliminating epitope predictions. Uses our trained DiscoTope3 + LoRA RYS ESM ensemble.

Two approaches:
1. Systematic: Mutate highest-confidence epitope residues to alanine
2. Monte Carlo: Random sampling of mutation combinations

Goal: Find minimum mutations to reduce epitope score below threshold.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import random
import time
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our training components
from train_struct import (
    StructDataset, StructureEpitopePredictionModel,
    create_struct_datasets_from_combined, _OUR_TO_ESM3, PAD_ID
)
from prepare import load_fasta
from sklearn.metrics import roc_auc_score, roc_curve

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard amino acids for mutation
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def load_ensemble_model(model_path: Path) -> StructureEpitopePredictionModel:
    """Load our trained ensemble model."""
    print(f"📂 Loading ensemble model from {model_path}...")
    model = StructureEpitopePredictionModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully")
    return model


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


def normalize_predictions_roc_based(y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    """Normalize predictions based on ROC curve and optimal threshold."""
    if len(np.unique(y_true)) < 2:
        return y_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_thresh = thresholds[optimal_idx]

    # Sigmoid-like transformation centered on optimal threshold
    normalized = 1 / (1 + np.exp(-10 * (y_scores - optimal_thresh)))
    return normalized


def create_mutated_sample(original_sample, mutations: Dict[int, str]):
    """Create a mutated version of a StructSample."""
    # Create a copy of the sample with mutations applied
    mutated_sequence = list(original_sample.sequence)
    for pos, new_aa in mutations.items():
        if 0 <= pos < len(mutated_sequence):
            mutated_sequence[pos] = new_aa

    # Create new sample with mutated sequence
    from train_struct import StructSample
    mutated_sample = StructSample(
        pdb_code=original_sample.pdb_code,
        antigen_chain=original_sample.antigen_chain,
        sequence=''.join(mutated_sequence),
        labels=original_sample.labels,  # Keep original labels for evaluation
        coords=original_sample.coords,
        rsa=original_sample.rsa
    )

    return mutated_sample


def predict_ensemble(model: StructureEpitopePredictionModel,
                    sample,
                    discotope_scores: np.ndarray,
                    ground_truth: np.ndarray) -> Tuple[float, np.ndarray]:
    """Get ensemble prediction for a single sample."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dataset and dataloader for single sample
    dataset = StructDataset([sample])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    with torch.no_grad():
        batch = next(iter(dataloader))

        # Move to device
        input_ids = batch['input_ids'].to(device)
        coords = batch.get('coords')
        if coords is not None:
            coords = coords.to(device)

        # Get LoRA predictions
        outputs = model(input_ids, coords)
        probs = torch.sigmoid(outputs).cpu().numpy()

        # Remove padding and special tokens
        valid_length = len(sample.sequence)
        lora_scores = probs[0, 1:valid_length+1]  # Skip CLS token

    # Align lengths
    min_len = min(len(ground_truth), len(discotope_scores), len(lora_scores))
    gt_labels = ground_truth[:min_len]
    disco_scores = discotope_scores[:min_len]
    lora_scores = lora_scores[:min_len]

    # Create ensemble using normalized max
    norm_discotope = normalize_predictions_roc_based(gt_labels, disco_scores)
    norm_lora = normalize_predictions_roc_based(gt_labels, lora_scores)
    ensemble_scores = np.maximum(norm_discotope, norm_lora)

    # Calculate overall immunogenicity score (mean of epitope predictions)
    immunogenicity_score = np.mean(ensemble_scores)

    return immunogenicity_score, ensemble_scores


def systematic_mutation_analysis(model, original_sample, discotope_scores, ground_truth,
                                target_reduction=0.5, max_mutations=10):
    """Systematically mutate highest-scoring epitope residues to alanine."""
    print("🔬 Systematic mutation analysis...")

    # Get baseline prediction
    baseline_score, baseline_ensemble = predict_ensemble(
        model, original_sample, discotope_scores, ground_truth
    )
    target_score = baseline_score * (1 - target_reduction)

    print(f"  Baseline immunogenicity: {baseline_score:.4f}")
    print(f"  Target score: {target_score:.4f}")

    # Find epitope residues sorted by confidence
    epitope_positions = []
    for i, (ensemble_score, gt_label) in enumerate(zip(baseline_ensemble, ground_truth)):
        if gt_label == 1:  # True epitope residue
            epitope_positions.append((i, ensemble_score))

    # Sort by ensemble score (highest confidence first)
    epitope_positions.sort(key=lambda x: x[1], reverse=True)

    print(f"  Found {len(epitope_positions)} epitope residues")

    # Try mutations incrementally
    mutations = {}
    results = []

    for mutation_count in range(1, min(max_mutations + 1, len(epitope_positions) + 1)):
        # Add next highest-scoring residue mutation
        pos, score = epitope_positions[mutation_count - 1]
        original_aa = original_sample.sequence[pos]

        # Skip if already alanine
        if original_aa == 'A':
            continue

        mutations[pos] = 'A'

        # Create mutated sample and predict
        mutated_sample = create_mutated_sample(original_sample, mutations)
        new_score, new_ensemble = predict_ensemble(
            model, mutated_sample, discotope_scores, ground_truth
        )

        reduction = (baseline_score - new_score) / baseline_score

        result = {
            'mutations': len(mutations),
            'positions': list(mutations.keys()),
            'score': new_score,
            'reduction': reduction,
            'target_reached': new_score <= target_score
        }
        results.append(result)

        print(f"  {len(mutations)} mutations: {original_aa}{pos+1}A -> score={new_score:.4f} ({reduction:.1%} reduction)")

        if new_score <= target_score:
            print(f"  ✅ Target reached with {len(mutations)} mutations!")
            break

    return results, mutations


def monte_carlo_mutation_analysis(model, original_sample, discotope_scores, ground_truth,
                                 target_reduction=0.5, max_mutations=10, n_iterations=1000):
    """Monte Carlo search for optimal mutation combinations."""
    print("🎲 Monte Carlo mutation analysis...")

    # Get baseline prediction
    baseline_score, baseline_ensemble = predict_ensemble(
        model, original_sample, discotope_scores, ground_truth
    )
    target_score = baseline_score * (1 - target_reduction)

    print(f"  Baseline immunogenicity: {baseline_score:.4f}")
    print(f"  Target score: {target_score:.4f}")
    print(f"  Running {n_iterations} iterations...")

    sequence_length = len(original_sample.sequence)
    best_results = {}  # mutation_count -> best result

    for iteration in range(n_iterations):
        if iteration % 200 == 0:
            print(f"  Iteration {iteration}/{n_iterations}")

        # Try different numbers of mutations
        for num_mutations in range(1, max_mutations + 1):
            # Random mutation positions
            mutation_positions = random.sample(range(sequence_length), num_mutations)

            # Create mutations (to alanine for now, could expand to other AAs)
            mutations = {}
            for pos in mutation_positions:
                original_aa = original_sample.sequence[pos]
                if original_aa != 'A':  # Don't mutate if already alanine
                    mutations[pos] = 'A'

            if len(mutations) == 0:
                continue

            # Predict with mutations
            mutated_sample = create_mutated_sample(original_sample, mutations)
            new_score, _ = predict_ensemble(
                model, mutated_sample, discotope_scores, ground_truth
            )

            reduction = (baseline_score - new_score) / baseline_score

            # Track best result for this number of mutations
            if (num_mutations not in best_results or
                new_score < best_results[num_mutations]['score']):

                best_results[num_mutations] = {
                    'mutations': len(mutations),
                    'positions': list(mutations.keys()),
                    'score': new_score,
                    'reduction': reduction,
                    'target_reached': new_score <= target_score
                }

    # Print results
    print("\n  Monte Carlo best results:")
    for num_mut in sorted(best_results.keys()):
        result = best_results[num_mut]
        print(f"  {num_mut} mutations: score={result['score']:.4f} ({result['reduction']:.1%} reduction)")
        if result['target_reached']:
            print(f"    ✅ Target reached!")

    return best_results


def analyze_antigen(antigen_id: str, model, gt_data, discotope_preds):
    """Complete mutation analysis for a single antigen."""
    print(f"\n{'='*80}")
    print(f"🧬 EPITOPE ESCAPE ANALYSIS: {antigen_id}")
    print(f"{'='*80}")

    # Get data for this antigen
    sequence, gt_labels = gt_data[antigen_id]
    discotope_scores, _ = discotope_preds[antigen_id]

    print(f"Sequence length: {len(sequence)}")
    print(f"Epitope residues: {sum(gt_labels)} ({sum(gt_labels)/len(gt_labels)*100:.1f}%)")
    print(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")

    # Load structural data for this antigen
    try:
        # Find the sample in our holdout data
        holdout2_data = parse_fasta(Path("data/holdout2.fasta"))

        # Create struct sample
        from train_struct import create_struct_datasets_from_combined

        # Create a temporary FASTA with just this antigen
        temp_fasta = Path("temp_single_antigen.fasta")
        pdb_code, antigen_chain = antigen_id.split('_')

        with open(temp_fasta, 'w') as f:
            f.write(f">{pdb_code}_{antigen_chain} {antigen_chain} 1\n")
            # Write sequence with epitope case encoding
            seq_with_case = ""
            for i, (aa, label) in enumerate(zip(sequence, gt_labels)):
                seq_with_case += aa.upper() if label == 1 else aa.lower()
            f.write(seq_with_case + '\n')

        # Load as struct dataset
        data, _ = create_struct_datasets_from_combined(
            fasta_path=temp_fasta,
            structures_dir=Path("data/structures2"),
            val_partition="999",
            max_length=512,
        )

        if not data:
            print(f"❌ Could not load structural data for {antigen_id}")
            temp_fasta.unlink()
            return None

        original_sample = data[0]
        temp_fasta.unlink()

    except Exception as e:
        print(f"❌ Error loading structural data: {e}")
        return None

    # Align data lengths
    min_len = min(len(gt_labels), len(discotope_scores), len(sequence))
    gt_labels = gt_labels[:min_len]
    discotope_scores = np.array(discotope_scores[:min_len])

    # Run both analyses
    systematic_results, systematic_mutations = systematic_mutation_analysis(
        model, original_sample, discotope_scores, gt_labels
    )

    monte_carlo_results = monte_carlo_mutation_analysis(
        model, original_sample, discotope_scores, gt_labels
    )

    return {
        'antigen_id': antigen_id,
        'sequence': sequence,
        'gt_labels': gt_labels,
        'systematic_results': systematic_results,
        'systematic_mutations': systematic_mutations,
        'monte_carlo_results': monte_carlo_results
    }


def main():
    print("🧬 Epitope Escape Mutation Analysis")
    print("=" * 60)
    print("Finding minimum mutations to eliminate epitope predictions")

    # Load model
    model_path = Path("ensemble_lora_model.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    model = load_ensemble_model(model_path)

    # Load data
    print("\n📋 Loading test set data...")
    holdout2_gt = parse_fasta(Path("data/holdout2.fasta"))
    discotope_preds = load_discotope_predictions()

    # Find common antigens with good ensemble predictions
    common_antigens = set(holdout2_gt.keys()) & set(discotope_preds.keys())
    print(f"Available test antigens: {len(common_antigens)}")

    if not common_antigens:
        print("❌ No common antigens found!")
        return 1

    # Select a few interesting antigens for analysis
    # Prefer antigens with multiple epitope residues
    candidates = []
    for antigen_id in common_antigens:
        sequence, gt_labels = holdout2_gt[antigen_id]
        n_epitopes = sum(gt_labels)
        if n_epitopes >= 5 and len(sequence) <= 200:  # Reasonable size for analysis
            candidates.append((antigen_id, n_epitopes, len(sequence)))

    # Sort by number of epitopes (most interesting first)
    candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"\nSelected candidates for analysis:")
    for i, (antigen_id, n_epi, length) in enumerate(candidates[:5]):
        print(f"  {i+1}. {antigen_id}: {n_epi} epitopes, {length} residues")

    # Analyze top candidates
    results = []
    for i, (antigen_id, n_epi, length) in enumerate(candidates[:3]):  # Analyze top 3
        print(f"\n🔍 Analyzing antigen {i+1}/3...")
        result = analyze_antigen(antigen_id, model, holdout2_gt, discotope_preds)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("📊 ANALYSIS SUMMARY")
    print(f"{'='*80}")

    for result in results:
        antigen_id = result['antigen_id']
        systematic = result['systematic_results']
        monte_carlo = result['monte_carlo_results']

        print(f"\n{antigen_id}:")

        # Find minimum mutations needed
        sys_min = None
        for res in systematic:
            if res['target_reached']:
                sys_min = res['mutations']
                break

        mc_min = None
        for num_mut in sorted(monte_carlo.keys()):
            if monte_carlo[num_mut]['target_reached']:
                mc_min = num_mut
                break

        print(f"  Systematic approach: {sys_min if sys_min else 'No solution'} mutations")
        print(f"  Monte Carlo approach: {mc_min if mc_min else 'No solution'} mutations")

        if sys_min and mc_min:
            print(f"  Best result: {min(sys_min, mc_min)} mutations")

    print(f"\n✅ Epitope escape analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())