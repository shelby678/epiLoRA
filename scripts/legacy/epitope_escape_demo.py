#!/usr/bin/env python3
"""
Simplified Epitope Escape Analysis Demo

This demonstrates the epitope escape concept using DiscoTope3 predictions only,
while we wait for the full ensemble model training to complete.

Shows both systematic and Monte Carlo approaches for finding minimum mutations
to eliminate epitope predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import random
from sklearn.metrics import roc_auc_score, roc_curve


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
                print(f"Warning: Could not load {csv_file}: {e}")

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


def calculate_immunogenicity_score(predictions: np.ndarray, method='mean') -> float:
    """Calculate overall immunogenicity score from epitope predictions."""
    if method == 'mean':
        return np.mean(predictions)
    elif method == 'max':
        return np.max(predictions)
    elif method == 'sum':
        return np.sum(predictions)
    else:
        return np.mean(predictions)


def simulate_mutation_effect(original_scores: np.ndarray, mutation_positions: List[int],
                            reduction_factor: float = 0.8) -> np.ndarray:
    """
    Simulate the effect of mutations on epitope predictions.

    This is a simplified model - assumes mutations to alanine reduce
    epitope scores at and near mutation sites.
    """
    mutated_scores = original_scores.copy()

    for pos in mutation_positions:
        if 0 <= pos < len(mutated_scores):
            # Direct effect: strong reduction at mutation site
            mutated_scores[pos] *= (1 - reduction_factor)

            # Local effects: moderate reduction at neighboring positions
            for neighbor in range(max(0, pos-1), min(len(mutated_scores), pos+2)):
                if neighbor != pos:
                    mutated_scores[neighbor] *= (1 - reduction_factor * 0.3)

    return mutated_scores


def systematic_mutation_analysis(original_scores: np.ndarray, gt_labels: np.ndarray,
                                sequence: str, target_reduction: float = 0.5,
                                max_mutations: int = 10) -> Dict:
    """Systematically mutate highest-scoring epitope residues."""
    print("🔬 Systematic mutation analysis...")

    baseline_score = calculate_immunogenicity_score(original_scores)
    target_score = baseline_score * (1 - target_reduction)

    print(f"  Baseline immunogenicity: {baseline_score:.4f}")
    print(f"  Target score: {target_score:.4f}")

    # Find epitope residues sorted by prediction score
    epitope_candidates = []
    for i, (score, gt_label) in enumerate(zip(original_scores, gt_labels)):
        if gt_label == 1 and score > 0.5:  # True epitope with decent prediction
            epitope_candidates.append((i, score, sequence[i]))

    # Sort by prediction score (highest first)
    epitope_candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"  Found {len(epitope_candidates)} high-scoring epitope residues")

    results = []
    mutation_positions = []

    for mutation_count in range(1, min(max_mutations + 1, len(epitope_candidates) + 1)):
        # Add next highest-scoring residue
        pos, score, aa = epitope_candidates[mutation_count - 1]

        if aa == 'A':  # Skip if already alanine
            continue

        mutation_positions.append(pos)

        # Simulate mutation effect
        mutated_scores = simulate_mutation_effect(original_scores, mutation_positions)
        new_score = calculate_immunogenicity_score(mutated_scores)
        reduction = (baseline_score - new_score) / baseline_score

        result = {
            'mutations': len(mutation_positions),
            'positions': mutation_positions.copy(),
            'residues': [sequence[p] for p in mutation_positions],
            'score': new_score,
            'reduction': reduction,
            'target_reached': new_score <= target_score
        }
        results.append(result)

        print(f"  {len(mutation_positions)} mutations: {aa}{pos+1}A -> score={new_score:.4f} ({reduction:.1%} reduction)")

        if new_score <= target_score:
            print(f"  ✅ Target reached with {len(mutation_positions)} mutations!")
            break

    return {'results': results, 'mutation_positions': mutation_positions}


def monte_carlo_mutation_analysis(original_scores: np.ndarray, gt_labels: np.ndarray,
                                 sequence: str, target_reduction: float = 0.5,
                                 max_mutations: int = 10, n_iterations: int = 500) -> Dict:
    """Monte Carlo search for optimal mutation combinations."""
    print("🎲 Monte Carlo mutation analysis...")

    baseline_score = calculate_immunogenicity_score(original_scores)
    target_score = baseline_score * (1 - target_reduction)

    print(f"  Baseline immunogenicity: {baseline_score:.4f}")
    print(f"  Target score: {target_score:.4f}")
    print(f"  Running {n_iterations} iterations...")

    # Find mutable positions (epitope residues that aren't already alanine)
    mutable_positions = []
    for i, (gt_label, aa) in enumerate(zip(gt_labels, sequence)):
        if gt_label == 1 and aa != 'A':
            mutable_positions.append(i)

    print(f"  {len(mutable_positions)} mutable epitope positions")

    best_results = {}  # num_mutations -> best result

    for iteration in range(n_iterations):
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}/{n_iterations}")

        # Try different numbers of mutations
        for num_mutations in range(1, min(max_mutations + 1, len(mutable_positions) + 1)):
            # Random selection of mutation positions
            mutation_positions = random.sample(mutable_positions, num_mutations)

            # Simulate mutation effect
            mutated_scores = simulate_mutation_effect(original_scores, mutation_positions)
            new_score = calculate_immunogenicity_score(mutated_scores)
            reduction = (baseline_score - new_score) / baseline_score

            # Track best result for this number of mutations
            if (num_mutations not in best_results or
                new_score < best_results[num_mutations]['score']):

                best_results[num_mutations] = {
                    'mutations': num_mutations,
                    'positions': mutation_positions,
                    'residues': [sequence[p] for p in mutation_positions],
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
            print("    ✅ Target reached!")

    return best_results


def analyze_antigen_demo(antigen_id: str, sequence: str, gt_labels: List[int],
                        discotope_scores: List[float]):
    """Demo analysis for a single antigen using DiscoTope3 predictions."""
    print(f"\n{'='*80}")
    print(f"🧬 EPITOPE ESCAPE DEMO: {antigen_id}")
    print(f"{'='*80}")

    # Convert to numpy arrays and align lengths
    min_len = min(len(sequence), len(gt_labels), len(discotope_scores))
    sequence = sequence[:min_len]
    gt_labels = np.array(gt_labels[:min_len])
    disco_scores = np.array(discotope_scores[:min_len])

    print(f"Sequence length: {min_len}")
    print(f"Epitope residues: {sum(gt_labels)} ({sum(gt_labels)/len(gt_labels)*100:.1f}%)")
    print(f"Sequence: {sequence[:60]}{'...' if len(sequence) > 60 else ''}")

    # Show epitope positions
    epitope_positions = [i+1 for i, label in enumerate(gt_labels) if label == 1]
    print(f"True epitope positions: {epitope_positions[:10]}{'...' if len(epitope_positions) > 10 else ''}")

    # Run both analyses
    systematic_result = systematic_mutation_analysis(
        disco_scores, gt_labels, sequence
    )

    monte_carlo_result = monte_carlo_mutation_analysis(
        disco_scores, gt_labels, sequence
    )

    return {
        'antigen_id': antigen_id,
        'sequence': sequence,
        'systematic': systematic_result,
        'monte_carlo': monte_carlo_result
    }


def main():
    print("🧬 Epitope Escape Mutation Analysis - DiscoTope3 Demo")
    print("=" * 70)
    print("Demonstrating mutation analysis using DiscoTope3 predictions")
    print("(Full ensemble analysis will run once model training completes)")

    # Load data
    print("\n📋 Loading test set data...")
    holdout2_gt = parse_fasta(Path("data/holdout2.fasta"))
    discotope_preds = load_discotope_predictions()

    # Find common antigens
    common_antigens = set(holdout2_gt.keys()) & set(discotope_preds.keys())
    print(f"Available antigens: {len(common_antigens)}")

    if not common_antigens:
        print("❌ No common antigens found!")
        return 1

    # Select interesting candidates
    candidates = []
    for antigen_id in common_antigens:
        sequence, gt_labels = holdout2_gt[antigen_id]
        discotope_scores, _ = discotope_preds[antigen_id]

        n_epitopes = sum(gt_labels)
        baseline_score = np.mean(discotope_scores[:len(gt_labels)])

        if n_epitopes >= 5 and len(sequence) <= 150 and baseline_score > 0.3:
            candidates.append((antigen_id, n_epitopes, len(sequence), baseline_score))

    # Sort by baseline score (most immunogenic first)
    candidates.sort(key=lambda x: x[3], reverse=True)

    print(f"\nSelected candidates:")
    for i, (antigen_id, n_epi, length, score) in enumerate(candidates[:5]):
        print(f"  {i+1}. {antigen_id}: {n_epi} epitopes, {length} residues, score={score:.3f}")

    # Analyze top candidates
    results = []
    for i, (antigen_id, n_epi, length, score) in enumerate(candidates[:2]):  # Demo with 2
        print(f"\n🔍 Analyzing antigen {i+1}/2...")
        sequence, gt_labels = holdout2_gt[antigen_id]
        discotope_scores, _ = discotope_preds[antigen_id]

        result = analyze_antigen_demo(antigen_id, sequence, gt_labels, discotope_scores)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("📊 DEMO SUMMARY")
    print(f"{'='*80}")

    for result in results:
        antigen_id = result['antigen_id']
        systematic = result['systematic']['results']
        monte_carlo = result['monte_carlo']

        print(f"\n{antigen_id}:")

        # Find minimum mutations needed (50% reduction target)
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
            print(f"  ✅ Best: {min(sys_min, mc_min)} mutations to reduce immunogenicity by 50%")
        elif sys_min or mc_min:
            best = sys_min or mc_min
            print(f"  ✅ Solution found: {best} mutations")

    print(f"\n🔬 This demonstrates the concept using DiscoTope3 predictions")
    print("📋 Full ensemble analysis will be more accurate once model training completes")
    print("✅ Demo complete!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())