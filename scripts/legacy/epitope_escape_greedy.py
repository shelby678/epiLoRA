#!/usr/bin/env python3
"""
Greedy Brute Force Epitope Escape Analysis

This implementation uses a greedy approach:
1. For each epitope residue, try all 20 amino acids
2. Calculate total immunogenicity for each substitution
3. Select the mutation that maximally decreases immunogenicity
4. Apply that mutation and repeat until target reached

This finds optimal single mutations at each step rather than just alanine substitutions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import time
from sklearn.metrics import roc_auc_score, roc_curve


# Standard amino acids
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


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


def calculate_immunogenicity_score(predictions: np.ndarray) -> float:
    """Calculate overall immunogenicity score from epitope predictions."""
    return np.mean(predictions)


def get_amino_acid_properties(aa: str) -> Dict:
    """Get basic amino acid properties for more realistic mutation effects."""
    properties = {
        'A': {'hydrophobic': 0.5, 'charged': 0, 'size': 1, 'aromatic': 0},
        'R': {'hydrophobic': -1, 'charged': 1, 'size': 4, 'aromatic': 0},
        'N': {'hydrophobic': -1, 'charged': 0, 'size': 2, 'aromatic': 0},
        'D': {'hydrophobic': -1, 'charged': -1, 'size': 2, 'aromatic': 0},
        'C': {'hydrophobic': 0.5, 'charged': 0, 'size': 2, 'aromatic': 0},
        'Q': {'hydrophobic': -1, 'charged': 0, 'size': 3, 'aromatic': 0},
        'E': {'hydrophobic': -1, 'charged': -1, 'size': 3, 'aromatic': 0},
        'G': {'hydrophobic': 0, 'charged': 0, 'size': 0, 'aromatic': 0},
        'H': {'hydrophobic': -0.5, 'charged': 0.5, 'size': 3, 'aromatic': 1},
        'I': {'hydrophobic': 1, 'charged': 0, 'size': 3, 'aromatic': 0},
        'L': {'hydrophobic': 1, 'charged': 0, 'size': 3, 'aromatic': 0},
        'K': {'hydrophobic': -1, 'charged': 1, 'size': 4, 'aromatic': 0},
        'M': {'hydrophobic': 0.5, 'charged': 0, 'size': 3, 'aromatic': 0},
        'F': {'hydrophobic': 1, 'charged': 0, 'size': 4, 'aromatic': 1},
        'P': {'hydrophobic': 0, 'charged': 0, 'size': 2, 'aromatic': 0},
        'S': {'hydrophobic': -0.5, 'charged': 0, 'size': 1, 'aromatic': 0},
        'T': {'hydrophobic': -0.5, 'charged': 0, 'size': 2, 'aromatic': 0},
        'W': {'hydrophobic': 1, 'charged': 0, 'size': 5, 'aromatic': 1},
        'Y': {'hydrophobic': 0.5, 'charged': 0, 'size': 4, 'aromatic': 1},
        'V': {'hydrophobic': 1, 'charged': 0, 'size': 2, 'aromatic': 0},
    }
    return properties.get(aa, {'hydrophobic': 0, 'charged': 0, 'size': 2, 'aromatic': 0})


def calculate_mutation_effect(original_aa: str, new_aa: str, original_score: float) -> float:
    """Calculate mutation effect based on amino acid properties."""
    if original_aa == new_aa:
        return original_score

    orig_props = get_amino_acid_properties(original_aa)
    new_props = get_amino_acid_properties(new_aa)

    # Calculate property differences
    hydrophobic_diff = abs(orig_props['hydrophobic'] - new_props['hydrophobic'])
    charge_diff = abs(orig_props['charged'] - new_props['charged'])
    size_diff = abs(orig_props['size'] - new_props['size'])
    aromatic_diff = abs(orig_props['aromatic'] - new_props['aromatic'])

    # More different properties = larger effect on epitope score
    property_change = (hydrophobic_diff + charge_diff + size_diff/5 + aromatic_diff) / 4

    # Reduction factor based on property changes
    # Large changes reduce epitope score more
    reduction_factor = min(0.9, 0.2 + 0.6 * property_change)

    # Special cases
    if new_aa == 'P':  # Proline is disruptive
        reduction_factor = max(reduction_factor, 0.7)
    elif new_aa == 'G':  # Glycine is flexible but small
        reduction_factor = max(reduction_factor, 0.5)

    return original_score * (1 - reduction_factor)


def simulate_single_mutation(original_scores: np.ndarray, sequence: str,
                           position: int, new_aa: str) -> np.ndarray:
    """Simulate effect of single mutation on all epitope scores."""
    mutated_scores = original_scores.copy()
    original_aa = sequence[position]

    if position < len(mutated_scores):
        # Direct effect at mutation site
        mutated_scores[position] = calculate_mutation_effect(
            original_aa, new_aa, original_scores[position]
        )

        # Local effects on neighboring positions (reduced impact)
        neighbor_effect = 0.3
        for offset in [-1, 1]:
            neighbor_pos = position + offset
            if 0 <= neighbor_pos < len(mutated_scores):
                neighbor_aa = sequence[neighbor_pos]
                neighbor_reduction = calculate_mutation_effect(
                    original_aa, new_aa, original_scores[neighbor_pos]
                )
                # Apply reduced neighbor effect
                mutated_scores[neighbor_pos] = (
                    mutated_scores[neighbor_pos] * (1 - neighbor_effect) +
                    neighbor_reduction * neighbor_effect
                )

    return mutated_scores


def greedy_brute_force_analysis(original_scores: np.ndarray, gt_labels: np.ndarray,
                               sequence: str, target_reduction: float = 0.5,
                               max_mutations: int = 10, verbose: bool = True) -> Dict:
    """
    Greedy brute force approach: at each step, try all possible single mutations
    and select the one that maximally decreases immunogenicity.
    """
    if verbose:
        print("🧬 Greedy Brute Force Mutation Analysis...")

    baseline_score = calculate_immunogenicity_score(original_scores)
    target_score = baseline_score * (1 - target_reduction)

    if verbose:
        print(f"  Baseline immunogenicity: {baseline_score:.4f}")
        print(f"  Target score: {target_score:.4f}")

    # Find epitope positions that can be mutated
    epitope_positions = [i for i, label in enumerate(gt_labels) if label == 1]

    if verbose:
        print(f"  Epitope positions to consider: {len(epitope_positions)}")

    # Track mutations applied
    current_sequence = list(sequence)
    current_scores = original_scores.copy()
    mutations_applied = []
    results = []

    for round_num in range(max_mutations):
        if verbose:
            print(f"\n  Round {round_num + 1}: Finding best mutation...")

        best_mutation = None
        best_score = float('inf')
        best_mutated_scores = None

        mutations_tested = 0

        # Try all possible mutations at all epitope positions
        for pos in epitope_positions:
            current_aa = current_sequence[pos]

            # Try all amino acids except current one
            for new_aa in AMINO_ACIDS:
                if new_aa == current_aa:
                    continue

                # Simulate this mutation
                test_sequence = current_sequence.copy()
                test_sequence[pos] = new_aa

                mutated_scores = simulate_single_mutation(
                    current_scores, ''.join(current_sequence), pos, new_aa
                )

                immunogenicity = calculate_immunogenicity_score(mutated_scores)
                mutations_tested += 1

                # Track best mutation
                if immunogenicity < best_score:
                    best_score = immunogenicity
                    best_mutation = (pos, current_aa, new_aa)
                    best_mutated_scores = mutated_scores.copy()

        if verbose:
            print(f"    Tested {mutations_tested} mutations")

        # Check if we found an improvement
        if best_mutation is None or best_score >= calculate_immunogenicity_score(current_scores):
            if verbose:
                print("    No beneficial mutations found - stopping")
            break

        # Apply best mutation
        pos, orig_aa, new_aa = best_mutation
        current_sequence[pos] = new_aa
        current_scores = best_mutated_scores
        mutations_applied.append(best_mutation)

        reduction = (baseline_score - best_score) / baseline_score

        result = {
            'round': round_num + 1,
            'mutation': f"{orig_aa}{pos+1}{new_aa}",
            'position': pos,
            'original_aa': orig_aa,
            'new_aa': new_aa,
            'score': best_score,
            'reduction': reduction,
            'cumulative_mutations': len(mutations_applied),
            'target_reached': best_score <= target_score
        }
        results.append(result)

        if verbose:
            print(f"    Best: {orig_aa}{pos+1}{new_aa} -> score={best_score:.4f} ({reduction:.1%} reduction)")

        # Check if target reached
        if best_score <= target_score:
            if verbose:
                print(f"  ✅ Target reached with {len(mutations_applied)} mutations!")
            break

    return {
        'results': results,
        'mutations_applied': mutations_applied,
        'final_sequence': ''.join(current_sequence),
        'final_score': calculate_immunogenicity_score(current_scores),
        'total_mutations': len(mutations_applied)
    }


def analyze_antigen_greedy(antigen_id: str, sequence: str, gt_labels: List[int],
                          discotope_scores: List[float]):
    """Greedy brute force analysis for a single antigen."""
    print(f"\n{'='*80}")
    print(f"🧬 GREEDY BRUTE FORCE ANALYSIS: {antigen_id}")
    print(f"{'='*80}")

    # Convert to numpy arrays and align lengths
    min_len = min(len(sequence), len(gt_labels), len(discotope_scores))
    sequence = sequence[:min_len]
    gt_labels = np.array(gt_labels[:min_len])
    disco_scores = np.array(discotope_scores[:min_len])

    print(f"Sequence length: {min_len}")
    print(f"Epitope residues: {sum(gt_labels)} ({sum(gt_labels)/len(gt_labels)*100:.1f}%)")
    print(f"Original sequence: {sequence}")

    # Show epitope positions and their scores
    epitope_info = []
    for i, (aa, label, score) in enumerate(zip(sequence, gt_labels, disco_scores)):
        if label == 1:
            epitope_info.append(f"{aa}{i+1}({score:.3f})")

    print(f"Epitope residues: {' '.join(epitope_info[:10])}{'...' if len(epitope_info) > 10 else ''}")

    # Run greedy analysis
    start_time = time.time()
    greedy_result = greedy_brute_force_analysis(
        disco_scores, gt_labels, sequence,
        target_reduction=0.5, max_mutations=15
    )
    elapsed_time = time.time() - start_time

    print(f"\n⏱️ Analysis completed in {elapsed_time:.1f} seconds")

    # Show mutation sequence
    print(f"\n📋 Optimal mutation sequence:")
    for i, result in enumerate(greedy_result['results']):
        marker = "✅" if result['target_reached'] else "🔄"
        print(f"  {i+1:2d}. {result['mutation']} -> {result['score']:.4f} ({result['reduction']:.1%}) {marker}")

    print(f"\n🧬 Final mutated sequence:")
    print(f"  Original: {sequence}")
    print(f"  Mutated:  {greedy_result['final_sequence']}")

    # Highlight changes
    changes = []
    for i, (orig, new) in enumerate(zip(sequence, greedy_result['final_sequence'])):
        if orig != new:
            changes.append(f"{orig}{i+1}{new}")

    print(f"  Changes:  {' '.join(changes)}")

    return greedy_result


def main():
    print("🧬 Greedy Brute Force Epitope Escape Analysis")
    print("=" * 70)
    print("Finding optimal mutations by testing all amino acid substitutions")

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

    # Select interesting candidates (smaller ones for brute force)
    candidates = []
    for antigen_id in common_antigens:
        sequence, gt_labels = holdout2_gt[antigen_id]
        discotope_scores, _ = discotope_preds[antigen_id]

        n_epitopes = sum(gt_labels)
        baseline_score = np.mean(discotope_scores[:len(gt_labels)])

        # Prefer smaller sequences with decent epitope scores for brute force
        if 5 <= n_epitopes <= 15 and 20 <= len(sequence) <= 80 and baseline_score > 0.35:
            candidates.append((antigen_id, n_epitopes, len(sequence), baseline_score))

    # Sort by epitope count (moderate complexity first)
    candidates.sort(key=lambda x: x[1])

    print(f"\nSelected candidates for greedy analysis:")
    for i, (antigen_id, n_epi, length, score) in enumerate(candidates[:5]):
        print(f"  {i+1}. {antigen_id}: {n_epi} epitopes, {length} residues, score={score:.3f}")

    # Analyze candidates
    results = []
    for i, (antigen_id, n_epi, length, score) in enumerate(candidates[:3]):  # Top 3
        print(f"\n🔍 Analyzing antigen {i+1}/3...")
        sequence, gt_labels = holdout2_gt[antigen_id]
        discotope_scores, _ = discotope_preds[antigen_id]

        result = analyze_antigen_greedy(antigen_id, sequence, gt_labels, discotope_scores)
        results.append((antigen_id, result))

    # Summary
    print(f"\n{'='*80}")
    print("📊 GREEDY ANALYSIS SUMMARY")
    print(f"{'='*80}")

    for antigen_id, result in results:
        print(f"\n{antigen_id}:")
        final_result = result['results'][-1] if result['results'] else None

        if final_result:
            if final_result['target_reached']:
                print(f"  ✅ SUCCESS: {result['total_mutations']} mutations achieved 50% reduction")
                print(f"     Final score: {result['final_score']:.4f} ({final_result['reduction']:.1%} reduction)")
            else:
                print(f"  🔄 PARTIAL: {result['total_mutations']} mutations, {final_result['reduction']:.1%} reduction")
                print(f"     Final score: {result['final_score']:.4f}")

            # Show key mutations
            mutations = [r['mutation'] for r in result['results'][:5]]
            print(f"     Key mutations: {' '.join(mutations)}{'...' if len(result['results']) > 5 else ''}")
        else:
            print("  ❌ No beneficial mutations found")

    print(f"\n🧬 Greedy brute force finds optimal single mutations at each step")
    print("✅ Analysis complete!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())