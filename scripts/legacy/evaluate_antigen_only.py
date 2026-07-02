#!/usr/bin/env python3
"""
Evaluate antigen-only Surf2Spot predictions against ground truth.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional

# Paths
AUTOPROT_DIR = Path(__file__).parent
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
ANTIGEN_ONLY_DIR = SURF2SPOT_DIR / "test_sabdab_antigen_only"
PREDICT_DIR = ANTIGEN_ONLY_DIR / "predict"

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

def parse_ply_predictions(ply_file: Path) -> Optional[List[float]]:
    """Parse PLY file to extract epitope scores."""
    try:
        # PLY files contain vertex data with x, y, z, score
        # Look for lines with numerical data
        scores = []

        with open(ply_file, 'r') as f:
            in_vertex_section = False
            vertex_count = 0
            target_vertices = 0

            for line in f:
                line = line.strip()

                if line.startswith('element vertex'):
                    target_vertices = int(line.split()[-1])
                    continue

                if line == 'end_header':
                    in_vertex_section = True
                    continue

                if in_vertex_section and vertex_count < target_vertices:
                    try:
                        parts = line.split()
                        if len(parts) >= 4:
                            # Last column should be the score
                            score = float(parts[-1])
                            scores.append(score)
                            vertex_count += 1
                    except (ValueError, IndexError):
                        continue

        return scores if scores else None

    except Exception as e:
        print(f"Error parsing {ply_file}: {e}")
        return None

def align_scores_to_residues(scores: List[float], sequence_length: int) -> List[float]:
    """Align surface scores to residue positions."""
    if len(scores) == sequence_length:
        return scores

    # If we have more scores than residues, average them
    if len(scores) > sequence_length:
        # Group scores and average
        group_size = len(scores) / sequence_length
        aligned_scores = []

        for i in range(sequence_length):
            start_idx = int(i * group_size)
            end_idx = int((i + 1) * group_size)
            residue_scores = scores[start_idx:end_idx]
            aligned_scores.append(np.mean(residue_scores) if residue_scores else 0.0)

        return aligned_scores

    # If we have fewer scores, pad with zeros
    else:
        return scores + [0.0] * (sequence_length - len(scores))

def evaluate_predictions(gt_data: Dict[str, Tuple[str, List[int]]]) -> Dict[str, float]:
    """Evaluate antigen-only predictions against ground truth."""

    all_y_true = []
    all_y_scores = []
    evaluated_count = 0
    failed_count = 0

    results = {}

    for antigen_id, (sequence, epitope_labels) in gt_data.items():
        # Look for prediction file
        pred_file = PREDICT_DIR / f"{antigen_id}_domain_0_pred.ply"

        if not pred_file.exists():
            failed_count += 1
            continue

        # Parse prediction scores
        raw_scores = parse_ply_predictions(pred_file)
        if raw_scores is None:
            failed_count += 1
            continue

        # Align scores to sequence length
        scores = align_scores_to_residues(raw_scores, len(epitope_labels))

        # Add to overall evaluation
        all_y_true.extend(epitope_labels)
        all_y_scores.extend(scores)

        evaluated_count += 1

        # Individual metrics
        try:
            if len(set(epitope_labels)) > 1:  # Need both classes for AUC
                auc = roc_auc_score(epitope_labels, scores)
                results[f"{antigen_id}_AUC"] = auc
        except:
            pass

    print(f"📊 Evaluation Summary:")
    print(f"   Total antigens in ground truth: {len(gt_data)}")
    print(f"   Successfully evaluated: {evaluated_count}")
    print(f"   Failed/missing predictions: {failed_count}")
    print(f"   Success rate: {evaluated_count/len(gt_data)*100:.1f}%")

    if all_y_true and all_y_scores:
        # Overall metrics
        overall_auc = roc_auc_score(all_y_true, all_y_scores)
        results["Overall_ROC_AUC"] = overall_auc

        # Find optimal threshold for F1
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_f1 = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in all_y_scores]
            _, _, f1, _ = precision_recall_fscore_support(all_y_true, y_pred, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Calculate final metrics with best threshold
        y_pred_best = [1 if score >= best_threshold else 0 for score in all_y_scores]
        precision, recall, f1, _ = precision_recall_fscore_support(all_y_true, y_pred_best, average='binary', zero_division=0)

        results.update({
            "Overall_Precision": precision,
            "Overall_Recall": recall,
            "Overall_F1": f1,
            "Optimal_Threshold": best_threshold
        })

        print(f"\n🎯 Overall Results (Antigen-Only Pipeline):")
        print(f"   ROC-AUC: {overall_auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Optimal Threshold: {best_threshold:.2f}")

    return results

def main():
    print("🔬 Antigen-Only Surf2Spot Evaluation")
    print("=" * 60)

    # Parse ground truth
    print("📖 Loading ground truth from sabdab_novel30.fasta...")
    gt_data = parse_sabdab_fasta()
    print(f"   Loaded {len(gt_data)} antigen sequences")

    # Check prediction files
    pred_files = list(PREDICT_DIR.glob("*_domain_0_pred.ply"))
    print(f"   Found {len(pred_files)} prediction files")

    if len(pred_files) == 0:
        print("❌ No prediction files found!")
        return

    # Evaluate
    results = evaluate_predictions(gt_data)

    if "Overall_ROC_AUC" in results:
        print(f"\n🏆 Key Result: Antigen-Only ROC-AUC = {results['Overall_ROC_AUC']:.4f}")
        print("   (This is the corrected result feeding only antigen chains to Surf2Spot)")
    else:
        print("❌ Could not calculate overall ROC-AUC")

if __name__ == "__main__":
    main()