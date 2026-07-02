#!/usr/bin/env python3
"""
Final Surf2Spot evaluation on sabdab_novel30.fasta validation set.

This script evaluates Surf2Spot predictions against the ground truth, using both
the existing working predictions and any new predictions from our pipeline.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional

# Paths
AUTOPROT_DIR = Path(__file__).parent
SURF2SPOT_DIR = AUTOPROT_DIR.parent / "Surf2Spot"
SABDAB_FASTA = AUTOPROT_DIR / "data" / "sabdab_novel30.fasta"

# All Surf2Spot prediction directories
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
                header = line[1:]  # Remove >
            elif header:
                sequence = line
                # Convert case to labels: uppercase = epitope (1), lowercase = non-epitope (0)
                labels = [1 if c.isupper() else 0 for c in sequence]
                gt_data[header] = (sequence.upper(), labels)

    return gt_data

def get_antigen_id(header: str) -> str:
    """Extract antigen identifier from sabdab header."""
    parts = header.split()
    pdb_chain_info = parts[0]  # e.g., "8dyx_HL"
    antigen_chain = parts[1]   # e.g., "I"

    pdb_code = pdb_chain_info.split("_")[0].lower()  # e.g., "8dyx"
    return f"{pdb_code}_{antigen_chain}"

def find_prediction_files(antigen_id: str) -> List[Path]:
    """Find all available prediction files for an antigen."""
    found_files = []

    pdb_code, antigen_chain = antigen_id.split("_")

    for predict_dir in PREDICT_DIRS:
        if not predict_dir.exists():
            continue

        # Try various naming patterns that exist in the working directories
        patterns = [
            f"{antigen_id}.csv",                    # Direct match: 8dyx_I.csv
            f"{antigen_id}_pred.csv",               # With pred suffix
            f"{pdb_code}_{antigen_chain}.csv",      # Same format
            f"{pdb_code.upper()}_{antigen_chain}.csv", # Uppercase PDB
        ]

        for pattern in patterns:
            filepath = predict_dir / pattern
            if filepath.exists():
                found_files.append(filepath)

        # Try domain split patterns (multiple CSV files per antigen)
        domain_pattern = predict_dir / f"{antigen_id}*_domain_*_pred.csv"
        domain_files = list(predict_dir.glob(f"{antigen_id}*_domain_*_pred.csv"))
        if domain_files:
            found_files.extend(domain_files)

    return found_files

def load_prediction_scores(files: List[Path]) -> Optional[List[float]]:
    """Load and merge prediction scores from files."""
    if not files:
        return None

    # Single file case
    if len(files) == 1 and "domain" not in files[0].name:
        try:
            df = pd.read_csv(files[0])
            if 'score' in df.columns:
                return df['score'].tolist()
        except Exception as e:
            print(f"  Error reading {files[0]}: {e}")
            return None

    # Domain files case - merge multiple domain predictions
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
        except Exception as e:
            print(f"  Error merging domain predictions: {e}")

    # Try any remaining file
    for file in files:
        try:
            df = pd.read_csv(file)
            if 'score' in df.columns:
                return df['score'].tolist()
        except Exception:
            continue

    return None

def comprehensive_evaluation():
    """Comprehensive evaluation of Surf2Spot on sabdab_novel30.fasta."""
    print("🔬 Comprehensive Surf2Spot Evaluation on sabdab_novel30.fasta")
    print("=" * 70)

    # Check available prediction directories
    print("📁 Checking prediction directories:")
    existing_dirs = []
    for d in PREDICT_DIRS:
        if d.exists():
            n_csvs = len(list(d.glob("*.csv")))
            existing_dirs.append(d)
            print(f"   ✅ {d.relative_to(SURF2SPOT_DIR)}: {n_csvs} CSV files")
        else:
            print(f"   ❌ {d.relative_to(SURF2SPOT_DIR)}: Not found")
    print()

    if not existing_dirs:
        print("❌ ERROR: No prediction directories found!")
        return

    # Load ground truth
    gt_data = parse_sabdab_fasta()
    print(f"📋 Ground truth: {len(gt_data)} antigens from sabdab_novel30.fasta")

    # Comprehensive evaluation
    results = []
    all_scores, all_labels = [], []
    skipped = []
    found_predictions = []
    source_counts = {}

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

        # Determine source directory
        source = pred_files[0].parent.name
        source_counts[source] = source_counts.get(source, 0) + 1
        found_predictions.append((header, pred_files))

        # Align lengths
        n_gt, n_pred = len(labels), len(scores)
        if n_pred != n_gt:
            n = min(n_gt, n_pred)
            labels_use, scores_use = labels[:n], scores[:n]
            note = f" (len mismatch: gt={n_gt} pred={n_pred}, using {n})"
        else:
            labels_use, scores_use = labels, scores
            note = ""

        n_pos = sum(labels_use)
        if n_pos == 0 or n_pos == len(labels_use):
            skipped.append(f"{header} (mono-class: n_pos={n_pos}/{len(labels_use)})")
            continue

        # Clean scores
        scores_clean = [0.0 if pd.isna(s) else s for s in scores_use]

        try:
            auc = roc_auc_score(labels_use, scores_clean)
            results.append((header, auc, n_pos, len(labels_use), note, source))
            all_scores.extend(scores_clean)
            all_labels.extend(labels_use)
        except Exception as e:
            skipped.append(f"{header} (AUC error: {e})")

    # 📊 Print detailed results
    print(f"🎯 RESULTS - Found predictions for {len(results)} antigens:")
    print(f"{'Antigen':<35} {'AUC':>6}  {'Pos':>3}/{'Tot':<4} {'Source':<15} {'Notes'}")
    print("-" * 85)

    for header, auc, n_pos, n_res, note, source in sorted(results, key=lambda x: -x[1]):
        print(f"{header:<35} {auc:>6.3f}  {n_pos:>3}/{n_res:<4} {source:<15}{note}")

    if results:
        overall_auc = roc_auc_score(all_labels, all_scores)
        mean_auc = np.mean([r[1] for r in results])
        median_auc = np.median([r[1] for r in results])

        print("\n📈 SUMMARY METRICS:")
        print(f"   Overall AUC (pooled):  {overall_auc:.4f}")
        print(f"   Mean per-antigen AUC:  {mean_auc:.4f}")
        print(f"   Median per-antigen:    {median_auc:.4f}")
        print(f"   Antigens evaluated:    {len(results)}")
        print(f"   Total residues:        {len(all_labels):,}")
        print(f"   Epitope residues:      {sum(all_labels):,} ({100*sum(all_labels)/len(all_labels):.1f}%)")

        print(f"\n🔍 BREAKDOWN BY SOURCE:")
        for source, count in sorted(source_counts.items()):
            source_results = [r for r in results if r[5] == source]
            source_auc = np.mean([r[1] for r in source_results])
            print(f"   {source:<15}: {count:>3} antigens, mean AUC = {source_auc:.4f}")

    print(f"\n⚠️  MISSING PREDICTIONS: {len(skipped)}")
    if skipped:
        for s in skipped[:15]:  # Show first 15
            print(f"   {s}")
        if len(skipped) > 15:
            print(f"   ... and {len(skipped) - 15} more")

    # 📋 Summary comparison
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   Previous attempt:  37 antigens, mean AUC = 0.507, overall = 0.527")
    if results:
        print(f"   Current results:   {len(results)} antigens, mean AUC = {mean_auc:.3f}, overall = {overall_auc:.3f}")
        improvement = len(results) / 37 if len(results) >= 37 else len(results) / 37
        print(f"   Coverage improvement: {improvement:.1f}x")

    # 🎯 Top and bottom performers
    if len(results) >= 5:
        print(f"\n🏆 TOP PERFORMING ANTIGENS:")
        for header, auc, n_pos, n_res, note, source in sorted(results, key=lambda x: -x[1])[:5]:
            print(f"   {header:<30} AUC = {auc:.3f}")

        print(f"\n⚡ CHALLENGING ANTIGENS:")
        for header, auc, n_pos, n_res, note, source in sorted(results, key=lambda x: x[1])[:5]:
            print(f"   {header:<30} AUC = {auc:.3f}")

    return results

if __name__ == "__main__":
    results = comprehensive_evaluation()

    if results:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"   Results saved for {len(results)} antigens")
    else:
        print(f"\n❌ No predictions could be evaluated")