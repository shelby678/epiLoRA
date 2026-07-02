#!/usr/bin/env python3
"""
Run DiscoTope-3.0 on test_C50.fasta sequences and compute ROC AUC.

This script:
1. Reads test_C50.fasta sequences with uppercase/lowercase epitope labels
2. Extracts individual chain PDB files from structures2/sabdab_dataset
3. Runs DiscoTope-3.0 predictions using the discotope3_web codebase
4. Computes ROC AUC comparing predictions vs ground truth labels

Ground truth format:
- Uppercase letters = epitope residues (label = 1)
- Lowercase letters = non-epitope residues (label = 0)

Run:
    cd discotope3_web && python ../autoprot/run_discotope3_test_c50_with_labels.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

# Add discotope3_web/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "discotope3_web/src"))

from Bio.PDB import PDBIO, PDBParser, Select
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# Paths
TEST_FASTA = Path("../conference_submission/data/test_C50.fasta")
STRUCTURES_DIR = Path("../autoprot/data/structures2/sabdab_dataset")
DISCOTOPE_MODELS_DIR = Path("models")

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "  # standard residues only

def load_fasta_with_labels(path: Path) -> List[Tuple[str, str, List[int]]]:
    """Return list of (header, sequence_uppercase, binary_labels) from FASTA file.

    Labels are extracted from case: uppercase = epitope (1), lowercase = non-epitope (0).
    """
    entries = []
    header = None
    seq_lines = []

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    # Process previous entry
                    sequence_mixed = "".join(seq_lines)
                    sequence_upper = sequence_mixed.upper()
                    labels = [1 if c.isupper() else 0 for c in sequence_mixed]
                    entries.append((header, sequence_upper, labels))
                header = line[1:]  # remove >
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        # Process last entry
        sequence_mixed = "".join(seq_lines)
        sequence_upper = sequence_mixed.upper()
        labels = [1 if c.isupper() else 0 for c in sequence_mixed]
        entries.append((header, sequence_upper, labels))

    return entries

def extract_chain_pdb(src_pdb: Path, chain_id: str, dst_pdb: Path) -> bool:
    """Extract a single chain from multi-chain PDB file."""
    try:
        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        structure = parser.get_structure("s", str(src_pdb))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(dst_pdb), ChainSelect(chain_id))
        return dst_pdb.stat().st_size > 0
    except Exception as e:
        logger.warning(f"Failed to extract chain {chain_id} from {src_pdb}: {e}")
        return False

def parse_header(header: str) -> Optional[Tuple[str, str]]:
    """Parse header like '1fsk_g' to return (pdb_id, chain_id)."""
    parts = header.split("_")
    if len(parts) >= 2:
        pdb_id = parts[0].lower()
        chain_id = parts[1].upper()
        return pdb_id, chain_id
    return None

def main():
    # Check if discotope3_web environment is available
    if not DISCOTOPE_MODELS_DIR.exists():
        logger.error(f"DiscoTope models not found at {DISCOTOPE_MODELS_DIR}")
        logger.error("Please ensure discotope3_web is set up correctly")
        sys.exit(1)

    # Load test sequences with labels
    entries = load_fasta_with_labels(TEST_FASTA)
    logger.info(f"Loaded {len(entries)} sequences from {TEST_FASTA}")

    # Show epitope statistics
    total_residues = sum(len(labels) for _, _, labels in entries)
    total_epitope = sum(sum(labels) for _, _, labels in entries)
    logger.info(f"Total residues: {total_residues}, epitope: {total_epitope} ({100*total_epitope/total_residues:.1f}%)")

    # Prepare individual chain PDB files
    prepared_pdbs = []
    skipped = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logger.info(f"Extracting chains to temporary directory: {tmpdir}")

        for header, sequence, labels in entries:
            parsed = parse_header(header)
            if not parsed:
                skipped.append((header, "Cannot parse header"))
                continue

            pdb_id, chain_id = parsed
            src_pdb = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"

            if not src_pdb.exists():
                skipped.append((header, f"PDB file not found: {src_pdb}"))
                continue

            dst_pdb = tmpdir / f"{header}.pdb"
            if extract_chain_pdb(src_pdb, chain_id, dst_pdb):
                prepared_pdbs.append((header, dst_pdb, sequence, labels))
                logger.info(f"  Extracted {header} -> {dst_pdb}")
            else:
                skipped.append((header, f"Chain extraction failed"))

        logger.info(f"Successfully prepared {len(prepared_pdbs)} PDB files")
        if skipped:
            logger.info(f"Skipped {len(skipped)} entries:")
            for header, reason in skipped[:5]:
                logger.info(f"  {header}: {reason}")
            if len(skipped) > 5:
                logger.info(f"  ... and {len(skipped)-5} more")

        if not prepared_pdbs:
            logger.error("No PDB files prepared for prediction")
            sys.exit(1)

        # Import discotope3_web modules
        try:
            import predict_webserver as _pw
            import make_dataset as _md
            # Fix logger issues in imported modules
            _pw.log = logging.getLogger("predict_webserver")
            _md.log = logging.getLogger("make_dataset")
            from predict_webserver import load_models, predict_using_models
            from make_dataset import Discotope_Dataset_web
        except ImportError as e:
            logger.error(f"Failed to import discotope3_web modules: {e}")
            logger.error("Make sure the discotope3_web environment is properly set up")
            sys.exit(1)

        # Load DiscoTope models
        logger.info("Loading DiscoTope-3.0 models...")
        models = load_models(str(DISCOTOPE_MODELS_DIR), verbose=0)
        logger.info(f"Loaded {len(models)} XGBoost models")

        # Build dataset (runs ESM-IF1 embeddings)
        logger.info("Building dataset and extracting ESM-IF1 embeddings...")
        dataset = Discotope_Dataset_web(
            pdb_dir=str(tmpdir),
            structure_type="solved",
            verbose=0,
        )
        logger.info(f"Dataset size: {len(dataset)}")

        # Run predictions
        valid_items = [dataset[i] for i in range(len(dataset))
                      if dataset[i]["X_arr"] is not False]

        if not valid_items:
            logger.error("No valid items in dataset for prediction")
            sys.exit(1)

        logger.info(f"Running predictions on {len(valid_items)} valid items...")
        X_all = np.concatenate([item["X_arr"] for item in valid_items])
        y_hat_all = predict_using_models(models, X_all)

        # Match predictions with ground truth labels
        all_predictions = []
        all_labels = []
        start = 0

        prepared_by_id = {header: (sequence, labels) for header, _, sequence, labels in prepared_pdbs}

        for item in valid_items:
            pdb_id = item["pdb_id"]  # should match the header
            L = len(item["X_arr"])
            end = start + L

            predictions = y_hat_all[start:end]

            if pdb_id in prepared_by_id:
                sequence, labels = prepared_by_id[pdb_id]

                if len(labels) == L:
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)

                    epi_count = sum(labels)
                    mean_epi_score = np.mean([predictions[i] for i in range(L) if labels[i] == 1]) if epi_count > 0 else 0
                    mean_non_epi_score = np.mean([predictions[i] for i in range(L) if labels[i] == 0]) if (L - epi_count) > 0 else 0

                    logger.info(f"  {pdb_id}: {L} residues, {epi_count} epitope ({100*epi_count/L:.1f}%), "
                               f"epi_score={mean_epi_score:.3f}, non_epi_score={mean_non_epi_score:.3f}")
                else:
                    logger.warning(f"  {pdb_id}: length mismatch - predictions={L}, labels={len(labels)}")
            else:
                logger.warning(f"  {pdb_id}: not found in prepared data")

            start = end

        # Compute ROC AUC
        if len(all_predictions) == 0:
            logger.error("No valid predictions collected")
            sys.exit(1)

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        if len(set(all_labels)) < 2:
            logger.error("Labels have only one class — cannot compute ROC-AUC")
            sys.exit(1)

        roc_auc = roc_auc_score(all_labels, all_predictions)

        n_proteins = len(valid_items)
        n_residues = len(all_labels)
        n_epitope = int(all_labels.sum())

        print(f"\n{'='*60}")
        print(f"DiscoTope-3.0 ROC-AUC on test_C50.fasta")
        print(f"{'='*60}")
        print(f"  Proteins evaluated     : {n_proteins}")
        print(f"  Total residues         : {n_residues:,}")
        print(f"  Epitope residues       : {n_epitope:,} ({100*n_epitope/n_residues:.1f}%)")
        print(f"  Mean prediction score  : {all_predictions.mean():.4f}")
        print(f"  ROC-AUC                : {roc_auc:.4f}")
        print(f"{'='*60}")

        return roc_auc

if __name__ == "__main__":
    roc_auc = main()
    print(f"\nFinal ROC-AUC: {roc_auc:.4f}")