#!/usr/bin/env python3
"""
Run DiscoTope-3.0 on test_C50.fasta sequences and compute ROC AUC if labels are available.

This script:
1. Reads test_C50.fasta sequences
2. Extracts individual chain PDB files from structures2/sabdab_dataset
3. Runs DiscoTope-3.0 predictions using the discotope3_web codebase
4. Computes ROC AUC if ground truth labels are found

Run:
    python run_discotope3_test_c50.py
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
TEST_FASTA = Path("../autoprot/data/test_C50.fasta")
STRUCTURES_DIR = Path("../autoprot/data/structures2/sabdab_dataset")
DISCOTOPE_MODELS_DIR = Path("models")

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "  # standard residues only

def load_fasta(path: Path) -> List[Tuple[str, str]]:
    """Return list of (header, sequence) from FASTA file."""
    entries = []
    header = None
    seq_lines = []

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_lines)))
                header = line[1:]  # remove >
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        entries.append((header, "".join(seq_lines)))

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

    # Load test sequences
    entries = load_fasta(TEST_FASTA)
    logger.info(f"Loaded {len(entries)} sequences from {TEST_FASTA}")

    # Prepare individual chain PDB files
    prepared_pdbs = []
    skipped = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logger.info(f"Extracting chains to temporary directory: {tmpdir}")

        for header, sequence in entries:
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
                prepared_pdbs.append((header, dst_pdb, sequence))
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

        # Collect results
        results = {}
        start = 0
        for item in valid_items:
            pdb_id = item["pdb_id"]
            L = len(item["X_arr"])
            end = start + L

            predictions = y_hat_all[start:end]
            results[pdb_id] = predictions
            start = end

            logger.info(f"  {pdb_id}: {L} residues, "
                       f"mean_score={predictions.mean():.4f}, "
                       f"max_score={predictions.max():.4f}")

        # Report summary
        all_predictions = np.concatenate(list(results.values()))
        logger.info(f"\nDiscoTope-3.0 Predictions Summary:")
        logger.info(f"  Total proteins: {len(results)}")
        logger.info(f"  Total residues: {len(all_predictions)}")
        logger.info(f"  Mean epitope score: {all_predictions.mean():.4f}")
        logger.info(f"  Max epitope score: {all_predictions.max():.4f}")
        logger.info(f"  Std epitope score: {all_predictions.std():.4f}")

        # TODO: Find ground truth labels and compute ROC AUC
        logger.info(f"\nROC AUC: Cannot compute without ground truth labels")
        logger.info(f"Ground truth labels needed for test_C50.fasta sequences")

        return results

if __name__ == "__main__":
    results = main()