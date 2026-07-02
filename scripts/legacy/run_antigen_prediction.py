#!/usr/bin/env python3
"""
Direct prediction on antigen-only files, bypassing domain partitioning issues.
"""

import os
import subprocess
import shutil
from pathlib import Path

# Paths
SURF2SPOT_DIR = Path("/home/sferrier/epitope_mapping/Surf2Spot")
ANTIGEN_ONLY_DIR = SURF2SPOT_DIR / "test_sabdab_antigen_only"
MICROMAMBA_PATH = "/home/sferrier/miniforge3/micromamba"
SURF2SPOT_ENV = "/home/sferrier/miniforge3/envs/surf2spot"

def create_filtered_ply_files():
    """Create the _all_5.0_filtered_domain_0.ply files expected by prediction."""
    print("📁 Creating filtered PLY files for prediction...")

    ply_files = list(ANTIGEN_ONLY_DIR.glob("*_all_5.0.ply"))
    print(f"   Found {len(ply_files)} PLY files to process")

    for ply_file in ply_files:
        # Create expected filtered domain file names
        base_name = ply_file.stem.replace("_all_5.0", "")

        # Main filtered file
        filtered_file = ANTIGEN_ONLY_DIR / f"{base_name}_all_5.0_filtered_domain_0.ply"

        if not filtered_file.exists():
            shutil.copy(ply_file, filtered_file)
            print(f"   Created: {filtered_file.name}")

def run_prediction():
    """Run the actual prediction."""
    print("\n🎯 Running antigen-only epitope prediction...")

    cmd = [
        MICROMAMBA_PATH, "run", "-p", SURF2SPOT_ENV,
        "python", "Surf2Spot/main.py", "NB-predict",
        "-i", str(ANTIGEN_ONLY_DIR),
        "-o", str(ANTIGEN_ONLY_DIR / "predict"),
        "-emb", str(ANTIGEN_ONLY_DIR / "seq_prottrans.h5"),
        "--model", "model/NB/model.pt",
        "--threshold", "0.4"
    ]

    print(f"   Running: {' '.join(cmd[5:])}")

    result = subprocess.run(cmd, cwd=SURF2SPOT_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Prediction failed!")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print("✅ Prediction completed successfully!")
        return True

def check_results():
    """Check prediction results."""
    predict_dir = ANTIGEN_ONLY_DIR / "predict"

    if predict_dir.exists():
        csv_files = list(predict_dir.glob("*.csv"))
        pred_files = list(predict_dir.glob("*pred*.csv"))

        print(f"\n📊 Results:")
        print(f"   Total files in predict/: {len(list(predict_dir.iterdir()))}")
        print(f"   CSV files: {len(csv_files)}")
        print(f"   Prediction files: {len(pred_files)}")

        if pred_files:
            print("   Sample prediction files:")
            for f in pred_files[:3]:
                print(f"     - {f.name}")

        return len(pred_files) > 0
    else:
        print("❌ Predict directory not found")
        return False

def main():
    print("🔬 Antigen-Only Epitope Prediction Pipeline")
    print("=" * 60)

    # Step 1: Create filtered PLY files
    create_filtered_ply_files()

    # Step 2: Run prediction
    if run_prediction():
        # Step 3: Check results
        success = check_results()

        if success:
            print("\n🎉 Antigen-only prediction pipeline completed successfully!")
            print("   Ready for ROC-AUC evaluation")
        else:
            print("\n⚠️  Pipeline completed but no prediction files found")
    else:
        print("\n❌ Pipeline failed at prediction step")

if __name__ == "__main__":
    main()