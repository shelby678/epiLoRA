#!/usr/bin/env python3
"""
Complete the Surf2Spot antigen-only pipeline after surface generation.
"""

import os
import subprocess
import time
from pathlib import Path

# Paths
SURF2SPOT_DIR = Path("/home/sferrier/epitope_mapping/Surf2Spot")
ANTIGEN_ONLY_DIR = SURF2SPOT_DIR / "test_sabdab_antigen_only"
MICROMAMBA_PATH = "/home/sferrier/miniforge3/micromamba"
SURF2SPOT_ENV = "/home/sferrier/miniforge3/envs/surf2spot"

def count_files(directory, extension):
    """Count files with given extension."""
    try:
        return len(list(directory.glob(f"*.{extension}")))
    except:
        return 0

def run_with_micromamba(env_path: str, cmd: list, cwd: Path):
    """Run command with micromamba environment."""
    full_cmd = [MICROMAMBA_PATH, "run", "-p", env_path] + cmd
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(full_cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def main():
    print("🔬 Completing Surf2Spot Antigen-Only Pipeline")
    print("=" * 60)

    # Wait for surface generation to complete
    print("📊 Monitoring surface generation progress...")
    target_pdb_count = count_files(ANTIGEN_ONLY_DIR, "pdb")
    print(f"   Total PDB files: {target_pdb_count}")

    while True:
        ply_count = count_files(ANTIGEN_ONLY_DIR, "ply")
        print(f"   PLY files generated: {ply_count}/{target_pdb_count}")

        if ply_count >= target_pdb_count * 0.95:  # 95% complete
            print("✅ Surface generation nearly complete!")
            break
        elif ply_count >= 50:  # Run on partial data for testing
            print("🧪 Running prediction on partial data...")
            break

        time.sleep(30)

    # Run atom feature engineering
    print("\n🧬 Running atom feature engineering...")
    result = run_with_micromamba(
        SURF2SPOT_ENV,
        ["python", "-c",
         "from Surf2Spot.run import atom_feature_engineering; "
         f"atom_feature_engineering('{ANTIGEN_ONLY_DIR}')"],
        SURF2SPOT_DIR
    )

    if result.returncode != 0:
        print("❌ Atom feature engineering failed!")
        return

    # Run surface partitioning
    print("\n🎯 Running surface partitioning...")
    result = run_with_micromamba(
        SURF2SPOT_ENV,
        ["python", "-c",
         "from Surf2Spot.run import NB_surfpart; "
         f"NB_surfpart('{ANTIGEN_ONLY_DIR}', '{ANTIGEN_ONLY_DIR}/chainsaw.tsv', 400)"],
        SURF2SPOT_DIR
    )

    if result.returncode != 0:
        print("❌ Surface partitioning failed!")
        return

    print("\n🎉 Pipeline steps completed!")
    print("   Check predict directory for CSV files")

    # Count prediction files
    predict_dir = ANTIGEN_ONLY_DIR / "predict"
    if predict_dir.exists():
        csv_count = count_files(predict_dir, "csv")
        print(f"   Generated {csv_count} prediction CSV files")

if __name__ == "__main__":
    main()