#!/bin/bash
# OpenDDE-backbone benchmark, phase 3 of 3: evaluate the 5 fold checkpoints on
# the final held-out benchmark (eval_final.py -- the 31-antigen min-resolution-5
# set, distinct from the CV folds), per fold and as a 5-fold ensemble
# (averaged sigmoid), on the warm embedding cache. Everything is teed into
# <repo>/weights/opendde_benchmark_results.txt next to the checkpoints.

#SBATCH --job-name=odd_ev
#SBATCH --partition=b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00

#SBATCH --output=/mnt/home/%u/logs/%x_%A.out
#SBATCH --error=/mnt/home/%u/logs/%x_%A.err
#SBATCH --open-mode=append

#SBATCH --container-image=/mnt/data/sferrier/containers/opendde-epilora-cu128-b200+v1.sqsh
# NB: no %u here -- pyxis does no substitution in --container-mounts (that's
# a Slurm --output/--error-only feature), so an unexpanded %u reaches enroot
# as a literal path and the container fails to start.
#SBATCH --container-mounts=/mnt/home/sferrier:/mnt/home/sferrier,/mnt/data:/mnt/data
#SBATCH --no-container-mount-home

set -euo pipefail

REPO="${REPO:-/mnt/home/$USER/epiLoRA}"
STRUCTURES="${STRUCTURES:-$REPO/data/raw/all-structures-extracted}"
RESULTS="$REPO/weights/opendde_benchmark_results.txt"
export OPENDDE_ROOT_DIR="${OPENDDE_ROOT_DIR:-/mnt/data/sferrier/opendde}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO/epilora"

{
    echo "# OpenDDE-backbone benchmark — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# per-fold val/test AUCs: weights/opendde_fold{1-5}.metrics.json"
    for fold in 1 2 3 4 5; do
        echo
        echo "## eval_final fold $fold"
        python3 eval_final.py --structures "$STRUCTURES" \
            --weights "$REPO/weights/opendde_fold${fold}.pt"
    done
    echo
    echo "## eval_final 5-fold ensemble"
    python3 eval_final.py --structures "$STRUCTURES" \
        --weights "$REPO"/weights/opendde_fold{1,2,3,4,5}.pt \
        --out "$REPO/weights/opendde_ensemble_eval.csv"
    echo
    echo "# done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$RESULTS"
