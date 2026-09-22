#!/bin/bash
# OpenDDE-backbone benchmark, phase 2 of 3: train the 5 CV folds of the
# champion dataset, one fold per array task (all on the warm embedding cache
# from phase 1). Each fold is exactly the train.py run the local/serial
# benchmark would do: --config configs/backbone_opendde.yaml, seed 42,
# default eval fastas (the shared benchmark), 4h budget with early stopping.
# Checkpoints + .metrics.json land in <repo>/weights/.

#SBATCH --job-name=odd_tr
#SBATCH --partition=b200
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --requeue

#SBATCH --output=/mnt/home/%u/logs/%x_%A_%a.out
#SBATCH --error=/mnt/home/%u/logs/%x_%A_%a.err
#SBATCH --open-mode=append

#SBATCH --container-image=/mnt/data/sferrier/containers/opendde-epilora-cu128-b200+v1.sqsh
#SBATCH --container-mounts=/mnt/home/%u:/mnt/home/%u,/mnt/data:/mnt/data
#SBATCH --no-container-mount-home

set -euo pipefail

REPO="${REPO:-/mnt/home/$USER/epiLoRA}"
STRUCTURES="${STRUCTURES:-$REPO/data/raw/all-structures-extracted}"
FASTA="${FASTA:-$REPO/data/train_test_eval/allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta}"
export OPENDDE_ROOT_DIR="${OPENDDE_ROOT_DIR:-/mnt/data/sferrier/opendde}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO/epilora"
exec python3 train.py \
    --fasta "$FASTA" --structures "$STRUCTURES" \
    --config configs/backbone_opendde.yaml \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --out "$REPO/weights/opendde_fold${SLURM_ARRAY_TASK_ID}.pt"
