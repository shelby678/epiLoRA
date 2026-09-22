#!/bin/bash
# OpenDDE-backbone benchmark, phase 1 of 3: trunk-feature precompute, sharded
# across a Slurm array (one GPU per task). Writes the shared embedding cache
# that the fold trainings and eval_final read; safe to resubmit/requeue
# (cached entries are skipped). Submitted by submit_all.sh.
#
# Edit the vars below for your cluster (or the #SBATCH headers directly).

#SBATCH --job-name=odd_pc
#SBATCH --partition=b200
#SBATCH --array=0-31%32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00
#SBATCH --requeue

#SBATCH --output=/mnt/home/%u/logs/%x_%A_%a.out
#SBATCH --error=/mnt/home/%u/logs/%x_%A_%a.err
#SBATCH --open-mode=append

#SBATCH --container-image=/mnt/data/sferrier/containers/opendde-epilora-cu128-b200+v1.sqsh
#SBATCH --container-mounts=/mnt/home/%u:/mnt/home/%u,/mnt/data:/mnt/data
#SBATCH --no-container-mount-home

set -euo pipefail

REPO="${REPO:-/mnt/home/$USER/epiLoRA}"            # the epiLoRA checkout
STRUCTURES="${STRUCTURES:-$REPO/data/raw/all-structures-extracted}"
# checkpoint/ + common/ (CCD assets) -- read-only is fine, they are only read
export OPENDDE_ROOT_DIR="${OPENDDE_ROOT_DIR:-/mnt/data/sferrier/opendde}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python3 "$REPO/epilora/models/opendde/hpc/precompute_shards.py" \
    --repo "$REPO" --structures "$STRUCTURES" \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS:-32}"
