#!/bin/bash
#
# Per-job sbatch script for run_ablation.py --slurm (one (row, fold) training
# job on a single GPU). Adapted from ../example_sbatch.sh: same .env /
# JOB_WORK_DIR / sync_job_dir() conventions, but single-GPU (no accelerate)
# and no checkpoint-resume support.
#
# Not meant to be run directly -- run_ablation.py submits it via
#   sbatch --job-name=... --export=ALL,FASTA=...,... ablation/slurm_job.sh
# Edit --partition/resources below for your cluster, or override per-call
# with run_ablation.py's --slurm-args (handy for a second cluster/machine).

# --- job ---
#SBATCH --job-name=epi_ablation
#SBATCH --partition=hpc-mid,hpc-low
#SBATCH --requeue

# --- resources ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00

# --- logs ---
#SBATCH --output=/mnt/home/%u/logs/%x_%A.out
#SBATCH --error=/mnt/home/%u/logs/%x_%A.err
#SBATCH --open-mode=append

# --- container ---
#SBATCH --container-image=/mnt/data/containers/deeplearning_v2026-05-26.sqsh
#SBATCH --container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp
#SBATCH --no-container-mount-home

# --- notifications ---
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user='slack:USER-ID' # TODO - fill with your slack UID to get notifications

set -euo pipefail

# per-job args, set by run_ablation.py via `sbatch --export=...`
: "${TRAIN_PYTHON:?}" "${TRAIN_PY:?}" "${FASTA:?}" "${STRUCTURES:?}" \
  "${FOLD:?}" "${OUT_NAME:?}" "${MAX_SECONDS:?}" "${SEED:?}"

# source env file
# if you've copied the example .env file, this will create the work dir
# (JOB_WORK_DIR), set object storage / wandb keys, and define sync_job_dir()
source /mnt/home/${SLURM_JOB_USER}/.env

# cd into the work dir (on local /tmp) so training artifacts land there
cd "$JOB_WORK_DIR"

# stash the config (if any) in the work dir so sync_job_dir uploads it with the run
CONFIG_ARGS=()
if [[ -n "${CONFIG_FILE:-}" ]]; then
    cp "$CONFIG_FILE" "$JOB_WORK_DIR/"
    CONFIG_ARGS=(--config "$CONFIG_FILE")
fi

WANDB_ARGS=()
if [[ "${WANDB:-0}" == "1" ]]; then
    # SLURM_JOB_ID is always set here (this *is* the Slurm job), so the run name
    # always gets a job-id suffix -- distinguishes reruns/requeues of the same job.
    RUN_NAME="${WANDB_RUN_NAME}${SLURM_JOB_ID:+_$SLURM_JOB_ID}${SLURM_RESTART_COUNT:+_r$SLURM_RESTART_COUNT}"
    WANDB_ARGS=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$RUN_NAME")
fi

# train
"$TRAIN_PYTHON" "$TRAIN_PY" \
    --fasta "$FASTA" --structures "$STRUCTURES" \
    --fold "$FOLD" --out "$JOB_WORK_DIR/$OUT_NAME" \
    --max-seconds "$MAX_SECONDS" --seed "$SEED" \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}

# transfer weights/metrics to object storage, then delete the files locally
# function defined in the example .env file
sync_job_dir
