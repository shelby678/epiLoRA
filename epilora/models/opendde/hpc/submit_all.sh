#!/bin/bash
# Submit the full OpenDDE-backbone benchmark (champion dataset, 5-fold CV) as
# a dependency chain:
#
#   precompute (32-way GPU array, ~2-3h)  ->  train folds 1-5 (GPU array)
#                                          ->  final eval (held-out set,
#                                             per fold + 5-fold ensemble)
#
# Everything resumes: cached trunk features are skipped, so any phase can be
# resubmitted standalone (see the slurm_*.sh files) if a job fails.
#
# Prereqs (see README.md): repo synced to $REPO with data/, the sqsh in
# /mnt/data/sferrier/containers/, OPENDDE_ROOT_DIR assets present.
#
# Usage:  bash submit_all.sh          # submit the chain
#         bash submit_all.sh status  # show the queue

set -euo pipefail
cd "$(dirname "$BASH_SOURCE")"

case "${1:-submit}" in
status)
    squeue -u "$USER" -o "%.10i %.12j %.8T %.10L %.6D %R" | head -40
    exit 0
    ;;
esac

PC=$(sbatch --parsable slurm_precompute.sh)
TR=$(sbatch --parsable --dependency=afterok:"$PC" slurm_train.sh)
EV=$(sbatch --parsable --dependency=afterok:"$TR" slurm_eval.sh)
echo "submitted:"
echo "  precompute array : $PC"
echo "  train folds 1-5   : $TR (after $PC)"
echo "  final eval        : $EV (after $TR)"
echo
echo "logs:     /mnt/home/$USER/logs/odd_{pc,tr,ev}_*.out"
echo "results:  \$REPO/weights/opendde_fold{1-5}.{pt,metrics.json},"
echo "          \$REPO/weights/opendde_benchmark_results.txt"
