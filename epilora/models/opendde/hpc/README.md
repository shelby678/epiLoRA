# OpenDDE backbone benchmark on Slurm (32-way GPU)

Runs the full benchmark — frozen OpenDDE ab_ag Pairformer trunk (sequence-only)
+ MLP head, champion dataset, 5-fold CV, held-out final eval — as a
three-phase Slurm dependency chain, using 32 GPUs in parallel for the
expensive part (the per-antigen trunk precompute; ~3 days serial, ~2-3 h
sharded 32-way).

```
slurm_precompute.sh  #SBATCH --array=0-31%32   shared trunk-feature cache
        | afterok
slurm_train.sh       #SBATCH --array=1-5      weights/opendde_fold{1-5}.pt
        | afterok                               (+ .metrics.json per fold)
slurm_eval.sh        one job: per-fold + 5-fold ensemble eval_final on the
                     held-out 31-antigen set -> weights/opendde_benchmark_results.txt
```

## One-time setup

1. **Sync the repo** to the cluster (`/mnt/home/$USER/epiLoRA` by default),
   including `epilora/models/opendde/`, the `train.py`/`model.py`/
   `predict.py`/`data.py`/`eval_final.py` changes, and
   `configs/backbone_opendde.yaml`. The champion dataset + structures must
   be under `data/` as usual.

2. **Build + deploy the container** (from this directory, on the workstation
   with podman + enroot; same recipe as the campaign's b200 image plus
   pyyaml — see Dockerfile.b200's header for the driver story):

   ```bash
   podman build -t opendde-epilora:cu128-b200 -f Dockerfile.b200 .
   podman run --rm opendde-epilora:cu128-b200 bash -lc \
     'python -c "import torch, yaml, opendde, sklearn, Bio, biotite; print(torch.__version__)"'
   enroot import -o opendde-epilora-cu128-b200+v1.sqsh podman://opendde-epilora:cu128-b200
   # copy to the cluster's read-only container store:
   scp opendde-epilora-cu128-b200+v1.sqsh \
       <cluster>:/mnt/data/sferrier/containers/
   ```

   Not baked into the image (only ever needed at runtime): `wandb` (only for
   `--wandb` runs), `freesasa` (only for the `rsa` extra head feature).

3. **Verify the assets** the jobs read: `OPENDDE_ROOT_DIR` (default
   `/mnt/data/sferrier/opendde`) must contain `checkpoint/opendde_abag.pt`
   and `common/components.cif` + `common/components.cif.rdkit_mol.pkl`
   (already there from the campaign).

4. **Sanity-check one GPU** (driver injection, per the campaign's
   container README):

   ```bash
   srun --partition=b200 --gpus=1 \
       --container-image=/mnt/data/sferrier/containers/opendde-epilora-cu128-b200+v1.sqsh \
       --container-mounts=/mnt/home/$USER:/mnt/home/$USER,/mnt/data:/mnt/data \
       python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
   ```

## Run

```bash
bash submit_all.sh           # submits the chain, prints job ids
bash submit_all.sh status
tail -f /mnt/home/$USER/logs/odd_pc_*.out   # precompute progress
```

Outputs land in `$REPO/weights/`: `opendde_fold{1-5}.pt` +
`opendde_fold{1-5}.metrics.json` (per-fold val/test AUCs on the shared
benchmarks), `opendde_benchmark_results.txt` (held-out eval: per-fold +
ensemble ROC-AUC), `opendde_ensemble_eval.csv` (ensemble per-residue
predictions, comparable against DiscoTope's).

## Notes

- **Resume/redo**: every phase is idempotent — the precompute skips cached
  entries, trainings overwrite their checkpoints, the eval overwrites its
  results file. A failed/preempted phase can be resubmitted standalone:
  `sbatch slurm_train.sh` etc.
- **Sharding**: records are sorted by sequence length (descending) and dealt
  round-robin, so each shard gets an ~equal sum of L^2 (the trunk's cost
  unit). Change `NSHARDS` / the array size in `slurm_precompute.sh` for a
  different GPU count (the user-space default is 32 concurrent).
- **Cache location**: the trunk-feature cache is the coords cache dir next to
  `--structures` (`data/raw/all-structures-extracted_coords_cache/`), i.e.
  shared storage — that's what makes 32 concurrent shards and instant fold
  restarts safe. Deleting `opendde_trunk_*.npy` from it forces a recompute.
- **rtxp6000 partition**: use the campaign's baked-driver image recipe
  instead (opendde_coreweave/container/Dockerfile) + pyyaml — the b200 image
  carries no driver libs and will silently fall back to CPU there.
- A handful of antigens without a usable structure (unparseable file, absent
  chain, or a sequence that doesn't line up with its residues) are skipped
  (logged) in training and eval alike — the shared data gate every backbone
  trains through, same behavior as the local run.
