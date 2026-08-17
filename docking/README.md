# docking

Snakemake pipeline that HADDOCK3-docks each antibody model against its
epiLoRA-constrained antigen surface, then scores every run with CAPRI
metrics.

## Pipeline (`Snakefile`)

For each `(dname, ab, ag)` combination discovered under `input_pdbs/{dname}/`:

1. **prepare_pdbs** — merge the antibody into chain B, antigen into chain A,
   strip non-protein.
2. **prepare_reference** — build a caprieval-ready reference complex from the
   experimental structure (only for dnames with a crystal in
   `input_pdbs/selected_32_pdbs/`).
3. **generate_restraints** — CDR-based ambiguous + VH/VL unambiguous
   restraints; antigen passive residues are surface ∩ epiLoRA-epitope
   (`cache/epitope/{dname}_{ag}.csv`, prob > 0.20).
4. **haddock_config** — render `template.cfg` into a per-pair `run.toml`.
5. **run_haddock** — run HADDOCK3 (rigidbody → flexref → emref → clustering).
6. **capri_analysis** — recompute lRMSD/iRMSD/fnat/DockQ for every cluster
   model directly against the crystal structure → `capri_results.csv`,
   `capri_plot.png`. Independent of HADDOCK3's own caprieval, so it covers
   every dname.
7. **compute_rmsd** — pull HADDOCK3's own caprieval numbers
   (`11_caprieval/capri_ss.tsv`) → `rmsd_results.csv`, `rmsd_summary.csv`, as
   a cross-check. Only meaningful for dnames with `runs/{dname}/reference.pdb`.

Run with:

```
snakemake -n            # dry run
snakemake --cores N
```

## Epitope cache prep (`epitope_cache.smk`)

`generate_restraints.py` needs `cache/epitope/{dname}_{ag}.csv` (per-antigen
epiLoRA epitope probabilities) already on disk, since computing them needs
the `epilora` torch+esm env, which this pipeline's python can't import. Build
that cache with a separate Snakefile that spans three environments (plain
stdlib, the epilora env, and this repo's own HADDOCK3 env):

```
snakemake -s epitope_cache.smk --cores 1 scratch/antigens   # first run only
snakemake -s epitope_cache.smk --cores N
```

Rules: **stage_antigens** (symlink antigen PDBs in from the vanilla HADDOCK
pipeline at `~/work/ab_MD/haddocking2`) → **dump_residue_keys** (fair-esm's
residue walk, epilora env) → **predict_ensemble** (epiLoRA champion ensemble,
epilora env, in `../epilora/`) → **build_epitope_cache** (join into
`cache/epitope/{stem}.csv`) → **surface_epitope_overlap** (QC: surface ∩
epitope-threshold overlap per antigen, via `env/bin/haddock3-restraints`).

## Directory layout

- `input_pdbs/` — symlink to the shared antibody/antigen PDBs (gitignored).
- `cache/epitope/` — precomputed epiLoRA epitope-probability CSVs consumed by
  `generate_restraints.py` (gitignored; rebuilt by `epitope_cache.smk`).
- `runs/{dname}/{ab}_vs_{ag}/` — per-pair working dir and HADDOCK3 output
  (gitignored).
- `env/` — HADDOCK3 install used by `run_haddock` (gitignored).

## `scripts/`

**Wired into the Snakefile:**
`prepare_pdbs.py`, `prepare_reference.py`, `generate_restraints.py`,
`make_haddock_config.sh`, `capri_analysis.py`, `compute_rmsd.py`.

**Wired into `epitope_cache.smk`:**
`stage_antigens.py`, `dump_residue_keys.py`, `build_epitope_cache.py`,
`surface_epitope_overlap.py` (this one also computes surface residues
in-memory via `haddock3-restraints calc_accessibility` — no separate
surface-calc script or intermediate file).

**Post-hoc analysis, run manually after the pipeline completes** (not
Snakefile rules — they compare against a separate vanilla-HADDOCK pipeline in
`~/work/ab_MD/haddocking2`, or launch extra HADDOCK3 scoring runs of their
own):
`score_reference.py`, `compare_vanilla_vs_epitope.py`, `ab_apo_gt_rmsd.py`,
`plot_by_antigen.py`, `plot_ab_rmsd_vs_acceptable.py`,
`plot_size_vs_acceptable.py`, `plot_score_vs_lrmsd_by_antigen.py`.
