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
   model directly against the crystal structure → `results/capri_results.csv`,
   `results/capri_plot.png`. Independent of HADDOCK3's own caprieval, so it
   covers every dname.
7. **compute_rmsd** — pull HADDOCK3's own caprieval numbers
   (`11_caprieval/capri_ss.tsv`) → `results/rmsd_results.csv`,
   `results/rmsd_summary.csv`, as a cross-check. Only meaningful for dnames
   with `runs/{dname}/reference.pdb`.
8. **capri_acceptable** — recompute CAPRI quality per cluster model for
   epiLoRA-constrained vs vanilla runs, for antigens with 5/5 ab runs complete
   in both conditions → `results/capri_acceptable_counts.csv`,
   `results/capri_model_details.csv`.
9. **capri_figures** — acceptable-count bars, HADDOCK-score violins and the
   combined two-panel figure from those CSVs.
10. **runtime_boxplot** — per-run HADDOCK3 wall-clock runtime, vanilla vs
    epiLoRA, over matched runs → `results/runtime_by_run.csv`,
    `results/runtime_boxplot.png`.

All analysis CSVs and plots land in `results/`. Steps 6–7 (and their
`*_vanilla` twins) wait for every run to finish; steps 8–10 work on whatever
runs are complete and refresh as more land.

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
- `results/` — all analysis CSVs and plots (gitignored).
- `env/` — HADDOCK3 install used by `run_haddock` (gitignored).

## `scripts/`

**Wired into the Snakefile:**
`prepare_pdbs.py`, `prepare_reference.py`, `generate_restraints.py`,
`make_haddock_config.sh`, `capri_analysis.py`, `compute_rmsd.py`,
`plot_capri_acceptable.py`, `plot_capri_figures.py`,
`plot_runtime_boxplot.py`.

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
