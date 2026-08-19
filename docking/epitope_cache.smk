"""
One-time prep pipeline that builds cache/epitope/{stem}.csv (per-antigen
epiLoRA epitope probabilities), the file generate_restraints.py (main
Snakefile) intersects with each antigen's surface to pick passive residues.

Kept out of the main Snakefile because it spans four different
environments:
  - stage_antigens, build_epitope_cache, build_epitope_cache_ensemble,
    surface_epitope_overlap: plain stdlib python (/opt/conda/bin/python).
  - dump_residue_keys, predict_ensemble: the epilora torch+esm env
    (../epilora/env/bin/python), since this pipeline's python can't import
    fair-esm.
  - run_discotope3: ../../benchmarking/discotope3_env/bin/python (DiscoTope-3.0's
    100-model XGBoost ensemble), for build_epitope_cache_ensemble's other input.
  - surface_epitope_overlap also shells out to this repo's own
    env/bin/haddock3-restraints.

cache/epitope_ensemble/{stem}.csv (epiLoRA rank-normalized 50/50 with
DiscoTope-3.0, per benchmarking/ensemble_with_discotope.py's finding that
this blend beats either model alone) is only built for the predicted (`ag`,
not `ag_gt`) antigen stems -- that's the only conditioning source the main
Snakefile's docking pipeline uses.

stage_antigens also reaches outside this repo, into the already-completed
vanilla HADDOCK pipeline at ~/work/ab_MD/haddocking2, to reuse its antigen
PDBs byte-for-byte.

Bootstrap (first run only): scratch/antigens/ doesn't exist yet, and the
per-antigen rules below discover their {stem} wildcards by globbing it at
parse time -- so a fresh checkout needs two invocations:

    snakemake -s epitope_cache.smk --cores 1 scratch/antigens
    snakemake -s epitope_cache.smk --cores N
"""
import glob
from pathlib import Path

ROOT_DIR = Path.cwd()

EPILORA_DIR = Path.home() / "work/epiLoRA/epilora"
EPILORA_PY  = str(EPILORA_DIR / "env" / "bin" / "python")

BENCHMARKING_DIR = Path.home() / "work/epiLoRA/benchmarking"
DISCOTOPE_PY     = str(BENCHMARKING_DIR / "discotope3_env" / "bin" / "python")
DISCOTOPE_DIR    = BENCHMARKING_DIR / "discotope3" / "discotope3"
DISCOTOPE_MODELS = BENCHMARKING_DIR / "discotope3" / "models"

STEMS = sorted(Path(p).stem for p in glob.glob("scratch/antigens/*.pdb"))
AG_STEMS = sorted(s for s in STEMS if s.endswith("_ag"))  # predicted antigen only, not _ag_gt


rule all:
    input:
        expand("cache/epitope/{stem}.csv", stem=STEMS),
        expand("cache/epitope_ensemble/{stem}.csv", stem=AG_STEMS),
        "scratch/surface_epitope_overlap.csv",


rule stage_antigens:
    """Symlink one representative merged antigen PDB per (dname, variant)
    from the vanilla HADDOCK pipeline into scratch/antigens/."""
    output:
        directory("scratch/antigens"),
    log:
        "logs/epitope_cache/stage_antigens.log"
    shell:
        "/opt/conda/bin/python scripts/stage_antigens.py > {log} 2>&1"


rule dump_residue_keys:
    """Dump (pdb_stem, idx, chain, res_id, ins_code) for every staged antigen
    in fair-esm's own residue-walk order."""
    input:
        pdbs = expand("scratch/antigens/{stem}.pdb", stem=STEMS),
    output:
        "scratch/residue_keys.csv",
    log:
        "logs/epitope_cache/dump_residue_keys.log"
    shell:
        "{EPILORA_PY} scripts/dump_residue_keys.py"
        " --pdb {input.pdbs} --chain A --out {output}"
        " > {log} 2>&1"


rule predict_ensemble:
    """Run the epiLoRA champion ensemble on every staged antigen in one
    process, so the 5 fold weights load once instead of once per antigen."""
    input:
        pdbs = expand("scratch/antigens/{stem}.pdb", stem=STEMS),
    output:
        expand("scratch/epitope_preds/{stem}_A.csv", stem=STEMS),
    log:
        "logs/epitope_cache/predict_ensemble.log"
    shell:
        "{EPILORA_PY} {EPILORA_DIR}/predict_ensemble.py"
        " --pdb {input.pdbs} --chain A --out-dir scratch/epitope_preds"
        " > {log} 2>&1"


rule build_epitope_cache:
    """Join residue_keys.csv with each antigen's epitope_preds into
    cache/epitope/{stem}.csv (res_id,prob)."""
    input:
        res_keys = "scratch/residue_keys.csv",
        preds = expand("scratch/epitope_preds/{stem}_A.csv", stem=STEMS),
    output:
        expand("cache/epitope/{stem}.csv", stem=STEMS),
    log:
        "logs/epitope_cache/build_epitope_cache.log"
    shell:
        "/opt/conda/bin/python scripts/build_epitope_cache.py > {log} 2>&1"


rule stage_antigens_ag_only:
    """Symlink just the predicted-antigen (`ag`, not `ag_gt`) PDBs into their
    own directory, since DiscoTope-3.0 predicts on a whole --pdb_dir at once
    and the ensemble cache is only needed for `ag`."""
    input:
        pdbs = expand("scratch/antigens/{stem}.pdb", stem=AG_STEMS),
    output:
        directory("scratch/antigens_ag"),
    log:
        "logs/epitope_cache/stage_antigens_ag_only.log"
    shell:
        "mkdir -p {output} && "
        "for f in {input.pdbs}; do "
        "ln -sf \"$(realpath \"$f\")\" \"{output}/$(basename \"$f\")\"; "
        "done > {log} 2>&1"


rule run_discotope3:
    """Run DiscoTope-3.0 (100-model XGBoost ensemble) on the predicted
    antigen structures. --struc_type alphafold keeps each residue's real
    per-residue confidence (B-factor) as a feature instead of flattening it
    to 100 -- these are predicted, not crystal, structures, and --pdb_dir is
    documented as being for AF2 PDBs specifically."""
    input:
        "scratch/antigens_ag",
    output:
        # main.py nests its real output under {out_dir}/{basename(pdb_dir)}/
        expand("scratch/discotope_out/antigens_ag/{stem}_A_discotope3.csv", stem=AG_STEMS),
    log:
        "logs/epitope_cache/run_discotope3.log"
    params:
        pdb_dir = lambda wc, input: str((ROOT_DIR / input[0]).resolve()),
        out_dir = str(ROOT_DIR / "scratch" / "discotope_out"),
        log_abs = str(ROOT_DIR / "logs" / "epitope_cache" / "run_discotope3.log"),
    shell:
        "mkdir -p {params.out_dir} $(dirname {params.log_abs}) && "
        "cd {DISCOTOPE_DIR} && "
        "{DISCOTOPE_PY} main.py"
        " --pdb_dir {params.pdb_dir}"
        " --struc_type alphafold"
        " --out_dir {params.out_dir}"
        " --models_dir {DISCOTOPE_MODELS}"
        " > {params.log_abs} 2>&1"


rule build_epitope_cache_ensemble:
    """Rank-normalize DiscoTope-3.0's raw score, average it 50/50 with
    epiLoRA's own probability (per benchmarking/ensemble_with_discotope.py,
    this blend beats either model alone: AUC 0.7892 vs 0.7734 epiLoRA-only /
    0.7774 DiscoTope-3.0-only), into cache/epitope_ensemble/{stem}.csv
    (res_id,prob) -- same format as cache/epitope/{stem}.csv, so
    generate_restraints.py needs no changes to consume it."""
    input:
        epilora = expand("cache/epitope/{stem}.csv", stem=AG_STEMS),
        disco   = expand("scratch/discotope_out/antigens_ag/{stem}_A_discotope3.csv", stem=AG_STEMS),
    output:
        expand("cache/epitope_ensemble/{stem}.csv", stem=AG_STEMS),
    log:
        "logs/epitope_cache/build_epitope_cache_ensemble.log"
    shell:
        "/opt/conda/bin/python scripts/build_epitope_cache_ensemble.py > {log} 2>&1"


rule surface_epitope_overlap:
    """QC report: what fraction of each antigen's surface residues clear the
    epitope-probability threshold generate_restraints.py uses."""
    input:
        res_keys = "scratch/residue_keys.csv",
        preds = expand("scratch/epitope_preds/{stem}_A.csv", stem=STEMS),
        pdbs  = expand("scratch/antigens/{stem}.pdb", stem=STEMS),
    output:
        "scratch/surface_epitope_overlap.csv",
    log:
        "logs/epitope_cache/surface_epitope_overlap.log"
    shell:
        "/opt/conda/bin/python scripts/surface_epitope_overlap.py"
        " --threshold 0.20 --out-csv {output}"
        " > {log} 2>&1"
