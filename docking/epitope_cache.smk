"""
One-time prep pipeline that builds cache/epitope/{stem}.csv (per-antigen
epiLoRA epitope probabilities), the file generate_restraints.py (main
Snakefile) intersects with each antigen's surface to pick passive residues.

Kept out of the main Snakefile because it spans three different
environments:
  - stage_antigens, build_epitope_cache, surface_epitope_overlap: plain
    stdlib python (/opt/conda/bin/python).
  - dump_residue_keys, predict_ensemble: the epilora torch+esm env
    (../epilora/env/bin/python), since this pipeline's python can't import
    fair-esm.
  - surface_epitope_overlap also shells out to this repo's own
    env/bin/haddock3-restraints.

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

EPILORA_DIR = Path.home() / "work/epiLoRA/epilora"
EPILORA_PY  = str(EPILORA_DIR / "env" / "bin" / "python")

STEMS = sorted(Path(p).stem for p in glob.glob("scratch/antigens/*.pdb"))


rule all:
    input:
        expand("cache/epitope/{stem}.csv", stem=STEMS),
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
