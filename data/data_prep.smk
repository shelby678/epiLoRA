RAW_CSV = config.get("raw_csv", "raw/sabdab_summary_all.csv")
RAW_TSV = config.get("raw_tsv", "raw/sabdab_summary_all.tsv")
STRUCTURES_DIR = config.get("structures_dir", "raw/all-structures-extracted")
RESULTS_DIR = config.get("results_dir", "results")
LOGS_DIR = config.get("logs_dir", "logs")
TTE_DIR = config.get("train_test_eval_dir", "train_test_eval")
EVAL_DIR = f"{TTE_DIR}/eval"
EVAL_CUTOFF_DATE = config.get("eval_cutoff_date", "2024-12-13")  # ISO date; a cluster's oldest member must be newer than this to be an eval candidate

rule all:
    input:
        f"{EVAL_DIR}/min_resolution_5_epitopes.fasta",
        f"{EVAL_DIR}/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta",
        f"{TTE_DIR}/all_epitopes.fasta",
        f"{TTE_DIR}/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
        f"{TTE_DIR}/allowed_species_homo_sapiens_epitopes.fasta",
        f"{TTE_DIR}/min_resolution_3_epitopes.fasta",
        f"{TTE_DIR}/min_resolution_4_epitopes.fasta",
        f"{TTE_DIR}/min_resolution_5_epitopes.fasta",
        f"{TTE_DIR}/min_resolution_10_epitopes.fasta",
        f"{TTE_DIR}/min_resolution_15_epitopes.fasta",
        f"{TTE_DIR}/allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta"

"""
Convert the raw SAbDab summary csv to tsv. Some fields (antigen_name,
authors, ...) contain literal commas inside quoted values, so this is a
proper quote-aware csv->tsv conversion, not a naive comma->tab replace.
"""
rule rule_csv_to_tsv:
    input:
        RAW_CSV
    output:
        RAW_TSV
    log:
        f"{LOGS_DIR}/rule_csv_to_tsv.log"
    shell:
        "python3 scripts/csv_to_tsv.py {input} {output} {log}"

"""
Filter the sabdab tsv according to following constraints:
- non-empty H-chain
- non-empty L-chain
- non-empty anticen chain
- antigen_type includes protien
"""
rule rule_filter_tsv:
    input:
        RAW_TSV
    output:
        f"{RESULTS_DIR}/filtered_summary.tsv"
    log:
        f"{LOGS_DIR}/rule_filter_tsv.log"
    shell:
        "python3 scripts/filter_tsv.py {input} {output} {log}"

"""
Return one fasta with all antigen sequences by reading from antigen chain(s) in PDB file
Restrict sequence length of the antigen to be between 60 and 1300 amino acids
"""
rule generate_fasta:
    input:
        tsv=f"{RESULTS_DIR}/filtered_summary.tsv"
    output:
        f"{RESULTS_DIR}/antigens.fasta"
    log:
        f"{LOGS_DIR}/generate_fasta.log"
    shell:
        "python3 scripts/generate_fasta.py {input.tsv} {STRUCTURES_DIR} {output} {log}"

"""
Output fasta with epitope residues marked as lower case (non-epitope = upper case)
Epitopes defined as residues with some heavy atom within 4A of a heavy atom on the immunoglobulin
"""
rule get_epitopes:
    input:
        fasta=f"{RESULTS_DIR}/antigens.fasta",
        tsv=f"{RESULTS_DIR}/filtered_summary.tsv"
    output:
        f"{RESULTS_DIR}/epitopes.fasta"
    log:
        f"{LOGS_DIR}/get_epitopes.log"
    shell:
        "python3 scripts/get_epitopes.py {input.fasta} {input.tsv} {STRUCTURES_DIR} {output} {log}"

"""
Cluster seqs at 95% sequence identity, output fasta organized by cluster.
Clustered sequences are aligned and keep their epitope markers (lowercase).

This is prepartion for data ablation, in which  we filter the clusters according to the restraint
we're testing and combine epitope markers into one representative sequence, i.e. if this residue is marked as epitope
in some seq in the cluster, mark it as epitope in the rep.

Fold labels are NOT drawn independently per cluster: cluster_fasta.py also clusters the 95%
representatives again at a much looser 40% identity, purely to decide which of the 5 CV groups
(the `i` in a `i.j` fold label) a cluster is confined to. Two 95%-clusters that are themselves
distinct epitope entries, but are near-duplicates of each other at 40% identity, always end up
in the same CV group and so won't be split across train/eval for any CV run

    >>CLUSTER {rep_instance} {fold_label}
    >{instance} {date} {resolution} {antigen_chains} {heavy_species} {light_species}
    {seq aligned to the cluster's shared frame, '-' where this member has no residue at that column}
    >{instance2} ...
    {seq2...}
    >>CLUSTER ...
"""
rule cluster_fasta:
    input:
        f"{RESULTS_DIR}/epitopes.fasta"
    output:
        f"{RESULTS_DIR}/clusters.fasta"
    log:
        f"{LOGS_DIR}/cluster_fasta.log"
    shell:
        "python3 scripts/cluster_fasta.py {input} {output} {log}"

"""
Split clusters.fasta into an eval set and a train set by deposit date (EVAL_CUTOFF_DATE): a
cluster is an eval candidate if its oldest member is newer than the cutoff, else it starts
out in the train set. Any eval candidate with a member >=40% similar to a train-cluster
member is reclaimed into the train set (not discarded) instead. Growing the train set this
way can turn a previously-safe eval cluster newly unsafe, so the reclaim + re-check (mmseqs
easy-search) repeats until a pass finds nothing left to move. 

Thus the eval set is leakage free of structures deposited prior to EVAL_CUTOFF_DATE
"""
rule split_eval_clusters:
    input:
        f"{RESULTS_DIR}/clusters.fasta"
    output:
        eval=f"{EVAL_DIR}/eval_clusters.fasta",
        train=f"{EVAL_DIR}/train_clusters.fasta"
    log:
        f"{LOGS_DIR}/split_eval_clusters.log"
    shell:
        "python3 scripts/split_eval_clusters.py {input} {output.eval} {output.train} {log} {EVAL_CUTOFF_DATE}"

rule combine_epitopes_eval_all_species:
    input:
        f"{EVAL_DIR}/eval_clusters.fasta"
    output:
        f"{EVAL_DIR}/min_resolution_5_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_eval_all_species.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {EVAL_DIR} {log} --min_resolution 5"

rule combine_epitopes_eval_homo_sapiens:
    input:
        f"{EVAL_DIR}/eval_clusters.fasta"
    output:
        f"{EVAL_DIR}/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_eval_homo_sapiens.log"
    shell:
        """python3 scripts/combine_epitopes.py {input} {EVAL_DIR} {log} --min_resolution 5 --allowed_species "homo sapiens" """

# One rule invocation per ablation. Output filename is derived by combine_epitopes.py itself
# from whichever args are non-default; these must match exactly.
rule combine_epitopes_all:
    input:
        f"{EVAL_DIR}/train_clusters.fasta"
    output:
        f"{TTE_DIR}/all_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_all.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log}"

rule combine_epitopes_species_homo_mus:
    input:
        f"{EVAL_DIR}/train_clusters.fasta"
    output:
        f"{TTE_DIR}/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_allowed_species_homo_sapiens_mus_musculus.log"
    shell:
        """python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --allowed_species "homo sapiens|mus musculus" """

rule combine_epitopes_species_homo_only:
    input:
        f"{EVAL_DIR}/train_clusters.fasta"
    output:
        f"{TTE_DIR}/allowed_species_homo_sapiens_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_allowed_species_homo_sapiens.log"
    shell:
        """python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --allowed_species "homo sapiens" """

# --- min_resolution 3 / 4 / 5 / 10 / 15 ablations ---
#
# combine_epitopes.py is the only place resolution filtering ever happens (filter_tsv.py
# doesn't filter by resolution at all), and every ablation reads the same train_clusters.fasta
# -- so each ceiling is just a --min_resolution value away, no separate re-clustering needed.
for _res in (3, 4, 5, 10, 15):
    rule:
        name: f"combine_epitopes_min_res{_res}"
        input:
            f"{EVAL_DIR}/train_clusters.fasta"
        output:
            f"{TTE_DIR}/min_resolution_{_res}_epitopes.fasta"
        log:
            f"{LOGS_DIR}/combine_epitopes_min_resolution_{_res}.log"
        params:
            res=_res
        shell:
            "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --min_resolution {params.res}"

# --- new pipeline default: human-only antibodies + <=10A resolution ---
# combine the two best-performing single-axis ablations
rule combine_epitopes_default_recipe:
    input:
        f"{EVAL_DIR}/train_clusters.fasta"
    output:
        f"{TTE_DIR}/allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_default_recipe.log"
    shell:
        """python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --allowed_species "homo sapiens" --min_resolution 10"""
