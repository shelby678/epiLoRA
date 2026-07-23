RAW_TSV = config.get("raw_tsv", "raw/sabdab_summary_all.tsv")
STRUCTURES_DIR = config.get("structures_dir", "raw/all-structures-extracted")
RESULTS_DIR = config.get("results_dir", "results")
LOGS_DIR = config.get("logs_dir", "logs")
TTE_DIR = config.get("train_test_eval_dir", "train_test_eval")

rule all:
    input:
        f"{RESULTS_DIR}/eval.fasta",
        f"{TTE_DIR}/all_epitopes.fasta",
        f"{TTE_DIR}/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
        f"{TTE_DIR}/allowed_species_homo_sapiens_epitopes.fasta",
        f"{TTE_DIR}/min_resolution_3_epitopes.fasta",
        f"{TTE_DIR}/min_resolution_4_epitopes.fasta",
        f"{TTE_DIR}/min_num_clusters_2_epitopes.fasta",
        f"{TTE_DIR}/min_num_clusters_3_epitopes.fasta",
        f"{TTE_DIR}/min_num_clusters_4_epitopes.fasta"

rule rule_filter_tsv:
    input:
        RAW_TSV
    output:
        f"{RESULTS_DIR}/filtered_summary.tsv"
    log:
        f"{LOGS_DIR}/rule_filter_tsv.log"
    shell:
        "python3 scripts/filter_tsv.py {input} {output} {log}"

rule generate_fasta:
    input:
        tsv=f"{RESULTS_DIR}/filtered_summary.tsv"
    output:
        f"{RESULTS_DIR}/antigens.fasta"
    log:
        f"{LOGS_DIR}/generate_fasta.log"
    shell:
        "python3 scripts/generate_fasta.py {input.tsv} {STRUCTURES_DIR} {output} {log}"

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

# Clusters seqs @ 95% identity and assigns each cluster a random fold label (1.0,1.1,...,5.0,5.1).
# Does NOT merge epitope between member sequences -- keeps every member's epitope-cased seq aligned
# to its rep, so combine_epitopes can decide, per ablation, which members' epitopes to merge.
rule cluster_fasta:
    input:
        f"{RESULTS_DIR}/epitopes.fasta"
    output:
        f"{RESULTS_DIR}/clusters.fasta"
    log:
        f"{LOGS_DIR}/cluster_fasta.log"
    shell:
        "python3 scripts/cluster_fasta.py {input} {output} {log}"

# Preliminary, default-settings combine to derive the held-out eval set from -- this is
# NOT the training all_epitopes.fasta (that comes later, from filtered_clusters.fasta).
# Lands in RESULTS_DIR (not TTE_DIR) so its auto-derived "all_epitopes.fasta" name can't
# collide with the real training output of the same name.
rule combine_epitopes_prelim_all:
    input:
        f"{RESULTS_DIR}/clusters.fasta"
    output:
        f"{RESULTS_DIR}/all_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_prelim_all.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {RESULTS_DIR} {log}"

rule get_eval_set:
    input:
        f"{RESULTS_DIR}/all_epitopes.fasta"
    output:
        f"{RESULTS_DIR}/eval.fasta"
    log:
        f"{LOGS_DIR}/get_eval_set.log"
    shell:
        "python3 scripts/get_eval_set.py {input} {output} {log}"

# Drops any clusters.fasta member that's too similar to a held-out eval sequence, so eval
# antigens (or their near-duplicates) can't leak into the training/ablation fastas below.
rule filter_clusters_against_eval:
    input:
        clusters=f"{RESULTS_DIR}/clusters.fasta",
        eval=f"{RESULTS_DIR}/eval.fasta"
    output:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    log:
        f"{LOGS_DIR}/filter_clusters_against_eval.log"
    shell:
        "python3 scripts/filter_clusters_against_eval.py {input.clusters} {input.eval} {output} {log}"

# One rule invocation per ablation. Output filename is derived by combine_epitopes.py itself
# from whichever args are non-default; these must match exactly.
rule combine_epitopes_all:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/all_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_all.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log}"

rule combine_epitopes_species_homo_mus:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_allowed_species_homo_sapiens_mus_musculus.log"
    shell:
        """python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --allowed_species "homo sapiens|mus musculus" """

rule combine_epitopes_species_homo_only:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/allowed_species_homo_sapiens_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_allowed_species_homo_sapiens.log"
    shell:
        """python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --allowed_species "homo sapiens" """

rule combine_epitopes_min_res_3:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/min_resolution_3_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_min_resolution_3.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --min_resolution 3"

rule combine_epitopes_min_res_4:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/min_resolution_4_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_min_resolution_4.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --min_resolution 4"

rule combine_epitopes_min_clusters_2:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/min_num_clusters_2_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_min_num_clusters_2.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --min_num_clusters 2"

rule combine_epitopes_min_clusters_3:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/min_num_clusters_3_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_min_num_clusters_3.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --min_num_clusters 3"

rule combine_epitopes_min_clusters_4:
    input:
        f"{RESULTS_DIR}/filtered_clusters.fasta"
    output:
        f"{TTE_DIR}/min_num_clusters_4_epitopes.fasta"
    log:
        f"{LOGS_DIR}/combine_epitopes_min_num_clusters_4.log"
    shell:
        "python3 scripts/combine_epitopes.py {input} {TTE_DIR} {log} --min_num_clusters 4"