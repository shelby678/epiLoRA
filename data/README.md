# Data prep

Snakemake pipeline (`data_prep.smk`) that turns raw SAbDab structures into
epitope-labeled antigen FASTAs for training and evaluation.

Run with:

```
snakemake -s data_prep.smk --cores <N>
```

## Pipeline

```
raw/sabdab_summary_all.tsv
  -> filter_tsv          drop rows missing chains/resolution, non-protein antigens
  -> generate_fasta      extract antigen sequences from CIF structures
  -> get_epitopes        mark epitope residues (lowercase) by 4A contact with the antibody
  -> cluster_fasta       cluster @ 95% identity (mmseqs2), assign each cluster a fold label
  -> combine_epitopes (prelim, defaults) -> get_eval_set   carve out results/eval.fasta
  -> filter_clusters_against_eval                          scrub eval leakage from clusters
  -> combine_epitopes (once per ablation)                  -> train_test_eval/*_epitopes.fasta
```

Each cluster's epitope calls come from `mmseqs2 result2msa`, aligned back to a
representative sequence; `combine_epitopes.py` then merges member epitope
calls onto that backbone, subject to the ablation's `--allowed_species` /
`--min_resolution` / `--min_num_clusters` filters.

Output FASTA header: `>instance date resolution antigen_chains heavy_species light_species n=<qualifying members> <fold_label>`.
Sequence casing carries the label: lowercase = epitope residue, UPPERCASE = non-epitope.

## No leakage into eval.fasta

`results/eval.fasta` is a **temporal** holdout: candidates newer than a fixed
date cutoff (2024-12-13). Leakage is prevented in both directions, each via
mmseqs2 sequence-identity search (>=40% identity = same underlying epitope):

- **train -> eval**: `get_eval_set.py` drops any newer candidate that is
  >=40% identical to an *older* (pre-cutoff) sequence, so eval never contains
  a near-duplicate of something abundant in the training pool.
- **eval -> train**: `filter_clusters_against_eval.py` drops any cluster
  member that is >=40% identical to a sequence that made it into
  `eval.fasta`, so a near-duplicate of an eval antigen can't leak back into
  the ablation training FASTAs built afterward. A cluster's representative
  can be dropped this way; `combine_epitopes.py` re-elects the qualifying
  member with the fewest MSA gaps as the new backbone.

## Fold label scheme (`1.0`, `1.1`, ... `5.0`, `5.1`)

`cluster_fasta.py` assigns each cluster one label `{i}.{j}` (seeded random
choice, reproducible):

- `i` (1-5): which of the 5 cross-validation holdout groups the cluster
  belongs to.
- `j`: role within that holdout — `0` = eval, `1` = test.

So for CV fold `i`, clusters labeled `i.0` are its eval set, `i.1` are its
test set, and everything else is training data for that fold.
