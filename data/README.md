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

## Analysis scripts

`scripts/pct_non_epitope_surface_hist.py` histograms, per antigen in
`train_test_eval/all_epitopes.fasta`, what percent of its *surface* residues
are non-epitope (UPPERCASE). Surface accessibility comes from freesasa
(default Shrake-Rupley parameters, default classifier -- antigen chain(s)
only, antibody excluded) with a residue called "surface" at relative SASA
>= 0.20, the same cutoff used by `benchmarking/discotope3` and
`benchmarking/webtools/ispred4`.

## Opendde epitope-dist datasets (soft labels)

`scripts/make_opendde_dataset.py` builds the opendde data ablation from an
opendde `epitope_dist` export (default
`../../opendde_coreweave/data/epitope_dist`): 250 docked campaign antibodies
against human-secreted-protein antigens, each residue labelled with the
*fraction of antibodies contacting it* -- continuous, so it cannot ride in
the FASTA casing. Per flavour (`best`, `all`) it writes
`train_test_eval/opendde_<flavour>_epitopes.fasta` with a
`<fasta stem>_soft_labels.tsv` companion (targets keyed by full header), and
the same records appended to the champion dataset as
`allowed_species_homo_sapiens_min_resolution_10_plus_opendde_<flavour>_epitopes.fasta`
(champion records verbatim -- binary labels, original folds).

Antigens >=40% identical to any benchmark/eval-set antigen are dropped
(mmseqs2, the standing leakage threshold of `split_eval_clusters.py`);
the rest are 40%-identity fold-grouped (connected components, the
`cluster_fasta.py` fold-group tier) and LPT-balanced into the 5 CV groups.
Each source CIF is a full predicted antibody-antigen complex, copied verbatim
to `raw/all-structures-extracted/pdb_<uniprot>/pdb_<uniprot>.cif` -- the
pipeline reads only the antigen chain (always A so far) named in the header.

Standalone (not part of `data_prep.smk`), since the source is an external
export rather than SAbDab; re-run it after a new epitope_dist export or a
champion-dataset regeneration.
