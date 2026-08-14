# Immune Escape Prediction with epiLoRA + UShER

A pipeline that builds a whole-genome SARS-CoV-2 phylogeny from **9.4 M
genomes**, identifies mutations under positive selection by counting tree
descendants, predicts B-cell epitopes on the spike structure with **epiLoRA**,
and tracks selection over **time**.

## Overview

```
metadata.tsv (9.4M rows, Nextclade substitutions)
      │
      ▼
 prepare_data.py ──► VCF + UPGMA starting tree (30k stratified subsample)
      │
      ▼
 UShER ──► mutation-annotated tree (MAT, parsimony-refined)
      │
      ├─► count_and_plot.py  ──► descendants_plot.png  (most→least descendants)
      │                       descendants.tsv
      │
      ├─► run_epilora_escape.py ──► escape_scores.tsv  (epitope × descendants)
      │                              spike_epitope_predictions.csv
      │
      └─► time_scaled.py ──► time_trajectories.png   (descendants per year)
                            time_growth_ratios.png   (fastest-growing)
                            time_descendants.tsv
```

## Why UShER (not IQ-TREE/RAxML/FastTree)?

With 9.4 M sequences, standard ML tree builders are infeasible. **UShER**
(Ultrafast Sample placement on Existing tRees) builds a parsimony-based
**mutation-annotated tree (MAT)** that scales to millions of genomes, annotates
every branch with its mutations, and supports **incremental placement** of new
samples — the same tool used to maintain the global SARS-CoV-2 phylogeny
(nextstrain/usher-merged).

## Pipeline

### 1. Prepare data (`scripts/prepare_data.py`)

- Loads `metadata.tsv` (Nextclade pre-computed substitutions — no need to
  re-parse the 282 GB FASTA).
- Filters: QC good, coverage > 0.9, valid date, non-empty substitutions.
- Stratified subsample by (pango-lineage × year) for diversity.
- Builds a VCF (one row per variant position) and a UPGMA starting tree from
  the Hamming-distance matrix.

```bash
python scripts/prepare_data.py --n-samples 30000 --outdir work
```

### 2. Build tree (`scripts/run_usher.sh`)

- Runs **UShER** to place all samples on the starting tree via maximum
  parsimony, producing a mutation-annotated tree (`.mat.pb`).
- Runs **matUtils** to export the tree as JSON (with mutations per node).

```bash
bash scripts/run_usher.sh work
```

### 3. Count descendants & plot (`scripts/count_and_plot.py`)

- Traverses the JSON tree; for each mutation on each branch, counts the
  number of descendant leaves below that branch.
- Mutations with many descendants were inherited widely → likely selected for.
- A mutation appearing on multiple branches (homoplasy) with high total
  descendants = even stronger signal.
- Plots a bar chart sorted most→least descendants, with known immune-escape
  mutations highlighted in red.

```bash
python scripts/count_and_plot.py --json work/tree.json --subsample work/subsample.tsv --outdir results --top 50
```

**Outputs:**
- `results/descendants_plot.png` — sorted bar chart
- `results/descendants.tsv` — full table
- `results/top_mutations.tsv` — top-N with amino-acid annotation

### 4. epiLoRA escape prediction (`scripts/run_epilora_escape.py`)

- Loads a trained epiLoRA checkpoint and predicts per-residue epitope
  probabilities on the SARS-CoV-2 spike structure (6VXX, chain A).
- Maps each nucleotide mutation to its spike amino-acid position.
- Scores each mutation: `escape_score = epitope_probability × log(descendants)`.
  A mutation at a high-epitope-probability residue with many tree descendants
  is a strong immune-escape candidate.

```bash
python scripts/run_epilora_escape.py \
    --pdb data/sars-cov2/6vxx.pdb --chain A \
    --weights /home/jovyan/work/epiLoRA/weights/ablation/all_fold1_esmif1.pt \
    --descendants results/descendants.tsv \
    --subsample work/subsample.tsv --outdir results
```

**Outputs:**
- `results/spike_epitope_predictions.csv` — per-residue epitope probabilities
- `results/escape_scores.tsv` — spike mutations ranked by escape score

### 5. Time-scaled analysis (`scripts/time_scaled.py`)

Two modes for tracking selection over time:

**Subset mode (fast, recommended):** reuses the full tree, but counts only
descendant leaves whose sample dates fall within each year. Shows how the
descendant count of each mutation changes over time — rising = selected for,
falling = purged/out-competed.

```bash
python scripts/time_scaled.py --mode subset --json work/tree.json --subsample work/subsample.tsv --outdir results
```

**Windowed mode (accurate, slower):** re-builds a separate tree for each year
with `--time-slice`, giving independent phylogenies per time window.

```bash
python scripts/time_scaled.py --mode windowed --years 2020 2021 2022 2023 2024 2025
```

**Outputs:**
- `results/time_trajectories.png` — descendant count per year for top mutations
- `results/time_growth_ratios.png` — fastest-growing mutations (last/first year)
- `results/time_descendants.tsv` — full mutation × year matrix

## Scaling with time

Three complementary strategies:

1. **Subset mode** (implemented): The full tree is built once; for each year we
   count only the leaves dated to that year. O(tree_size) per year, very fast.
   Ideal for monitoring: run it whenever new samples are added.

2. **UShER incremental placement**: The MAT is persistent (`tree.mat.pb`).
   When new sequences arrive, place them with `usher -i tree.mat.pb -v new.vcf`
   — no need to rebuild from scratch. This is how the live SARS-CoV-2 phylogeny
   is maintained. After placement, re-run `count_and_plot.py` to update
   descendant counts.

3. **Windowed rebuild**: For the most accurate time-scaled picture, build
   independent trees per time window (`--mode windowed`). Each year's tree
   captures the lineage structure as it was at that time, without hindsight
   from later samples.

## Requirements

- **conda env `phylo`**: `mamba create -n phylo -c bioconda -c conda-forge usher`
- **epiLoRA env**: `/home/jovyan/work/epiLoRA/epilora/env/` (ESM-IF1 + torch)
- Python packages: pandas, numpy, scipy, matplotlib, biopython

## Results

The current run used 30,000 stratified samples (5,295 pango lineages, 2019-2026):

| Output | Description |
|--------|-------------|
| `descendants_plot.png` | 28,483 mutations ranked by tree descendants |
| `escape_scores.tsv` | 3,263 spike mutations scored by epitope prob × descendants |
| `time_trajectories.png` | Top-30 mutation trajectories over 2019-2026 |
| `time_growth_ratios.png` | Fastest-growing mutations (2025/2020 ratio) |

Top immune-escape candidates include S:V445P, S:N440K, S:F486P, S:T19I — all in
or near the receptor-binding domain, consistent with known escape biology.
