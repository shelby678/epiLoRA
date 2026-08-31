# figures

Snakemake pipeline producing up to three structures carrying epitope calls
in the B-factor column (for coloring by B-factor in PyMOL/ChimeraX):

1. query antigen with ground-truth epitope sites (by homology) marked
2. query antigen with epiLoRA-predicted epitope sites marked
3. conjoined ground truth: one reference antigen with every surviving
   antibody's Fv superposed onto it (overlaps OK)

Every run needs the fair-esm environment (`../epilora/env`, overridable via
`--config env_python=...`) and `mmseqs` on `PATH` for results 1 & 2; both are
assumed, not checked.

## 1 & 2. Annotate an arbitrary structure

```
snakemake --cores 4 --config input_pdb=/path/to/query.pdb
snakemake --cores 4 --config input_pdb=/path/to/query.cif chain=A
snakemake --cores 4 --config input_pdb=complex.cif "chain_groups=A|D;B|E|O;C|F|K"
```

Produces, in `outdir` (default `results/{stem}_{tag}`):

- `{tag}_groundtruth_bfactor.{ext}` — epitope-by-homology: unioned across
  every `train_fasta` hit ≥ `min_seq_id` (lowercase residues in the training
  fasta mark epitopes)
- `{tag}_prediction_bfactor.{ext}` — epiLoRA per-residue epitope probability

`chain` (or each spec in `chain_groups`) is one chain id, or `|`-separated
ids for a multi-chain antigen (SAbDab's `antigen_chain` convention).
`chain_groups` scores each antigen copy in a multi-copy file independently,
then merges them into ONE combined output per call.

Per chain group: mmseqs-search the extracted sequence against the training
corpus (`train_fasta`, → `groundtruth.csv` via union-across-hits) and the
champion model's training fasta (`champion_fasta`, → `fold_choice.json`: use
the single fold that never saw this antigen, else the 5-fold ensemble) →
`prediction.csv`. Then all groups' CSVs merge into the two output
structures via `5_write_bfactor_structure.py`.

## 3. Conjoined ground-truth structure

```
snakemake --cores 4 --config conjoined_tsv=ebola/ebola_summary.tsv
```

Filters the Ebola SAbDab complexes (`conjoined_tsv`: human-only antibody;
placeable on the 7kfe GP trimer reference without the antibody clashing
through it, which rejects cryptic epitopes and non-spike antigens;
non-redundant by binding pose; antigen ≈ the reference antigen) and writes
`results/conjoined_groundtruth/combined_on_pdb_00006qd8.cif`: one GP trimer
with every surviving antibody's Fv superposed onto it (overlaps OK), antigen
B-factors marking the union of all antibody contacts, plus a `_mapping.tsv`
of antibody chain ids. Runs `6_finalize_groundtruth_structs.py` (which imports
`chain_residues`/`chain_sequence` from `../data/scripts/structures.py`)
against `structures_dir` (default `../data/raw/all-structures-extracted`).

`--config input_pdb=... conjoined_tsv=...` builds all three results in one
run. `ebola/ebola_summary.tsv` is the filtered SAbDab summary
(`../data/raw/sabdab_summary_all.tsv` restricted to Ebola protein antigens
with annotated H/L/antigen chains).

### Config

| key | default | meaning |
|---|---|---|
| `input_pdb` | — | `.pdb` or `.cif` structure to annotate (results 1 & 2) |
| `chain` | first chain in the file | chain(s) for a single group |
| `chain_groups` | — | `;`-separated chain specs, one per antigen copy |
| `outdir` | `results/{stem}_{tag}` | where results 1 & 2 land |
| `train_fasta` | `../data/results/epitopes.fasta` | ground-truth-transfer corpus (one row per raw antigen instance — the cluster-collapsed `train_test_eval/*_epitopes.fasta` files can miss a near-exact match that wasn't elected its cluster's representative) |
| `champion_fasta` | `../data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta` | champion model's training fasta, for fold selection |
| `champion_weights_glob` | `../weights/ablation/allowed_species_homo_sapiens_fold?.pt` | the 5 per-fold champion checkpoints |
| `min_seq_id` | `0.95` | identity cutoff for ground-truth transfer |
| `min_aln_len` | `20` | minimum aligned residues (absolute floor, not a coverage-ratio filter — short-subsequence matches should still count) |
| `fold_avoid_min_seq_id` | `0.80` | more lenient cutoff used only for fold selection |
| `conjoined_tsv` | — | SAbDab summary TSV to build result 3 from (omit to skip) |
| `structures_dir` | `../data/raw/all-structures-extracted` | raw SAbDab structures (`<pdb>/<pdb>_sabdab.cif`) |
| `ref_pdb` / `ref_chains` | `pdb_00007kfe` / `A,B,C,D,E,F` | reference trimer antibodies are placed against |
| `ref_antigen_pdb` | `pdb_00006qd8` | PDB whose antigen all survivors are superposed onto |
| `conjoined_outdir` | `results/conjoined_groundtruth` | where result 3 lands |
| `env_python` | `../epilora/env/bin/python3` | python used inside the rules |

## Known limitations

- Each chain group is scored as a single concatenated pseudo-chain, so
  contacts depending on interaction *between* two groups scored separately
  aren't captured within one group's prediction.
- `chain_groups` must be told explicitly which chains belong together.
- Result 3's filters are tuned for Ebola GP (GP1/GP2 protomer pairing, GP2
  requirement to reject sGP/NP); `6_finalize_groundtruth_structs.py
  --single_chain_protomer` covers single-chain antigens like a SARS-CoV-2
  spike, but that mode isn't exposed through the Snakefile.

## `scripts/`

Numbered by pipeline step: `0_prepare_corpus_db.py`,
`1_homology_search.py`, `2_groundtruth_from_homology.py`,
`3_pick_fold.py`, `4_predict_epitope.py`, `5_write_bfactor_structure.py`
(steps 0-1 run once per corpus, 2-5 per query chain group), and
`6_finalize_groundtruth_structs.py` (the conjoined structure, an independent
branch). `epitope_pipeline_common.py` is shared helpers, not a step.
