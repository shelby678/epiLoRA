# figures

Snakemake pipeline that annotates an arbitrary input structure (not
necessarily one of ours, not necessarily bound to an antibody) with two
independent epitope calls, written into the B-factor column so they can be
loaded straight into PyMOL/ChimeraX and colored by B-factor:

- **ground truth by homology** — does this antigen (or something close to
  it) already have a known epitope in the training data?
- **epiLoRA's prediction** — using whichever champion checkpoint, if any,
  is guaranteed not to have trained on something too similar to this query.

`ebola/`, `scripts/filter_ebola_tsv.py`, `scripts/mark_bfactors.py` and
`scripts/msa_ebola.py` are an earlier, Ebola-specific one-off analysis (its
ground truth comes from direct antibody-contact distances within the same
SAbDab entry, not homology) and aren't part of this Snakefile.

## Usage

```
snakemake --cores 4 --config input_pdb=/path/to/query.pdb
snakemake --cores 4 --config input_pdb=/path/to/query.cif chain=A
snakemake --cores 4 --config input_pdb=complex.cif "chain_groups=A|D;B|E|O;C|F|K"
```

`chain` (or each spec inside `chain_groups`) accepts a single chain id, or
`|`-separated ids for a multi-chain antigen (e.g. `A|D`, concatenated in
order) — same convention SAbDab's own `antigen_chain` column uses.
`chain_groups` is for a file holding *several separate* antigen copies
(e.g. three independent Fv+antigen pairs in one cryo-EM asymmetric unit):
each group is scored independently, then merged into ONE combined output
covering every group's chains — not one file per group.

Every run needs the fair-esm environment (`../epilora/env`) and `mmseqs`
on `PATH`; both are assumed, not checked.

### Config

| key | default | meaning |
|---|---|---|
| `input_pdb` | *(required)* | `.pdb` or `.cif` structure to annotate |
| `chain` | first chain in the file | chain(s) for a single group |
| `chain_groups` | — | `;`-separated chain specs, one per independently-scored antigen copy |
| `outdir` | `results/{stem}_{tag}` | where output lands |
| `train_fasta` | `../data/results/epitopes.fasta` | ground-truth-transfer corpus (one row per raw antigen instance — see below for why not `all_epitopes.fasta`) |
| `champion_fasta` | `../data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta` | champion model's own training fasta, for fold selection |
| `champion_weights_glob` | `../weights/ablation/allowed_species_homo_sapiens_fold?.pt` | the 5 per-fold champion checkpoints |
| `member_fasta` | `../data/train_test_eval/eval/train_clusters.fasta` | per-structure (not per-cluster) corpus, for `matches_members.csv` |
| `min_seq_id` | `0.95` | identity cutoff for ground-truth transfer and the matches list |
| `min_aln_len` | `20` | minimum aligned residues (absolute floor, not a length-ratio coverage filter — see below) |
| `fold_avoid_min_seq_id` | `0.80` | separate, more lenient cutoff used only for fold selection |

## Pipeline

Per chain group:

1. **homology_search** (×3: `train`, `champion`, `members` corpora) —
   mmseqs2-search the group's extracted sequence against each corpus.
   Reports the single best hit (`homology_{corpus}.json`), every
   above-threshold hit ranked best-first (`matches_{corpus}.csv`), and
   every hit's full alignment mapping (`all_hits_{corpus}.json`).
2. **groundtruth_from_homology** — unions epitope calls (lowercase in the
   training fasta) across *every* `train`-corpus hit ≥ `min_seq_id` →
   `groundtruth.csv`.
3. **pick_fold** — if any `champion`-corpus hit is ≥ `fold_avoid_min_seq_id`,
   use the single checkpoint whose training split excluded that cluster
   (fold `i`, where `i` is the cluster's fold-group label — every *other*
   fold trained on it); otherwise use the full 5-fold ensemble mean →
   `fold_choice.json`.
4. **predict_epitope** — run the checkpoint(s) from step 3 → `prediction.csv`.

Then, across all groups:

5. **combine_groundtruth_structure** / **combine_prediction_structure** —
   merge every group's CSV into one output structure each, covering every
   group's chains:
   `{outdir}/{tag}_groundtruth_bfactor.{ext}`, `..._prediction_bfactor.{ext}`.

## Why two different corpora, two different thresholds

- **`train_fasta` defaults to `data/results/epitopes.fasta`, not
  `all_epitopes.fasta`.** The `train_test_eval/*_epitopes.fasta` files keep
  only ONE sequence per 95%-identity cluster (whichever `combine_epitopes.py`
  elected as that cluster's representative). A query can be a near-exact
  match to a real SAbDab structure and still miss the search entirely if
  that particular structure wasn't the one elected — `data/results/
  epitopes.fasta` (one row per raw, pre-clustering antigen instance) doesn't
  have that blind spot.
- **`fold_avoid_min_seq_id` (0.80) is lower than `min_seq_id` (0.95).**
  Ground-truth transfer needs real confidence before asserting a label.
  Fold avoidance is cheap insurance — even a "somewhat similar" training
  cluster is reason enough to route around the one fold that saw it, so
  this threshold errs lenient.
- **No mmseqs `-c`/`--cov-mode` length-ratio filter, anywhere.** Those modes
  (including mode 5, "short seq. needs to be at least x% of the *other*
  seq. length" in the installed mmseqs version) reject exactly the case we
  want to allow: a query that's a short subsequence of a much longer
  training antigen, or vice versa. `min_aln_len` is an absolute residue
  floor instead, just to keep a coincidental few-residue high-identity
  stretch from counting as a hit.

## Known limitations

- Each chain group is scored as a single concatenated pseudo-chain (matching
  how the training data itself was built), so contacts that depend on
  interaction *between* two groups scored separately aren't captured within
  one group's prediction.
- `chain_groups` must be told explicitly which chains belong together —
  there's no automatic detection of "these N chains are one antigen copy."

## `scripts/`

**Wired into the Snakefile:** `homology_search.py` (also extracts the query
sequence — folded in from a separate script since every other stage already
takes `--pdb`/`--chain` directly), `prepare_corpus_db.py` (stages a corpus
for mmseqs; auto-detects flat vs. `cluster_fasta.py` block format),
`groundtruth_from_homology.py`, `pick_fold.py`, `predict_epitope.py`,
`write_bfactor_structure.py` (writes the final B-factor structure; also
handles merging multiple chain groups into one file). `epitope_pipeline_common.py`
is shared helpers, not its own rule.

**Not part of this Snakefile** (the earlier Ebola one-off, see above):
`filter_ebola_tsv.py`, `mark_bfactors.py`, `msa_ebola.py`.
