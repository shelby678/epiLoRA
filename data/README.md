# Data

Training data is **not committed**. Put your files here (or point `train.py` at
them with `--fasta` / `--structures`).

## Expected layout

```
data/
  BEPIPRED.fasta                         labelled antigen sequences
  structures2/sabdab_dataset/
    <pdb_id>/structure/<pdb_id>.pdb      one PDB per antigen
```

## FASTA format

One record per antigen chain. The header is space-separated; the sequence
casing carries the label:

```
>4qci_A 4qci A 1
mnfprkleQEKLLNGWA...
```

- header fields: `<id>_<chain>`, `<pdb_id>`, `<chain>`, `<partition>`
- **UPPERCASE** residue = epitope, **lowercase** = non-epitope
- partition `EVAL` is always held out; `train.py --val <p>` holds out one more
  partition for early stopping

Only chains whose PDB is found under `structures2/...` (and whose residue count
matches the sequence) are used — ESM-IF1 requires backbone coordinates.

For inference you don't need any of this: `predict.py` reads the sequence and
coordinates directly from a single PDB file.
