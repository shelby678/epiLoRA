# eperase

In-silico epitope substitution scan for epiLoRA's ESM-IF1 ensemble.

- `substitution_scan.py` — swap each eval-set epitope residue to all 19
  alternatives (fixed backbone) and record the change in predicted epitope
  probability, pooled into a 20x20 (substitution x original) delta matrix.
- `antisymmetric_heatmap.py` — collapse that matrix into an antisymmetric,
  order-independent heatmap of which residues are more epitope-favoring
  than others.

Outputs land in `results/`.
