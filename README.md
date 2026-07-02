# epiLoRA

Per-residue **B-cell epitope prediction** on antigen structures with the
best-performing model from the epiLoRA study: **ESM-IF1 + LoRA + RYS**.

- **5-fold ROC-AUC: 0.824 ± 0.064** (pooled 0.833) — ahead of an ESM3 LoRA
  model (~0.71), ESM2 (~0.67), and a DiscoTope-style XGBoost recipe (~0.75).

ESM-IF1 is an inverse-folding model, so it reads protein **backbone geometry**:
the frozen ESM-IF1 GVP-Transformer encoder is adapted with **LoRA** on its
attention projections, its top encoder layers are replayed once (**RYS** =
"Repeat Yourself"), and a small linear head scores each residue. Inputs are a
**PDB structure + chain**, not a bare sequence.

## Layout

```
model.py            ESM-IF1 + LoRA + RYS + head (the model)
data.py             training-data loading (labelled FASTA + PDB structures)
train.py            train the model, save a checkpoint
predict.py          run a trained checkpoint on a PDB -> per-residue scores
requirements.txt    pinned dependencies (Python 3.9)
weights/            trained checkpoints go here (not committed) — see weights/README.md
data/               your training data goes here (not committed) — see data/README.md
```

## Install

Requires **Python 3.9** and the fair-esm inverse-folding stack (torch-geometric
et al.). ESM-IF1's frozen backbone (~140 MB) downloads automatically on first
use.

```bash
python -m venv env && source env/bin/activate
pip install torch==2.8.0
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html   # match your torch/CUDA
pip install -r requirements.txt
```

A GPU is recommended for training; prediction runs fine on CPU.

## Weights

Trained checkpoints are **not committed**. Put one at `weights/epilora_if1.pt`
(the default path for both scripts):

- **download** the released checkpoint into `weights/`, **or**
- **train your own** (below).

See [`weights/README.md`](weights/README.md) for the checkpoint format.

## Predict

```bash
python predict.py --pdb antigen.pdb --chain A
# writes a CSV too:
python predict.py --pdb antigen.pdb --chain A --out scores.csv
```

Output is one row per residue: position, amino acid, epitope probability, and a
binary call at `--threshold` (default 0.5). If `--chain` is omitted the first
chain is used.

## Train

Put your labelled data under `data/` (format in [`data/README.md`](data/README.md)),
then:

```bash
python train.py \
    --fasta data/BEPIPRED.fasta \
    --structures data/structures2/sabdab_dataset \
    --out weights/epilora_if1.pt
```

Trains on every non-`EVAL` partition, holds out one partition (`--val`, default
`5`) for early stopping, and saves the trainable weights (LoRA adapters + head,
~0.5 MB) plus config to `--out`. The frozen ESM-IF1 backbone is not stored — it
is re-downloaded on load.
