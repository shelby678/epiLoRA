# epiLoRA

Per-residue **B-cell epitope prediction** on antigen structures with the
best-performing model from the epiLoRA study: **ESM-IF1 + LoRA + RYS**.

ESM-IF1 is an inverse-folding model, so it reads protein **backbone geometry**:
the frozen ESM-IF1 GVP-Transformer encoder is adapted with **LoRA** on its
attention projections, its top encoder layers are replayed once (**RYS** =
"Repeat Yourself"), and a small linear head scores each residue. Inputs are a
**PDB structure + chain**, not a bare sequence.

## Layout

```
model.py            ESM-IF1 + LoRA + RYS + head (the model)
data.py             training-data loading (labelled FASTA + mmCIF structures)
train.py            train one fold of one ablation, save a checkpoint
predict.py          run a trained checkpoint on a PDB -> per-residue scores
ablation/           run_ablation.py -- 5-fold CV sweep across data ablations
requirements.txt    pinned dependencies (Python 3.9)
weights/            trained checkpoints go here (not committed) — see weights/README.md
data/               train/test/eval set generation from raw datasets (sabdab)
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

Data prep (`data/data_prep.smk`, see [`data/README.md`](data/README.md)) turns raw
SAbDab structures into a set of ablation FASTAs under `data/train_test_eval/`,
each carrying a 5-fold CV label (`i.j`) per record. Train one fold of one
ablation with:

```bash
python train.py \
    --fasta data/train_test_eval/all_epitopes.fasta \
    --structures data/raw/all-structures-extracted \
    --fold 1 \
    --out weights/all_epitopes_fold1.pt
```

Trains on every record in `--fasta` whose fold != `--fold`. Early stopping and
test-set reporting always use a **fixed shared benchmark** (`--eval-fastas`,
default the homo-sapiens and homo-sapiens+mus-musculus ablations) rather than
`--fasta`'s own fold split — so every ablation's model is judged on the same
held-out human epitopes, making cross-ablation comparisons apples-to-apples.
Saves the trainable weights (LoRA adapters + head, ~0.5 MB) plus config and
metrics to `--out`. The frozen ESM-IF1 backbone is not stored — it is
re-downloaded on load.

To compare ablations, run the full sweep (8 datasets x 5 folds) and get a
ranked summary:

```bash
python ablation/run_ablation.py --max-seconds 3600
```

### Extra head features

The per-residue head normally reads the ESM-IF1 embedding alone. It can also be
given scalar per-residue features to weigh alongside it — **relative solvent
accessibility** (freesasa, computed on the antigen chain(s) with the antibody
excluded) and **antigen length** (log-scaled, same value for every residue):

```bash
python train.py --config configs/feats_rsa_length.yaml \
    --fold 1 --out weights/feats_rsa_length_fold1.pt
```

That config is a single line — `extra_feats: [rsa, length]` (see
`TrainConfig`); write it yourself if it isn't there, since `configs/` is
machine-local. The
features are appended to the head's input, so the champion's `Linear(512,1)`
head becomes `Linear(514,1)` and nothing else about the recipe changes. This
needs `freesasa` installed, and works with `backbone: esmif1`.

Each checkpoint records which features its head expects, so `predict.py`,
`predict_ensemble.py` and `eval_final.py` recompute them from the structure
automatically — checkpoints trained without them keep working untouched. RSA is
cached next to the backbone coordinates (`*_coords_cache/*.rsa.npy`).
