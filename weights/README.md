# Weights

Trained model checkpoints go here. **Weights are not committed to git** (only
this README is).

`predict.py` and `train.py` default to `weights/epilora_if1.pt`.

## Getting the checkpoint

- **Download** the released checkpoint and save it as `weights/epilora_if1.pt`,
  **or**
- **Train your own** with `python train.py --out weights/epilora_if1.pt`
  (see the top-level README).

A checkpoint is a small (~a few MB) dict:

```python
{"config": {...}, "trainable_state": {...}, "val_auc": 0.83}
```

It stores only the trainable parts (LoRA adapters, the RYS-replayed encoder
layers, and the head). The frozen ESM-IF1 backbone is downloaded automatically
by `fair-esm` the first time you load the model, so it is not part of this file.
