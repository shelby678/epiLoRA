"""Shared per-residue prediction format + CV loader-with-headers for the
backbone-ensemble experiment.

Each backbone model (ESM3, ESM2, ESM-IF1) trains one model per CV fold and dumps
per-residue *test* predictions for that fold. Predictions across the 3 folds are
pooled into one file per model (every residue in partitions 1/2/3 is scored once,
by the fold in which its partition is the holdout).

File format (numpy .npz), parallel arrays of equal length:
    key   : str   "<header>|<residue_pos>"  (residue_pos = 0-based, BOS/EOS stripped)
    fold  : str   the holdout partition ("1"/"2"/"3")
    prob  : f32   sigmoid(logit) epitope probability
    label : i8    ground-truth epitope label (0/1)

`key` is identical across backbones for the same residue, so the ensemble step
just intersects keys and takes max() of prob over the chosen models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_preds(path: Path, records: list[tuple[str, str, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array([r[0] for r in records], dtype=object)
    fold = np.array([r[1] for r in records], dtype=object)
    prob = np.array([r[2] for r in records], dtype=np.float32)
    label = np.array([r[3] for r in records], dtype=np.int8)
    np.savez(path, key=keys, fold=fold, prob=prob, label=label)


def load_preds(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "key": d["key"].astype(str),
        "fold": d["fold"].astype(str),
        "prob": d["prob"].astype(np.float32),
        "label": d["label"].astype(np.int64),
    }


def cv_test_with_headers(fasta_path, structures_dir, test_partition, max_length=512):
    """Like train_struct.create_cv_datasets but returns the TEST split as a list of
    (header, token_ids, labels, coords) so predictions can be keyed by header.

    Returns (train_data, val_data, test_with_headers) where train/val are plain
    StructSamples (val_partition == test_partition, matching the CV convention)."""
    from prepare import load_combined_fasta_partitioned
    from train_struct import _load_coords_rsa

    by_part = load_combined_fasta_partitioned(fasta_path, exclude_partitions=frozenset({"EVAL"}))
    train, val, test_h = [], [], []
    for part_id, samples in by_part.items():
        for header, (token_ids, labels) in samples:
            if len(token_ids) > max_length:
                continue
            seq_len = len(token_ids) - 2
            coords, rsa = _load_coords_rsa(header, seq_len, structures_dir)
            if part_id == test_partition:
                test_h.append((header, token_ids, labels, coords))
                val.append((token_ids, labels, coords, rsa))   # holdout = val (early stop)
            else:
                train.append((token_ids, labels, coords, rsa))
    return train, val, test_h


def predict_venv_model(model, test_with_headers, device, fold, batch_size=1):
    """Run a .venv backbone model (ESM3/ESM2 — forward(input_ids, attention_mask=,
    structure_coords=)) over the test set, one sequence at a time, returning
    per-residue records (key, fold, prob, label) for labelled residues only."""
    import torch

    model.eval()
    records: list[tuple[str, str, float, int]] = []
    with torch.no_grad():
        for header, token_ids, labels, coords in test_with_headers:
            L = len(token_ids)
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            mask = torch.ones((1, L), dtype=torch.long, device=device)
            sc = torch.full((1, L, 3, 3), float("nan"), dtype=torch.float32, device=device)
            if coords is not None:
                sc[0, 1:L - 1] = coords.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                logits = model(ids, attention_mask=mask, structure_coords=sc)[0]  # (L,)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for pos in range(L - 2):  # residue index, BOS/EOS stripped
                lab = labels[pos + 1]
                if lab < 0:
                    continue
                records.append((f"{header}|{pos}", fold, float(probs[pos + 1]), int(lab)))
    return records
