"""Ensemble member: ESM-IF1 + LoRA + RYS. Self-contained; runs in the
discotope3_web env (fair-esm, py3.9):

    /home/sferrier/epitope_mapping/discotope3_web/env/bin/python run_ensemble_esmif1.py

Trains one ESM-IF1 + LoRA(+RYS) model per CV fold and dumps per-residue test
predictions to data/ensemble_preds/esmif1.npz, keyed identically to the ESM3/
ESM2 members (header|pos). Only structure-backed sequences get IF1 predictions.

Config: LoRA rank=4 on all 8 encoder layers + RYS(4,8) (replay top 4 layers).
Adapted from scripts/legacy/train_esmif1_lora.py.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent
BEPIPRED_FASTA = REPO / "data/BEPIPRED.fasta"
STRUCTURES_DIR = REPO / "data/structures2/sabdab_dataset"
OUT = REPO / "data/ensemble_preds/esmif1.npz"
DISCOTOPE_SRC = Path("/home/sferrier/epitope_mapping/discotope3_web/src")

CV_FOLDS = [{"holdout": "1", "train": ["2", "3", "4", "5"]},
            {"holdout": "2", "train": ["1", "3", "4", "5"]},
            {"holdout": "3", "train": ["1", "2", "4", "5"]}]

LR, LORA_RANK, LORA_ALPHA, LORA_LAYERS = 1e-4, 4, 8.0, 8
RYS_START, RYS_END = 4, 8
WARMUP_STEPS, MAX_SECONDS, PATIENCE, VAL_INTERVAL = 200, 1200, 5, 200


# ── FASTA / structure parsing (matches all other experiments' header keys) ──────
def parse_bepipred(path: Path) -> dict:
    by_part: dict = {}
    header, seq = None, []
    def add(h, s):
        parts = h.split()
        part = parts[2] if len(parts) >= 3 else "?"
        if part == "EVAL":
            return
        by_part.setdefault(part, []).append((h, s.upper(), [1 if c.isupper() else 0 for c in s]))
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                add(header, "".join(seq))
            header, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if header is not None:
        add(header, "".join(seq))
    return by_part


def parse_seq_id(header: str):
    parts = header.split()
    if len(parts) >= 2:
        return parts[0].split("_")[0].lower(), parts[1]
    return None


def load_coords(pdb_path: Path, chain_id: str, seq_len: int):
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
    try:
        f = pdb_io.PDBFile.read(str(pdb_path))
        st = pdb_io.get_structure(f, model=1)
    except Exception:
        return None
    st = st[(st.chain_id == chain_id) & struc.filter_amino_acids(st)]
    res_ids = struc.get_residues(st)[0]
    if len(res_ids) != seq_len:
        return None
    coords = np.full((seq_len, 3, 3), np.nan, dtype=np.float32)
    for ri, rid in enumerate(res_ids):
        ra = st[st.res_id == rid]
        for ai, an in enumerate(["N", "CA", "C"]):
            m = ra.atom_name == an
            if m.any():
                coords[ri, ai] = ra.coord[m][0]
    return coords


def load_samples(entries):
    out = []
    for header, aa_seq, labels in entries:
        parsed = parse_seq_id(header)
        coords = None
        if parsed:
            pdb_id, chain = parsed
            pp = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
            if pp.exists():
                coords = load_coords(pp, chain, len(aa_seq))
        out.append((header, aa_seq, labels, coords))
    return out


# ── LoRA + RYS ──────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.orig = orig
        for p in self.orig.parameters():
            p.requires_grad = False
        out_f, in_f = orig.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.orig(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    # Passthroughs so any code reading the wrapped layer's params still works.
    @property
    def weight(self):
        return self.orig.weight

    @property
    def bias(self):
        return self.orig.bias

    @property
    def in_features(self):
        return self.orig.in_features

    @property
    def out_features(self):
        return self.orig.out_features


def inject_lora(model, rank, alpha, n_layers):
    start = max(0, 8 - n_layers)
    for i in range(start, 8):
        layer = model.encoder.layers[i]
        # Force the manual q/k/v projection path so LoRA.forward is actually
        # invoked: the fused F.multi_head_attention_forward fast path uses
        # q_proj.weight directly and would bypass the adapter.
        layer.self_attn.enable_torch_version = False
        for attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(layer.self_attn, attr, LoRALinear(getattr(layer.self_attn, attr), rank, alpha))


def patch_encoder_rys(encoder, rys_start, rys_end):
    """Replay encoder.layers[rys_start:rys_end] a second time (RYS)."""
    def new_forward(self, coords, encoder_padding_mask, confidence, return_all_hiddens=False):
        x, encoder_embedding = self.forward_embedding(coords, encoder_padding_mask, confidence)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        layers = self.layers
        for i in range(rys_start):
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        for i in range(rys_start, rys_end):
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        for i in range(rys_start, rys_end):   # RYS replay
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        for i in range(rys_end, len(layers)):
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return {"encoder_out": [x], "encoder_padding_mask": [encoder_padding_mask],
                "encoder_embedding": [encoder_embedding], "encoder_states": []}

    encoder.forward = types.MethodType(new_forward, encoder)


class ESMIF1Model(nn.Module):
    def __init__(self, esm_model, alphabet, rank, alpha, n_lora_layers, dropout=0.1):
        super().__init__()
        self.esm, self.alpha, self.hidden = esm_model, alphabet, 512
        for p in self.esm.parameters():
            p.requires_grad = False
        inject_lora(self.esm, rank, alpha, n_lora_layers)
        if RYS_END > RYS_START:
            patch_encoder_rys(self.esm.encoder, RYS_START, RYS_END)
        self.head_ln = nn.LayerNorm(self.hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden, 1)

    def _encode(self, coords_batch, seq_batch):
        sys.path.insert(0, str(DISCOTOPE_SRC))
        from esm_util_custom import CoordBatchConverter
        bc = CoordBatchConverter(self.alpha)
        batch = [(c, None, s) for c, s in zip(coords_batch, seq_batch)]
        coords_t, confidence, _, _, padding_mask = bc(batch, device=DEVICE)
        self.esm.to(DEVICE)
        enc = self.esm.encoder.forward(coords_t, padding_mask, confidence, return_all_hiddens=False)
        return enc["encoder_out"][0].permute(1, 0, 2)  # (B, L, 512)

    def forward(self, coords_batch, seq_batch):
        hidden = self._encode(coords_batch, seq_batch)
        out = []
        for b in range(len(seq_batch)):
            L = len(seq_batch[b])
            h = self.head_drop(self.head_ln(hidden[b, 1:L + 1]))
            out.append(self.head(h).squeeze(-1))
        return out


@torch.no_grad()
def evaluate_auc(model, samples):
    model.eval()
    logits_all, labels_all = [], []
    for header, aa_seq, labels, coords in samples:
        if coords is None:
            continue
        try:
            lg = model([coords], [aa_seq])[0].cpu().numpy()
        except Exception:
            continue
        logits_all.append(lg)
        labels_all.append(labels)
    if not logits_all:
        return float("nan")
    y, s = np.concatenate(labels_all), np.concatenate(logits_all)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan")


def train_fold(model, train_samples, val_samples, max_seconds=MAX_SECONDS):
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, WARMUP_STEPS)))
    best_auc, best_state, no_improve, step, tl = -1.0, None, 0, 0, 0.0
    rng = np.random.default_rng(42)
    idxs = list(range(len(train_samples)))
    start = time.time()
    model.train(); model.to(DEVICE)
    while time.time() - start < max_seconds:
        rng.shuffle(idxs)
        stop = False
        for idx in idxs:
            if time.time() - start >= max_seconds:
                break
            header, aa_seq, labels, coords = train_samples[idx]
            if coords is None:
                continue
            try:
                logits = model([coords], [aa_seq])[0]
            except Exception as e:
                logger.debug(f"  skip {header}: {e}")
                continue
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.tensor(labels, dtype=torch.float32, device=DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
            step += 1
            tl = 0.9 * tl + 0.1 * loss.item()
            if step % VAL_INTERVAL == 0:
                va = evaluate_auc(model, val_samples); model.train()
                logger.info(f"  step={step} train_loss={tl:.4f} val_auc={va:.4f} {time.time()-start:.0f}s")
                if va > best_auc:
                    best_auc, no_improve = va, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                                  if any(k.startswith(p) for p in ("head", "esm.encoder.layers"))}
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        logger.info(f"  Early stop at step {step}"); stop = True; break
        if stop:
            break
    if best_state is not None:
        cur = model.state_dict()
        cur.update({k: v.to(DEVICE) for k, v in best_state.items() if k in cur})
        model.load_state_dict(cur)
    return {"steps": step, "val_auc": evaluate_auc(model, val_samples)}


@torch.no_grad()
def predict_records(model, samples, fold):
    """Per-residue (key, fold, prob, label) for structure-backed test samples."""
    model.eval()
    recs = []
    for header, aa_seq, labels, coords in samples:
        if coords is None:
            continue
        try:
            logits = model([coords], [aa_seq])[0].cpu().numpy()
        except Exception:
            continue
        probs = 1.0 / (1.0 + np.exp(-logits))
        for pos, (p, lab) in enumerate(zip(probs, labels)):
            recs.append((f"{header}|{pos}", fold, float(p), int(lab)))
    return recs


if __name__ == "__main__":
    import esm
    logger.info("Loading BEPIPRED + ESM-IF1 ...")
    by_part = parse_bepipred(BEPIPRED_FASTA)
    esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_model = esm_model.eval()

    all_records = []
    for fold in CV_FOLDS:
        holdout = fold["holdout"]
        train_entries = [e for k in fold["train"] for e in by_part.get(k, [])]
        test_entries = by_part.get(holdout, [])
        logger.info(f"\n=== IF1 fold holdout={holdout}: loading structures ===")
        train_samples = load_samples(train_entries)
        test_samples = load_samples(test_entries)
        n_struct = sum(1 for *_, c in test_samples if c is not None)
        logger.info(f"  train={len(train_samples)} test={len(test_samples)} ({n_struct} w/ struct)")

        model = ESMIF1Model(esm_model, alphabet, LORA_RANK, LORA_ALPHA, LORA_LAYERS, 0.1).to(DEVICE)
        t0 = time.time()
        res = train_fold(model, train_samples, test_samples, MAX_SECONDS)
        logger.info(f"  fold {holdout}: val_auc={res['val_auc']:.4f} {time.time()-t0:.0f}s")

        recs = predict_records(model, test_samples, holdout)
        all_records.extend(recs)
        logger.info(f"  fold {holdout}: {len(recs)} residue predictions")
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT,
             key=np.array([r[0] for r in all_records], dtype=object),
             fold=np.array([r[1] for r in all_records], dtype=object),
             prob=np.array([r[2] for r in all_records], dtype=np.float32),
             label=np.array([r[3] for r in all_records], dtype=np.int8))
    logger.info(f"\nSaved {len(all_records)} predictions to {OUT}")
