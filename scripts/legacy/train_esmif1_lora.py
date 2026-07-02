"""ESM-IF1 + LoRA experiments on BEPIPRED.

Run with the discotope3_web environment:
    /home/sferrier/epitope_mapping/discotope3_web/env/bin/python train_esmif1_lora.py

Experiment 1 (exp1-lora-head):
    ESM-IF1 frozen backbone + LoRA on encoder transformer layers + linear head
    → BCE loss on epitope labels → ROC-AUC

Experiment 2 (exp2-lora-xgb):
    Same LoRA fine-tuning as exp1, then extract fine-tuned embeddings
    → XGBoost ensemble (100 models, DiscoTope-3.0 hyperparams) → ROC-AUC

Both use the same 3-fold CV as all other experiments:
    fold 1: train=2+3+4+5, holdout=1
    fold 2: train=1+3+4+5, holdout=2
    fold 3: train=1+2+4+5, holdout=3
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

AUTOPROT_DIR   = Path(__file__).parent
BEPIPRED_FASTA = AUTOPROT_DIR / "data/BEPIPRED.fasta"
STRUCTURES_DIR = AUTOPROT_DIR / "data/structures2/sabdab_dataset"
RESULTS_TSV    = AUTOPROT_DIR / "results.tsv"
LORA_CACHE_DIR = AUTOPROT_DIR / "data/esmif1_lora_embed_cache"

CV_FOLDS = [
    {"holdout": "1", "train": ["2", "3", "4", "5"]},
    {"holdout": "2", "train": ["1", "3", "4", "5"]},
    {"holdout": "3", "train": ["1", "2", "4", "5"]},
]

LR           = 1e-4
LORA_RANK    = 4
LORA_ALPHA   = 8.0
LORA_LAYERS  = 8          # all 8 encoder transformer layers
WARMUP_STEPS = 200
MAX_SECONDS  = 1200
PATIENCE     = 5
VAL_INTERVAL = 200

try:
    COMMIT = subprocess.check_output(
        ["git", "-C", str(AUTOPROT_DIR), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"


# ── FASTA parsing ──────────────────────────────────────────────────────────────

def parse_bepipred(path: Path) -> dict[str, list[tuple[str, str, list[int]]]]:
    """Parse BEPIPRED.fasta. Returns dict partition → list of (header, AA_seq, labels).

    Epitope (uppercase) = 1, non-epitope (lowercase) = 0. EVAL partition excluded.
    """
    by_part: dict[str, list] = {}
    header, seq = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                _add_entry(by_part, header, "".join(seq))
            header = line[1:].strip()
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        _add_entry(by_part, header, "".join(seq))
    return by_part


def _add_entry(by_part, header, seq):
    parts = header.split()
    partition = parts[2] if len(parts) >= 3 else "?"
    if partition == "EVAL":
        return
    labels = [1 if c.isupper() else 0 for c in seq]
    aa_seq = seq.upper()
    by_part.setdefault(partition, []).append((header, aa_seq, labels))


def parse_seq_id(header: str):
    parts = header.split()
    if len(parts) >= 2:
        pdb_id = parts[0].split("_")[0].lower()
        chain  = parts[1]
        return pdb_id, chain
    return None


# ── Structure loading ──────────────────────────────────────────────────────────

def load_coords(pdb_path: Path, chain_id: str, seq_len: int) -> np.ndarray | None:
    """Return (L, 3, 3) N/CA/C backbone coords, or None on failure."""
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure as struc
    try:
        f = pdb_io.PDBFile.read(str(pdb_path))
        struct = pdb_io.get_structure(f, model=1)
    except Exception:
        return None
    struct = struct[(struct.chain_id == chain_id) & struc.filter_amino_acids(struct)]
    res_ids = struc.get_residues(struct)[0]
    if len(res_ids) != seq_len:
        return None
    coords = np.full((seq_len, 3, 3), np.nan, dtype=np.float32)
    for ri, rid in enumerate(res_ids):
        res_atoms = struct[struct.res_id == rid]
        for ai, aname in enumerate(["N", "CA", "C"]):
            mask = res_atoms.atom_name == aname
            if mask.any():
                coords[ri, ai] = res_atoms.coord[mask][0]
    return coords


# ── LoRA implementation ────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank trainable adaptation."""

    def __init__(self, orig: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_f, out_f = orig.in_features, orig.out_features
        self.orig   = orig
        self.rank   = rank
        self.scale  = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        # lora_B initialised to zero → identity at start

    @property
    def bias(self):
        return self.orig.bias

    @property
    def weight(self):
        return self.orig.weight

    @property
    def in_features(self):
        return self.orig.in_features

    @property
    def out_features(self):
        return self.orig.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orig(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(model: nn.Module, rank: int, alpha: float, n_layers: int) -> int:
    """Inject LoRA into q/k/v/out_proj of the last n_layers encoder transformer layers.

    Returns number of trainable parameters added.
    """
    layer_start = max(0, 8 - n_layers)  # ESM-IF1 has 8 encoder transformer layers
    n_params = 0
    for i in range(layer_start, 8):
        layer = model.encoder.layers[i]
        for attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            orig = getattr(layer.self_attn, attr)
            lora = LoRALinear(orig, rank, alpha)
            setattr(layer.self_attn, attr, lora)
            n_params += lora.lora_A.numel() + lora.lora_B.numel()
    return n_params


# ── Model: ESM-IF1 + LoRA + linear head ───────────────────────────────────────

class ESMIF1LoRAModel(nn.Module):
    def __init__(self, esm_model, alphabet, rank: int, alpha: float,
                 n_lora_layers: int, dropout: float = 0.1):
        super().__init__()
        self.esm    = esm_model
        self.alpha  = alphabet
        self.hidden = 512

        # Freeze all ESM-IF1 parameters
        for p in self.esm.parameters():
            p.requires_grad = False

        # Inject LoRA
        n_lora_params = inject_lora(self.esm, rank, alpha, n_lora_layers)
        logger.info(f"LoRA params added: {n_lora_params:,}")

        # Head
        self.head_ln   = nn.LayerNorm(self.hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head      = nn.Linear(self.hidden, 1)

    def forward(self, coords_batch, seq_batch: list[str]) -> list[torch.Tensor]:
        """Run encoder on a list of (coords, seq) pairs. Returns list of (L,) logit tensors."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "discotope3_web/src"))
        from esm_util_custom import CoordBatchConverter

        batch_converter = CoordBatchConverter(self.alpha)
        batch = [(c, None, s) for c, s in zip(coords_batch, seq_batch)]
        coords_t, confidence, _, _, padding_mask = batch_converter(batch, device=DEVICE)
        self.esm.to(DEVICE)

        encoder_out = self.esm.encoder.forward(
            coords_t, padding_mask, confidence, return_all_hiddens=False
        )
        # encoder_out["encoder_out"][0]: (L, B, 512)
        hidden = encoder_out["encoder_out"][0].permute(1, 0, 2)  # (B, L, 512)

        results = []
        for b in range(len(seq_batch)):
            L = len(seq_batch[b])
            h = hidden[b, 1:L+1]           # strip BOS/EOS
            h = self.head_drop(self.head_ln(h))
            logits = self.head(h).squeeze(-1)  # (L,)
            results.append(logits)
        return results

    def get_embeddings(self, coords_batch, seq_batch: list[str]) -> list[np.ndarray]:
        """Extract LoRA-fine-tuned embeddings (L, 512) without head. No grad."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "discotope3_web/src"))
        from esm_util_custom import CoordBatchConverter

        batch_converter = CoordBatchConverter(self.alpha)
        batch = [(c, None, s) for c, s in zip(coords_batch, seq_batch)]
        coords_t, confidence, _, _, padding_mask = batch_converter(batch, device=DEVICE)
        self.esm.to(DEVICE)

        with torch.no_grad():
            encoder_out = self.esm.encoder.forward(
                coords_t, padding_mask, confidence, return_all_hiddens=False
            )
        hidden = encoder_out["encoder_out"][0].permute(1, 0, 2)  # (B, L, 512)
        out = []
        for b in range(len(seq_batch)):
            L = len(seq_batch[b])
            h = hidden[b, 1:L+1].cpu().numpy()   # (L, 512)
            out.append(h)
        return out


# ── Data loading helpers ───────────────────────────────────────────────────────

def load_samples(entries: list[tuple[str, str, list[int]]]):
    """Load coords for all entries. Returns list of (header, aa_seq, labels, coords_or_None)."""
    samples = []
    for header, aa_seq, labels in entries:
        parsed = parse_seq_id(header)
        coords = None
        if parsed:
            pdb_id, chain = parsed
            pdb_path = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
            if pdb_path.exists():
                coords = load_coords(pdb_path, chain, len(aa_seq))
        samples.append((header, aa_seq, labels, coords))
    return samples


# ── Training loop ──────────────────────────────────────────────────────────────

def train_fold(
    model: ESMIF1LoRAModel,
    train_samples: list,
    val_samples:   list,
    max_seconds:   int = MAX_SECONDS,
) -> dict:
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params   = sum(p.numel() for p in trainable)
    logger.info(f"  Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)

    # Warmup scheduler
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_auc = -1.0
    best_state   = None
    no_improve   = 0
    step         = 0
    train_loss   = 0.0
    start_time   = time.time()

    # Shuffle order
    rng = np.random.default_rng(42)
    indices = list(range(len(train_samples)))

    model.train()
    model.to(DEVICE)

    while True:
        if time.time() - start_time >= max_seconds:
            break
        rng.shuffle(indices)

        for idx in indices:
            if time.time() - start_time >= max_seconds:
                break

            header, aa_seq, labels, coords = train_samples[idx]
            coords_in = [coords if coords is not None else np.full((len(aa_seq), 3, 3), np.nan, dtype=np.float32)]

            try:
                logits_list = model([coords_in[0]], [aa_seq])
            except Exception as e:
                logger.debug(f"  Skip {header}: {e}")
                continue

            logits = logits_list[0]
            lbl_t  = torch.tensor(labels, dtype=torch.float32, device=DEVICE)
            loss   = F.binary_cross_entropy_with_logits(logits, lbl_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            train_loss = 0.9 * train_loss + 0.1 * loss.item()

            if step % VAL_INTERVAL == 0:
                val_auc = evaluate_auc(model, val_samples)
                elapsed = time.time() - start_time
                logger.info(f"  step={step}  train_loss={train_loss:.4f}  val_auc={val_auc:.4f}  {elapsed:.0f}s")
                model.train()

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state   = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()
                                    if any(k.startswith(p) for p in ("head", "esm.encoder.layers"))}
                    no_improve   = 0
                else:
                    no_improve  += 1
                    if no_improve >= PATIENCE:
                        logger.info(f"  Early stop at step {step}")
                        break
        else:
            continue
        break

    # Restore best
    if best_state is not None:
        cur = model.state_dict()
        cur.update({k: v.to(DEVICE) for k, v in best_state.items() if k in cur})
        model.load_state_dict(cur)

    final_val_auc = evaluate_auc(model, val_samples)
    return {"steps": step, "val_auc": final_val_auc}


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_auc(model: ESMIF1LoRAModel, samples: list) -> float:
    model.eval()
    all_logits, all_labels = [], []
    for header, aa_seq, labels, coords in samples:
        coords_in = coords if coords is not None else np.full((len(aa_seq), 3, 3), np.nan, dtype=np.float32)
        try:
            logits_list = model([coords_in], [aa_seq])
        except Exception:
            continue
        logits = logits_list[0].cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)
    if not all_logits:
        return float("nan")
    y_score = np.concatenate(all_logits)
    y_true  = np.concatenate(all_labels)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ── XGBoost helpers ────────────────────────────────────────────────────────────

_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX   = {aa: i for i, aa in enumerate(_AA_ORDER)}

def one_hot_aa(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), 20), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in _AA_IDX:
            arr[i, _AA_IDX[aa]] = 1.0
    return arr


def load_rsa(header: str, seq_len: int) -> np.ndarray:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
    parsed = parse_seq_id(header)
    if not parsed:
        return np.zeros(seq_len, dtype=np.float32)
    pdb_id, chain = parsed
    pdb_path = STRUCTURES_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return np.zeros(seq_len, dtype=np.float32)
    # Sander scale for RSA normalisation (same as discotope3_web)
    sander = {"A":106,"R":248,"N":157,"D":163,"C":135,"Q":198,"E":194,"G":84,
              "H":184,"I":169,"L":164,"K":205,"M":188,"F":197,"P":136,"S":130,
              "T":142,"W":227,"Y":222,"V":142}
    try:
        f      = pdb_io.PDBFile.read(str(pdb_path))
        struct = pdb_io.get_structure(f, model=1)
        struct = struct[(struct.chain_id == chain) & struc.filter_amino_acids(struct)]
        atom_sasa   = struc.sasa(struct, vdw_radii="ProtOr")
        res_ids     = struc.get_residues(struct)[0]
        if len(res_ids) != seq_len:
            return np.zeros(seq_len, dtype=np.float32)
        res_sasa = struc.apply_residue_wise(struct, atom_sasa, np.nansum)
        seq_3 = [struct[struct.res_id == rid].res_name[0] for rid in res_ids]
        from Bio.SeqUtils import seq1
        aa1 = [seq1(r) or "X" for r in seq_3]
        max_sasa = np.array([sander.get(a, 169.55) for a in aa1], dtype=np.float32)
        rsa = (res_sasa / max_sasa).astype(np.float32)
        return np.clip(rsa, 0, 1)
    except Exception:
        return np.zeros(seq_len, dtype=np.float32)


def build_xgb_features(emb: np.ndarray, aa_seq: str, header: str) -> np.ndarray:
    """Concatenate embedding + one-hot AA + RSA → (L, 533) feature matrix."""
    oh  = one_hot_aa(aa_seq)
    rsa = load_rsa(header, len(aa_seq)).reshape(-1, 1)
    return np.concatenate([emb, oh, rsa], axis=1)


def train_xgb_ensemble(X: np.ndarray, y: np.ndarray, n_models: int = 100) -> list:
    import xgboost as xgb
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(42)
    models = []
    for i in range(n_models):
        n_pos  = max(1, int(len(pos_idx) * 0.7))
        n_neg  = min(len(neg_idx), int(n_pos * 2.5 / 0.7))
        s_pos  = rng.choice(pos_idx, size=n_pos,  replace=False)
        s_neg  = rng.choice(neg_idx, size=n_neg,  replace=False)
        idx    = rng.permutation(np.concatenate([s_pos, s_neg]))
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.3, subsample=0.5,
            objective="binary:logistic", eval_metric="logloss",
            verbosity=0, random_state=int(rng.integers(0, 2**31)),
            device="cuda" if DEVICE == "cuda" else "cpu",
        )
        m.fit(X[idx], y[idx])
        models.append(m)
        if (i + 1) % 25 == 0:
            logger.info(f"    XGB {i+1}/{n_models}")
    return models


def predict_xgb(models: list, X: np.ndarray) -> np.ndarray:
    preds = np.zeros(len(X), dtype=np.float64)
    for m in models:
        preds += m.predict_proba(X)[:, 1]
    return preds / len(models)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_experiments():
    # Ensure results.tsv header exists
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                    "steps\tpeak_vram_mb\telapsed_s\tdesc\n")

    logger.info("Loading BEPIPRED data ...")
    by_part = parse_bepipred(BEPIPRED_FASTA)
    logger.info(f"Partitions: { {k: len(v) for k, v in by_part.items()} }")

    logger.info("Loading ESM-IF1 ...")
    import esm
    esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_model = esm_model.eval()

    for exp_name, run_xgb in [("exp1-lora-head", False), ("exp2-lora-xgb", True)]:
        desc = ("ESM-IF1 + LoRA(rank=4, all 8 enc layers) + linear head → BCE"
                if not run_xgb else
                "ESM-IF1 + LoRA(rank=4) fine-tuned → XGBoost-100 on embeddings")

        print(f"\n{'='*60}\nEXPERIMENT: {exp_name}\n  {desc}\n{'='*60}")
        fold_aucs = []

        for fold in CV_FOLDS:
            holdout  = fold["holdout"]
            train_keys = fold["train"]
            t0 = time.time()

            train_entries = [e for k in train_keys for e in by_part.get(k, [])]
            test_entries  = by_part.get(holdout, [])

            logger.info(f"\nFold holdout={holdout}: loading structures ...")
            train_samples = load_samples(train_entries)
            test_samples  = load_samples(test_entries)
            logger.info(f"  train={len(train_samples)}, test={len(test_samples)}")

            # Fresh model for each fold
            model = ESMIF1LoRAModel(
                esm_model, alphabet,
                rank=LORA_RANK, alpha=LORA_ALPHA,
                n_lora_layers=LORA_LAYERS,
                dropout=0.1,
            ).to(DEVICE)

            logger.info(f"  Training LoRA (max {MAX_SECONDS}s) ...")
            train_result = train_fold(model, train_samples, test_samples,
                                      max_seconds=MAX_SECONDS)

            if not run_xgb:
                # Experiment 1: evaluate with the linear head
                test_auc = evaluate_auc(model, test_samples)
                val_auc  = train_result["val_auc"]
            else:
                # Experiment 2: extract fine-tuned embeddings → XGBoost
                logger.info("  Extracting LoRA embeddings for XGBoost ...")
                model.eval()

                def _collect_features(samples):
                    Xs, ys = [], []
                    for header, aa_seq, labels, coords in samples:
                        coords_in = (coords if coords is not None
                                     else np.full((len(aa_seq), 3, 3), np.nan, dtype=np.float32))
                        try:
                            embs = model.get_embeddings([coords_in], [aa_seq])
                        except Exception:
                            continue
                        X = build_xgb_features(embs[0], aa_seq, header)
                        y = np.array(labels, dtype=np.float32)
                        Xs.append(X); ys.append(y)
                    return np.vstack(Xs), np.concatenate(ys)

                X_train, y_train = _collect_features(train_samples)
                X_test,  y_test  = _collect_features(test_samples)
                logger.info(f"  XGB features: train={X_train.shape}, test={X_test.shape}")

                xgb_models = train_xgb_ensemble(X_train, y_train)
                preds      = predict_xgb(xgb_models, X_test)
                test_auc   = float(roc_auc_score(y_test, preds))
                val_auc    = train_result["val_auc"]
                del xgb_models

            elapsed = time.time() - t0
            fold_aucs.append(test_auc)
            print(f"  fold={holdout}: val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  {elapsed:.0f}s")

            with open(RESULTS_TSV, "a") as f:
                f.write(f"{COMMIT}\t{exp_name}\t{holdout}\t0\t"
                        f"nan\t{val_auc:.6f}\t{test_auc:.6f}\t"
                        f"{train_result['steps']}\t0\t{elapsed:.1f}\t{desc}\n")

            del model
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        print(f"\n{'─'*60}\nSUMMARY {exp_name}")
        print(f"  test_auc = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"{'─'*60}")


if __name__ == "__main__":
    run_experiments()
