"""MUTABLE: ESM2 frozen backbone + trainable binary classification head.

This is the file the autonomous agent modifies between experiments.
The backbone (ESM2) is frozen; only the prediction head is trained.
"""

from __future__ import annotations

import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import EsmModel

from prepare import (
    PAD_ID,
    PAD_VOCAB_SIZE,
    create_dataloader,
    evaluate_loss,
)

try:
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# === HYPERPARAMETERS (agent-tunable) ===
ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # frozen backbone (320-dim, 6 layers)
DROPOUT = 0.1
MAX_SEQ_LEN = 512
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

# ---------------------------------------------------------------------------
# Vocab ID mapping: our custom IDs → ESM2 vocab IDs
# Our vocab:  cls=0 pad=1 eos=2 unk=3 mask=4 A=5 C=6 D=7 E=8 F=9 G=10
#             H=11 I=12 K=13 L=14 M=15 N=16 P=17 Q=18 R=19 S=20 T=21 V=22 W=23 Y=24
# ESM2 vocab: cls=0 pad=1 eos=2 unk=3 L=4  A=5 G=6 V=7 S=8 E=9  R=10 T=11 I=12 D=13
#             P=14 K=15 Q=16 N=17 F=18 Y=19 M=20 H=21 W=22 C=23 ... mask=32
# ---------------------------------------------------------------------------
_OUR_TO_ESM2: list[int] = [3] * PAD_VOCAB_SIZE  # default to UNK
_OUR_TO_ESM2[0] = 0    # <cls>  → <cls>
_OUR_TO_ESM2[1] = 1    # <pad>  → <pad>
_OUR_TO_ESM2[2] = 2    # <eos>  → <eos>
_OUR_TO_ESM2[3] = 3    # <unk>  → <unk>
_OUR_TO_ESM2[4] = 32   # <mask> → <mask>
_OUR_TO_ESM2[5] = 5    # A
_OUR_TO_ESM2[6] = 23   # C
_OUR_TO_ESM2[7] = 13   # D
_OUR_TO_ESM2[8] = 9    # E
_OUR_TO_ESM2[9] = 18   # F
_OUR_TO_ESM2[10] = 6   # G
_OUR_TO_ESM2[11] = 21  # H
_OUR_TO_ESM2[12] = 12  # I
_OUR_TO_ESM2[13] = 15  # K
_OUR_TO_ESM2[14] = 4   # L
_OUR_TO_ESM2[15] = 20  # M
_OUR_TO_ESM2[16] = 17  # N
_OUR_TO_ESM2[17] = 14  # P
_OUR_TO_ESM2[18] = 16  # Q
_OUR_TO_ESM2[19] = 10  # R
_OUR_TO_ESM2[20] = 8   # S
_OUR_TO_ESM2[21] = 11  # T
_OUR_TO_ESM2[22] = 7   # V
_OUR_TO_ESM2[23] = 22  # W
_OUR_TO_ESM2[24] = 19  # Y


# ---------------------------------------------------------------------------
# Model: frozen ESM2 + trainable head
# ---------------------------------------------------------------------------


class EpitopePredictionModel(nn.Module):
    """Frozen ESM2 backbone with a trainable per-token binary classification head.

    Token IDs in our custom vocabulary are remapped to ESM2's vocabulary
    internally, so this model is compatible with prepare.py's data loading
    and evaluate_loss function.
    """

    def __init__(
        self,
        esm2_model_name: str = ESM2_MODEL_NAME,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        # Load ESM2 and freeze all its parameters
        self.esm2 = EsmModel.from_pretrained(esm2_model_name)
        for param in self.esm2.parameters():
            param.requires_grad = False
        self.esm2.eval()

        hidden_dim = self.esm2.config.hidden_size

        # Trainable binary classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=True),
        )

        # Learnable per-layer weights for the 6 ESM2 transformer layers
        self.layer_weights = nn.Parameter(torch.zeros(6))

        # Register vocab ID mapping as a buffer (moves with .to(device))
        self.register_buffer(
            "id_map", torch.tensor(_OUR_TO_ESM2, dtype=torch.long)
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (B, L) token IDs in our custom vocabulary.
            attention_mask: (B, L) with 1=real token, 0=padding.

        Returns:
            Logits of shape (B, L) — one binary logit per token position.
        """
        # Remap to ESM2 vocabulary
        esm2_ids = self.id_map[input_ids]

        with torch.no_grad():
            outputs = self.esm2(
                input_ids=esm2_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Softmax-weighted sum of all 6 transformer layer outputs (skip embedding at index 0)
        weights = torch.softmax(self.layer_weights, dim=0)  # (6,)
        stacked = torch.stack(outputs.hidden_states[1:], dim=0)  # (6, B, L, hidden_dim)
        hidden = (weights[:, None, None, None] * stacked).sum(0)  # (B, L, hidden_dim)
        return self.head(hidden).squeeze(-1)  # (B, L)

    def num_parameters(self) -> int:
        """Count trainable parameters (head only; ESM2 is frozen)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def _get_lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# ROC-AUC evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_roc_auc(
    model: nn.Module,
    val_data: list,
    batch_size: int = 16,
    device: str = "cpu",
) -> float:
    """Compute token-level ROC-AUC on the validation set."""
    if not HAS_SKLEARN:
        return float("nan")

    model.eval()
    loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)

    all_probs: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)

        valid = labels != -100
        probs = torch.sigmoid(logits[valid]).cpu().float().numpy().tolist()
        lbls = labels[valid].cpu().float().numpy().tolist()

        all_probs.extend(probs)
        all_labels.extend(lbls)

    model.train()

    if len(set(all_labels)) < 2:
        return float("nan")

    return float(roc_auc_score(all_labels, all_probs))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    train_data: list,
    val_data: list,
    max_seconds: int = 300,
    device: str = "cpu",
    compute_auc: bool = False,
    val_eval_interval: int = 200,
) -> dict:
    """Train the ESM2-based epitope prediction model.

    Args:
        train_data: List of (token_ids, epitope_labels) training samples.
        val_data: List of (token_ids, epitope_labels) validation samples.
        max_seconds: Wall-clock time budget (seconds).
        device: Device to run on.
        compute_auc: Whether to compute ROC-AUC after training.

    Returns:
        Dict with val_loss, train_loss, steps, params, etc.
    """
    model = EpitopePredictionModel(
        esm2_model_name=ESM2_MODEL_NAME,
        dropout=DROPOUT,
    ).to(device)

    n_params = model.num_parameters()
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {n_params:,}  (total incl. frozen: {n_total:,})")

    # Only optimize the trainable head
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loader = create_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    step = 0
    total_tokens = 0
    running_loss = 0.0
    log_interval = 10
    total_estimate = max_seconds * 84  # ~84 steps/sec on this GPU

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    peak_vram_mb = 0

    start_time = time.time()
    model.train()

    while True:
        for batch in loader:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attention_mask)  # (B, L)
                valid = labels != -100
                loss = F.binary_cross_entropy_with_logits(
                    logits[valid], labels[valid].float()
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # LR schedule
            lr_scale = _get_lr_scale(step, WARMUP_STEPS, total_estimate)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            running_loss += loss.item()
            step += 1
            total_tokens += int(attention_mask.sum())

            if step % log_interval == 0:
                avg = running_loss / log_interval
                logger.info(f"step={step}  train_loss={avg:.4f}  elapsed={elapsed:.0f}s")
                running_loss = 0.0

            if val_eval_interval > 0 and step % val_eval_interval == 0:
                v = evaluate_loss(model, val_data, batch_size=BATCH_SIZE, device=device)
                logger.info(f"step={step}  val_loss={v:.6f}  [periodic]")
                model.train()

        if time.time() - start_time >= max_seconds:
            break

    train_loss = running_loss / max(1, step % log_interval) if step % log_interval != 0 else 0.0

    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)

    val_loss = evaluate_loss(model, val_data, batch_size=BATCH_SIZE, device=device)
    logger.info(f"val_loss={val_loss:.6f} after {step} steps")

    roc_auc = float("nan")
    if compute_auc:
        roc_auc = compute_roc_auc(model, val_data, batch_size=BATCH_SIZE, device=device)
        logger.info(f"roc_auc={roc_auc:.6f}")

    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "steps": step,
        "params": n_params,
        "peak_vram_mb": peak_vram_mb,
        "total_tokens": total_tokens,
        "depth": 6,  # ESM2 t6 has 6 transformer layers
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    from prepare import create_datasets

    # === RUN CONFIGURATION ===
    DATA_DIR = Path("data/combined/1")
    TIME_BUDGET = 300  # 5 minutes
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    train_data, val_data = create_datasets(DATA_DIR, max_length=MAX_SEQ_LEN)
    print(
        f"Train: {len(train_data)} sequences, Val: {len(val_data)} sequences",
        file=sys.stderr,
    )

    t0 = time.time()
    result = train(
        train_data, val_data,
        max_seconds=TIME_BUDGET,
        device=DEVICE,
        compute_auc=True,
    )
    total = time.time() - t0

    # Print results in parseable format (agent greps for ^val_loss:)
    print("---")
    print(f"val_loss:            {result['val_loss']:.6f}")
    print(f"roc_auc:             {result['roc_auc']:.6f}")
    print(f"train_loss:          {result['train_loss']:.6f}")
    print(f"training_seconds:    {TIME_BUDGET}")
    print(f"total_seconds:       {total:.1f}")
    print(f"num_steps:           {result['steps']}")
    print(f"num_params_M:        {result['params'] / 1e6:.3f}")
    print(f"peak_vram_mb:        {result['peak_vram_mb']}")
    print(f"total_tokens_M:      {result['total_tokens'] / 1e6:.1f}")
    print(f"depth:               {result['depth']}")
