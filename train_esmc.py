"""MUTABLE: ESMC (ESM Cambrian) frozen backbone + LoRA + trainable epitope head.

A sequence-only counterpart to ``train_struct.py`` (which uses ESM3 + backbone
coordinates).  ESMC ("ESM Cambrian", EvolutionaryScale) is a pure-sequence
protein LM; there is no structure track, so backbone coordinates are ignored.

ESMC conveniently shares ESM3's amino-acid token vocabulary (A=5, C=23, ...,
BOS=0, PAD=1, EOS=2, MASK=32) and transformer block layout
(``transformer.blocks[i].attn.layernorm_qkv[1]`` / ``.attn.out_proj``), so we
reuse ``_OUR_TO_ESM3``, ``LoRALinear`` and ``_inject_lora`` from
``train_struct`` directly.

Architecture
------------
* ESMC-600M (1152-dim, 36 layers) or ESMC-300M (960-dim, 30 layers) — frozen.
* LoRA on attention QKV + out_proj of the last N blocks (rank/alpha tunable).
* Trainable head: LayerNorm -> Dropout -> Linear(d_model -> 1) per token.

The forward signature matches ``StructureEpitopePredictionModel`` so the shared
``evaluate_struct_loss`` / ``compute_roc_auc`` helpers work unchanged; the
``structure_coords`` and ``extra_features`` kwargs are accepted and ignored.
"""

from __future__ import annotations

import gc
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from train_struct import (
    _OUR_TO_ESM3,
    _inject_lora,
    _get_lr_scale,
    create_struct_dataloader,
    evaluate_struct_loss,
    compute_roc_auc,
    BATCH_SIZE,
    DROPOUT,
    LR,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)

# ESMC pretrained-loader name -> (loader, d_model, n_layers) for logging/asserts.
_ESMC_LOADERS = {
    "300m": ("ESMC_300M_202412", 960, 30),
    "600m": ("ESMC_600M_202412", 1152, 36),
}


class ESMCEpitopeModel(nn.Module):
    """Frozen ESMC backbone + optional LoRA + trainable per-token binary head."""

    def __init__(
        self,
        size: str = "600m",
        dropout: float = DROPOUT,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_n_blocks: int = 8,
        lora_block_start: int = -1,
        head_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        import esm.pretrained

        loader_name, _, _ = _ESMC_LOADERS[size]
        loader = getattr(esm.pretrained, loader_name)
        self.esmc = loader(device="cpu")
        for p in self.esmc.parameters():
            p.requires_grad = False
        self.esmc.eval()

        self._d_model = self.esmc.transformer.blocks[0].attn.out_proj.out_features

        self._use_lora = lora_rank > 0
        if self._use_lora:
            # ESMC exposes transformer.blocks (no top-level .layers), so the
            # ESM3 injector targets attn.layernorm_qkv[1] + attn.out_proj here.
            _inject_lora(
                self.esmc, lora_rank, lora_alpha, lora_n_blocks,
                lora_block_start=lora_block_start,
            )

        self.head_ln = nn.LayerNorm(self._d_model)
        self.head_drop = nn.Dropout(dropout)
        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(self._d_model, head_hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(head_hidden_dim, 1, bias=True),
            )
        else:
            self.head = nn.Linear(self._d_model, 1, bias=True)

        self.register_buffer("id_map", torch.tensor(_OUR_TO_ESM3, dtype=torch.long))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,   # unused (ESMC infers from PAD)
        structure_coords: Tensor | None = None,  # ignored (sequence-only model)
        extra_features: Tensor | None = None,    # ignored (kept for API parity)
    ) -> Tensor:
        esmc_ids = self.id_map[input_ids]
        if self._use_lora:
            out = self.esmc(sequence_tokens=esmc_ids)
        else:
            with torch.no_grad():
                out = self.esmc(sequence_tokens=esmc_ids)
        emb = out.embeddings.to(self.head_ln.weight.dtype)  # (B, L, d_model)
        hidden = self.head_drop(self.head_ln(emb))
        return self.head(hidden).squeeze(-1)  # (B, L)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_esmc(
    train_data: list,
    val_data: list,
    *,
    size: str = "600m",
    max_seconds: int = 1200,
    device: str = "cuda",
    compute_auc: bool = True,
    val_eval_interval: int = 200,
    lora_rank: int = 8,
    lora_alpha: float = 8.0,
    lora_n_blocks: int = 8,
    lora_block_start: int = -1,
    head_hidden_dim: int = 0,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    warmup_steps: int = WARMUP_STEPS,
    patience: int = 5,
) -> dict:
    """Train the ESMC-based epitope predictor. Mirrors the core of
    ``train_struct.train`` (BCE + warmup/cosine LR + early stopping on val loss)
    without the structure / RYS / paper-dropout machinery."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    model = ESMCEpitopeModel(
        size=size,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_n_blocks=lora_n_blocks,
        lora_block_start=lora_block_start,
        head_hidden_dim=head_hidden_dim,
    ).to(device)

    n_params = model.num_parameters()
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"ESMC-{size}: trainable {n_params:,}  (total incl. frozen: {n_total:,})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    loader = create_struct_dataloader(train_data, batch_size=batch_size, shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    step = 0
    running_loss = 0.0
    log_interval = 10
    total_estimate = max_seconds * 3

    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve_count = 0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    model.train()
    early_stop = False
    optimizer.zero_grad()

    while True:
        for batch in loader:
            if time.time() - start_time >= max_seconds or early_stop:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                logits = model(input_ids)
                valid = labels != -100
                loss = F.binary_cross_entropy_with_logits(
                    logits[valid], labels[valid].float()
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            lr_scale = _get_lr_scale(step, warmup_steps, total_estimate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * lr_scale

            running_loss += loss.item()
            step += 1

            if step % log_interval == 0:
                logger.info(
                    f"step={step}  train_loss={running_loss/log_interval:.4f}  "
                    f"elapsed={time.time()-start_time:.0f}s"
                )
                running_loss = 0.0

            if val_eval_interval > 0 and step % val_eval_interval == 0:
                v = evaluate_struct_loss(model, val_data, batch_size=batch_size, device=device)
                logger.info(f"step={step}  val_loss={v:.6f}  [periodic]")
                if v < best_val_loss:
                    best_val_loss = v
                    no_improve_count = 0
                    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
                    best_state = {
                        k: val.cpu().clone()
                        for k, val in model.state_dict().items()
                        if k in trainable_keys
                    }
                else:
                    no_improve_count += 1
                    if patience > 0 and no_improve_count >= patience:
                        logger.info(f"Early stop at step {step} (best val_loss={best_val_loss:.6f})")
                        early_stop = True
                model.train()

        if time.time() - start_time >= max_seconds or early_stop:
            break

    if best_state is not None:
        current = model.state_dict()
        current.update({k: v.to(device) for k, v in best_state.items()})
        model.load_state_dict(current)
        logger.info(f"Restored best checkpoint (val_loss={best_val_loss:.6f})")

    peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024) if device == "cuda" else 0
    val_loss = evaluate_struct_loss(model, val_data, batch_size=batch_size, device=device)
    roc_auc = float("nan")
    if compute_auc:
        roc_auc = compute_roc_auc(model, val_data, batch_size=batch_size, device=device)
        logger.info(f"val roc_auc={roc_auc:.6f}")

    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_state = {
        k: v.cpu().clone() for k, v in model.state_dict().items() if k in trainable_keys
    }

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_loss": val_loss,
        "steps": step,
        "params": n_params,
        "peak_vram_mb": peak_vram_mb,
        "roc_auc": roc_auc,
        "trainable_state": trainable_state,
    }
