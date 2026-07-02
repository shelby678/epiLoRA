"""MUTABLE: ESM2-650M (HuggingFace) frozen backbone + LoRA + RYS + epitope head.

Sequence-only counterpart using the original ESM2 (facebook/esm2_t33_650M_UR50D,
1280-dim, 33 layers) via transformers.EsmModel. ESM2 shares the ESM3/our token
vocab (A=5, C=23, ..., BOS=0, PAD=1, EOS=2, MASK=32), so `_OUR_TO_ESM3` remaps
our token IDs to ESM2 IDs unchanged.

* LoRA on attention query/key/value + attention.output.dense of the last N layers.
* RYS (Repeat Yourself): replay encoder layers [rys_start, rys_end) a second
  time, via a patched EsmEncoder.forward (keeps embeddings + final norm intact).
* Head: LayerNorm -> Dropout -> Linear(1280 -> 1) per token.

Forward signature matches the ESM3 model (accepts/ignores structure_coords,
extra_features) so ensemble_io.predict_venv_model and the shared eval helpers
work unchanged.
"""

from __future__ import annotations

import gc
import logging
import math
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from train_struct import (
    _OUR_TO_ESM3, LoRALinear, _get_lr_scale,
    create_struct_dataloader, evaluate_struct_loss, compute_roc_auc,
    DROPOUT, LR, WARMUP_STEPS, WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)

ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 4  # 650M is larger than ESM3-sm per-token; keep batch modest


def _inject_lora_esm2(esm, rank: int, alpha: float, n_layers: int) -> None:
    """Wrap query/key/value + attention.output.dense of the last n_layers
    EsmLayers with LoRALinear (base frozen)."""
    layers = esm.encoder.layer
    n = len(layers)
    start = max(0, n - n_layers)
    for i in range(start, n):
        attn = layers[i].attention
        attn.self.query = LoRALinear(attn.self.query, rank, alpha)
        attn.self.key = LoRALinear(attn.self.key, rank, alpha)
        attn.self.value = LoRALinear(attn.self.value, rank, alpha)
        attn.output.dense = LoRALinear(attn.output.dense, rank, alpha)


def _patch_encoder_rys(esm, rys_start: int, rys_end: int) -> None:
    """Replace EsmEncoder.forward with a version that replays layers
    [rys_start, rys_end) a second time (RYS). Final emb_layer_norm_after is
    preserved."""
    from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions

    def new_forward(self, hidden_states, attention_mask=None,
                    encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        def run(layer, x):
            out = layer(x, attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask, **kwargs)
            return out[0] if isinstance(out, tuple) else out

        x = hidden_states
        for i in range(rys_start):
            x = run(self.layer[i], x)
        for i in range(rys_start, rys_end):
            x = run(self.layer[i], x)
        for i in range(rys_start, rys_end):   # RYS replay
            x = run(self.layer[i], x)
        for i in range(rys_end, len(self.layer)):
            x = run(self.layer[i], x)
        if self.emb_layer_norm_after:
            x = self.emb_layer_norm_after(x)
        return BaseModelOutputWithCrossAttentions(last_hidden_state=x)

    esm.encoder.forward = types.MethodType(new_forward, esm.encoder)


class ESM2EpitopeModel(nn.Module):
    _HIDDEN = 1280

    def __init__(self, dropout: float = DROPOUT, lora_rank: int = 8, lora_alpha: float = 8.0,
                 lora_n_blocks: int = 16, rys_start: int = 24, rys_end: int = 30,
                 head_hidden_dim: int = 0) -> None:
        super().__init__()
        from transformers import EsmModel

        self.esm = EsmModel.from_pretrained(ESM2_NAME, add_pooling_layer=False)
        for p in self.esm.parameters():
            p.requires_grad = False
        self.esm.eval()

        self._use_lora = lora_rank > 0
        if self._use_lora:
            _inject_lora_esm2(self.esm, lora_rank, lora_alpha, lora_n_blocks)
        if rys_end > rys_start:
            _patch_encoder_rys(self.esm, rys_start, rys_end)

        self.head_ln = nn.LayerNorm(self._HIDDEN)
        self.head_drop = nn.Dropout(dropout)
        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(self._HIDDEN, head_hidden_dim), nn.GELU(),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            self.head = nn.Linear(self._HIDDEN, 1)

        self.register_buffer("id_map", torch.tensor(_OUR_TO_ESM3, dtype=torch.long))

    def forward(self, input_ids, attention_mask=None, structure_coords=None, extra_features=None):
        esm2_ids = self.id_map[input_ids]
        if attention_mask is None:
            attention_mask = (input_ids != 1).long()
        if self._use_lora:
            out = self.esm(input_ids=esm2_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.esm(input_ids=esm2_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state.to(self.head_ln.weight.dtype)
        hidden = self.head_drop(self.head_ln(emb))
        return self.head(hidden).squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_esm2(train_data, val_data, *, max_seconds=1200, device="cuda", compute_auc=True,
               val_eval_interval=200, lora_rank=8, lora_alpha=8.0, lora_n_blocks=16,
               rys_start=24, rys_end=30, dropout=DROPOUT, batch_size=BATCH_SIZE,
               lr=LR, weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS, patience=5):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    model = ESM2EpitopeModel(
        dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha,
        lora_n_blocks=lora_n_blocks, rys_start=rys_start, rys_end=rys_end,
    ).to(device)
    logger.info(f"ESM2-650M: trainable {model.num_parameters():,}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    loader = create_struct_dataloader(train_data, batch_size=batch_size, shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    step = 0
    running = 0.0
    total_estimate = max_seconds * 3
    best_val = float("inf")
    best_state = None
    no_improve = 0
    start = time.time()
    model.train()
    early = False
    optimizer.zero_grad()

    while True:
        for batch in loader:
            if time.time() - start >= max_seconds or early:
                break
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                logits = model(ids)
                valid = labels != -100
                loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            for pg in optimizer.param_groups:
                pg["lr"] = lr * _get_lr_scale(step, warmup_steps, total_estimate)
            running += loss.item()
            step += 1
            if step % 10 == 0:
                logger.info(f"step={step} train_loss={running/10:.4f} elapsed={time.time()-start:.0f}s")
                running = 0.0
            if val_eval_interval > 0 and step % val_eval_interval == 0:
                v = evaluate_struct_loss(model, val_data, batch_size=batch_size, device=device)
                logger.info(f"step={step} val_loss={v:.6f}")
                if v < best_val:
                    best_val = v
                    no_improve = 0
                    tk = {n for n, p in model.named_parameters() if p.requires_grad}
                    best_state = {k: val.cpu().clone() for k, val in model.state_dict().items() if k in tk}
                else:
                    no_improve += 1
                    if patience > 0 and no_improve >= patience:
                        logger.info(f"Early stop at step {step}")
                        early = True
                model.train()
        if time.time() - start >= max_seconds or early:
            break

    if best_state is not None:
        cur = model.state_dict()
        cur.update({k: v.to(device) for k, v in best_state.items()})
        model.load_state_dict(cur)

    val_loss = evaluate_struct_loss(model, val_data, batch_size=batch_size, device=device)
    roc = compute_roc_auc(model, val_data, batch_size=batch_size, device=device) if compute_auc else float("nan")
    tk = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if k in tk}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"val_loss": val_loss, "roc_auc": roc, "steps": step, "trainable_state": trainable_state}
