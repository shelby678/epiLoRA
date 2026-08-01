"""epiLoRA backbone variant: frozen ESM3 (sequence-only track) + LoRA + per-residue head.

Like model_esm2.py, this is a sequence-only comparison point for the epiLoRA
backbone ablation -- no backbone coordinates are used. ESM3 is multimodal
(sequence/structure/function tracks) but we only ever feed it sequence
tokens; every other track's input defaults to its own mask/pad token inside
ESM3.forward, which is how the reference API itself supports sequence-only
use. RYS (an ESM-IF1-encoder-specific trick) is not implemented here.

ESM3's attention module (esm.layers.attention.MultiHeadAttention) has a fused
QKV projection (``layernorm_qkv`` = LayerNorm -> Linear(d_model, 3*d_model))
plus a separate ``out_proj``, unlike fair-esm's separate q/k/v/out_proj
layout -- so it needs its own LoRA injection helper (inject_lora_layers in
model.py assumes the fair-esm layout and won't work here).

Must run under epilora/env_esm3/bin/python3: EvolutionaryScale's ``esm`` pip
package and fair-esm's ``esm`` package (used by model.py/model_esm2.py)
collide on the same import name and cannot coexist in one environment.

Same trainable-tensor convention as model.py: only LoRA adapters + head are
saved; the frozen ESM3 backbone is re-downloaded (from HuggingFace, gated --
requires an authenticated + license-accepted account) at load time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import LoRALinear, build_head

HIDDEN = 1536
N_LAYERS = 48


def inject_lora_esm3(blocks, rank: int, alpha: float, n_layers: int) -> None:
    """Add LoRA adapters to the fused QKV projection + out_proj of the last
    ``n_layers`` of ``blocks`` (ESM3's UnifiedTransformerBlock stack)."""
    total = len(blocks)
    start = max(0, total - n_layers)
    for i in range(start, total):
        attn = blocks[i].attn
        attn.layernorm_qkv[1] = LoRALinear(attn.layernorm_qkv[1], rank, alpha)
        attn.out_proj = LoRALinear(attn.out_proj, rank, alpha)


class ESM3EpitopeModel(nn.Module):
    """Frozen ESM3 (sequence track) + LoRA + per-residue epitope head."""

    def __init__(self, esm_model, rank: int = 4, alpha: float = 8.0,
                 n_lora_layers: int = 8, dropout: float = 0.1, head_dim: int | None = None):
        super().__init__()
        self.esm = esm_model
        self.hidden = HIDDEN
        self._cfg = dict(rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         dropout=dropout, head_dim=head_dim)
        for p in self.esm.parameters():
            p.requires_grad = False
        inject_lora_esm3(self.esm.transformer.blocks, rank, alpha, n_lora_layers)
        self.head_ln = nn.LayerNorm(self.hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head = build_head(self.hidden, head_dim, dropout)

    @property
    def device(self):
        return self.head_ln.weight.device

    def _encode(self, seq_batch):
        # train.py only ever calls forward() with single-element lists (see
        # its per-sample training loop), so no padding/attention-masking
        # across differently-sized sequences is needed here.
        from esm.utils.encoding import tokenize_sequence
        assert len(seq_batch) == 1, "model_esm3 only supports batch size 1"
        tokens = tokenize_sequence(seq_batch[0], self.esm.tokenizers.sequence,
                                   add_special_tokens=True).unsqueeze(0).to(self.device)
        out = self.esm.forward(sequence_tokens=tokens)
        return out.embeddings  # (1, L+2, hidden): bos, residues..., eos

    def forward(self, coords_batch, seq_batch):
        """``coords_batch`` is accepted (for interface parity with the ESM-IF1
        model) but ignored -- this backbone reads ESM3's sequence track only."""
        hidden = self._encode(seq_batch)
        L = len(seq_batch[0])
        h = self.head_drop(self.head_ln(hidden[0, 1:L + 1]))  # drop bos token
        return [self.head(h).squeeze(-1)]

    def config(self) -> dict:
        return dict(self._cfg)

    def trainable_state_dict(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k in names}

    def load_trainable_state_dict(self, trainable: dict) -> None:
        cur = self.state_dict()
        cur.update({k: v.to(self.device) for k, v in trainable.items() if k in cur})
        self.load_state_dict(cur)


def load_base_esm3():
    """Load the pretrained (frozen) ESM3-small-open backbone.

    Loaded on CPU first: ``ESM3.from_pretrained`` auto-casts to bfloat16 for
    any non-cpu device, which we don't want (our LoRA/head params are
    float32); the caller moves the wrapped model to the target device after
    construction, staying in float32 throughout.
    """
    from esm.models.esm3 import ESM3
    return ESM3.from_pretrained(device=torch.device("cpu"))


def build_model(device: str = "cpu", rank: int = 4, alpha: float = 8.0,
                n_lora_layers: int = 8, dropout: float = 0.1,
                head_dim: int | None = None) -> "ESM3EpitopeModel":
    """Build an (untrained) ESM3-backbone epiLoRA model on ``device``."""
    esm_model = load_base_esm3()
    model = ESM3EpitopeModel(esm_model, rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                             dropout=dropout, head_dim=head_dim).to(device)
    return model
