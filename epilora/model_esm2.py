"""epiLoRA backbone variant: frozen ESM2 (sequence-only) + LoRA + per-residue head.

Unlike ESM-IF1, ESM2 is a sequence-only protein language model -- no backbone
coordinates are used, only the residue sequence. This exists to answer one of
the epiLoRA ablation questions: how much does reading structure (ESM-IF1) vs.
sequence alone (ESM2) matter for epitope prediction, holding the LoRA-adapter
recipe and head fixed. RYS (see model.py) is an ESM-IF1-encoder-specific trick
and is not implemented here.

Same trainable-tensor convention as model.py: only LoRA adapters + head are
saved; the frozen ESM2 backbone is re-downloaded at load time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import build_head, inject_lora_layers

# name -> (fair-esm loader attr, hidden dim, total transformer layers)
ESM2_SIZES = {
    "35M": ("esm2_t12_35M_UR50D", 480, 12),
    "150M": ("esm2_t30_150M_UR50D", 640, 30),
    "650M": ("esm2_t33_650M_UR50D", 1280, 33),
}


class ESM2EpitopeModel(nn.Module):
    """Frozen ESM2 + LoRA + per-residue epitope head (sequence-only input)."""

    def __init__(self, esm_model, alphabet, size: str, rank: int, alpha: float,
                 n_lora_layers: int, dropout: float, head_dim: int | None):
        super().__init__()
        self.esm = esm_model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.size = size
        self.hidden = ESM2_SIZES[size][1]
        self.n_layers = ESM2_SIZES[size][2]
        self._cfg = dict(size=size, rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         dropout=dropout, head_dim=head_dim)
        for p in self.esm.parameters():
            p.requires_grad = False
        inject_lora_layers(self.esm.layers, rank, alpha, n_lora_layers)
        self.head_ln = nn.LayerNorm(self.hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head = build_head(self.hidden, head_dim, dropout)

    @property
    def device(self):
        return self.head_ln.weight.device

    def _encode(self, seq_batch):
        data = [(f"seq{i}", s) for i, s in enumerate(seq_batch)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        out = self.esm(tokens, repr_layers=[self.n_layers])
        return out["representations"][self.n_layers]  # (B, L+2, hidden): bos, residues..., eos

    def forward(self, coords_batch, seq_batch):
        """``coords_batch`` is accepted (for interface parity with the ESM-IF1
        model) but ignored -- ESM2 reads sequence only."""
        hidden = self._encode(seq_batch)
        out = []
        for b in range(len(seq_batch)):
            L = len(seq_batch[b])
            h = self.head_drop(self.head_ln(hidden[b, 1:L + 1]))  # drop bos token
            out.append(self.head(h).squeeze(-1))
        return out

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


def load_base_esm2(size: str):
    """Load the pretrained (frozen) ESM2 backbone + alphabet."""
    import esm
    loader_name = ESM2_SIZES[size][0]
    esm_model, alphabet = getattr(esm.pretrained, loader_name)()
    return esm_model.eval(), alphabet


def build_model(device: str = "cpu", size: str = "650M", rank: int = 4, alpha: float = 8.0,
                n_lora_layers: int = 8, dropout: float = 0.1, head_dim: int | None = None) -> "ESM2EpitopeModel":
    """Build an (untrained) ESM2-backbone epiLoRA model on ``device``."""
    esm_model, alphabet = load_base_esm2(size)
    model = ESM2EpitopeModel(esm_model, alphabet, size=size, rank=rank, alpha=alpha,
                             n_lora_layers=n_lora_layers, dropout=dropout, head_dim=head_dim).to(device)
    return model
