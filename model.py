"""epiLoRA model: ESM-IF1 (inverse folding) + LoRA + RYS + per-residue head.

Best-performing B-cell epitope predictor from the epiLoRA study
(5-fold ROC-AUC 0.824 ± 0.064). The frozen ESM-IF1 GVP-Transformer encoder is
adapted with LoRA on the attention projections of its encoder layers, its top
layers are replayed once (RYS = "Repeat Yourself"), and a small linear head
maps each residue's 512-d encoding to an epitope logit.

Because ESM-IF1 is an inverse-folding model, inputs are backbone coordinates
(N, CA, C) plus the sequence — i.e. a protein *structure*, not a bare sequence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Champion hyperparameters (see README / findings).
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_LAYERS = 8          # LoRA on all 8 ESM-IF1 encoder layers
RYS_START, RYS_END = 4, 8  # replay encoder layers [4, 8) once
DROPOUT = 0.1
HIDDEN = 512             # ESM-IF1 encoder output dim
N_ENCODER_LAYERS = 8

# The only trainable tensors are the LoRA adapters (…lora_A/lora_B) and the
# head (head_ln + head). Everything else is frozen ESM-IF1 and comes from the
# pretrained download, so a checkpoint stores just these (~a few MB). RYS adds
# no parameters — it replays existing layers.


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update B @ A."""

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

    # Passthroughs so code reading the wrapped layer's params still works.
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


def inject_lora(esm_model, rank: int, alpha: float, n_layers: int) -> None:
    """Add LoRA adapters to q/k/v/out projections of the last ``n_layers``."""
    start = max(0, N_ENCODER_LAYERS - n_layers)
    for i in range(start, N_ENCODER_LAYERS):
        attn = esm_model.encoder.layers[i].self_attn
        # Force the manual q/k/v path so LoRA.forward is actually invoked; the
        # fused F.multi_head_attention_forward fast path reads q_proj.weight
        # directly and would silently bypass the adapter.
        attn.enable_torch_version = False
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha))


def patch_encoder_rys(encoder, rys_start: int, rys_end: int) -> None:
    """Replay ``encoder.layers[rys_start:rys_end]`` a second time (RYS)."""
    import types

    def new_forward(self, coords, encoder_padding_mask, confidence, return_all_hiddens=False):
        x, encoder_embedding = self.forward_embedding(coords, encoder_padding_mask, confidence)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        layers = self.layers
        for i in range(rys_start):
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        for i in range(rys_start, rys_end):
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        for i in range(rys_start, rys_end):            # RYS replay
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        for i in range(rys_end, len(layers)):
            x = layers[i](x, encoder_padding_mask=encoder_padding_mask)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return {"encoder_out": [x], "encoder_padding_mask": [encoder_padding_mask],
                "encoder_embedding": [encoder_embedding], "encoder_states": []}

    encoder.forward = types.MethodType(new_forward, encoder)


class ESMIF1EpitopeModel(nn.Module):
    """Frozen ESM-IF1 + LoRA + RYS + per-residue epitope head."""

    def __init__(self, esm_model, alphabet, rank=LORA_RANK, alpha=LORA_ALPHA,
                 n_lora_layers=LORA_LAYERS, rys_start=RYS_START, rys_end=RYS_END,
                 dropout=DROPOUT):
        super().__init__()
        self.esm = esm_model
        self.alphabet = alphabet
        self.hidden = HIDDEN
        self._cfg = dict(rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         rys_start=rys_start, rys_end=rys_end, dropout=dropout)
        for p in self.esm.parameters():
            p.requires_grad = False
        inject_lora(self.esm, rank, alpha, n_lora_layers)
        if rys_end > rys_start:
            patch_encoder_rys(self.esm.encoder, rys_start, rys_end)
        self.head_ln = nn.LayerNorm(self.hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden, 1)

    @property
    def device(self):
        return self.head.weight.device

    def _encode(self, coords_batch, seq_batch):
        # Stock fair-esm inverse-folding collate (pads coords, builds mask/confidence).
        from esm.inverse_folding.util import CoordBatchConverter
        dev = self.device
        bc = CoordBatchConverter(self.alphabet)
        batch = [(c, None, s) for c, s in zip(coords_batch, seq_batch)]
        coords_t, confidence, _, _, padding_mask = bc(batch, device=dev)
        enc = self.esm.encoder.forward(coords_t, padding_mask, confidence, return_all_hiddens=False)
        return enc["encoder_out"][0].permute(1, 0, 2)  # (B, L, 512)

    def forward(self, coords_batch, seq_batch):
        """Return a list of per-residue logit tensors, one per input protein."""
        hidden = self._encode(coords_batch, seq_batch)
        out = []
        for b in range(len(seq_batch)):
            L = len(seq_batch[b])
            h = self.head_drop(self.head_ln(hidden[b, 1:L + 1]))  # drop begin token
            out.append(self.head(h).squeeze(-1))
        return out

    # ---- checkpoint helpers -------------------------------------------------
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


def load_base_esmif1():
    """Load the pretrained (frozen) ESM-IF1 backbone + alphabet."""
    import esm
    esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    return esm_model.eval(), alphabet


def build_model(device: str = "cpu", **cfg) -> "ESMIF1EpitopeModel":
    """Build an (untrained) epiLoRA model on ``device``.

    ``cfg`` overrides the champion defaults (rank, alpha, n_lora_layers,
    rys_start, rys_end, dropout).
    """
    esm_model, alphabet = load_base_esmif1()
    model = ESMIF1EpitopeModel(esm_model, alphabet, **cfg).to(device)
    return model
