"""epiLoRA model: frozen backbone + LoRA + per-residue head, for every
epiLoRA backbone variant (ESM-IF1, ESM2, ESM3, ESMc).

ESM-IF1 (the champion backbone) is an inverse-folding model and reads
backbone geometry; ESM2/ESM3/ESMc are sequence-only protein language models
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# fair-esm's esm.inverse_folding.util imports biotite.structure.filter_backbone,
# which newer biotite (>=1.0) renamed to filter_peptide_backbone.
import biotite.structure as _struc
if not hasattr(_struc, "filter_backbone"):
    _struc.filter_backbone = _struc.filter_peptide_backbone

HIDDEN = 512             # ESM-IF1 encoder output dim

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


def inject_lora_layers(layers, rank: int, alpha: float, n_layers: int) -> None:
    """Add LoRA adapters to q/k/v/out projections of the last ``n_layers`` of ``layers``.

    ``layers`` is any indexable sequence of transformer blocks exposing a
    ``.self_attn`` with fair-esm's ``MultiheadAttention`` (q/k/v/out_proj) --
    true for ESM-IF1's encoder, ESM2, and ESM3's transformer stack alike, so
    this one function backs LoRA injection for every backbone.
    """
    total = len(layers)
    start = max(0, total - n_layers)
    for i in range(start, total):
        attn = layers[i].self_attn
        # Force the manual q/k/v path so LoRA.forward is actually invoked; the
        # fused F.multi_head_attention_forward fast path reads q_proj.weight
        # directly and would silently bypass the adapter.
        attn.enable_torch_version = False
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha))


def build_head(hidden: int, head_dim: int | None, dropout: float) -> nn.Module:
    """Per-residue scoring head: direct Linear(hidden,1) if ``head_dim`` is None,
    else an MLP Linear(hidden,head_dim) -> GELU -> Dropout -> Linear(head_dim,1)."""
    if head_dim is None:
        return nn.Linear(hidden, 1)
    return nn.Sequential(
        nn.Linear(hidden, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, 1)
    )


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


class EpitopeModel(nn.Module):
    """Shared per-residue scoring head + checkpoint plumbing for every frozen-
    backbone + LoRA epiLoRA variant (ESM-IF1, ESM2, ESM3, ESMc, ...).

    Subclasses build their own frozen backbone + LoRA adapters in ``__init__``
    (call ``self._init_head(hidden, dropout, head_dim)`` once ``self.hidden``
    is known), then implement ``_encode(coords_batch, seq_batch) -> hidden``
    returning a ``(B, L+1, hidden)`` tensor (leading special/bos token then one
    row per residue) -- ``forward`` here handles the rest identically for
    every backbone.
    """

    def _init_head(self, hidden: int, dropout: float, head_dim: int | None) -> None:
        self.hidden = hidden
        self.head_ln = nn.LayerNorm(hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head = build_head(hidden, head_dim, dropout)

    @property
    def device(self):
        return self.head_ln.weight.device

    def _encode(self, coords_batch, seq_batch):
        raise NotImplementedError

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


# ==== ESM-IF1 backbone (structure, champion) ===============================


class ESMIF1EpitopeModel(EpitopeModel):
    """Frozen ESM-IF1 + LoRA + RYS + per-residue epitope head."""

    def __init__(self, esm_model, alphabet, rank=4, alpha=8.0,
                 n_lora_layers=8, rys_start=4, rys_end=8,
                 dropout=0.1, head_dim=None):
        super().__init__()
        self.esm = esm_model
        self.alphabet = alphabet
        self._cfg = dict(rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         rys_start=rys_start, rys_end=rys_end, dropout=dropout,
                         head_dim=head_dim)
        for p in self.esm.parameters():
            p.requires_grad = False
        inject_lora_layers(self.esm.encoder.layers, rank, alpha, n_lora_layers)
        if rys_end > rys_start:
            patch_encoder_rys(self.esm.encoder, rys_start, rys_end)
        self._init_head(HIDDEN, dropout, head_dim)

    def _encode(self, coords_batch, seq_batch):
        # Stock fair-esm inverse-folding collate (pads coords, builds mask/confidence).
        from esm.inverse_folding.util import CoordBatchConverter
        dev = self.device
        bc = CoordBatchConverter(self.alphabet)
        batch = [(c, None, s) for c, s in zip(coords_batch, seq_batch)]
        coords_t, confidence, _, _, padding_mask = bc(batch, device=dev)
        enc = self.esm.encoder.forward(coords_t, padding_mask, confidence, return_all_hiddens=False)
        return enc["encoder_out"][0].permute(1, 0, 2)  # (B, L, 512)


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


# ==== ESM2 backbone (sequence-only) =========================================
#
# ESM2 is a sequence-only protein language model -- no backbone coordinates
# are used, only the residue sequence. This exists to answer one of the
# epiLoRA ablation questions: how much does reading structure (ESM-IF1) vs.
# sequence alone (ESM2) matter for epitope prediction, holding the
# LoRA-adapter recipe and head fixed. RYS (ESM-IF1-encoder-specific) is not
# implemented here.

# name -> (fair-esm loader attr, hidden dim, total transformer layers)
ESM2_SIZES = {
    "35M": ("esm2_t12_35M_UR50D", 480, 12),
    "150M": ("esm2_t30_150M_UR50D", 640, 30),
    "650M": ("esm2_t33_650M_UR50D", 1280, 33),
}


class ESM2EpitopeModel(EpitopeModel):
    """Frozen ESM2 + LoRA + per-residue epitope head (sequence-only input)."""

    def __init__(self, esm_model, alphabet, size: str, rank: int, alpha: float,
                 n_lora_layers: int, dropout: float, head_dim: int | None):
        super().__init__()
        self.esm = esm_model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.size = size
        self.n_layers = ESM2_SIZES[size][2]
        self._cfg = dict(size=size, rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         dropout=dropout, head_dim=head_dim)
        for p in self.esm.parameters():
            p.requires_grad = False
        inject_lora_layers(self.esm.layers, rank, alpha, n_lora_layers)
        self._init_head(ESM2_SIZES[size][1], dropout, head_dim)

    def _encode(self, coords_batch, seq_batch):
        """``coords_batch`` is accepted (for interface parity with the ESM-IF1
        model) but ignored -- ESM2 reads sequence only."""
        data = [(f"seq{i}", s) for i, s in enumerate(seq_batch)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        out = self.esm(tokens, repr_layers=[self.n_layers])
        return out["representations"][self.n_layers]  # (B, L+2, hidden): bos, residues..., eos


def load_base_esm2(size: str):
    """Load the pretrained (frozen) ESM2 backbone + alphabet."""
    import esm
    loader_name = ESM2_SIZES[size][0]
    esm_model, alphabet = getattr(esm.pretrained, loader_name)()
    return esm_model.eval(), alphabet


def build_model_esm2(device: str = "cpu", size: str = "650M", rank: int = 4, alpha: float = 8.0,
                n_lora_layers: int = 8, dropout: float = 0.1, head_dim: int | None = None) -> "ESM2EpitopeModel":
    """Build an (untrained) ESM2-backbone epiLoRA model on ``device``."""
    esm_model, alphabet = load_base_esm2(size)
    model = ESM2EpitopeModel(esm_model, alphabet, size=size, rank=rank, alpha=alpha,
                             n_lora_layers=n_lora_layers, dropout=dropout, head_dim=head_dim).to(device)
    return model


# ==== ESM3 backbone (sequence track) ========================================
#
# Like ESM2 above, this is a sequence-only comparison point for the epiLoRA
# backbone ablation -- no backbone coordinates are used. ESM3 is multimodal
# (sequence/structure/function tracks) but we only ever feed it sequence
# tokens; every other track's input defaults to its own mask/pad token inside
# ESM3.forward, which is how the reference API itself supports sequence-only
# use. RYS (an ESM-IF1-encoder-specific trick) is not implemented here.
#
# Must run under epilora/env_esm3/bin/python3: EvolutionaryScale's ``esm`` pip
# package and fair-esm's ``esm`` package (used by the ESM-IF1/ESM2 sections
# above) collide on the same import name and cannot coexist in one
# environment.

ESM3_HIDDEN = 1536
ESM3_N_LAYERS = 48


def inject_lora_esm3(blocks, rank: int, alpha: float, n_layers: int) -> None:
    """Add LoRA adapters to the fused QKV projection + out_proj of the last
    ``n_layers`` of ``blocks`` (ESM3/ESMc's UnifiedTransformerBlock stack).

    ESM3's attention module (esm.layers.attention.MultiHeadAttention) has a
    fused QKV projection (``layernorm_qkv`` = LayerNorm -> Linear(d_model,
    3*d_model)) plus a separate ``out_proj``, unlike fair-esm's separate
    q/k/v/out_proj layout -- so it needs its own LoRA injection helper
    (``inject_lora_layers`` above assumes the fair-esm layout and won't work
    here). ESMc shares this same block layout, so it reuses this helper too.
    """
    total = len(blocks)
    start = max(0, total - n_layers)
    for i in range(start, total):
        attn = blocks[i].attn
        attn.layernorm_qkv[1] = LoRALinear(attn.layernorm_qkv[1], rank, alpha)
        attn.out_proj = LoRALinear(attn.out_proj, rank, alpha)


class ESM3EpitopeModel(EpitopeModel):
    """Frozen ESM3 (sequence track) + LoRA + per-residue epitope head."""

    def __init__(self, esm_model, rank: int = 4, alpha: float = 8.0,
                 n_lora_layers: int = 8, dropout: float = 0.1, head_dim: int | None = None):
        super().__init__()
        self.esm = esm_model
        self._cfg = dict(rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         dropout=dropout, head_dim=head_dim)
        for p in self.esm.parameters():
            p.requires_grad = False
        inject_lora_esm3(self.esm.transformer.blocks, rank, alpha, n_lora_layers)
        self._init_head(ESM3_HIDDEN, dropout, head_dim)

    def _encode(self, coords_batch, seq_batch):
        """``coords_batch`` is accepted (for interface parity with the ESM-IF1
        model) but ignored -- this backbone reads ESM3's sequence track only.

        train.py only ever calls forward() with single-element lists (see its
        per-sample training loop), so no padding/attention-masking across
        differently-sized sequences is needed here.
        """
        from esm.utils.encoding import tokenize_sequence
        if len(seq_batch) != 1:
            raise ValueError("ESM3EpitopeModel only supports batch size 1")
        tokens = tokenize_sequence(seq_batch[0], self.esm.tokenizers.sequence,
                                   add_special_tokens=True).unsqueeze(0).to(self.device)
        out = self.esm.forward(sequence_tokens=tokens)
        return out.embeddings  # (1, L+2, hidden): bos, residues..., eos


def load_base_esm3():
    """Load the pretrained (frozen) ESM3-small-open backbone.

    Loaded on CPU first: ``ESM3.from_pretrained`` auto-casts to bfloat16 for
    any non-cpu device, which we don't want (our LoRA/head params are
    float32); the caller moves the wrapped model to the target device after
    construction, staying in float32 throughout.
    """
    from esm.models.esm3 import ESM3
    return ESM3.from_pretrained(device=torch.device("cpu"))


def build_model_esm3(device: str = "cpu", rank: int = 4, alpha: float = 8.0,
                n_lora_layers: int = 8, dropout: float = 0.1,
                head_dim: int | None = None) -> "ESM3EpitopeModel":
    """Build an (untrained) ESM3-backbone epiLoRA model on ``device``."""
    esm_model = load_base_esm3()
    model = ESM3EpitopeModel(esm_model, rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                             dropout=dropout, head_dim=head_dim).to(device)
    return model


# ==== ESMc backbone (sequence-only) =========================================
#
# ESMc is a sequence-only protein language model released by EvolutionaryScale
# (the ESM3 team) in the same ``esm`` pip package as ESM3, built from the same
# ``UnifiedTransformerBlock`` stack (fused QKV ``layernorm_qkv`` + separate
# ``out_proj``) -- so it reuses ``inject_lora_esm3`` above unchanged rather
# than duplicating a LoRA-injection helper for it.
#
# Must run under epilora/env_esm3/bin/python3, same as the ESM3 section above
# (same package collision with fair-esm's ``esm``).
#
# UNVERIFIED -- written by close analogy to the ESM3 section since this
# machine only has the fair-esm environment installed (no env_esm3, so
# EvolutionaryScale's ``esm`` package can't be imported here to confirm
# against). Once run in a real env_esm3, check:
#   * ``ESMC.from_pretrained`` accepts (name, device=...) the same way
#     ``ESM3.from_pretrained`` does.
#   * the tokenizer is exposed as ``self.esm.tokenizer`` (singular -- ESMc has
#     only one, sequence, track, unlike ESM3's ``tokenizers.sequence``/etc.
#     namespace); adjust ``_encode`` if that attribute name is different.
#   * ``self.esm.forward(sequence_tokens=...)`` returns an object with an
#     ``.embeddings`` field, mirroring ESM3's ``ESMOutput``.
#   * ``self.esm.transformer.blocks[i].attn`` exposes ``layernorm_qkv``/
#     ``out_proj`` exactly as ESM3's blocks do (this is why LoRA injection is
#     reused from the ESM3 section -- if it silently no-ops or errors, that
#     assumption was wrong).
#
# Hidden size and layer count are read off the loaded model at construction
# time (not hardcoded), so a wrong guess about ESMc's published dimensions
# can't silently produce a mis-shaped head.

# name -> ESMC.from_pretrained() loader name
ESMC_SIZES = {
    "300M": "esmc_300m",
    "600M": "esmc_600m",
}


class ESMCEpitopeModel(EpitopeModel):
    """Frozen ESMc + LoRA + per-residue epitope head (sequence-only input)."""

    def __init__(self, esm_model, size: str, rank: int = 4, alpha: float = 8.0,
                 n_lora_layers: int = 8, dropout: float = 0.1, head_dim: int | None = None):
        super().__init__()
        self.esm = esm_model
        self.size = size
        self._cfg = dict(size=size, rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         dropout=dropout, head_dim=head_dim)
        for p in self.esm.parameters():
            p.requires_grad = False
        blocks = self.esm.transformer.blocks
        inject_lora_esm3(blocks, rank, alpha, n_lora_layers)
        hidden = blocks[0].attn.out_proj.out_features
        self._init_head(hidden, dropout, head_dim)

    def _encode(self, coords_batch, seq_batch):
        """``coords_batch`` is accepted (for interface parity with the ESM-IF1
        model) but ignored -- ESMc is sequence-only.

        Only batch size 1 is supported (mirrors ESM3EpitopeModel; train.py
        only ever calls forward() with single-element lists)."""
        from esm.utils.encoding import tokenize_sequence
        if len(seq_batch) != 1:
            raise ValueError("ESMCEpitopeModel only supports batch size 1")
        tokens = tokenize_sequence(seq_batch[0], self.esm.tokenizer,
                                   add_special_tokens=True).unsqueeze(0).to(self.device)
        out = self.esm.forward(sequence_tokens=tokens)
        return out.embeddings  # (1, L+2, hidden): bos, residues..., eos


def load_base_esmc(size: str):
    """Load the pretrained (frozen) ESMc backbone.

    Loaded on CPU first, same rationale as load_base_esm3: avoid an
    auto-cast to bfloat16 that would conflict with our float32 LoRA/head
    params; the caller moves the wrapped model to the target device after
    construction.
    """
    from esm.models.esmc import ESMC
    return ESMC.from_pretrained(ESMC_SIZES[size], device=torch.device("cpu"))


def build_model_esmc(device: str = "cpu", size: str = "600M", rank: int = 4, alpha: float = 8.0,
                n_lora_layers: int = 8, dropout: float = 0.1,
                head_dim: int | None = None) -> "ESMCEpitopeModel":
    """Build an (untrained) ESMc-backbone epiLoRA model on ``device``."""
    esm_model = load_base_esmc(size)
    model = ESMCEpitopeModel(esm_model, size=size, rank=rank, alpha=alpha,
                             n_lora_layers=n_lora_layers, dropout=dropout, head_dim=head_dim).to(device)
    return model


# ==== ProstT5 backbone (sequence-only) ======================================
#
# ProstT5 (Rostlab/ProstT5) is a T5 encoder-decoder pretrained on AA<->3Di
# "translation"; we only ever use its encoder in AA2fold direction as a
# sequence-only protein embedder (no backbone coordinates used), same
# ablation role as ESM2/ESM3/ESMc. Reconstructed from a checkpoint's saved
# config -- ``{'name': 'Rostlab/ProstT5', 'rank': 4, 'alpha': 8.0,
# 'n_lora_layers': 8, 'dropout': ..., 'head_dim': None}`` -- and the
# checkpoint's trainable-state key names (``t5.encoder.block.{16..23}.layer.0.
# SelfAttention.{q,k,v,o}.lora_{A,B}``, ``head_ln.*``, ``head.*``): this repo's
# own training code for this backbone was never committed, so this follows
# Rostlab's documented ProstT5 usage (github.com/mheinzinger/ProstT5) --
# uppercase sequence, rare AAs (U,Z,O,B) mapped to X, space-joined characters,
# "<AA2fold>" direction-prefix token -- rather than a from-scratch guess.

PROSTT5_HIDDEN = 1024
PROSTT5_N_LAYERS = 24


def inject_lora_t5(blocks, rank: int, alpha: float, n_layers: int) -> None:
    """Add LoRA adapters to q/k/v/o of the last ``n_layers`` T5 encoder blocks.

    ``blocks`` is a T5Stack's ``.block`` ModuleList; each block's self-attention
    lives at ``block[i].layer[0].SelfAttention`` with separate q/k/v/o Linears
    (T5's own layout, unlike fair-esm's fused-vs-separate variants above).
    """
    total = len(blocks)
    start = max(0, total - n_layers)
    for i in range(start, total):
        attn = blocks[i].layer[0].SelfAttention
        for name in ("q", "k", "v", "o"):
            setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha))


class ProstT5EpitopeModel(EpitopeModel):
    """Frozen ProstT5 encoder (AA2fold direction) + LoRA + per-residue epitope head."""

    def __init__(self, t5_model, tokenizer, name: str = "Rostlab/ProstT5",
                 rank: int = 4, alpha: float = 8.0, n_lora_layers: int = 8,
                 dropout: float = 0.1, head_dim: int | None = None):
        super().__init__()
        self.t5 = t5_model
        self.tokenizer = tokenizer
        self._cfg = dict(name=name, rank=rank, alpha=alpha, n_lora_layers=n_lora_layers,
                         dropout=dropout, head_dim=head_dim)
        for p in self.t5.parameters():
            p.requires_grad = False
        inject_lora_t5(self.t5.encoder.block, rank, alpha, n_lora_layers)
        self._init_head(PROSTT5_HIDDEN, dropout, head_dim)

    def _encode(self, coords_batch, seq_batch):
        """``coords_batch`` is accepted (for interface parity with the ESM-IF1
        model) but ignored -- ProstT5 here is sequence-only.

        Sequences are uppercased, rare amino acids (U,Z,O,B) mapped to X,
        space-joined, and prefixed with the "<AA2fold>" direction token per
        Rostlab's documented usage -- so token 0 is the prefix token and
        tokens 1..L are the L residues, matching the base class's
        ``hidden[b, 1:L+1]`` convention exactly.
        """
        import re
        prepped = ["<AA2fold> " + " ".join(re.sub(r"[UZOB]", "X", s.upper()))
                   for s in seq_batch]
        enc = self.tokenizer(
            prepped, add_special_tokens=True, padding="longest", return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.t5.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state  # (B, 1+L+..., hidden): AA2fold prefix, residues..., pad


def load_base_prostt5(name: str = "Rostlab/ProstT5"):
    """Load the pretrained (frozen) ProstT5 encoder + tokenizer."""
    from transformers import T5EncoderModel, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(name)
    return model.eval(), tokenizer


def build_model_prostt5(device: str = "cpu", name: str = "Rostlab/ProstT5",
                rank: int = 4, alpha: float = 8.0, n_lora_layers: int = 8,
                dropout: float = 0.1, head_dim: int | None = None) -> "ProstT5EpitopeModel":
    """Build an (untrained) ProstT5-backbone epiLoRA model on ``device``."""
    t5_model, tokenizer = load_base_prostt5(name)
    model = ProstT5EpitopeModel(t5_model, tokenizer, name=name, rank=rank, alpha=alpha,
                                n_lora_layers=n_lora_layers, dropout=dropout, head_dim=head_dim).to(device)
    return model
