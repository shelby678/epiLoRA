"""MUTABLE: ESM3 frozen backbone + structure-aware trainable classification head.

Sequences are tokenised with ESM3's native vocabulary.  For each sample whose
FASTA ID matches a PDB file (format: ``{pdb}_{heavy}_{light}_{antigen}``,
e.g. ``7wvg_C_D_B``), backbone N/CA/C coordinates for the antigen chain are
extracted and passed to ESM3 as ``structure_coords`` (shape B×L×3×3).  Token
positions without structure—either because no PDB matched or because a
residue was missing—receive NaN coordinates; ESM3 treats these as structurally
undefined and uses sequence context only.

Architecture
------------
* ESM3-small-open (1536-dim, 48 layers) — frozen backbone.
* Optional LoRA: inject trainable rank-decomposition into attention QKV and
  out_proj of the last N blocks.
* Optional RYS (Repeat Yourself): replay a contiguous segment of layers twice.
* Trainable head:  Dropout → Linear(1536 → 1) per token position.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from prepare import (
    PAD_ID, load_fasta, load_fasta_pairs, load_combined_fasta,
    load_combined_fasta_partitioned, pu_loss, Sample,
)
from features import compute_rsa, compute_extra_features, extra_feature_dim

try:
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# === HYPERPARAMETERS (agent-tunable) ===
DROPOUT = 0.1
MAX_SEQ_LEN = 512
BATCH_SIZE = 8          # smaller than ESM2 runs — ESM3 is much larger
LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 100

# RYS (Repeat Yourself) — repeat a segment of ESM3 transformer blocks.
# Layers 0..RYS_START-1 run once; RYS_START..RYS_END-1 run twice; RYS_END..47 run once.
# Set RYS_END <= RYS_START to disable.
RYS_START = 36
RYS_END = 44

# PU learning prior: estimated fraction of positives in the unlabelled set.
# Set to 0.0 to use standard BCE loss (disabled).
PU_PRIOR = 0.0

# ESM3 sequence vocabulary constants (same as ESM2)
# BOS=0, PAD=1, EOS=2, L=4, A=5, G=6, V=7 ... mask=32
_ESM3_BOS = 0
_ESM3_PAD = 1
_ESM3_EOS = 2
_ESM3_MASK = 32

# Mapping from prepare.py token IDs → ESM3 token IDs (identical to ESM2)
from prepare import PAD_VOCAB_SIZE
_OUR_TO_ESM3: list[int] = [3] * PAD_VOCAB_SIZE  # default → UNK
_OUR_TO_ESM3[0] = 0    # <cls>  → BOS
_OUR_TO_ESM3[1] = 1    # <pad>  → PAD
_OUR_TO_ESM3[2] = 2    # <eos>  → EOS
_OUR_TO_ESM3[3] = 3    # <unk>  → UNK
_OUR_TO_ESM3[4] = 32   # <mask> → MASK
_OUR_TO_ESM3[5] = 5    # A
_OUR_TO_ESM3[6] = 23   # C
_OUR_TO_ESM3[7] = 13   # D
_OUR_TO_ESM3[8] = 9    # E
_OUR_TO_ESM3[9] = 18   # F
_OUR_TO_ESM3[10] = 6   # G
_OUR_TO_ESM3[11] = 21  # H
_OUR_TO_ESM3[12] = 12  # I
_OUR_TO_ESM3[13] = 15  # K
_OUR_TO_ESM3[14] = 4   # L
_OUR_TO_ESM3[15] = 20  # M
_OUR_TO_ESM3[16] = 17  # N
_OUR_TO_ESM3[17] = 14  # P
_OUR_TO_ESM3[18] = 16  # Q
_OUR_TO_ESM3[19] = 10  # R
_OUR_TO_ESM3[20] = 8   # S
_OUR_TO_ESM3[21] = 11  # T
_OUR_TO_ESM3[22] = 7   # V
_OUR_TO_ESM3[23] = 22  # W
_OUR_TO_ESM3[24] = 19  # Y


# ---------------------------------------------------------------------------
# PDB backbone parsing — pure Python, no BioPython needed
# ---------------------------------------------------------------------------

_BACKBONE_ATOMS = ("N", "CA", "C")


def parse_pdb_backbone(
    pdb_path: Path,
    chain_id: str,
) -> list[tuple[float, float, float, float, float, float, float, float, float]] | None:
    """Extract backbone N, CA, C coordinates per residue from a PDB file.

    Args:
        pdb_path:  Path to the PDB file.
        chain_id:  Chain identifier (single character, e.g. ``"B"``).

    Returns:
        A list of (N_x, N_y, N_z, CA_x, CA_y, CA_z, C_x, C_y, C_z) per residue,
        in residue order.  Returns ``None`` if the chain is absent or any
        residue is missing one of the three backbone atoms.
    """
    # Accumulate per-residue atom coords:  key=(resseq, icode) → {atom_name: (x,y,z)}
    residue_atoms: dict[tuple[int, str], dict[str, tuple[float, float, float]]] = {}
    residue_order: list[tuple[int, str]] = []

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in _BACKBONE_ATOMS:
                continue
            if line[21] != chain_id:
                continue
            alt_loc = line[16]
            if alt_loc not in (" ", "A"):
                continue
            try:
                resseq = int(line[22:26])
                icode = line[26]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (resseq, icode)
            if key not in residue_atoms:
                residue_atoms[key] = {}
                residue_order.append(key)
            if atom_name not in residue_atoms[key]:
                residue_atoms[key][atom_name] = (x, y, z)

    if not residue_order:
        return None

    # Build flat list in residue order; require all three backbone atoms
    result = []
    for key in sorted(residue_order):
        atoms = residue_atoms.get(key, {})
        if not all(a in atoms for a in _BACKBONE_ATOMS):
            return None  # any missing atom breaks the alignment
        n = atoms["N"]
        ca = atoms["CA"]
        c = atoms["C"]
        result.append((*n, *ca, *c))

    return result


def _parse_seq_id(seq_id: str) -> tuple[str, str] | None:
    """Return (pdb_id_lower, antigen_chain) from a FASTA header string.

    Supports two formats:

    * **New** (pdb_chains.fasta): ``"1A2Y_BA C 2"``  — space-separated fields;
      first field is ``PDBID_antibody``, second is the antigen chain.
    * **Old** (combined/1/): ``"7wvg_C_D_B"`` — underscore-separated;
      first field is pdb_id, last field is the antigen chain.
    """
    if " " in seq_id:
        # New format: "1A2Y_BA C 2"
        parts = seq_id.split()
        pdb_id = parts[0].split("_")[0].lower()
        antigen = parts[1] if len(parts) >= 2 else None
    else:
        # Old format: "7wvg_C_D_B"
        parts = seq_id.split("_")
        if len(parts) < 4:
            return None
        pdb_id = parts[0].lower()
        antigen = parts[3]

    if not antigen:
        return None
    return pdb_id, antigen




# ---------------------------------------------------------------------------
# Extended sample type
# ---------------------------------------------------------------------------

# (token_ids, epitope_labels, structure_coords_or_None, rsa_or_None)
# structure_coords shape: (seq_len_aa, 3, 3)  — aa positions only, no BOS/EOS
# rsa shape: (seq_len_aa,) — relative surface area in [0,1], no BOS/EOS
StructSample = tuple[list[int], list[int], Tensor | None, Tensor | None]


def _load_struct_pairs(
    aa_path: Path,
    bce_path: Path,
    structures_dir: Path,
    max_length: int,
) -> list[StructSample]:
    base_samples: list[Sample] = load_fasta_pairs(aa_path, bce_path)
    aa_seqs = load_fasta(aa_path)

    struct_samples: list[StructSample] = []
    for (token_ids, labels), (seq_id, aa_seq) in zip(base_samples, aa_seqs.items()):
        if len(token_ids) > max_length:
            continue
        coords, rsa = _load_coords_rsa(seq_id, len(aa_seq), structures_dir)
        struct_samples.append((token_ids, labels, coords, rsa))

    return struct_samples


def _load_coords_rsa(
    seq_id: str,
    seq_len: int,
    structures_dir: Path,
) -> tuple[Tensor | None, Tensor | None]:
    """Return (coords, rsa) for one sequence, or (None, None) if PDB unavailable."""
    parsed = _parse_seq_id(seq_id)
    if parsed is None:
        return None, None
    pdb_id, antigen = parsed

    pdb_path = structures_dir / pdb_id / "structure" / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return None, None

    flat = parse_pdb_backbone(pdb_path, antigen)
    if flat is None or len(flat) != seq_len:
        return None, None

    t = torch.tensor(flat, dtype=torch.float32)
    coords = t.view(seq_len, 3, 3)

    rsa_arr = compute_rsa(pdb_path, antigen, seq_len)
    rsa = torch.tensor(rsa_arr, dtype=torch.float32) if rsa_arr is not None else None

    return coords, rsa


def create_struct_datasets(
    data_dir: Path,
    structures_dir: Path,
    max_length: int = 512,
) -> tuple[list[StructSample], list[StructSample]]:
    """Load BCR epitope data with optional per-sample backbone coordinates."""
    for fname in ("train_aa.fasta", "train_bce.fasta", "valid_aa.fasta", "valid_bce.fasta"):
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    train_data = _load_struct_pairs(
        data_dir / "train_aa.fasta", data_dir / "train_bce.fasta", structures_dir, max_length
    )
    val_data = _load_struct_pairs(
        data_dir / "valid_aa.fasta", data_dir / "valid_bce.fasta", structures_dir, max_length
    )

    if not train_data:
        raise ValueError("No training sequences after filtering.")
    if not val_data:
        raise ValueError("No validation sequences after filtering.")

    return train_data, val_data


def create_struct_datasets_from_combined(
    fasta_path: Path,
    structures_dir: Path,
    max_length: int = 512,
    val_partition: str = "5",
    pretrain_fasta: Path | None = None,
    exclude_partitions: frozenset[str] = frozenset(),
) -> tuple[list[StructSample], list[StructSample]]:
    """Load data from a combined-FASTA file (pdb_chains.fasta format).

    Sequences are split into train/val by the ``val_partition`` field in
    each FASTA header.  Backbone coordinates are loaded for entries whose
    PDB structure exists in ``structures_dir``.

    Args:
        fasta_path:      Path to the primary combined-FASTA (e.g. ``data/pdb_chains.fasta``).
        structures_dir:  Root of SAbDab PDB dataset.
        max_length:      Maximum tokenised sequence length (incl. BOS/EOS).
        val_partition:   Header partition field to use as validation split.
        pretrain_fasta:  Optional path to an additional combined-FASTA whose
                         entries are appended to the *training* set only
                         (structures are not loaded for these).  Useful for
                         mixing in ``data/pdb_contacts.fasta`` pretraining
                         sequences.

    Returns:
        ``(train_data, val_data)`` lists of ``StructSample``.
    """
    train_headers, val_headers = load_combined_fasta(
        fasta_path, val_partition=val_partition, exclude_partitions=exclude_partitions
    )

    def _to_struct(header_samples: list, load_struct: bool) -> list[StructSample]:
        out: list[StructSample] = []
        for header, (token_ids, labels) in header_samples:
            if len(token_ids) > max_length:
                continue
            if load_struct:
                seq_len = len(token_ids) - 2  # strip BOS/EOS
                coords, rsa = _load_coords_rsa(header, seq_len, structures_dir)
            else:
                coords, rsa = None, None
            out.append((token_ids, labels, coords, rsa))
        return out

    train_data = _to_struct(train_headers, load_struct=True)
    val_data   = _to_struct(val_headers,   load_struct=True)

    # Mix in pretraining sequences (no structure)
    if pretrain_fasta is not None and pretrain_fasta.exists():
        pretrain_headers, _ = load_combined_fasta(pretrain_fasta, val_partition="__none__")
        pretrain_samples = _to_struct(pretrain_headers, load_struct=False)
        train_data = train_data + pretrain_samples
        logger.info(
            f"Pretraining data: {len(pretrain_samples):,} additional sequences "
            f"from {pretrain_fasta}"
        )

    if not train_data:
        raise ValueError("No training sequences after filtering.")
    if not val_data:
        raise ValueError("No validation sequences after filtering.")

    return train_data, val_data


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def struct_collate_fn(batch: list[StructSample]) -> dict[str, Tensor]:
    """Pad sequences, structure coords, and RSA to max-in-batch length.

    ``structure_coords`` and ``rsa`` are filled with ``nan`` at:
      * BOS and EOS positions (position 0 and -1),
      * padding positions,
      * all positions when a sample has no structure / RSA.
    """
    max_len = max(len(token_ids) for token_ids, _, _, _ in batch)
    B = len(batch)

    padded = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    coords = torch.full((B, max_len, 3, 3), float("nan"), dtype=torch.float32)
    rsa = torch.full((B, max_len), float("nan"), dtype=torch.float32)

    for i, (token_ids, epi_labels, sample_coords, sample_rsa) in enumerate(batch):
        length = len(token_ids)
        padded[i, :length] = torch.tensor(token_ids, dtype=torch.long)
        labels[i, :length] = torch.tensor(epi_labels, dtype=torch.long)
        attention_mask[i, :length] = 1
        if sample_coords is not None:
            coords[i, 1 : length - 1] = sample_coords
        if sample_rsa is not None:
            rsa[i, 1 : length - 1] = sample_rsa

    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "labels": labels,
        "structure_coords": coords,
        "rsa": rsa,
    }


class StructDataset(Dataset):
    def __init__(self, data: list[StructSample]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> StructSample:
        return self.data[idx]


def create_struct_dataloader(
    data: list[StructSample],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        StructDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=struct_collate_fn,
        num_workers=0,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper.

    The wrapped ``linear`` stays frozen.  Only ``lora_A`` and ``lora_B`` are
    trainable.  Forward: ``W x + scale * B A x``  where scale = alpha / rank.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        d_out, d_in = linear.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        base = self.linear(x)
        # cast to input dtype so bfloat16 autocast works correctly
        A = self.lora_A.to(dtype=x.dtype)
        B = self.lora_B.to(dtype=x.dtype)
        return base + self.scale * (x @ A.T @ B.T)


class HeadMaskedLoRA(nn.Module):
    """LoRA on the QKV fused linear (1536→4608) whose delta only writes to
    output dimensions belonging to a specific subset of attention heads.

    For each target head h, the delta is applied to three slices of the
    output (1536-d Q, K, V each, with head h occupying dims [h*head_dim,
    (h+1)*head_dim) inside its chunk). All other output dimensions are
    untouched, so the un-selected heads behave identically to the frozen
    base.

    Total trainable params: rank * (d_in + len(target_heads) * 3 * head_dim).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float,
                 n_heads: int, head_dim: int, target_heads: list[int]) -> None:
        super().__init__()
        self.linear = linear
        d_out, d_in = linear.weight.shape  # (4608, 1536)
        assert d_out == 3 * n_heads * head_dim, (
            f"HeadMaskedLoRA expects 3*n_heads*head_dim output dims, got "
            f"{d_out} != 3*{n_heads}*{head_dim}"
        )
        active_dim = len(target_heads) * head_dim * 3
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(active_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank
        # Output indices for active heads' Q, K, V slices.
        idx: list[int] = []
        for h in target_heads:
            for chunk in range(3):
                start = chunk * n_heads * head_dim + h * head_dim
                idx.extend(range(start, start + head_dim))
        self.register_buffer("out_idx", torch.tensor(idx, dtype=torch.long))
        self.target_heads = list(target_heads)

    def forward(self, x: Tensor) -> Tensor:
        base = self.linear(x)
        A = self.lora_A.to(dtype=x.dtype)
        B = self.lora_B.to(dtype=x.dtype)
        delta = self.scale * (x @ A.T @ B.T)  # (..., active_dim)
        return torch.index_add(base, -1, self.out_idx, delta)


def _inject_head_lora(
    esm3: nn.Module,
    rank: int,
    alpha: float,
    targets: dict[int, list[int]],
) -> None:
    """Inject HeadMaskedLoRA on the QKV linear of each block listed in
    ``targets`` (block_idx → list of head indices to adapt).

    Untargeted blocks remain fully frozen. Out_proj is not adapted by this
    function — the selective experiment focuses on input-side QKV slices,
    keeping the param count proportional to the number of selected heads.
    """
    blocks = esm3.layers if hasattr(esm3, "layers") else esm3.transformer.blocks
    for block_idx, heads in targets.items():
        if not heads:
            continue
        block = blocks[block_idx]
        if not hasattr(block, "attn") or block.attn is None:
            continue
        n_heads = block.attn.n_heads
        head_dim = block.attn.layernorm_qkv[1].weight.shape[0] // (3 * n_heads)
        qkv_lin = block.attn.layernorm_qkv[1]
        if isinstance(qkv_lin, (LoRALinear, HeadMaskedLoRA)):
            continue
        block.attn.layernorm_qkv[1] = HeadMaskedLoRA(
            qkv_lin, rank, alpha, n_heads, head_dim, heads
        )


def _inject_lora(
    esm3: nn.Module,
    rank: int,
    alpha: float,
    n_blocks: int,
    inject_ffn: bool = False,
    inject_geom_attn: bool = False,
    lora_block_start: int = -1,
) -> None:
    """Replace attention (and optionally FFN) linears with LoRA wrappers.

    Attention: ``attn.layernorm_qkv[1]`` (QKV fused, 1536→4608) and
    ``attn.out_proj`` (1536→1536).
    FFN (when inject_ffn=True): ``ffn[1]`` (1536→8192) and ``ffn[3]`` (4096→1536).
    Geometric attention (when inject_geom_attn=True): ``geom_attn.proj`` and
    ``geom_attn.out_proj`` in block 0 only (the sole block with geom_attn).
    lora_block_start: if >=0, inject into blocks [lora_block_start, lora_block_start+n_blocks).
                      If -1 (default), inject into the last n_blocks blocks.
    """
    # ESM2 uses model.layers instead of model.transformer.blocks
    blocks = esm3.layers if hasattr(esm3, 'layers') else esm3.transformer.blocks
    n = len(blocks)
    start = lora_block_start if lora_block_start >= 0 else max(0, n - n_blocks)
    end = min(n, start + n_blocks)
    for i in range(start, end):
        block = blocks[i]
        if not hasattr(block, "attn"):
            continue
        # QKV projection (layernorm_qkv is Sequential[LayerNorm, Linear])
        qkv_lin = block.attn.layernorm_qkv[1]
        if not isinstance(qkv_lin, LoRALinear):  # safe to call multiple times
            block.attn.layernorm_qkv[1] = LoRALinear(qkv_lin, rank, alpha)
        # output projection
        if not isinstance(block.attn.out_proj, LoRALinear):
            block.attn.out_proj = LoRALinear(block.attn.out_proj, rank, alpha)
        # FFN up/down projections
        if inject_ffn and hasattr(block, "ffn"):
            block.ffn[1] = LoRALinear(block.ffn[1], rank, alpha)
            block.ffn[3] = LoRALinear(block.ffn[3], rank, alpha)
    # Geometric attention lives only in block 0
    if inject_geom_attn and hasattr(blocks[0], "geom_attn") and blocks[0].geom_attn is not None:
        ga = blocks[0].geom_attn
        ga.proj = LoRALinear(ga.proj, rank, alpha)
        ga.out_proj = LoRALinear(ga.out_proj, rank, alpha)


# ---------------------------------------------------------------------------
# Dropout variants (head dropout, LayerDrop)
# ---------------------------------------------------------------------------
# These are applied via monkey-patch on the ESM3 module forwards. State is
# stored on the parent transformer module (`_dropout_cfg`) and on each block /
# attention submodule (`_drop_mode`).
#
# Modes:
#   head_drop_mode:
#     "off"     — no head dropout
#     "uniform" — each head zeroed iid with prob `head_drop_prob` per batch
#     "active"  — pick top-K heads by per-batch |context| magnitude and zero
#                 them (with batch-level keep-prob = 1 - head_drop_prob)
#   layer_drop_mode:
#     "off"     — no layer dropout
#     "uniform" — each block's attn+ffn residual zeroed iid with prob
#                 `layer_drop_prob`
#     "active"  — drop the layer(s) whose recent EMA-averaged residual norm
#                 is highest (top-K), each independently with prob
#                 `layer_drop_prob`

import einops as _einops


def _patch_attention_for_head_drop(attn_module: nn.Module) -> None:
    """Replace MultiHeadAttention.forward with a version that supports head dropout,
    DropKey (column-wise drop of attention logits) and DropAttention (column-wise
    drop of attention weights with post-softmax renormalisation).

    Methods set via attributes on the attention module:
        _head_drop_mode / _head_drop_prob / _head_drop_topk  — see train()
        _dropkey_prob                                        — DropKey paper (column)
        _dropattn_prob                                       — DropAttention (NoGrad'd)
    """
    import functools as _functools
    orig_layernorm_qkv = attn_module.layernorm_qkv
    orig_out_proj = attn_module.out_proj
    orig_q_ln = attn_module.q_ln
    orig_k_ln = attn_module.k_ln
    orig_rotary_method = attn_module._apply_rotary
    n_heads = attn_module.n_heads

    def new_forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD = self.q_ln(query_BLD).to(query_BLD.dtype)
        key_BLD = self.k_ln(key_BLD).to(query_BLD.dtype)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        reshaper = _functools.partial(
            _einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )
        query_BHLD, key_BHLD, value_BHLD = map(reshaper, (query_BLD, key_BLD, value_BLD))

        dk_prob = getattr(self, "_dropkey_prob", 0.0) if self.training else 0.0
        da_prob = getattr(self, "_dropattn_prob", 0.0) if self.training else 0.0

        if dk_prob <= 0.0 and da_prob <= 0.0:
            # Fast path: fused SDPA, identical to the previous behaviour.
            if seq_id is not None:
                mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                mask_BHLL = mask_BLL.unsqueeze(1)
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD, mask_BHLL
                )
            else:
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD
                )
        else:
            # Manual SDPA so we can mask logits/weights before/after softmax.
            d_head = query_BHLD.shape[-1]
            scores = torch.matmul(
                query_BHLD, key_BHLD.transpose(-1, -2)
            ) / math.sqrt(d_head)  # (B,H,Lq,Lk)
            if seq_id is not None:
                seq_mask = (seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)).unsqueeze(1)
                scores = scores.masked_fill(~seq_mask, float("-inf"))
            if dk_prob > 0.0:
                # Column-wise DropKey: each (B, head) chooses keys to mask iid;
                # the mask is shared across all queries within the head.
                B, H, _, Lk = scores.shape
                drop_BHK = torch.rand(B, H, Lk, device=scores.device) < dk_prob
                scores = scores.masked_fill(drop_BHK.unsqueeze(2), float("-inf"))
            weights = F.softmax(scores, dim=-1)
            if da_prob > 0.0:
                # DropAttention: zero out columns AFTER softmax, then renormalise
                # so each row sums to 1. NoGrad on the rescaling factor — matches
                # the paper's formulation and intentionally introduces gradient
                # noise that the authors report as harmful (kept for ablation).
                B, H, Lq, Lk = weights.shape
                drop_BHK = torch.rand(B, H, Lk, device=weights.device) < da_prob
                keep = (~drop_BHK).to(weights.dtype).unsqueeze(2)  # (B,H,1,Lk)
                weights = weights * keep
                with torch.no_grad():
                    row_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                weights = weights / row_sum
            context_BHLD = torch.matmul(weights, value_BHLD)

        # Apply head dropout
        mode = getattr(self, "_head_drop_mode", "off")
        prob = getattr(self, "_head_drop_prob", 0.0)
        topk = getattr(self, "_head_drop_topk", 0)
        if self.training and mode != "off" and prob > 0.0:
            H = context_BHLD.shape[1]
            if mode == "uniform":
                # iid per-head Bernoulli mask, inverted-dropout scaling
                keep = (torch.rand(H, device=context_BHLD.device) > prob).to(context_BHLD.dtype)
                scale = 1.0 / max(1e-6, 1.0 - prob)
                mask = keep * scale
                context_BHLD = context_BHLD * mask.view(1, -1, 1, 1)
            elif mode == "active":
                if torch.rand(1, device=context_BHLD.device).item() < prob:
                    head_mag = context_BHLD.detach().abs().mean(dim=(0, 2, 3))  # (H,)
                    k = min(int(topk), H - 1)
                    if k > 0:
                        top_idx = torch.topk(head_mag, k).indices
                        keep = torch.ones(H, device=context_BHLD.device, dtype=context_BHLD.dtype)
                        keep[top_idx] = 0.0
                        # inverted-dropout scale to preserve expected magnitude
                        scale = H / max(1.0, float(H - k))
                        context_BHLD = context_BHLD * (keep * scale).view(1, -1, 1, 1)

        context_BLD = _einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)

    import types as _types
    attn_module.forward = _types.MethodType(new_forward, attn_module)


def _patch_block_for_layer_drop(block: nn.Module, layer_idx: int) -> None:
    """Replace block.forward with version supporting LayerDrop residual scaling."""
    import types as _types

    def new_forward(self, x, sequence_id, frames, frames_mask, chain_id):
        mode = getattr(self, "_layer_drop_mode", "off")
        prob = getattr(self, "_layer_drop_prob", 0.0)
        scale_factor = 1.0  # multiplier applied to attn+ffn residual

        drop = False
        if self.training and mode != "off" and prob > 0.0:
            if mode == "uniform":
                if torch.rand(1, device=x.device).item() < prob:
                    drop = True
            elif mode == "active":
                # Caller selects which layers to drop and sets a per-step flag.
                if getattr(self, "_drop_this_step", False):
                    drop = True

        if drop:
            # Skip attn + ffn residuals; identity layer for this batch.
            # Still record activation magnitude (==0) for EMA bookkeeping.
            if hasattr(self, "_act_norm_log"):
                self._act_norm_log.append(0.0)
            self._drop_this_step = False
            return x

        out = x
        if self.use_plain_attn:
            r1 = self.attn(out, sequence_id)
            out = out + (r1 * scale_factor) / self.scaling_factor
        if self.use_geom_attn:
            r2 = self.geom_attn(out, frames, frames_mask, sequence_id, chain_id)
            out = out + r2 / self.scaling_factor
        r3 = self.ffn(out) / self.scaling_factor
        out = out + r3

        # EMA bookkeeping for active-mode selection
        if mode == "active" or getattr(self, "_track_act_norm", False):
            with torch.no_grad():
                delta = (out - x).detach().float().abs().mean().item()
            ema_decay = 0.95
            cur = getattr(self, "_act_norm_ema", None)
            self._act_norm_ema = delta if cur is None else ema_decay * cur + (1 - ema_decay) * delta

        self._drop_this_step = False
        return out

    block.forward = _types.MethodType(new_forward, block)
    block._layer_idx = layer_idx


def _hiddencut_hook(module, _input, output):
    """Forward hook for the FFN activation: element-wise Bernoulli dropout
    with inverted-dropout scaling. Element-wise HiddenCut from Wang et al.
    ACL 2024 Findings (LoRA + dropout unified framework)."""
    prob = getattr(module, "_hiddencut_prob", 0.0)
    if not module.training or prob <= 0.0:
        return output
    keep_p = 1.0 - prob
    mask = (torch.rand_like(output) > prob).to(output.dtype)
    return output * (mask / keep_p)


def _attach_hiddencut(block: nn.Module, target_idx: int = 2) -> None:
    """Attach a HiddenCut forward hook on ``block.ffn[target_idx]``.

    ESM3 FFN is Sequential[LayerNorm, Linear-up (1536→8192), SwiGLU (8192→4096),
    Linear-down (4096→1536)]; index 2 outputs the 4096-d activated hidden
    representation, the same position HiddenCut targets in the paper."""
    if not hasattr(block, "ffn"):
        return
    if isinstance(block.ffn, nn.Sequential) and len(block.ffn) > target_idx:
        target = block.ffn[target_idx]
    else:
        target = block.ffn
    if not getattr(target, "_hiddencut_patched", False):
        target.register_forward_hook(_hiddencut_hook)
        target._hiddencut_patched = True
    block._hiddencut_target = target


def _configure_dropout(
    esm3: nn.Module,
    *,
    head_drop_mode: str = "off",
    head_drop_prob: float = 0.0,
    head_drop_topk: int = 0,
    layer_drop_mode: str = "off",
    layer_drop_prob: float = 0.0,
    layer_drop_topk: int = 0,
    layer_drop_layers: list[int] | None = None,
    dropkey_prob: float = 0.0,
    dropattn_prob: float = 0.0,
    hiddencut_prob: float = 0.0,
    paper_drop_layers: list[int] | None = None,
) -> None:
    """Patch ESM3 blocks/attention to support head + layer dropout AND
    paper-style DropKey/DropAttention/HiddenCut.

    ``paper_drop_layers`` (default: all blocks) restricts DropKey/DropAttention/
    HiddenCut to a subset of blocks, mirroring the LoRA target layers."""
    blocks = esm3.layers if hasattr(esm3, "layers") else esm3.transformer.blocks
    n_blocks = len(blocks)
    paper_set = (
        set(paper_drop_layers) if paper_drop_layers is not None else set(range(n_blocks))
    )

    for i, block in enumerate(blocks):
        if not getattr(block, "_attn_patched_for_drop", False):
            if hasattr(block, "attn") and block.attn is not None:
                _patch_attention_for_head_drop(block.attn)
            block._attn_patched_for_drop = True
        if not getattr(block, "_block_patched_for_drop", False):
            _patch_block_for_layer_drop(block, i)
            block._block_patched_for_drop = True

        if hasattr(block, "attn") and block.attn is not None:
            block.attn._head_drop_mode = head_drop_mode
            block.attn._head_drop_prob = float(head_drop_prob)
            block.attn._head_drop_topk = int(head_drop_topk)
            if i in paper_set:
                block.attn._dropkey_prob = float(dropkey_prob)
                block.attn._dropattn_prob = float(dropattn_prob)
            else:
                block.attn._dropkey_prob = 0.0
                block.attn._dropattn_prob = 0.0
        block._layer_drop_mode = layer_drop_mode
        # Restrict eligibility: when layer_drop_layers is provided, only blocks
        # in that set get a nonzero drop prob. (Active mode also filters via
        # cfg["layers"] in _layer_drop_step, but uniform mode reads _layer_drop_prob
        # directly so we have to zero it out here for non-eligible blocks.)
        if layer_drop_layers is not None and i not in layer_drop_layers:
            block._layer_drop_prob = 0.0
        else:
            block._layer_drop_prob = float(layer_drop_prob)
        block._track_act_norm = (layer_drop_mode == "active")
        block._drop_this_step = False

        if hiddencut_prob > 0.0 and i in paper_set:
            _attach_hiddencut(block)
            tgt = getattr(block, "_hiddencut_target", None)
            if tgt is not None:
                tgt._hiddencut_prob = float(hiddencut_prob)
        else:
            tgt = getattr(block, "_hiddencut_target", None)
            if tgt is not None:
                tgt._hiddencut_prob = 0.0

    # Attach controller state to the transformer/model
    parent = esm3.transformer if hasattr(esm3, "transformer") else esm3
    parent._layer_drop_cfg = {
        "mode": layer_drop_mode,
        "prob": float(layer_drop_prob),
        "topk": int(layer_drop_topk),
        "layers": list(layer_drop_layers) if layer_drop_layers is not None else list(range(n_blocks)),
    }


def _bidir_bernoulli_kl(z1: Tensor, z2: Tensor) -> Tensor:
    """Bidirectional KL divergence between Bernoulli(sigmoid(z1)) and
    Bernoulli(sigmoid(z2)), computed in a numerically-stable log-sigmoid form.

    Returns the per-element symmetric KL (shape matches ``z1``).
    """
    p1 = torch.sigmoid(z1)
    p2 = torch.sigmoid(z2)
    lp1, lp1m = F.logsigmoid(z1), F.logsigmoid(-z1)
    lp2, lp2m = F.logsigmoid(z2), F.logsigmoid(-z2)
    kl_12 = p1 * (lp1 - lp2) + (1.0 - p1) * (lp1m - lp2m)
    kl_21 = p2 * (lp2 - lp1) + (1.0 - p2) * (lp2m - lp1m)
    return 0.5 * (kl_12 + kl_21)


def _layer_drop_step(esm3: nn.Module) -> None:
    """Called once per training step BEFORE the forward pass to decide which
    blocks should drop this step (active mode only). Uses the EMA activation
    norms computed during previous step(s)."""
    parent = esm3.transformer if hasattr(esm3, "transformer") else esm3
    cfg = getattr(parent, "_layer_drop_cfg", None)
    if cfg is None or cfg["mode"] != "active" or cfg["prob"] <= 0.0:
        return
    blocks = esm3.layers if hasattr(esm3, "layers") else esm3.transformer.blocks
    eligible = cfg["layers"]
    topk = max(0, min(cfg["topk"], len(eligible)))
    if topk == 0:
        return
    # Rank eligible blocks by EMA activation norm
    ema_pairs = [(i, getattr(blocks[i], "_act_norm_ema", 0.0) or 0.0) for i in eligible]
    ema_pairs.sort(key=lambda p: p[1], reverse=True)
    chosen = [i for i, _ in ema_pairs[:topk]]
    for i in chosen:
        if torch.rand(1).item() < cfg["prob"]:
            blocks[i]._drop_this_step = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class StructureEpitopePredictionModel(nn.Module):
    """Frozen ESM3 backbone + optional LoRA + trainable per-token binary head.

    ESM3 receives both sequence tokens and backbone coordinates.  Positions
    with NaN coordinates are treated as structurally undefined by ESM3
    (sequence context only, identical to training without structure).

    Args:
        dropout:      Dropout probability before the linear head.
        rys_start:    First layer index of the RYS repeat segment.
        rys_end:      One-past-last layer index of the RYS repeat segment.
                      Set ``rys_end <= rys_start`` to disable RYS.
        lora_rank:    LoRA rank (0 = disable LoRA).
        lora_alpha:   LoRA scaling factor (effective scale = alpha / rank).
        lora_n_blocks: Number of trailing ESM3 blocks to inject LoRA into.
        lora_inject_ffn: Also inject LoRA into FFN (ffn[1] and ffn[3]) of each block.
        lora_inject_geom_attn: Also inject LoRA into geom_attn of block 0.
    """

    _ESM3_HIDDEN = 1536  # ESM3-small-open hidden dimension

    def __init__(
        self,
        dropout: float = DROPOUT,
        rys_start: int = RYS_START,
        rys_end: int = RYS_END,
        lora_rank: int = 0,
        lora_alpha: float = 8.0,
        lora_n_blocks: int = 8,
        lora_inject_ffn: bool = False,
        lora_inject_geom_attn: bool = False,
        lora_block_start: int = -1,
        lora_also_first_n: int = 0,
        head_hidden_dim: int = 0,
        extra_dim: int = 0,
        head_lora_targets: dict[int, list[int]] | None = None,
    ) -> None:
        super().__init__()

        import esm.pretrained

        import types

        # Load ESM3 small open model
        self.esm3 = esm.pretrained.ESM3_sm_open_v0(device="cpu")  # Start on CPU, will move to device later
        for param in self.esm3.parameters():
            param.requires_grad = False
        self.esm3.eval()

        # Inject LoRA before RYS (LoRA params are trainable; originals stay frozen)
        self._use_lora = lora_rank > 0
        self._head_lora_targets = head_lora_targets
        if self._use_lora:
            if head_lora_targets is not None:
                # Per-head LoRA on a subset of (block, head) pairs from the
                # activation probe. The dict is {block_idx: [head_idx, ...]}.
                _inject_head_lora(self.esm3, lora_rank, lora_alpha, head_lora_targets)
            else:
                _inject_lora(self.esm3, lora_rank, lora_alpha, lora_n_blocks,
                             inject_ffn=lora_inject_ffn,
                             inject_geom_attn=lora_inject_geom_attn,
                             lora_block_start=lora_block_start)
                if lora_also_first_n > 0:
                    # Second injection into first N blocks (safe: skips already-wrapped layers)
                    _inject_lora(self.esm3, lora_rank, lora_alpha, lora_also_first_n,
                                 lora_block_start=0)

        # Apply RYS: repeat transformer blocks [rys_start, rys_end) a second time.
        if rys_end > rys_start:
            def _rys_forward(self_t, x, sequence_id=None, affine=None,
                              affine_mask=None, chain_id=None):
                import torch as _torch
                if affine_mask is not None:
                    affine_mask = affine_mask.to(x.device)
                if affine is not None:
                    affine = affine.to(device=x.device)
                *batch_dims, _ = x.shape
                if chain_id is None:
                    chain_id = _torch.ones(
                        size=batch_dims, dtype=_torch.int64, device=x.device
                    )
                hiddens = []
                # Handle ESM2 vs ESM3 structure differences
                blocks = self_t.layers if hasattr(self_t, 'layers') else self_t.blocks
                for block in blocks[:rys_start]:
                    x = block(x, sequence_id, affine, affine_mask, chain_id)
                    hiddens.append(x)
                for block in blocks[rys_start:rys_end]:   # first pass
                    x = block(x, sequence_id, affine, affine_mask, chain_id)
                    hiddens.append(x)
                for block in blocks[rys_start:rys_end]:   # repeat (RYS)
                    x = block(x, sequence_id, affine, affine_mask, chain_id)
                for block in blocks[rys_end:]:
                    x = block(x, sequence_id, affine, affine_mask, chain_id)
                    hiddens.append(x)
                return self_t.norm(x), x, hiddens

            # Handle ESM2 vs ESM3 structure differences for RYS
            if hasattr(self.esm3, 'transformer'):
                # ESM3 case
                self.esm3.transformer.forward = types.MethodType(
                    _rys_forward, self.esm3.transformer
                )
            else:
                # ESM2 case - patch the model itself
                self.esm3._original_forward = self.esm3.forward
                self.esm3.forward = types.MethodType(_rys_forward, self.esm3)

        self._extra_dim = extra_dim
        head_in = self._ESM3_HIDDEN + extra_dim
        self.head_ln   = nn.LayerNorm(self._ESM3_HIDDEN)
        self.head_drop = nn.Dropout(dropout)
        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(head_in, head_hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(head_hidden_dim, 1, bias=True),
            )
        else:
            self.head = nn.Linear(head_in, 1, bias=True)

        self.register_buffer(
            "id_map", torch.tensor(_OUR_TO_ESM3, dtype=torch.long)
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        structure_coords: Tensor | None = None,
        extra_features: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            input_ids:        (B, L) token IDs in prepare.py's vocabulary.
            attention_mask:   (B, L) — unused here (ESM3 infers from PAD tokens).
            structure_coords: (B, L, 3, 3) backbone N/CA/C coordinates.
                              ``nan`` → no structure at that position.
            extra_features:   (B, L, D) optional per-residue features (RSA, bio, BLOSUM).

        Returns:
            Logits of shape (B, L).
        """
        esm3_ids = self.id_map[input_ids]  # remap to ESM3 vocabulary

        if self._use_lora:
            output = self.esm3(
                sequence_tokens=esm3_ids,
                structure_coords=structure_coords,
            )
        else:
            with torch.no_grad():
                output = self.esm3(
                    sequence_tokens=esm3_ids,
                    structure_coords=structure_coords,
                )

        # output.embeddings: (B, L, 1536) — pre-norm last hidden state
        emb = output.embeddings.to(self.head_ln.weight.dtype)
        hidden = self.head_drop(self.head_ln(emb))
        if extra_features is not None and self._extra_dim > 0:
            hidden = torch.cat([hidden, extra_features.to(hidden.dtype)], dim=-1)
        return self.head(hidden).squeeze(-1)  # (B, L)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_struct_loss(
    model: nn.Module,
    val_data: list[StructSample],
    batch_size: int = 8,
    device: str = "cuda",
    extra_fn=None,
) -> float:
    """Compute val loss. ``extra_fn(batch, device)`` → extra_features tensor or None."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loader = create_struct_dataloader(val_data, batch_size=batch_size, shuffle=False)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        coords = batch["structure_coords"].to(device)
        extra = extra_fn(batch, device) if extra_fn is not None else None

        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask=attention_mask,
                           structure_coords=coords, extra_features=extra)

        valid = labels != -100
        loss = F.binary_cross_entropy_with_logits(
            logits[valid].float(), labels[valid].float(), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += valid.sum().item()

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


@torch.no_grad()
def compute_roc_auc(
    model: nn.Module,
    val_data: list[StructSample],
    batch_size: int = 8,
    device: str = "cpu",
    extra_fn=None,
) -> float:
    """Compute ROC-AUC. ``extra_fn(batch, device)`` → extra_features tensor or None."""
    if not HAS_SKLEARN:
        return float("nan")

    model.eval()
    loader = create_struct_dataloader(val_data, batch_size=batch_size, shuffle=False)

    all_probs: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        coords = batch["structure_coords"].to(device)
        extra = extra_fn(batch, device) if extra_fn is not None else None

        logits = model(input_ids, attention_mask=attention_mask,
                       structure_coords=coords, extra_features=extra)

        valid = labels != -100
        all_probs.extend(torch.sigmoid(logits[valid]).cpu().float().tolist())
        all_labels.extend(labels[valid].cpu().float().tolist())

    model.train()
    if len(set(all_labels)) < 2:
        return float("nan")

    return float(roc_auc_score(all_labels, all_probs))


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _get_lr_scale(step: int, warmup_steps: int, total_steps: int,
                  cosine_restart_period: int = 0) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if cosine_restart_period > 0:
        # Cosine annealing with warm restarts (SGDR-style)
        t = (step - warmup_steps) % cosine_restart_period
        return 0.5 * (1.0 + math.cos(math.pi * t / cosine_restart_period))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    train_data: list[StructSample],
    val_data: list[StructSample],
    max_seconds: int = 300,
    device: str = "cpu",
    compute_auc: bool = False,
    val_eval_interval: int = 200,
    rys_start: int = RYS_START,
    rys_end: int = RYS_END,
    lora_rank: int = 0,
    lora_alpha: float = 8.0,
    lora_n_blocks: int = 8,
    lora_inject_ffn: bool = False,
    lora_inject_geom_attn: bool = False,
    head_hidden_dim: int = 0,
    label_smooth: float = 0.0,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    warmup_steps: int = WARMUP_STEPS,
    patience: int = 3,
    lora_warmup_steps: int = 0,
    cosine_restart_period: int = 0,
    grad_accum_steps: int = 1,
    pos_weight: float = 0.0,
    head_lr_mult: float = 1.0,
    lora_block_start: int = -1,
    lora_also_first_n: int = 0,
    coord_noise_std: float = 0.0,
    pu_prior: float = 0.0,
    initial_state_dict: dict | None = None,
    rsa_surface_threshold: float = 0.0,
    rsa_as_feature: bool = False,
    bio_features: bool = False,
    blosum_features: bool = False,
    structure_dropout_prob: float = 0.0,
    head_drop_mode: str = "off",
    head_drop_prob: float = 0.0,
    head_drop_topk: int = 0,
    layer_drop_mode: str = "off",
    layer_drop_prob: float = 0.0,
    layer_drop_topk: int = 0,
    layer_drop_only_lora: bool = True,
    layer_drop_target_blocks: list[int] | None = None,  # explicit eligibility list (overrides layer_drop_only_lora)
    # ACL 2024 Findings: LoRA Meets Dropout under a Unified Framework
    # (see docs/lora_dropout_hiddenkey_acl2024.pdf).
    dropkey_prob: float = 0.0,        # column-wise drop of attention logits
    dropattn_prob: float = 0.0,       # column-wise drop of attention weights (NoGrad'd)
    hiddencut_prob: float = 0.0,      # element-wise drop of FFN activation hidden state
    kl_loss_weight: float = 0.0,      # weight on bidirectional Bernoulli KL between two stochastic forward passes
    paper_drop_only_lora: bool = True,  # restrict the above to the LoRA-targeted blocks
    head_lora_targets: dict[int, list[int]] | None = None,  # selective per-head LoRA: {block_idx: [head, ...]}
) -> dict:
    """Train the ESM3-based structure-aware epitope prediction model."""
    # Force-collect any GPU tensors leaked from previous runs before loading ESM3.
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    model = StructureEpitopePredictionModel(
        dropout=dropout,
        rys_start=rys_start,
        rys_end=rys_end,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_n_blocks=lora_n_blocks,
        lora_inject_ffn=lora_inject_ffn,
        lora_inject_geom_attn=lora_inject_geom_attn,
        lora_block_start=lora_block_start,
        lora_also_first_n=lora_also_first_n,
        head_hidden_dim=head_hidden_dim,
        extra_dim=extra_feature_dim(rsa_as_feature, bio_features, blosum_features),
        head_lora_targets=head_lora_targets,
    ).to(device)

    # Configure head/layer dropout (no-op if all modes are "off" and probs are 0)
    _need_paper_drop = (dropkey_prob > 0.0 or dropattn_prob > 0.0 or hiddencut_prob > 0.0)
    if (head_drop_mode != "off" and head_drop_prob > 0.0) or \
       (layer_drop_mode != "off" and layer_drop_prob > 0.0) or _need_paper_drop:
        # LoRA block range — used to target both LayerDrop and paper-style drops.
        _lora_target_layers: list[int] | None = None
        if lora_rank > 0:
            if head_lora_targets is not None:
                _lora_target_layers = sorted(head_lora_targets.keys())
            else:
                blocks_ref = model.esm3.layers if hasattr(model.esm3, "layers") else model.esm3.transformer.blocks
                n = len(blocks_ref)
                start = lora_block_start if lora_block_start >= 0 else max(0, n - lora_n_blocks)
                end = min(n, start + lora_n_blocks)
                _lora_target_layers = list(range(start, end))
        if layer_drop_target_blocks is not None:
            layer_target_layers = list(layer_drop_target_blocks)
        else:
            layer_target_layers = (
                _lora_target_layers
                if (layer_drop_mode != "off" and layer_drop_only_lora and lora_rank > 0)
                else None
            )
        paper_target_layers = (
            _lora_target_layers
            if (_need_paper_drop and paper_drop_only_lora and lora_rank > 0)
            else None
        )
        _configure_dropout(
            model.esm3,
            head_drop_mode=head_drop_mode,
            head_drop_prob=head_drop_prob,
            head_drop_topk=head_drop_topk,
            layer_drop_mode=layer_drop_mode,
            layer_drop_prob=layer_drop_prob,
            layer_drop_topk=layer_drop_topk,
            layer_drop_layers=layer_target_layers,
            dropkey_prob=dropkey_prob,
            dropattn_prob=dropattn_prob,
            hiddencut_prob=hiddencut_prob,
            paper_drop_layers=paper_target_layers,
        )
        logger.info(
            f"dropout cfg: head_drop={head_drop_mode}/p={head_drop_prob}/k={head_drop_topk}, "
            f"layer_drop={layer_drop_mode}/p={layer_drop_prob}/k={layer_drop_topk}, "
            f"struct_dropout_prob={structure_dropout_prob}, "
            f"dropkey_prob={dropkey_prob}, dropattn_prob={dropattn_prob}, "
            f"hiddencut_prob={hiddencut_prob}, kl_loss_weight={kl_loss_weight}"
        )

    # Build extra-feature function (used in training loop and evaluation).
    _extra_fn = None
    if rsa_as_feature or bio_features or blosum_features:
        _rsa_feat, _bio, _bl = rsa_as_feature, bio_features, blosum_features
        def _extra_fn(batch, dev):
            return compute_extra_features(batch, dev, _rsa_feat, _bio, _bl)

    # Load pretrained weights if provided (two-stage training: stage 2 starts here)
    if initial_state_dict is not None:
        current = model.state_dict()
        matched = {k: v.to(device) for k, v in initial_state_dict.items() if k in current}
        current.update(matched)
        model.load_state_dict(current)
        logger.info(f"Loaded {len(matched)} pretrained param tensors from stage 1")

    n_params = model.num_parameters()
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {n_params:,}  (total incl. frozen: {n_total:,})")

    # Separate param groups to support head_lr_mult and/or LoRA warmup.
    # "head" includes head_ln, head_drop, and head linear(s).
    _head_modules = [model.head, model.head_ln, model.head_drop]
    _head_ids = {id(p) for m in _head_modules for p in m.parameters()}
    _head_params = [p for m in _head_modules for p in m.parameters()]
    _other_params = [p for p in model.parameters()
                     if p.requires_grad and id(p) not in _head_ids]
    _use_groups = head_lr_mult != 1.0 or (lora_warmup_steps > 0 and model._use_lora)
    if _use_groups:
        _other_init_lr = 0.0 if (lora_warmup_steps > 0 and model._use_lora) else lr
        optimizer = torch.optim.AdamW([
            {"params": _head_params, "lr": lr * head_lr_mult, "base_lr": lr * head_lr_mult},
            {"params": _other_params, "lr": _other_init_lr, "base_lr": lr},
        ], lr=lr, weight_decay=weight_decay)
        if lora_warmup_steps > 0 and model._use_lora:
            logger.info(f"LoRA warmup: head-only for first {lora_warmup_steps} steps")
        if head_lr_mult != 1.0:
            logger.info(f"head_lr_mult={head_lr_mult}: head lr={lr*head_lr_mult:.2e}, other lr={lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

    loader = create_struct_dataloader(train_data, batch_size=batch_size, shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    step = 0
    total_tokens = 0
    running_loss = 0.0
    log_interval = 10
    total_estimate = max_seconds * 3  # ESM3-small on A6000: ~3 steps/sec

    best_val_loss = float("inf")
    best_state: dict | None = None  # CPU copy of all trainable weights at best val
    no_improve_count = 0  # consecutive periodic evals without improvement

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    model.train()
    early_stop = False
    optimizer.zero_grad()

    while True:
        for batch in loader:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds or early_stop:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            coords = batch["structure_coords"].to(device)

            # Surface masking: exclude buried residues (RSA < threshold) from loss.
            if rsa_surface_threshold > 0.0:
                rsa_vals = batch["rsa"].to(device)  # (B, L), NaN where unavailable
                buried = (rsa_vals < rsa_surface_threshold) | torch.isnan(rsa_vals)
                labels = labels.clone()
                labels[buried] = -100

            # Coordinate noise augmentation: add Gaussian jitter to known backbone coords.
            if coord_noise_std > 0.0:
                nan_mask = torch.isnan(coords)
                coords = coords + torch.randn_like(coords) * coord_noise_std
                coords[nan_mask] = float("nan")

            # Structure dropout: per-sample, NaN out coords with prob `structure_dropout_prob`.
            # Forces the model to handle both with-structure and sequence-only inputs.
            if structure_dropout_prob > 0.0:
                B = coords.shape[0]
                drop_mask = torch.rand(B, device=coords.device) < structure_dropout_prob
                if drop_mask.any():
                    idx = drop_mask.nonzero(as_tuple=True)[0]
                    coords[idx] = float("nan")

            # Active LayerDrop: pick which blocks to drop this step (uses
            # EMA from previous batches' activation norms).
            if layer_drop_mode == "active" and layer_drop_prob > 0.0:
                _layer_drop_step(model.esm3)

            extra = _extra_fn(batch, device) if _extra_fn is not None else None
            do_two_pass = kl_loss_weight > 0.0 and _need_paper_drop
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                logits = model(
                    input_ids,
                    attention_mask=attention_mask,
                    structure_coords=coords,
                    extra_features=extra,
                )
                if pu_prior > 0.0:
                    loss = pu_loss(logits, labels, prior=pu_prior)
                else:
                    valid = labels != -100
                    tgt = labels[valid].float()
                    if label_smooth > 0.0:
                        tgt = tgt * (1.0 - label_smooth) + label_smooth / 2.0
                    _pw = (torch.tensor([pos_weight], device=device, dtype=torch.float32)
                           if pos_weight > 0.0 else None)
                    loss = F.binary_cross_entropy_with_logits(logits[valid], tgt, pos_weight=_pw)

                # HiddenKey / R-drop: second stochastic forward pass + bidirectional
                # KL between the two output distributions, computed only over
                # labelled positions (valid mask). See Wang et al. ACL 2024 Findings.
                if do_two_pass:
                    logits2 = model(
                        input_ids,
                        attention_mask=attention_mask,
                        structure_coords=coords,
                        extra_features=extra,
                    )
                    if pu_prior > 0.0:
                        loss2 = pu_loss(logits2, labels, prior=pu_prior)
                    else:
                        valid2 = labels != -100
                        tgt2 = labels[valid2].float()
                        if label_smooth > 0.0:
                            tgt2 = tgt2 * (1.0 - label_smooth) + label_smooth / 2.0
                        loss2 = F.binary_cross_entropy_with_logits(logits2[valid2], tgt2, pos_weight=_pw)
                    valid_kl = labels != -100
                    kl_per = _bidir_bernoulli_kl(
                        logits.float()[valid_kl], logits2.float()[valid_kl]
                    )
                    kl_term = kl_per.mean()
                    loss = 0.5 * (loss + loss2) + kl_loss_weight * kl_term

            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
            if step % grad_accum_steps == grad_accum_steps - 1:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            lr_scale = _get_lr_scale(step, warmup_steps, total_estimate,
                                      cosine_restart_period=cosine_restart_period)
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("base_lr", lr) * lr_scale
            # Unlock LoRA params at lora_warmup boundary
            if lora_warmup_steps > 0 and step == lora_warmup_steps and len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = optimizer.param_groups[1].get("base_lr", lr) * lr_scale
                logger.info(f"step={step}: LoRA params unlocked (lr={optimizer.param_groups[1]['lr']:.2e})")

            running_loss += loss.item()
            step += 1
            total_tokens += int(attention_mask.sum())

            if step % log_interval == 0:
                avg = running_loss / log_interval
                logger.info(f"step={step}  train_loss={avg:.4f}  elapsed={elapsed:.0f}s")
                running_loss = 0.0

            if val_eval_interval > 0 and step % val_eval_interval == 0:
                v = evaluate_struct_loss(model, val_data, batch_size=batch_size, device=device, extra_fn=_extra_fn)
                logger.info(f"step={step}  val_loss={v:.6f}  [periodic]")
                if v < best_val_loss:
                    best_val_loss = v
                    no_improve_count = 0
                    # Save all trainable params (head + LoRA if present)
                    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                        if k in trainable_keys
                    }
                else:
                    no_improve_count += 1
                    if patience > 0 and no_improve_count >= patience:
                        logger.info(
                            f"Early stop: no improvement for {no_improve_count} "
                            f"consecutive evals (best={best_val_loss:.6f})"
                        )
                        early_stop = True
                model.train()

        if time.time() - start_time >= max_seconds or early_stop:
            break

    # Restore best trainable weights observed during training.
    if best_state is not None:
        current = model.state_dict()
        current.update({k: v.to(device) for k, v in best_state.items()})
        model.load_state_dict(current)
        logger.info(f"Restored best checkpoint (val_loss={best_val_loss:.6f})")

    train_loss = running_loss / max(1, step % log_interval) if step % log_interval != 0 else 0.0
    peak_vram_mb = 0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)

    val_loss = evaluate_struct_loss(model, val_data, batch_size=batch_size, device=device, extra_fn=_extra_fn)
    logger.info(f"val_loss={val_loss:.6f} after {step} steps")

    roc_auc = float("nan")
    if compute_auc:
        roc_auc = compute_roc_auc(model, val_data, batch_size=batch_size, device=device, extra_fn=_extra_fn)
        logger.info(f"roc_auc={roc_auc:.6f}")

    # Capture trainable weights before freeing the model (used for two-stage chaining)
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_state = {
        k: v.cpu().clone() for k, v in model.state_dict().items() if k in trainable_keys
    }

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
        "depth": 48,  # ESM3-small-open has 48 transformer layers
        "roc_auc": roc_auc,
        "trainable_state": trainable_state,
    }



# ---------------------------------------------------------------------------
# Cross-validation data loading
# ---------------------------------------------------------------------------


def create_cv_datasets(
    fasta_path: Path,
    structures_dir: Path,
    test_partition: str,
    val_partition: str = "5",
    max_length: int = MAX_SEQ_LEN,
) -> tuple[list[StructSample], list[StructSample], list[StructSample]]:
    """Load BEPIPRED data for one CV fold.

    Train: all partitions except ``test_partition``, ``val_partition``, and "EVAL".
    Val:   ``val_partition`` (used for early stopping / checkpoint selection).
    Test:  ``test_partition`` (held out; never seen during training or HP tuning).

    Args:
        fasta_path:      Path to BEPIPRED.fasta (or any combined-FASTA).
        structures_dir:  Root of the PDB structure dataset.
        test_partition:  Partition label to hold out for final test evaluation.
        val_partition:   Partition label to use for early stopping.
        max_length:      Maximum tokenised sequence length (incl. BOS/EOS).

    Returns:
        ``(train_data, val_data, test_data)`` lists of StructSample.
    """
    by_part = load_combined_fasta_partitioned(
        fasta_path, exclude_partitions=frozenset({"EVAL"})
    )

    train_headers: list = []
    val_headers:   list = []
    test_headers:  list = []

    for part_id, samples in by_part.items():
        if part_id == test_partition:
            test_headers.extend(samples)
            val_headers.extend(samples)  # holdout is both val (early stop) and test
        elif part_id == val_partition:
            val_headers.extend(samples)
        else:
            train_headers.extend(samples)

    def _to_struct(header_samples: list) -> list[StructSample]:
        out: list[StructSample] = []
        for header, (token_ids, labels) in header_samples:
            if len(token_ids) > max_length:
                continue
            seq_len = len(token_ids) - 2  # strip BOS/EOS
            coords, rsa = _load_coords_rsa(header, seq_len, structures_dir)
            out.append((token_ids, labels, coords, rsa))
        return out

    return _to_struct(train_headers), _to_struct(val_headers), _to_struct(test_headers)


def load_contacts_data(
    contacts_fasta: Path,
    max_length: int = MAX_SEQ_LEN,
    interface_min: int = 0,
    interface_max: int = 9999,
) -> list[StructSample]:
    """Load PDB contacts pretraining sequences (no structures or RSA).

    Args:
        contacts_fasta:  Path to pdb_contacts.fasta.
        max_length:      Maximum tokenised sequence length.
        interface_min:   Minimum number of interface residues to keep.
        interface_max:   Maximum number of interface residues to keep.

    Returns:
        List of StructSample with coords=None, rsa=None.
    """
    from features import filter_contacts_by_size

    headers, _ = load_combined_fasta(contacts_fasta, val_partition="__none__")
    if interface_min > 0 or interface_max < 9999:
        headers = filter_contacts_by_size(headers, interface_min, interface_max)

    out: list[StructSample] = []
    for _, (tids, lbls) in headers:
        if len(tids) <= max_length:
            out.append((tids, lbls, None, None))
    return out
