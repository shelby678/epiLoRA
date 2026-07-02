"""Tokenizer, data loading, and evaluation for BCR epitope prediction.

Per-token binary prediction: each amino acid position is labeled 0 (non-epitope)
or 1 (epitope). Data is loaded from paired FASTA files (amino acid sequences +
binary label strings).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# Type alias used by combined-FASTA loaders: (header_str, Sample)
HeaderedSample = tuple[str, "Sample"]

# ---------------------------------------------------------------------------
# Vocabulary & Tokenizer
# ---------------------------------------------------------------------------

VOCAB: dict[str, int] = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "A": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "V": 22,
    "W": 23,
    "Y": 24,
}
VOCAB_SIZE = 25
PAD_VOCAB_SIZE = 32  # padded for tensor core alignment

ID_TO_TOKEN: dict[int, str] = {v: k for k, v in VOCAB.items()}

# Token IDs for convenience
CLS_ID = VOCAB["<cls>"]
PAD_ID = VOCAB["<pad>"]
EOS_ID = VOCAB["<eos>"]
UNK_ID = VOCAB["<unk>"]
MASK_ID = VOCAB["<mask>"]

# Amino acid token ID range (inclusive)
AA_ID_MIN = 5
AA_ID_MAX = 24


def encode(sequence: str) -> list[int]:
    """Encode an amino acid sequence to token IDs with <cls> and <eos>."""
    ids = [CLS_ID]
    for ch in sequence.upper():
        ids.append(VOCAB.get(ch, UNK_ID))
    ids.append(EOS_ID)
    return ids


def decode(token_ids: list[int]) -> str:
    """Decode token IDs back to a string (special tokens included)."""
    return "".join(ID_TO_TOKEN.get(tid, "?") for tid in token_ids)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

# Each sample is a pair of parallel lists of equal length:
#   token_ids:      [CLS, aa1, aa2, ..., aan, EOS]
#   epitope_labels: [-100, l1,  l2,  ..., ln,  -100]
# where l_i in {0, 1} and -100 marks positions excluded from the loss.
Sample = tuple[list[int], list[int]]


def load_fasta(path: Path) -> dict[str, str]:
    """Load sequences from a FASTA file.

    Returns:
        Dict mapping sequence ID to sequence string.
    """
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:]
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)
    return sequences


def load_fasta_pairs(aa_path: Path, bce_path: Path) -> list[Sample]:
    """Load paired amino acid and epitope label FASTA files.

    Args:
        aa_path: Path to amino acid FASTA file.
        bce_path: Path to binary epitope label FASTA file (0/1 character strings).

    Returns:
        List of (token_ids, epitope_labels) where epitope_labels has -100 at
        CLS/EOS positions and 0 or 1 at amino acid positions.
    """
    aa_seqs = load_fasta(aa_path)
    bce_seqs = load_fasta(bce_path)

    samples: list[Sample] = []
    for seq_id, aa_seq in aa_seqs.items():
        if seq_id not in bce_seqs:
            continue
        label_str = bce_seqs[seq_id]
        if len(aa_seq) != len(label_str):
            continue

        token_ids = encode(aa_seq)
        # -100 for CLS and EOS (excluded from loss); 0/1 for each AA position
        epitope_labels = [-100] + [int(c) for c in label_str] + [-100]
        samples.append((token_ids, epitope_labels))

    return samples


def create_datasets(
    data_dir: Path,
    max_length: int = 512,
) -> tuple[list[Sample], list[Sample]]:
    """Load BCR epitope data from paired FASTA files.

    Expects four files in ``data_dir``:
      - ``train_aa.fasta``  — amino acid training sequences
      - ``train_bce.fasta`` — binary epitope labels for training
      - ``valid_aa.fasta``  — amino acid validation sequences
      - ``valid_bce.fasta`` — binary epitope labels for validation

    Args:
        data_dir: Directory containing the four FASTA files.
        max_length: Maximum sequence length (including <cls> and <eos>).

    Returns:
        (train_data, val_data) — each a list of (token_ids, epitope_labels) tuples.
    """
    train_aa = data_dir / "train_aa.fasta"
    train_bce = data_dir / "train_bce.fasta"
    val_aa = data_dir / "valid_aa.fasta"
    val_bce = data_dir / "valid_bce.fasta"

    for p in [train_aa, train_bce, val_aa, val_bce]:
        if not p.exists():
            raise FileNotFoundError(f"Required data file not found: {p}")

    def _load_and_filter(aa_path: Path, bce_path: Path) -> list[Sample]:
        samples = load_fasta_pairs(aa_path, bce_path)
        return [(t, l) for t, l in samples if len(t) <= max_length]

    train_data = _load_and_filter(train_aa, train_bce)
    if not train_data:
        raise ValueError(
            f"No training sequences remaining after filtering to max_length={max_length}"
        )

    val_data = _load_and_filter(val_aa, val_bce)
    if not val_data:
        raise ValueError(
            f"No validation sequences remaining after filtering to max_length={max_length}"
        )

    return train_data, val_data


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """Dataset wrapping a list of (token_ids, epitope_labels) samples."""

    def __init__(self, data: list[Sample]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        return self.data[idx]


def collate_fn(batch: list[Sample]) -> dict[str, Tensor]:
    """Pad sequences and labels to max-in-batch length, create attention mask.

    Returns:
        Dict with keys: input_ids, attention_mask, labels
          - input_ids: (B, L) token IDs, padded with PAD_ID
          - attention_mask: (B, L) with 1 for real tokens, 0 for padding
          - labels: (B, L) epitope labels (0/1), with -100 at CLS/EOS/PAD positions
    """
    max_len = max(len(token_ids) for token_ids, _ in batch)

    padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (token_ids, epitope_labels) in enumerate(batch):
        length = len(token_ids)
        padded[i, :length] = torch.tensor(token_ids, dtype=torch.long)
        labels[i, :length] = torch.tensor(epitope_labels, dtype=torch.long)
        attention_mask[i, :length] = 1

    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# DataLoader Helper
# ---------------------------------------------------------------------------


def create_dataloader(
    data: list[Sample],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for BCR epitope prediction."""
    dataset = SequenceDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    val_data: list[Sample],
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """Compute average binary cross-entropy loss on epitope-labeled positions.

    Args:
        model: The protein model to evaluate.
        val_data: List of (token_ids, epitope_labels) samples.
        batch_size: Batch size for evaluation.
        device: Device to run on.

    Returns:
        Average validation loss (float).
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask=attention_mask)  # (B, L)

        # Binary cross-entropy on labeled AA positions only (labels != -100)
        valid = labels != -100
        loss = F.binary_cross_entropy_with_logits(
            logits[valid].float(), labels[valid].float(), reduction="sum"
        )

        total_loss += loss.item()
        total_tokens += valid.sum().item()

    model.train()

    if total_tokens == 0:
        return float("inf")

    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Combined-FASTA loader  (pdb_chains.fasta / pdb_contacts.fasta format)
# ---------------------------------------------------------------------------


def _parse_combined_partition(header: str) -> str:
    """Return the partition field from a combined-FASTA header.

    Header format: ``PDBID_AB CHAIN PARTITION``  e.g. ``1A2Y_BA C 2``
    Returns ``""`` when the header has fewer than 3 space-separated fields.
    """
    parts = header.split()
    return parts[2] if len(parts) >= 3 else ""


def load_combined_fasta(
    path: Path,
    val_partition: str = "5",
    exclude_partitions: frozenset[str] = frozenset(),
) -> tuple[list["HeaderedSample"], list["HeaderedSample"]]:
    """Load a combined-FASTA file where case encodes labels.

    Format (same as ``data/pdb_chains.fasta`` and ``data/pdb_contacts.fasta``)::

        >PDBID_AB CHAIN PARTITION
        kvfgrCELAamkrhgla...   # uppercase = label 1, lowercase = label 0

    Args:
        path:               Path to the FASTA file.
        val_partition:      Header partition field whose entries go to the
                            validation split.  All other partitions (including
                            ``"pretrain"``) go to the training split.
        exclude_partitions: Set of partition labels to drop entirely (excluded
                            from both train and val).  Useful for held-out test
                            sets such as ``frozenset({"EVAL"})``.

    Returns:
        ``(train_samples, val_samples)`` — each a list of
        ``(header, (token_ids, epitope_labels))`` pairs.
    """
    train: list[HeaderedSample] = []
    val:   list[HeaderedSample] = []

    current_id: str | None = None
    current_seq: list[str] = []

    def _emit(hdr: str, seq_parts: list[str]) -> None:
        seq = "".join(seq_parts)
        if not seq:
            return
        partition = _parse_combined_partition(hdr)
        if partition in exclude_partitions:
            return
        token_ids = encode(seq.upper())
        labels = [-100] + [1 if ch.isupper() else 0 for ch in seq] + [-100]
        sample: HeaderedSample = (hdr, (token_ids, labels))
        if partition == val_partition:
            val.append(sample)
        else:
            train.append(sample)

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_id is not None:
                    _emit(current_id, current_seq)
                current_id = line[1:]
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)

    if current_id is not None:
        _emit(current_id, current_seq)

    return train, val


def load_combined_fasta_partitioned(
    path: Path,
    exclude_partitions: frozenset[str] = frozenset({"EVAL"}),
) -> dict[str, list["HeaderedSample"]]:
    """Load a combined-FASTA file and group samples by partition label.

    Args:
        path:               Path to the FASTA file.
        exclude_partitions: Partition labels to drop entirely (default: ``{"EVAL"}``).

    Returns:
        Dict mapping partition label → list of ``(header, (token_ids, labels))`` samples.
        Entries with excluded partitions are omitted.
    """
    by_partition: dict[str, list[HeaderedSample]] = {}

    current_id: str | None = None
    current_seq: list[str] = []

    def _emit(hdr: str, seq_parts: list[str]) -> None:
        seq = "".join(seq_parts)
        if not seq:
            return
        partition = _parse_combined_partition(hdr)
        if partition in exclude_partitions:
            return
        token_ids = encode(seq.upper())
        labels = [-100] + [1 if ch.isupper() else 0 for ch in seq] + [-100]
        sample: HeaderedSample = (hdr, (token_ids, labels))
        by_partition.setdefault(partition, []).append(sample)

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_id is not None:
                    _emit(current_id, current_seq)
                current_id = line[1:]
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)

    if current_id is not None:
        _emit(current_id, current_seq)

    return by_partition


# ---------------------------------------------------------------------------
# Non-negative PU loss  (Kiryo et al., NeurIPS 2017)
# ---------------------------------------------------------------------------


def pu_loss(
    logits: Tensor,
    labels: Tensor,
    prior: float = 0.1,
) -> Tensor:
    """Non-negative Positive-Unlabelled (nnPU) loss.

    Treats positions with ``labels == 1`` as *confirmed positives* and
    positions with ``labels == 0`` as *unlabelled* (may contain hidden
    positives).  Positions with ``labels == -100`` are excluded.

    The risk estimate is::

        R_pu = π · E[l(f,+1) | P] + max(0, E[l(f,−1) | U] − π · E[l(f,−1) | P])

    where π = ``prior`` ≈ fraction of positives in the population.

    Args:
        logits: Raw model output tensor (any shape, flattened internally).
        labels: Integer labels, same shape as ``logits``.
                  1  → confirmed positive
                  0  → unlabelled
                 -100 → excluded from loss
        prior:  Estimated fraction of positive examples (π).

    Returns:
        Scalar loss tensor.
    """
    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)

    valid = flat_labels != -100
    lg = flat_logits[valid]
    lb = flat_labels[valid].float()

    pos = lb == 1.0
    unl = lb == 0.0

    ones  = torch.ones_like(lg)
    zeros = torch.zeros_like(lg)

    l_pos = F.binary_cross_entropy_with_logits(lg, ones,  reduction="none")
    l_neg = F.binary_cross_entropy_with_logits(lg, zeros, reduction="none")

    if pos.any():
        r_p_plus = l_pos[pos].mean()
        r_p_minus = l_neg[pos].mean()
    else:
        r_p_plus = lg.new_zeros(())
        r_p_minus = lg.new_zeros(())

    r_u_minus = l_neg[unl].mean() if unl.any() else lg.new_zeros(())

    pu = prior * r_p_plus + torch.clamp(r_u_minus - prior * r_p_minus, min=0.0)
    return pu
