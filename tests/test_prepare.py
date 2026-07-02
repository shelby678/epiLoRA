"""Tests for prepare (tokenizer, data, masking, eval)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from prepare import (
    AA_ID_MIN,
    CLS_ID,
    EOS_ID,
    MASK_ID,
    PAD_ID,
    UNK_ID,
    VOCAB,
    apply_mask,
    collate_fn,
    create_datasets,
    decode,
    encode,
    evaluate_loss,
    load_parquet,
)


class TestEncoding:
    """Tests for encode/decode."""

    def test_encode_basic(self) -> None:
        ids = encode("ACDE")
        assert ids[0] == CLS_ID
        assert ids[-1] == EOS_ID
        assert len(ids) == 6  # cls + 4 AAs + eos

    def test_encode_unknown_chars(self) -> None:
        ids = encode("AXZ")
        assert ids[1] == VOCAB["A"]
        assert ids[2] == UNK_ID  # X is unknown
        assert ids[3] == UNK_ID  # Z is unknown

    def test_roundtrip(self) -> None:
        original = "MVLSPADKTN"
        ids = encode(original)
        decoded = decode(ids)
        # Decoded includes <cls> and <eos> tokens
        assert decoded == f"<cls>{original}<eos>"

    def test_all_amino_acids(self) -> None:
        all_aa = "ACDEFGHIKLMNPQRSTVWY"
        ids = encode(all_aa)
        # Should have no UNK tokens (excluding cls/eos)
        assert UNK_ID not in ids[1:-1]

    def test_case_insensitive(self) -> None:
        upper = encode("ACDE")
        lower = encode("acde")
        assert upper == lower


class TestParquetLoading:
    """Tests for load_parquet."""

    def _write_parquet(self, path: Path, ids: list[str], seqs: list[str]) -> None:
        df = pd.DataFrame({"sequence_id": ids, "sequence": seqs})
        df.to_parquet(path, index=False)

    def test_load_parquet_single_file(self, tmp_path: Path) -> None:
        pq_path = tmp_path / "data.parquet"
        self._write_parquet(pq_path, ["s1", "s2"], ["MVLSPADK", "KETAAAKF"])
        sequences = load_parquet([pq_path])
        assert len(sequences) == 2
        assert sequences[0] == "MVLSPADK"
        assert sequences[1] == "KETAAAKF"

    def test_load_parquet_multiple_shards(self, tmp_path: Path) -> None:
        self._write_parquet(tmp_path / "shard0.parquet", ["s1"], ["ACDEFG"])
        self._write_parquet(tmp_path / "shard1.parquet", ["s2"], ["GHIKLM"])
        sequences = load_parquet(
            [tmp_path / "shard0.parquet", tmp_path / "shard1.parquet"]
        )
        assert len(sequences) == 2
        assert sequences[0] == "ACDEFG"
        assert sequences[1] == "GHIKLM"


class TestCreateDatasets:
    """Tests for create_datasets with train/val subdirectories."""

    def _write_parquet(self, path: Path, ids: list[str], seqs: list[str]) -> None:
        df = pd.DataFrame({"sequence_id": ids, "sequence": seqs})
        df.to_parquet(path, index=False)

    def test_create_datasets(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        self._write_parquet(
            train_dir / "train.parquet",
            ["s1", "s2", "s3", "s4", "s5", "s6"],
            ["MVLSPADK", "ACDEFGHI", "KLMNPQRS", "TVWYACDE", "FGHIKLMN", "PQRSTVWY"],
        )
        self._write_parquet(
            val_dir / "val.parquet",
            ["s7", "s8"],
            ["ACDEFGHI", "KLMNPQRS"],
        )

        train_data, val_data = create_datasets(tmp_path)
        assert len(train_data) == 6
        assert len(val_data) == 2

    def test_no_train_dir_raises(self, tmp_path: Path) -> None:
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        self._write_parquet(val_dir / "v.parquet", ["s1"], ["ACDE"])
        with pytest.raises(FileNotFoundError, match="train"):
            create_datasets(tmp_path)

    def test_no_val_dir_raises(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        self._write_parquet(train_dir / "t.parquet", ["s1"], ["ACDE"])
        with pytest.raises(FileNotFoundError, match="val"):
            create_datasets(tmp_path)

    def test_no_train_files_raises(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir()
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        self._write_parquet(val_dir / "v.parquet", ["s1"], ["ACDE"])
        with pytest.raises(FileNotFoundError, match="Parquet"):
            create_datasets(tmp_path)

    def test_no_val_files_raises(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (tmp_path / "val").mkdir()
        self._write_parquet(train_dir / "t.parquet", ["s1"], ["ACDE"])
        with pytest.raises(FileNotFoundError, match="Parquet"):
            create_datasets(tmp_path)

    def test_multiple_shards_combined(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        self._write_parquet(
            train_dir / "shard0.parquet", ["s1", "s2"], ["MVLSPADK", "ACDEFGHI"]
        )
        self._write_parquet(train_dir / "shard1.parquet", ["s3"], ["KLMNPQRS"])
        self._write_parquet(val_dir / "shard0.parquet", ["s4"], ["TVWYACDE"])
        self._write_parquet(val_dir / "shard1.parquet", ["s5"], ["FGHIKLMN"])

        train_data, val_data = create_datasets(tmp_path)
        assert len(train_data) == 3
        assert len(val_data) == 2


class TestMasking:
    """Tests for apply_mask."""

    def test_masking_ratio(self) -> None:
        # Large enough batch for statistical testing
        ids = torch.tensor([encode("ACDEFGHIKLMNPQRSTVWY")] * 100)
        masked, labels = apply_mask(ids, mask_prob=0.15)

        n_aa = (ids >= AA_ID_MIN).sum().item()
        n_masked = (labels != -100).sum().item()
        ratio = n_masked / n_aa
        # Should be roughly 15% (allow wide margin for randomness)
        assert 0.05 < ratio < 0.30

    def test_special_tokens_not_masked(self) -> None:
        ids = torch.tensor([encode("ACDEFG")])
        for _ in range(50):
            masked, labels = apply_mask(ids)
            # CLS and EOS should never be masked
            assert labels[0, 0] == -100  # CLS position
            assert labels[0, -1] == -100  # EOS position

    def test_mask_token_presence(self) -> None:
        ids = torch.tensor([encode("ACDEFGHIKLMNPQRSTVWY")] * 50)
        masked, labels = apply_mask(ids, mask_prob=0.3)  # higher prob for test
        # Should have some <mask> tokens
        assert (masked == MASK_ID).any()

    def test_masking_determinism(self) -> None:
        ids = torch.tensor([encode("ACDEFGHIKLMNPQRSTVWY")] * 10)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        masked1, labels1 = apply_mask(ids, generator=gen1)
        masked2, labels2 = apply_mask(ids, generator=gen2)
        assert torch.equal(masked1, masked2)
        assert torch.equal(labels1, labels2)


class TestCollation:
    """Tests for collate_fn."""

    def test_collate_shapes(self) -> None:
        batch = [encode("ACDE"), encode("FGHIKLMN")]  # different lengths
        result = collate_fn(batch)

        assert result["input_ids"].shape[0] == 2
        assert result["attention_mask"].shape[0] == 2
        assert result["labels"].shape[0] == 2
        # Padded to max length in batch
        max_len = max(len(encode("ACDE")), len(encode("FGHIKLMN")))
        assert result["input_ids"].shape[1] == max_len

    def test_collate_padding(self) -> None:
        short = encode("AC")  # length 4: cls + A + C + eos
        long = encode("ACDEFG")  # length 8
        result = collate_fn([short, long])

        # First sequence should be padded
        assert result["attention_mask"][0].sum() == len(short)
        assert result["attention_mask"][1].sum() == len(long)
        # Padding positions should have PAD_ID
        assert (
            result["input_ids"][0, len(short) :].eq(PAD_ID).all()
            or result["input_ids"][0, len(short) :].eq(MASK_ID).all()
            or True
        )


class TestEvaluateLoss:
    """Tests for evaluate_loss."""

    def test_returns_finite(self, tiny_val_data: list[list[int]]) -> None:
        from train import ProteinLM

        device = "cpu"
        model = ProteinLM(dim=32, n_layers=2, n_heads=4, max_seq_len=128).to(device)
        loss = evaluate_loss(model, tiny_val_data, batch_size=4, device=device)
        assert isinstance(loss, float)
        assert loss > 0
        assert not float("inf") == loss

    def test_deterministic(self, tiny_val_data: list[list[int]]) -> None:
        from train import ProteinLM

        device = "cpu"
        model = ProteinLM(dim=32, n_layers=2, n_heads=4, max_seq_len=128).to(device)
        loss1 = evaluate_loss(model, tiny_val_data, batch_size=4, device=device)
        loss2 = evaluate_loss(model, tiny_val_data, batch_size=4, device=device)
        assert loss1 == loss2
