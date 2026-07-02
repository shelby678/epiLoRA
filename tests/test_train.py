"""Tests for train (model, optimizer, training loop)."""

from __future__ import annotations

import pytest
import torch

from prepare import PAD_VOCAB_SIZE
from train import (
    Muon,
    ProteinLM,
    SelfAttention,
    SwiGLU,
    TransformerBlock,
    apply_rope,
    build_rope_cache,
    train,
)


class TestRoPE:
    """Tests for rotary position embeddings."""

    def test_build_rope_cache_shape(self) -> None:
        dim, seq_len = 32, 64
        cos, sin = build_rope_cache(dim, seq_len)
        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

    def test_apply_rope_preserves_shape(self) -> None:
        B, H, L, D = 2, 4, 16, 32
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        cos, sin = build_rope_cache(D, L)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.shape == (B, H, L, D)
        assert k_rot.shape == (B, H, L, D)


class TestSwiGLU:
    """Tests for SwiGLU FFN."""

    def test_shape(self) -> None:
        B, L, D = 2, 16, 64
        ffn = SwiGLU(D, mult=4)
        x = torch.randn(B, L, D)
        out = ffn(x)
        assert out.shape == (B, L, D)


class TestSelfAttention:
    """Tests for self-attention."""

    def test_shape(self) -> None:
        B, L, D, H = 2, 16, 64, 4
        attn = SelfAttention(D, H)
        x = torch.randn(B, L, D)
        cos, sin = build_rope_cache(D // H, L)
        out = attn(x, cos, sin)
        assert out.shape == (B, L, D)

    def test_with_mask(self) -> None:
        B, L, D, H = 2, 16, 64, 4
        attn = SelfAttention(D, H)
        x = torch.randn(B, L, D)
        cos, sin = build_rope_cache(D // H, L)
        mask = torch.ones(B, L, dtype=torch.long)
        mask[:, -4:] = 0  # pad last 4 positions
        out = attn(x, cos, sin, attention_mask=mask)
        assert out.shape == (B, L, D)


class TestTransformerBlock:
    """Tests for transformer block."""

    @pytest.mark.parametrize("norm_type", ["pre", "post"])
    def test_both_norms(self, norm_type: str) -> None:
        B, L, D, H = 2, 16, 64, 4
        block = TransformerBlock(D, H, norm_type=norm_type)
        x = torch.randn(B, L, D)
        cos, sin = build_rope_cache(D // H, L)
        out = block(x, cos, sin)
        assert out.shape == (B, L, D)
        assert torch.isfinite(out).all()


class TestProteinLM:
    """Tests for the full model."""

    def test_forward_shape(self) -> None:
        B, L = 2, 20
        model = ProteinLM(dim=32, n_layers=2, n_heads=4, max_seq_len=64)
        ids = torch.randint(0, PAD_VOCAB_SIZE, (B, L))
        mask = torch.ones(B, L, dtype=torch.long)
        logits = model(ids, attention_mask=mask)
        assert logits.shape == (B, L, PAD_VOCAB_SIZE)

    def test_weight_tying(self) -> None:
        model = ProteinLM(dim=32, n_layers=2, n_heads=4)
        assert model.lm_head.weight is model.embed.weight

    def test_num_parameters(self) -> None:
        model = ProteinLM(dim=32, n_layers=2, n_heads=4)
        n = model.num_parameters()
        assert n > 0
        assert isinstance(n, int)

    @pytest.mark.parametrize("norm_type", ["pre", "post"])
    def test_norm_types(self, norm_type: str) -> None:
        model = ProteinLM(dim=32, n_layers=2, n_heads=4, norm_type=norm_type)
        ids = torch.randint(0, PAD_VOCAB_SIZE, (2, 10))
        out = model(ids)
        assert torch.isfinite(out).all()


class TestMuon:
    """Tests for Muon optimizer."""

    def test_step_no_error(self) -> None:
        linear = torch.nn.Linear(16, 16, bias=False)
        opt = Muon([linear.weight], lr=0.01)
        x = torch.randn(4, 16)
        loss = (linear(x) ** 2).sum()
        loss.backward()
        opt.step()  # should not raise
        opt.zero_grad()


class TestTrainLoop:
    """Integration tests for the training loop."""

    def test_smoke(self, tiny_train_data: list[list[int]], tiny_val_data: list[list[int]]) -> None:
        """Train for a few seconds on tiny data — should not error."""
        result = train(
            tiny_train_data,
            tiny_val_data,
            max_seconds=10,
            device="cpu",
        )
        assert "val_loss" in result
        assert "train_loss" in result
        assert "steps" in result
        assert "params" in result
        assert result["val_loss"] > 0
        assert result["steps"] > 0
        assert result["params"] > 0
