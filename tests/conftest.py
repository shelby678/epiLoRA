"""Shared fixtures for autoprot tests."""

from __future__ import annotations

import pytest

from prepare import encode

# Real protein sequences (short fragments from well-known proteins)
REAL_SEQUENCES = [
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",  # Hemoglobin alpha
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRN",  # Lysozyme fragment
    "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESL",  # Ubiquitin fragment
    "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMI",  # Calmodulin
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY",  # KRAS
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL",  # GFP fragment
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVL",  # BSA fragment
    "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSL",  # Beta-2 AR fragment
]


@pytest.fixture
def tiny_sequences() -> list[str]:
    """A handful of real short protein sequences."""
    return REAL_SEQUENCES


@pytest.fixture
def tiny_train_data() -> list[list[int]]:
    """Pre-encoded token ID lists for training."""
    return [encode(seq) for seq in REAL_SEQUENCES[:6]]


@pytest.fixture
def tiny_val_data() -> list[list[int]]:
    """Pre-encoded token ID lists for validation."""
    return [encode(seq) for seq in REAL_SEQUENCES[6:]]
