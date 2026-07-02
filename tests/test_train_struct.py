"""Tests for train_struct: PDB parsing, structure loading, collation, and model."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from train_struct import (
    _BACKBONE_ATOMS,
    _OUR_TO_ESM3,
    PAD_VOCAB_SIZE,
    load_structure_coords,
    parse_pdb_backbone,
    struct_collate_fn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
EXAMPLE_PDB_ID = "7l7d"
EXAMPLE_PDB = STRUCTURES_DIR / EXAMPLE_PDB_ID / "structure" / f"{EXAMPLE_PDB_ID}.pdb"


def _minimal_pdb(tmp_path: Path, chain: str, residues: list[tuple[int, str]]) -> Path:
    """Write a minimal PDB with perfect N/CA/C coords for *residues*.

    Each residue gets N at (r, 0, 0), CA at (r, 1, 0), C at (r, 2, 0)
    where r is the residue sequence number — easy to check in tests.
    """
    lines = []
    serial = 1
    for resseq, _ in residues:
        for atom_idx, atom_name in enumerate(_BACKBONE_ATOMS):
            x = float(resseq)
            y = float(atom_idx)
            z = 0.0
            line = (
                f"ATOM  {serial:5d}  {atom_name:<3s} ALA {chain}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
            )
            lines.append(line)
            serial += 1
    path = tmp_path / "test.pdb"
    path.write_text("".join(lines))
    return path


# ---------------------------------------------------------------------------
# PDB backbone parser
# ---------------------------------------------------------------------------


class TestParsePdbBackbone:
    def test_returns_one_entry_per_residue(self, tmp_path: Path) -> None:
        residues = [(1, " "), (2, " "), (3, " ")]
        pdb = _minimal_pdb(tmp_path, "A", residues)
        result = parse_pdb_backbone(pdb, "A")
        assert result is not None
        assert len(result) == 3

    def test_coords_match_expected_values(self, tmp_path: Path) -> None:
        residues = [(5, " "), (6, " ")]
        pdb = _minimal_pdb(tmp_path, "B", residues)
        result = parse_pdb_backbone(pdb, "B")
        assert result is not None
        # Residue 5: N=(5,0,0), CA=(5,1,0), C=(5,2,0)
        flat = result[0]
        assert flat[0] == pytest.approx(5.0)  # N_x
        assert flat[1] == pytest.approx(0.0)  # N_y
        assert flat[3] == pytest.approx(5.0)  # CA_x
        assert flat[4] == pytest.approx(1.0)  # CA_y
        assert flat[6] == pytest.approx(5.0)  # C_x
        assert flat[7] == pytest.approx(2.0)  # C_y

    def test_wrong_chain_returns_none(self, tmp_path: Path) -> None:
        pdb = _minimal_pdb(tmp_path, "A", [(1, " "), (2, " ")])
        result = parse_pdb_backbone(pdb, "Z")
        assert result is None

    def test_missing_backbone_atom_returns_none(self, tmp_path: Path) -> None:
        # Write a PDB with only CA atoms — N and C are missing.
        lines = []
        for resseq in (1, 2, 3):
            lines.append(
                f"ATOM      1  CA  ALA A{resseq:4d}    "
                f"{float(resseq):8.3f}   1.000   0.000  1.00  0.00           C  \n"
            )
        pdb = tmp_path / "ca_only.pdb"
        pdb.write_text("".join(lines))
        result = parse_pdb_backbone(pdb, "A")
        assert result is None

    def test_alternate_location_skipped(self, tmp_path: Path) -> None:
        # Alt-loc 'B' records should be ignored; only ' ' and 'A' are kept.
        lines = []
        for resseq in (1, 2):
            for atom_name in _BACKBONE_ATOMS:
                # Primary location
                lines.append(
                    f"ATOM      1  {atom_name:<3s} ALA A{resseq:4d}    "
                    f"   1.000   0.000   0.000  1.00  0.00           C  \n"
                )
                # Alt-loc B — should be ignored
                lines.append(
                    f"ATOM      2  {atom_name:<3s}BALA A{resseq:4d}    "
                    f"  99.000  99.000  99.000  0.50  0.00           C  \n"
                )
        pdb = tmp_path / "alt_loc.pdb"
        pdb.write_text("".join(lines))
        result = parse_pdb_backbone(pdb, "A")
        assert result is not None
        assert len(result) == 2
        # CA_x should be 1.0, not 99.0
        assert result[0][3] == pytest.approx(1.0)

    def test_sorts_by_residue_number(self, tmp_path: Path) -> None:
        # Write residues out of order: 3, 1, 2
        lines = []
        for resseq in (3, 1, 2):
            for atom_name in _BACKBONE_ATOMS:
                lines.append(
                    f"ATOM      1  {atom_name:<3s} ALA A{resseq:4d}    "
                    f"{float(resseq):8.3f}   0.000   0.000  1.00  0.00           C  \n"
                )
        pdb = tmp_path / "unsorted.pdb"
        pdb.write_text("".join(lines))
        result = parse_pdb_backbone(pdb, "A")
        assert result is not None
        # Should be sorted: residue 1 first
        assert result[0][0] == pytest.approx(1.0)  # N_x of residue 1
        assert result[2][0] == pytest.approx(3.0)  # N_x of residue 3

    @pytest.mark.skipif(not EXAMPLE_PDB.exists(), reason="7l7d PDB not available")
    def test_real_pdb_7l7d_antigen_chain_e(self) -> None:
        """Antigen chain E of 7l7d (SARS-CoV-2 RBD) should parse cleanly."""
        result = parse_pdb_backbone(EXAMPLE_PDB, "E")
        assert result is not None
        assert len(result) > 0
        # All coords should be finite numbers
        for row in result:
            for val in row:
                assert math.isfinite(val), f"Non-finite coordinate: {val}"


# ---------------------------------------------------------------------------
# Structure coordinate loading
# ---------------------------------------------------------------------------


class TestLoadStructureCoords:
    def test_returns_none_for_short_id(self, tmp_path: Path) -> None:
        coords = load_structure_coords("nopdb", 10, tmp_path)
        assert coords is None

    def test_returns_none_when_pdb_missing(self, tmp_path: Path) -> None:
        coords = load_structure_coords("xxxx_H_L_A", 10, tmp_path)
        assert coords is None

    def test_returns_none_when_length_mismatches(self, tmp_path: Path) -> None:
        # Create a PDB with 3 residues, request seq_len=5
        pdb_dir = tmp_path / "test1" / "structure"
        pdb_dir.mkdir(parents=True)
        pdb_file = _minimal_pdb(pdb_dir / "..", "A", [(1, " "), (2, " "), (3, " ")])
        pdb_file.rename(pdb_dir / "test1.pdb")
        coords = load_structure_coords("test1_H_L_A", 5, tmp_path)
        assert coords is None

    def test_correct_shape_and_values(self, tmp_path: Path) -> None:
        pdb_dir = tmp_path / "mypdb" / "structure"
        pdb_dir.mkdir(parents=True)
        residues = [(1, " "), (2, " "), (4, " ")]  # seq_len=3 even with gap in numbering
        tmp_pdb = _minimal_pdb(tmp_path, "A", residues)
        tmp_pdb.rename(pdb_dir / "mypdb.pdb")

        coords = load_structure_coords("mypdb_H_L_A", 3, tmp_path)
        assert coords is not None
        assert coords.shape == (3, 3, 3)
        # Residue 1: N=(1,0,0) → coords[0, 0, :] = [1.0, 0.0, 0.0]
        assert coords[0, 0, 0].item() == pytest.approx(1.0)  # N_x
        assert coords[0, 0, 1].item() == pytest.approx(0.0)  # N_y
        # Residue 1: CA=(1,1,0) → coords[0, 1, :] = [1.0, 1.0, 0.0]
        assert coords[0, 1, 0].item() == pytest.approx(1.0)  # CA_x
        assert coords[0, 1, 1].item() == pytest.approx(1.0)  # CA_y

    @pytest.mark.skipif(not EXAMPLE_PDB.exists(), reason="7l7d PDB not available")
    def test_real_pdb_7l7d(self) -> None:
        """Test load_structure_coords using the real 7l7d PDB (antigen chain E).

        7l7d is SARS-CoV-2 RBD (chain E) bound to antibody AZD8895 (H/L chains).
        The seq_len is derived directly from the parsed CA count so it always matches.
        """
        # Derive seq_len from the actual PDB
        flat = parse_pdb_backbone(EXAMPLE_PDB, "E")
        assert flat is not None, "Could not parse 7l7d chain E"
        seq_len = len(flat)

        coords = load_structure_coords(f"7l7d_H_L_E", seq_len, STRUCTURES_DIR)
        assert coords is not None, "load_structure_coords returned None for 7l7d_H_L_E"
        assert coords.shape == (seq_len, 3, 3)
        assert torch.isfinite(coords).all(), "Non-finite coordinates in 7l7d structure"


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


class TestStructCollateFn:
    def _make_sample(
        self,
        seq: str,
        has_coords: bool = False,
    ) -> tuple:
        """Build a minimal StructSample from a raw AA string."""
        from prepare import encode

        token_ids = encode(seq)
        seq_len = len(seq)
        labels = [-100] + [0] * seq_len + [-100]
        coords = torch.zeros(seq_len, 3, 3) if has_coords else None
        return token_ids, labels, coords

    def test_output_keys(self) -> None:
        batch = [self._make_sample("ACDE")]
        result = struct_collate_fn(batch)
        assert set(result.keys()) == {"input_ids", "attention_mask", "labels", "structure_coords"}

    def test_padding_to_max_length(self) -> None:
        from prepare import encode

        batch = [self._make_sample("AC"), self._make_sample("ACDEFGHI")]
        result = struct_collate_fn(batch)
        max_len = max(len(encode("AC")), len(encode("ACDEFGHI")))
        assert result["input_ids"].shape == (2, max_len)
        assert result["structure_coords"].shape == (2, max_len, 3, 3)

    def test_no_structure_gives_nan_coords(self) -> None:
        batch = [self._make_sample("ACDE", has_coords=False)]
        result = struct_collate_fn(batch)
        assert torch.isnan(result["structure_coords"]).all()

    def test_with_structure_fills_aa_positions(self) -> None:
        """AA positions (1 … L-2) should have the supplied coords; BOS/EOS/pad are NaN."""
        token_ids, labels, coords = self._make_sample("ACG", has_coords=True)
        # Give distinctive values so we can check the right positions are filled.
        coords[:] = 42.0
        batch = [(token_ids, labels, coords)]
        result = struct_collate_fn(batch)
        sc = result["structure_coords"][0]  # (L, 3, 3)
        L = len(token_ids)
        # BOS (pos 0) and EOS (pos L-1) should be NaN
        assert torch.isnan(sc[0]).all(), "BOS position should be NaN"
        assert torch.isnan(sc[L - 1]).all(), "EOS position should be NaN"
        # AA positions 1 … L-2 should be 42.0
        for pos in range(1, L - 1):
            assert (sc[pos] == 42.0).all(), f"AA position {pos} should be 42.0"

    def test_attention_mask_marks_padding(self) -> None:
        from prepare import encode

        short = self._make_sample("AC")
        long = self._make_sample("ACDEFGHI")
        result = struct_collate_fn([short, long])
        short_len = len(encode("AC"))
        assert result["attention_mask"][0, :short_len].all()
        assert not result["attention_mask"][0, short_len:].any()

    def test_labels_exclude_pad_positions(self) -> None:
        short = self._make_sample("AC")
        long = self._make_sample("ACDE")
        result = struct_collate_fn([short, long])
        from prepare import encode

        short_len = len(encode("AC"))
        # Padded positions of first sample should be -100
        assert (result["labels"][0, short_len:] == -100).all()

    def test_mixed_structure_availability(self) -> None:
        """Batch where only some samples have structure."""
        with_struct = self._make_sample("ACD", has_coords=True)
        without_struct = self._make_sample("ACG", has_coords=False)
        result = struct_collate_fn([with_struct, without_struct])
        sc = result["structure_coords"]
        # Sample without structure: all NaN
        assert torch.isnan(sc[1]).all()
        # Sample with structure: AA positions not NaN
        from prepare import encode

        L = len(encode("ACD"))
        assert not torch.isnan(sc[0, 1 : L - 1]).any()


# ---------------------------------------------------------------------------
# Vocabulary mapping
# ---------------------------------------------------------------------------


class TestVocabMapping:
    def test_special_tokens_map_correctly(self) -> None:
        assert _OUR_TO_ESM3[0] == 0   # BOS
        assert _OUR_TO_ESM3[1] == 1   # PAD
        assert _OUR_TO_ESM3[2] == 2   # EOS
        assert _OUR_TO_ESM3[4] == 32  # MASK

    def test_amino_acid_tokens_are_distinct(self) -> None:
        """All 20 standard AA IDs (positions 5-24) must map to distinct ESM3 IDs."""
        aa_mappings = [_OUR_TO_ESM3[i] for i in range(5, 25)]
        assert len(set(aa_mappings)) == 20, "Duplicate AA token mapping detected"

    def test_mapping_covers_full_vocab(self) -> None:
        assert len(_OUR_TO_ESM3) == PAD_VOCAB_SIZE

    def test_alanine_maps_to_5(self) -> None:
        """A=5 in prepare.py should map to A=5 in ESM3 (both agree)."""
        from prepare import VOCAB

        our_a = VOCAB["A"]
        assert _OUR_TO_ESM3[our_a] == 5

    def test_leucine_maps_to_4(self) -> None:
        """L=14 in prepare.py should map to L=4 in ESM3."""
        from prepare import VOCAB

        our_l = VOCAB["L"]
        assert _OUR_TO_ESM3[our_l] == 4


# ---------------------------------------------------------------------------
# Model integration (requires ESM3 weights — marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestStructureEpitopePredictionModel:
    @pytest.fixture(scope="class")
    def model(self):
        from train_struct import StructureEpitopePredictionModel

        return StructureEpitopePredictionModel(dropout=0.0).eval()

    def test_trainable_params_are_head_only(self, model) -> None:
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert all(n.startswith("head") for n in trainable), (
            f"Unexpected trainable params outside head: "
            f"{[n for n in trainable if not n.startswith('head')]}"
        )

    def test_esm3_backbone_is_frozen(self, model) -> None:
        for name, param in model.esm3.named_parameters():
            assert not param.requires_grad, f"ESM3 param {name} is not frozen"

    def test_forward_sequence_only(self, model) -> None:
        """Forward pass with no structure (all-NaN coords)."""
        from prepare import encode

        token_ids = [encode("ACDEFG")]
        batch = struct_collate_fn(
            [(tid, [-100] + [0] * (len(tid) - 2) + [-100], None) for tid in token_ids]
        )
        with torch.no_grad():
            logits = model(batch["input_ids"], structure_coords=batch["structure_coords"])
        assert logits.shape == batch["input_ids"].shape
        assert torch.isfinite(logits).all()

    def test_forward_with_structure_differs_from_sequence_only(self, model) -> None:
        """Providing real structure coords should change the output logits."""
        from prepare import encode

        seq = "ACDEFGHIKLMN"
        token_ids = encode(seq)
        labels = [-100] + [0] * len(seq) + [-100]
        coords = torch.randn(len(seq), 3, 3) * 10.0  # random but non-NaN structure

        batch_no_struct = struct_collate_fn([(token_ids, labels, None)])
        batch_with_struct = struct_collate_fn([(token_ids, labels, coords)])

        with torch.no_grad():
            logits_no = model(
                batch_no_struct["input_ids"],
                structure_coords=batch_no_struct["structure_coords"],
            )
            logits_yes = model(
                batch_with_struct["input_ids"],
                structure_coords=batch_with_struct["structure_coords"],
            )

        assert not torch.allclose(logits_no, logits_yes, atol=1e-4), (
            "Structure coords had no effect on model output — structure is not being used"
        )

    def test_structure_from_7l7d_changes_output(self, model) -> None:
        """Real 7l7d structure should produce different logits than no structure."""
        if not EXAMPLE_PDB.exists():
            pytest.skip("7l7d PDB not available")

        from prepare import encode, load_fasta

        fasta_path = Path("data/combined/1/train_aa.fasta")
        if not fasta_path.exists():
            pytest.skip("Training FASTA not available")

        aa_seqs = load_fasta(fasta_path)
        entry = next((sid for sid in aa_seqs if sid.startswith("7l7d_")), None)
        if entry is None:
            pytest.skip("7l7d not found in training FASTA")

        seq = aa_seqs[entry]
        token_ids = encode(seq)
        labels = [-100] + [0] * len(seq) + [-100]
        coords = load_structure_coords(entry, len(seq), STRUCTURES_DIR)
        if coords is None:
            pytest.skip(f"Could not load structure for {entry}")

        batch_no = struct_collate_fn([(token_ids, labels, None)])
        batch_yes = struct_collate_fn([(token_ids, labels, coords)])

        with torch.no_grad():
            logits_no = model(batch_no["input_ids"], structure_coords=batch_no["structure_coords"])
            logits_yes = model(batch_yes["input_ids"], structure_coords=batch_yes["structure_coords"])

        assert not torch.allclose(logits_no, logits_yes, atol=1e-4), (
            "Real structure had no effect on model output"
        )

    def test_gradients_flow_through_head(self, model) -> None:
        """Check backprop works and only head params receive gradients."""
        from prepare import encode

        model.train()
        token_ids = [encode("ACDE")]
        batch = struct_collate_fn(
            [(tid, [-100] + [0] * (len(tid) - 2) + [-100], None) for tid in token_ids]
        )
        logits = model(batch["input_ids"], structure_coords=batch["structure_coords"])
        valid = batch["labels"] != -100
        import torch.nn.functional as F

        loss = F.binary_cross_entropy_with_logits(
            logits[valid], batch["labels"][valid].float()
        )
        loss.backward()
        # Head params should have gradients
        for name, param in model.head.named_parameters():
            assert param.grad is not None, f"No gradient for head param {name}"
        # ESM3 params should have no gradient
        for name, param in model.esm3.named_parameters():
            assert param.grad is None, f"ESM3 param {name} has unexpected gradient"
        model.eval()
