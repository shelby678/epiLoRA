"""BEPIPRED 3-fold CV experiments: ESM3 + LoRA + RYS.

One config-driven runner that replaces the former ``run_bepipred_<name>.py``
scripts. Each experiment *set* is a named batch of configs; pick one on the
command line::

    uv run python run_bepipred.py baseline
    uv run python run_bepipred.py hiddenkey
    uv run python run_bepipred.py --list
    uv run python run_bepipred.py all

Available sets:
    baseline         best SAbDab config (RYS + LoRA), no extra features
    rsa              RSA masking / RSA-bio-BLOSUM features (round 1)
    features         RSA / bio / BLOSUM as head inputs, no masking (round 2)
    dropout          structure / LayerDrop / DropHead sweep (no RYS)
    hiddenkey        DropKey / HiddenCut / KL sweep (ACL'24 paper)
    hky_pscan        HiddenKey probability scan
    hkx              HiddenKey + LayerDrop / RYS / DropAttention combos
    lora_all_active  LoRA on all 48 blocks + active LayerDrop
    lora_scale       LoRA rank / coverage scaling on HiddenKey
    lora_select      selective per-head LoRA on probe-activated heads
    probe_ldrop      probe-derived LayerDrop schedule
    pretrain         PDB-contacts mixing / two-stage pretraining
    ultra            best features + best pretraining combined

Results append to ``results.tsv`` (schema: commit exp test_fold run val_loss
val_auc test_auc steps peak_vram_mb elapsed_s desc). Folds already logged for
the same (exp, test_fold, run) are skipped, so a crashed run can resume.

Sets that read ``data/activation_probe.pt`` (lora_select, probe_ldrop) or
``data/pdb_contacts.fasta`` (pretrain, ultra) load those artifacts lazily, only
when the set is selected.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train_struct import (
    BATCH_SIZE, DROPOUT, LR, MAX_SEQ_LEN, WARMUP_STEPS, WEIGHT_DECAY,
    StructureEpitopePredictionModel,
    create_cv_datasets, load_contacts_data, train, compute_roc_auc,
)
from features import compute_extra_features, extra_feature_dim

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
CONTACTS_FASTA = Path("data/pdb_contacts.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
RESULTS_TSV    = Path("results.tsv")
PROBE_PATH     = Path("data/activation_probe.pt")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 3-fold CV: hold out each of the three largest partitions in turn.
CV_FOLDS = [
    {"test": "1", "val": "1"},  # train on 2+3+4+5, holdout=1
    {"test": "2", "val": "2"},  # train on 1+3+4+5, holdout=2
    {"test": "3", "val": "3"},  # train on 1+2+4+5, holdout=3
]
N_RERUNS    = 1
TIME_BUDGET = 1200  # 20 min ceiling per fold; early stop usually kicks in first
RNG         = random.Random(42)

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

# ---------------------------------------------------------------------------
# Shared base hyperparameter blocks
# ---------------------------------------------------------------------------
COMMON = dict(
    dropout=DROPOUT, batch_size=BATCH_SIZE, lr=LR,
    weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS,
    val_eval_interval=200, patience=5,
    compute_auc=True, device=DEVICE,
)
RYS_BEST   = dict(rys_start=36, rys_end=44)
NO_RYS     = dict(rys_start=0, rys_end=0)
LORA_LAST8 = dict(lora_rank=4, lora_alpha=8.0, lora_n_blocks=8, lora_block_start=-1)
LORA_ALL48 = dict(lora_rank=4, lora_alpha=8.0, lora_n_blocks=48, lora_block_start=0)

# best SAbDab config (baseline + RSA experiments share it)
BEST_CONFIG = {**COMMON, **RYS_BEST, **LORA_LAST8}
# HiddenKey family: LoRA last-8, no RYS, paper dropout restricted to LoRA blocks
HK_BASE     = {**COMMON, **NO_RYS, **LORA_LAST8, "paper_drop_only_lora": True}
HKX_BASE    = {**HK_BASE, "layer_drop_only_lora": True}
HK_ARROW    = {**HKX_BASE, "dropkey_prob": 0.10, "hiddencut_prob": 0.10}
# structure / LayerDrop / DropHead sweep: LoRA last-8, no RYS, plain dropout
DROPOUT_BASE   = {**COMMON, **NO_RYS, **LORA_LAST8}
ALLACTIVE_BASE = {**COMMON, **NO_RYS, **LORA_ALL48, "layer_drop_only_lora": True}
# LoRA scaling / selective LoRA: HiddenKey-arrow dropout, rank/coverage per-exp
SCALE_BASE  = {**COMMON, **NO_RYS, "lora_alpha": 8.0,
               "paper_drop_only_lora": True, "dropkey_prob": 0.10, "hiddencut_prob": 0.10}
SELECT_BASE = {**COMMON, **NO_RYS, "lora_rank": 4, "lora_alpha": 8.0,
               "paper_drop_only_lora": True, "dropkey_prob": 0.10, "hiddencut_prob": 0.10}
# probe-derived LayerDrop: pass explicit target blocks (overrides only-LoRA)
PLD_BASE    = {**COMMON, **NO_RYS, **LORA_LAST8, "paper_drop_only_lora": True,
               "dropkey_prob": 0.10, "hiddencut_prob": 0.10, "layer_drop_only_lora": False}
# pretraining / ultra: best config with the best RSA masking on by default
PRETRAIN_BASE = {**COMMON, **RYS_BEST, **LORA_LAST8, "rsa_surface_threshold": 0.15}
ULTRA_BASE    = {**COMMON, **RYS_BEST, **LORA_LAST8}

# Directive keys consumed by run_one (stripped before **hparams reaches train()).
_DIRECTIVE_KEYS = ("_stage1", "_stage1_seconds", "_stage2_seconds",
                   "_mixin", "_mixin_balanced", "_n_reruns")

# ---------------------------------------------------------------------------
# Lazy artifact loaders (probe + PDB contacts)
# ---------------------------------------------------------------------------
_CONTACTS_CACHE: dict = {}


def _contacts(interface: tuple[int, int] | None = None) -> list:
    """PDB-contact StructSamples, optionally filtered by interface size. Cached."""
    if interface not in _CONTACTS_CACHE:
        if not CONTACTS_FASTA.exists():
            logger.warning(f"{CONTACTS_FASTA} not found — using empty contact set")
            _CONTACTS_CACHE[interface] = []
        elif interface is None:
            _CONTACTS_CACHE[interface] = load_contacts_data(CONTACTS_FASTA)
        else:
            lo, hi = interface
            _CONTACTS_CACHE[interface] = load_contacts_data(
                CONTACTS_FASTA, interface_min=lo, interface_max=hi)
    return _CONTACTS_CACHE[interface]


def load_probe_topk(k: int) -> dict[int, list[int]]:
    """Return {block_idx: [head_idx, ...]} for the top-k (block, head) pairs."""
    data = torch.load(PROBE_PATH, weights_only=False)
    flat = data["per_block_head_mag"].flatten()  # (n_blocks * n_heads,)
    n_heads = int(data["n_heads"])
    top_idx = torch.topk(flat, k=k).indices.tolist()
    out: dict[int, list[int]] = {}
    for idx in top_idx:
        b, h = divmod(idx, n_heads)
        out.setdefault(b, []).append(h)
    return {b: sorted(hs) for b, hs in out.items()}


def all_heads_in_blocks(blocks: list[int]) -> dict[int, list[int]]:
    return {b: list(range(24)) for b in blocks}


def top_k_blocks_from_probe(k: int) -> list[int]:
    """Return indices of the k most-activated blocks (descending |context|)."""
    data = torch.load(PROBE_PATH, weights_only=False)
    block_mag = data["per_block_mag"]  # (n_blocks,) sum over heads
    return torch.topk(block_mag, k=k).indices.sort().values.tolist()


# ---------------------------------------------------------------------------
# Experiment sets — each returns [(name, desc, cfg), ...]
# ---------------------------------------------------------------------------

def _set_baseline():
    return [("bepipred-baseline",
             "RYS(36,44)+LoRA rank4 last-8, best SAbDab config, no extra features",
             dict(BEST_CONFIG))]


def _set_rsa():
    return [
        ("rsa-mask-15", "Surface masking RSA>0.15 (exclude buried from loss)",
         {**BEST_CONFIG, "rsa_surface_threshold": 0.15}),
        ("rsa-mask-25", "Surface masking RSA>0.25 (stricter surface filter)",
         {**BEST_CONFIG, "rsa_surface_threshold": 0.25}),
        ("rsa-feat", "RSA as 1-dim head input (no masking)",
         {**BEST_CONFIG, "rsa_as_feature": True}),
        ("rsa-feat-mask15", "RSA as head feature + surface masking RSA>0.15",
         {**BEST_CONFIG, "rsa_as_feature": True, "rsa_surface_threshold": 0.15}),
        ("bio-feat", "Biophysical AA properties (4-dim: hydrophobicity, charge, volume, polarity)",
         {**BEST_CONFIG, "bio_features": True}),
        ("bio-rsa-mask15", "Biophysical features + RSA masking at 0.15",
         {**BEST_CONFIG, "bio_features": True, "rsa_surface_threshold": 0.15}),
        ("blosum-feat", "BLOSUM62 rows (20-dim) as head input",
         {**BEST_CONFIG, "blosum_features": True}),
        ("all-feat-mask15", "RSA feature + biophysical + BLOSUM62 + surface mask 0.15",
         {**BEST_CONFIG, "rsa_as_feature": True, "bio_features": True,
          "blosum_features": True, "rsa_surface_threshold": 0.15}),
    ]


def _set_features():
    base = {**BEST_CONFIG, "rsa_surface_threshold": 0.0}  # no masking
    return [
        ("rsa-feat", "RSA as 1-dim head input (no masking)",
         {**base, "rsa_as_feature": True}),
        ("bio-feat", "Biophysical AA props (4-dim: hydrophob/charge/vol/polarity), no masking",
         {**base, "bio_features": True}),
        ("blosum-feat", "BLOSUM62 rows (20-dim) as head input, no masking",
         {**base, "blosum_features": True}),
        ("rsa-bio", "RSA (1-dim) + biophysical (4-dim) as head inputs, no masking",
         {**base, "rsa_as_feature": True, "bio_features": True}),
        ("all-feat", "RSA + biophysical + BLOSUM (25-dim total), no masking",
         {**base, "rsa_as_feature": True, "bio_features": True, "blosum_features": True}),
    ]


def _set_dropout():
    return [
        ("baseline-no-rys", "no-RYS baseline: LoRA rank=4 last-8 only",
         dict(DROPOUT_BASE)),
        ("struct-dropout-50", "structure dropout: NaN coords per-sample with p=0.5",
         {**DROPOUT_BASE, "structure_dropout_prob": 0.50}),
        ("layerdrop-uniform-15", "LayerDrop uniform: skip each LoRA block iid p=0.15",
         {**DROPOUT_BASE, "layer_drop_mode": "uniform", "layer_drop_prob": 0.15,
          "layer_drop_only_lora": True}),
        ("layerdrop-active-2-30", "LayerDrop active: drop top-2 LoRA blocks by EMA-activation, p=0.30",
         {**DROPOUT_BASE, "layer_drop_mode": "active", "layer_drop_prob": 0.30,
          "layer_drop_topk": 2, "layer_drop_only_lora": True}),
        ("drophead-uniform-15", "DropHead uniform: each head zeroed iid p=0.15",
         {**DROPOUT_BASE, "head_drop_mode": "uniform", "head_drop_prob": 0.15}),
        ("drophead-active-2-30", "DropHead active: per-batch zero top-2 highest-activation heads, p=0.30",
         {**DROPOUT_BASE, "head_drop_mode": "active", "head_drop_prob": 0.30,
          "head_drop_topk": 2}),
    ]


def _set_hiddenkey():
    return [
        ("hk-dropkey-col-10", "DropKey column-wise p=0.10 (paper best position+pattern, no FFN drop, no KL)",
         {**HK_BASE, "dropkey_prob": 0.10}),
        ("hk-dropkey-col-20", "DropKey column-wise p=0.20",
         {**HK_BASE, "dropkey_prob": 0.20}),
        ("hk-hiddencut-elem-10", "HiddenCut element-wise on FFN SwiGLU output p=0.10 (paper LoRA preference)",
         {**HK_BASE, "hiddencut_prob": 0.10}),
        ("hk-hiddencut-elem-20", "HiddenCut element-wise p=0.20",
         {**HK_BASE, "hiddencut_prob": 0.20}),
        ("hk-dropattn-col-10", "DropAttention column-wise p=0.10 (paper: worst — NoGrad gradient noise)",
         {**HK_BASE, "dropattn_prob": 0.10}),
        ("hk-hiddenkey-arrow", "HiddenKey↗: DropKey 0.10 col + HiddenCut 0.10 elem, no KL",
         {**HK_BASE, "dropkey_prob": 0.10, "hiddencut_prob": 0.10}),
        ("hk-hiddenkey-kl05", "HiddenKey full: DropKey 0.10 + HiddenCut 0.10 + bidir KL weight=0.05",
         {**HK_BASE, "dropkey_prob": 0.10, "hiddencut_prob": 0.10, "kl_loss_weight": 0.05}),
        ("hk-hiddenkey-kl10", "HiddenKey full: DropKey 0.10 + HiddenCut 0.10 + bidir KL weight=0.10",
         {**HK_BASE, "dropkey_prob": 0.10, "hiddencut_prob": 0.10, "kl_loss_weight": 0.10}),
        ("hk-dropkey-col-30", "DropKey column-wise p=0.30 (matches LayerDrop active-2-30 regime)",
         {**HK_BASE, "dropkey_prob": 0.30}),
        ("hk-hiddencut-elem-30", "HiddenCut element-wise p=0.30",
         {**HK_BASE, "hiddencut_prob": 0.30}),
        ("hk-hiddenkey-2020-kl10", "HiddenKey full: DropKey 0.20 col + HiddenCut 0.20 elem + KL weight=0.10",
         {**HK_BASE, "dropkey_prob": 0.20, "hiddencut_prob": 0.20, "kl_loss_weight": 0.10}),
    ]


def _set_hky_pscan():
    return [
        ("hky-pscan-15", "HiddenKey↗ at p=0.15/0.15 (between sweep-1's 0.10 and 0.20)",
         {**HK_BASE, "dropkey_prob": 0.15, "hiddencut_prob": 0.15}),
        ("hky-pscan-20", "HiddenKey↗ at p=0.20/0.20 — sweep-1 only tested this with KL=0.10 (which hurt)",
         {**HK_BASE, "dropkey_prob": 0.20, "hiddencut_prob": 0.20}),
        ("hky-pscan-25", "HiddenKey↗ at p=0.25/0.25 — slightly past the 0.20 sweet spot",
         {**HK_BASE, "dropkey_prob": 0.25, "hiddencut_prob": 0.25}),
    ]


def _set_hkx():
    return [
        ("hkx-arrow-layerdrop",
         "HiddenKey↗ (DropKey 0.10 col + HiddenCut 0.10 elem) + LayerDrop active top-2 p=0.30",
         {**HK_ARROW, "layer_drop_mode": "active", "layer_drop_prob": 0.30, "layer_drop_topk": 2}),
        ("hkx-arrow-rys",
         "HiddenKey↗ + RYS(36, 44) — free depth boost on top of best dropout",
         {**HK_ARROW, "rys_start": 36, "rys_end": 44}),
        ("hkx-arrow-layerdrop-rys",
         "HiddenKey↗ + LayerDrop active-2-30 + RYS(36, 44) — all three regularizers",
         {**HK_ARROW, "rys_start": 36, "rys_end": 44,
          "layer_drop_mode": "active", "layer_drop_prob": 0.30, "layer_drop_topk": 2}),
        ("hkx-arrow-dropattn",
         "HiddenKey↗ + DropAttention col 0.05 — fold in the third paper drop at low rate",
         {**HK_ARROW, "dropattn_prob": 0.05}),
    ]


def _set_lora_all_active():
    return [
        ("lora-all48-no-drop", "LoRA rank=4 on all 48 blocks (~1.77M params), no LayerDrop",
         dict(ALLACTIVE_BASE)),
        ("lora-all48-active-6-40", "LoRA all 48 + active LayerDrop top-6, p=0.40 (~2.4 layers/step dropped)",
         {**ALLACTIVE_BASE, "layer_drop_mode": "active", "layer_drop_prob": 0.40, "layer_drop_topk": 6}),
        ("lora-all48-active-12-30", "LoRA all 48 + active LayerDrop top-12, p=0.30 (~3.6 layers/step)",
         {**ALLACTIVE_BASE, "layer_drop_mode": "active", "layer_drop_prob": 0.30, "layer_drop_topk": 12}),
        ("lora-all48-active-12-50", "LoRA all 48 + active LayerDrop top-12, p=0.50 (~6 layers/step, aggressive)",
         {**ALLACTIVE_BASE, "layer_drop_mode": "active", "layer_drop_prob": 0.50, "layer_drop_topk": 12}),
    ]


def _set_lora_scale():
    return [
        ("ls-blocks-16", "HK↗ + LoRA rank=4 last-16 blocks",
         {**SCALE_BASE, "lora_rank": 4, "lora_n_blocks": 16, "lora_block_start": -1}),
        ("ls-blocks-24", "HK↗ + LoRA rank=4 last-24 blocks",
         {**SCALE_BASE, "lora_rank": 4, "lora_n_blocks": 24, "lora_block_start": -1}),
        ("ls-blocks-48", "HK↗ + LoRA rank=4 all 48 blocks",
         {**SCALE_BASE, "lora_rank": 4, "lora_n_blocks": 48, "lora_block_start": 0}),
        ("ls-rank-2", "HK↗ + LoRA rank=2 last-8 blocks",
         {**SCALE_BASE, "lora_rank": 2, "lora_n_blocks": 8, "lora_block_start": -1}),
        ("ls-rank-8", "HK↗ + LoRA rank=8 last-8 blocks",
         {**SCALE_BASE, "lora_rank": 8, "lora_n_blocks": 8, "lora_block_start": -1}),
        ("ls-rank-16", "HK↗ + LoRA rank=16 last-8 blocks",
         {**SCALE_BASE, "lora_rank": 16, "lora_n_blocks": 8, "lora_block_start": -1}),
        ("ls-rank8-blocks-16", "HK↗ + LoRA rank=8 last-16 blocks",
         {**SCALE_BASE, "lora_rank": 8, "lora_n_blocks": 16, "lora_block_start": -1}),
        ("ls-rank8-blocks-48", "HK↗ + LoRA rank=8 all 48 blocks",
         {**SCALE_BASE, "lora_rank": 8, "lora_n_blocks": 48, "lora_block_start": 0}),
    ]


def _set_lora_select():
    top8, top16, top32 = load_probe_topk(8), load_probe_topk(16), load_probe_topk(32)
    print(f"[probe targets] top-8:  {top8}")
    print(f"[probe targets] top-16: {top16}")
    print(f"[probe targets] top-32: {top32}")
    return [
        ("sl-top-08-r4", "Selective LoRA rank=4 on top-8 activated (block, head) pairs + HK↗",
         {**SELECT_BASE, "lora_rank": 4, "head_lora_targets": top8}),
        ("sl-top-16-r4", "Selective LoRA rank=4 on top-16 activated (block, head) pairs + HK↗",
         {**SELECT_BASE, "lora_rank": 4, "head_lora_targets": top16}),
        ("sl-top-32-r4", "Selective LoRA rank=4 on top-32 activated (block, head) pairs + HK↗",
         {**SELECT_BASE, "lora_rank": 4, "head_lora_targets": top32}),
        ("sl-top-16-r8", "Selective LoRA rank=8 on top-16 — higher per-head capacity + HK↗",
         {**SELECT_BASE, "lora_rank": 8, "head_lora_targets": top16}),
        ("sl-top-32-r2", "Selective LoRA rank=2 on top-32 — lower per-head capacity + HK↗",
         {**SELECT_BASE, "lora_rank": 2, "head_lora_targets": top32}),
        ("sl-last4-all", "Selective LoRA rank=4 on ALL heads of blocks 44-47 + HK↗ (layer-only selection control)",
         {**SELECT_BASE, "lora_rank": 4, "head_lora_targets": all_heads_in_blocks([44, 45, 46, 47])}),
    ]


def _set_probe_ldrop():
    top4 = top_k_blocks_from_probe(4)  # expect [44, 45, 46, 47]
    top1 = top_k_blocks_from_probe(1)  # expect [47]
    print(f"[probe] top-4 blocks (sorted): {top4}")
    print(f"[probe] top-1 block:          {top1}")
    return [
        ("pld-top4-p20", f"HK↗ + probe-uniform LayerDrop on blocks {top4}, each iid p=0.20",
         {**PLD_BASE, "layer_drop_mode": "uniform", "layer_drop_prob": 0.20,
          "layer_drop_target_blocks": top4}),
        ("pld-top4-p30", f"HK↗ + probe-uniform LayerDrop on blocks {top4}, each iid p=0.30",
         {**PLD_BASE, "layer_drop_mode": "uniform", "layer_drop_prob": 0.30,
          "layer_drop_target_blocks": top4}),
        ("pld-47-p50", f"HK↗ + drop ONLY block 47 (probe top-1) with p=0.50",
         {**PLD_BASE, "layer_drop_mode": "uniform", "layer_drop_prob": 0.50,
          "layer_drop_target_blocks": top1}),
    ]


def _set_pretrain():
    half = TIME_BUDGET // 2
    return [
        ("contacts-all", "Mix all PDB contacts with BEPIPRED train",
         {**PRETRAIN_BASE, "_mixin": _contacts(None)}),
        ("contacts-small", "Mix small interfaces (10-35 res) + BEPIPRED",
         {**PRETRAIN_BASE, "_mixin": _contacts((10, 35))}),
        ("contacts-tiny", "Mix tiny interfaces (5-20 res, CDR-sized) + BEPIPRED",
         {**PRETRAIN_BASE, "_mixin": _contacts((5, 20))}),
        ("2stage-small", "2-stage: pretrain on small contacts → finetune BEPIPRED",
         {**PRETRAIN_BASE, "_stage1": _contacts((10, 35)),
          "_stage1_seconds": half, "_stage2_seconds": half}),
        ("2stage-tiny", "2-stage: pretrain on tiny contacts → finetune BEPIPRED",
         {**PRETRAIN_BASE, "_stage1": _contacts((5, 20)),
          "_stage1_seconds": half, "_stage2_seconds": half}),
        ("mix-small", "Balanced 1:1 mix: BEPIPRED + small contacts",
         {**PRETRAIN_BASE, "_mixin": _contacts((10, 35)), "_mixin_balanced": True}),
    ]


def _set_ultra():
    tiny, small = _contacts((5, 20)), _contacts((10, 35))
    two_stage = dict(_stage1_seconds=600, _stage2_seconds=600)
    return [
        ("ultra-1", "2-stage tiny-contacts (600s) → BEPIPRED (600s) + RSA mask 0.15 + RSA feature",
         {**ULTRA_BASE, "_stage1": tiny, **two_stage,
          "rsa_as_feature": True, "rsa_surface_threshold": 0.15}),
        ("ultra-2", "2-stage tiny-contacts + RSA mask 0.15 + RSA + bio + BLOSUM (all features)",
         {**ULTRA_BASE, "_stage1": tiny, **two_stage,
          "rsa_as_feature": True, "bio_features": True, "blosum_features": True,
          "rsa_surface_threshold": 0.15}),
        ("ultra-3", "2-stage small-contacts (600s) → BEPIPRED + RSA mask 0.15 + RSA feature",
         {**ULTRA_BASE, "_stage1": small, **two_stage,
          "rsa_as_feature": True, "rsa_surface_threshold": 0.15}),
        ("ultra-4", "2-stage tiny-contacts + RSA mask 0.15 + bio features (no BLOSUM)",
         {**ULTRA_BASE, "_stage1": tiny, **two_stage,
          "rsa_as_feature": True, "bio_features": True, "rsa_surface_threshold": 0.15}),
        ("ultra-best", "Best ultra config (5 reruns) — 2-stage tiny + RSA mask 0.15 + RSA feat",
         {**ULTRA_BASE, "_stage1": tiny, **two_stage,
          "rsa_as_feature": True, "rsa_surface_threshold": 0.15}),
    ]


SETS = {
    "baseline":        _set_baseline,
    "rsa":             _set_rsa,
    "features":        _set_features,
    "dropout":         _set_dropout,
    "hiddenkey":       _set_hiddenkey,
    "hky_pscan":       _set_hky_pscan,
    "hkx":             _set_hkx,
    "lora_all_active": _set_lora_all_active,
    "lora_scale":      _set_lora_scale,
    "lora_select":     _set_lora_select,
    "probe_ldrop":     _set_probe_ldrop,
    "pretrain":        _set_pretrain,
    "ultra":           _set_ultra,
}

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _already_done() -> set[tuple[str, str, int]]:
    """Set of (exp, test_fold, run) tuples already present in RESULTS_TSV."""
    if not RESULTS_TSV.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with open(RESULTS_TSV) as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                done.add((parts[1], parts[2], int(parts[3])))
            except ValueError:
                continue
    return done


def _balanced_mix(epitope_data: list, contact_data: list) -> list:
    """Downsample contacts to match epitope count, then concatenate."""
    sampled = RNG.sample(contact_data, min(len(contact_data), len(epitope_data)))
    return epitope_data + sampled


def _run_two_stage(train_data, val_data, stage1_data, s1: int, s2: int, hparams: dict) -> dict:
    """Stage 1: pretrain on contacts (no periodic eval). Stage 2: finetune on epitopes."""
    h = dict(hparams)
    vei = h.pop("val_eval_interval", 200)  # stage 1 evaluates never, stage 2 uses this
    r1 = train(stage1_data, val_data, max_seconds=s1, val_eval_interval=0, **h)
    logger.info(f"Stage-1 done ({r1['steps']} steps)")
    r2 = train(train_data, val_data, max_seconds=s2,
               initial_state_dict=r1["trainable_state"],
               val_eval_interval=vei, **h)
    r2["peak_vram_mb"] = max(r1["peak_vram_mb"], r2["peak_vram_mb"])
    return r2


def run_one(name: str, desc: str, cfg: dict) -> dict:
    """Run one experiment across all CV folds; append rows to RESULTS_TSV."""
    cfg = dict(cfg)
    stage1         = cfg.pop("_stage1", None)
    stage1_seconds = cfg.pop("_stage1_seconds", TIME_BUDGET // 2)
    stage2_seconds = cfg.pop("_stage2_seconds", TIME_BUDGET // 2)
    mixin          = cfg.pop("_mixin", None)
    mixin_balanced = cfg.pop("_mixin_balanced", False)
    n_reruns       = cfg.pop("_n_reruns", N_RERUNS)
    hparams        = cfg  # everything else is a valid train() kwarg

    rsa_f = hparams.get("rsa_as_feature", False)
    bio_f = hparams.get("bio_features", False)
    blo_f = hparams.get("blosum_features", False)
    ed = extra_feature_dim(rsa_f, bio_f, blo_f)
    extra_fn = None
    if ed:
        def extra_fn(batch, dev):
            return compute_extra_features(batch, dev, rsa_as_feature=rsa_f, bio=bio_f, blosum=blo_f)

    done = _already_done()
    print(f"\n{'=' * 60}\nEXPERIMENT: {name}\n  {desc}\n{'=' * 60}")

    fold_val_aucs: list[float] = []
    fold_test_aucs: list[float] = []

    for fold in CV_FOLDS:
        test_part, val_part = fold["test"], fold["val"]
        if all((name, test_part, r) in done for r in range(n_reruns)):
            print(f"  fold test={test_part}: SKIP (already in results.tsv)")
            continue

        logger.info(f"Loading fold: test={test_part}, val={val_part}")
        train_data, val_data, test_data = create_cv_datasets(
            BEPIPRED_FASTA, STRUCTURES_DIR,
            test_partition=test_part, val_partition=val_part, max_length=MAX_SEQ_LEN,
        )
        n_struct_tr = sum(1 for _, _, c, _ in train_data if c is not None)
        n_struct_ts = sum(1 for _, _, c, _ in test_data if c is not None)
        logger.info(
            f"  train={len(train_data)} ({n_struct_tr} w/ struct)  "
            f"val={len(val_data)}  test={len(test_data)} ({n_struct_ts} w/ struct)"
        )

        for run_idx in range(n_reruns):
            if (name, test_part, run_idx) in done:
                continue
            t0 = time.time()

            if stage1 is not None:
                result = _run_two_stage(train_data, val_data, stage1,
                                        stage1_seconds, stage2_seconds, hparams)
            elif mixin is not None:
                td = _balanced_mix(train_data, mixin) if mixin_balanced else train_data + mixin
                result = train(td, val_data, max_seconds=TIME_BUDGET, **hparams)
            else:
                result = train(train_data, val_data, max_seconds=TIME_BUDGET, **hparams)

            elapsed = time.time() - t0

            # Rebuild model with the same architecture to load best state for test eval.
            model = StructureEpitopePredictionModel(
                dropout=hparams.get("dropout", DROPOUT),
                rys_start=hparams.get("rys_start", 0),
                rys_end=hparams.get("rys_end", 0),
                lora_rank=hparams.get("lora_rank", 0),
                lora_alpha=hparams.get("lora_alpha", 8.0),
                lora_n_blocks=hparams.get("lora_n_blocks", 8),
                lora_block_start=hparams.get("lora_block_start", -1),
                extra_dim=ed,
                head_lora_targets=hparams.get("head_lora_targets", None),
            ).to(DEVICE)
            cur = model.state_dict()
            cur.update({k: v.to(DEVICE) for k, v in result["trainable_state"].items() if k in cur})
            model.load_state_dict(cur)

            test_auc = compute_roc_auc(model, test_data, batch_size=BATCH_SIZE,
                                       device=DEVICE, extra_fn=extra_fn)
            del model, cur
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            val_auc = result["roc_auc"]
            fold_val_aucs.append(val_auc)
            fold_test_aucs.append(test_auc)
            print(
                f"  fold test={test_part} run={run_idx + 1}/{n_reruns}:  "
                f"val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  "
                f"val_loss={result['val_loss']:.4f}  steps={result['steps']}  "
                f"elapsed={elapsed:.0f}s"
            )
            with open(RESULTS_TSV, "a") as f:
                f.write(
                    f"{COMMIT}\t{name}\t{test_part}\t{run_idx}\t"
                    f"{result['val_loss']:.6f}\t{val_auc:.6f}\t{test_auc:.6f}\t"
                    f"{result['steps']}\t{result['peak_vram_mb']}\t{elapsed:.1f}\t"
                    f"{desc}\n"
                )

    # Summarize across all logged rows for this experiment (spans skipped folds).
    all_val, all_test = _summary_rows(name)
    if not all_test:
        all_val, all_test = fold_val_aucs, fold_test_aucs
    mean = float(np.mean(all_test)) if all_test else float("nan")
    std = float(np.std(all_test)) if all_test else float("nan")
    print(f"\n{'─' * 60}\nSUMMARY {name}")
    print(f"  test_auc = {mean:.4f} ± {std:.4f}   "
          f"(val_auc={np.mean(all_val) if all_val else float('nan'):.4f})  [n={len(all_test)}]")
    print(f"{'─' * 60}")
    return {"name": name, "test_auc_mean": mean, "test_auc_std": std}


def _summary_rows(name: str) -> tuple[list[float], list[float]]:
    """Return (val_aucs, test_aucs) for all RESULTS_TSV rows of this experiment."""
    all_val: list[float] = []
    all_test: list[float] = []
    if RESULTS_TSV.exists():
        with open(RESULTS_TSV) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7 and parts[1] == name:
                    try:
                        all_val.append(float(parts[5]))
                        all_test.append(float(parts[6]))
                    except ValueError:
                        pass
    return all_val, all_test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_set", nargs="?",
                        help="name of the experiment set to run, or 'all'")
    parser.add_argument("--list", action="store_true", help="list available sets and exit")
    args = parser.parse_args()

    if args.list or not args.experiment_set:
        print("Available experiment sets:")
        for name in SETS:
            print(f"  {name}")
        print("  all   (run every set)")
        return

    if args.experiment_set != "all" and args.experiment_set not in SETS:
        parser.error(f"unknown set '{args.experiment_set}'. Use --list to see options.")

    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\texp\ttest_fold\trun\tval_loss\tval_auc\ttest_auc\t"
                    "steps\tpeak_vram_mb\telapsed_s\tdesc\n")

    set_names = list(SETS) if args.experiment_set == "all" else [args.experiment_set]
    summaries = []
    for set_name in set_names:
        try:
            experiments = SETS[set_name]()
        except FileNotFoundError as e:
            print(f"\n### SKIP set '{set_name}': missing artifact ({e})")
            continue
        print(f"\n########## SET: {set_name} ({len(experiments)} experiments) ##########")
        for name, desc, cfg in experiments:
            summaries.append(run_one(name, desc, cfg))

    print(f"\n{'=' * 60}\nFINAL SUMMARY\n{'=' * 60}")
    print(f"{'method':<28}  test_auc")
    for s in summaries:
        print(f"{s['name']:<28}  {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f}")


if __name__ == "__main__":
    main()
