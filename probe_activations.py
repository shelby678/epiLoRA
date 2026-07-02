"""Probe per-(block, head) attention activation magnitudes on BEPIPRED antigens.

Runs the frozen ESM3-small-open backbone over every BEPIPRED antigen sequence
(EVAL partition excluded) and records the mean |context_BHLD| per attention
head per layer. The result is a (48, 24) tensor saved to
``data/activation_probe.pt`` along with the per-block global magnitudes.

Used to select which heads/layers to LoRA-adapt in the selective-LoRA
experiment.

Run:
    .venv/bin/python probe_activations.py > /tmp/probe.log 2>&1
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
import types
from pathlib import Path

import einops
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from prepare import load_combined_fasta_partitioned, PAD_ID
from train_struct import (
    StructureEpitopePredictionModel, MAX_SEQ_LEN, BATCH_SIZE, struct_collate_fn,
    create_struct_dataloader, _load_coords_rsa,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

BEPIPRED_FASTA = Path("data/BEPIPRED.fasta")
STRUCTURES_DIR = Path("data/structures2/sabdab_dataset")
OUT_PATH = Path("data/activation_probe.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _patch_attention_for_probe(attn_module, sink, block_idx):
    """Replace MultiHeadAttention.forward to record per-head context |context|."""
    def new_forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD = self.q_ln(query_BLD).to(query_BLD.dtype)
        key_BLD = self.k_ln(key_BLD).to(query_BLD.dtype)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )
        query_BHLD, key_BHLD, value_BHLD = map(reshaper, (query_BLD, key_BLD, value_BLD))
        if seq_id is not None:
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)
            context_BHLD = torch.nn.functional.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
        else:
            context_BHLD = torch.nn.functional.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )

        # Record per-head mean |context| for this batch.
        with torch.no_grad():
            head_mag = context_BHLD.detach().float().abs().mean(dim=(0, 2, 3))  # (H,)
            n_tok = context_BHLD.shape[0] * context_BHLD.shape[2]
            sink["sum"][block_idx] += head_mag.cpu() * n_tok
            sink["count"][block_idx] += n_tok

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)

    attn_module.forward = types.MethodType(new_forward, attn_module)


def load_all_antigens() -> list[tuple[list[int], list[int], torch.Tensor | None, torch.Tensor | None]]:
    by_part = load_combined_fasta_partitioned(
        BEPIPRED_FASTA, exclude_partitions=frozenset({"EVAL"})
    )
    samples = []
    for part_id, header_samples in by_part.items():
        for header, (token_ids, labels) in header_samples:
            if len(token_ids) > MAX_SEQ_LEN:
                continue
            seq_len = len(token_ids) - 2
            coords, rsa = _load_coords_rsa(header, seq_len, STRUCTURES_DIR)
            samples.append((token_ids, labels, coords, rsa))
    return samples


def main() -> None:
    logger.info("Loading antigens (EVAL excluded)...")
    samples = load_all_antigens()
    logger.info(f"  {len(samples):,} sequences")

    logger.info("Building frozen ESM3-small-open + LoRA(rank=0)...")
    model = StructureEpitopePredictionModel(
        dropout=0.0,
        rys_start=0, rys_end=0,
        lora_rank=0,
    ).to(DEVICE)
    model.eval()

    blocks = model.esm3.layers if hasattr(model.esm3, "layers") else model.esm3.transformer.blocks
    n_blocks = len(blocks)
    n_heads = blocks[0].attn.n_heads
    logger.info(f"  {n_blocks} blocks, {n_heads} heads/block")

    sink = {
        "sum": torch.zeros(n_blocks, n_heads, dtype=torch.float32),
        "count": torch.zeros(n_blocks, dtype=torch.float64),
    }
    for i, block in enumerate(blocks):
        if hasattr(block, "attn") and block.attn is not None:
            _patch_attention_for_probe(block.attn, sink, i)

    loader = create_struct_dataloader(samples, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"Forwarding {len(samples)} sequences in {len(loader)} batches...")
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            coords = batch["structure_coords"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
                _ = model(input_ids, attention_mask=attention_mask, structure_coords=coords)
            if (bi + 1) % 10 == 0:
                logger.info(f"  batch {bi+1}/{len(loader)}  elapsed={time.time()-t0:.0f}s")

    # Normalise: per-(block, head) average magnitude across tokens
    counts = sink["count"].clamp_min(1).unsqueeze(-1)  # (B, 1)
    avg = sink["sum"] / counts  # (B, H)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "per_block_head_mag": avg,             # (n_blocks, n_heads)
            "per_block_mag": avg.sum(dim=-1),      # (n_blocks,)
            "per_head_mag_flat": avg.flatten(),    # (n_blocks * n_heads,)
            "n_sequences": len(samples),
            "n_blocks": n_blocks,
            "n_heads": n_heads,
        },
        OUT_PATH,
    )
    logger.info(f"Saved probe results to {OUT_PATH} (elapsed {time.time()-t0:.0f}s)")

    # Print quick top-K summary.
    print("\nTop-16 most-activated (block, head) pairs:")
    print(f"{'rank':<4}  {'block':>5}  {'head':>4}  mean_|ctx|")
    flat = avg.flatten()
    top_idx = torch.topk(flat, k=16).indices
    for rk, idx in enumerate(top_idx.tolist()):
        b, h = divmod(idx, n_heads)
        print(f"{rk+1:<4}  {b:>5}  {h:>4}  {flat[idx]:.5f}")

    print("\nTop-16 most-activated blocks (sum over heads):")
    block_mag = avg.sum(dim=-1)
    top_blocks = torch.topk(block_mag, k=16).indices
    for rk, b in enumerate(top_blocks.tolist()):
        print(f"  {rk+1:<3}  block {b:>2}   sum_|ctx| = {block_mag[b]:.5f}")


if __name__ == "__main__":
    main()
