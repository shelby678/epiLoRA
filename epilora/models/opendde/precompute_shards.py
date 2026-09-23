"""Shard the OpenDDE trunk-feature precompute across parallel tasks.

    python precompute_shards.py --shard $SLURM_ARRAY_TASK_ID --nshards 32

Each task takes a disjoint slice of the records every training/eval run
needs (the champion training FASTA + the two shared-benchmark eval FASTAs +
the final held-out eval set), runs the frozen sequence-only trunk once per
record, and writes the per-residue features to the shared embedding cache.
The cache (atomic writes) makes shards safe to run concurrently and the whole
thing resumable: re-running a shard skips every entry already on disk, so a
preempted (--requeue) or resubmitted array loses at most the one in-flight
record.

Run this (sharded over as many GPUs as you can get -- e.g. via your own
`sbatch --array` wrapper) *before* submitting an opendde sweep with
run_ablation.py: the fold jobs read the same cache, and on a cold one each
of them recomputes the ~1 min/antigen trunk pass itself, which no single
fold's time budget covers. The trunk-feature cache is the coords cache dir
next to data/raw/all-structures-extracted (shared storage), so shards warmed
from any node feed every fold.

Needs only the FASTAs -- no structures (the trunk is sequence-only). A few
records that training will skip anyway (no usable structure, so usable()
drops them) get cached too; harmless, their entries just go unread.

Load balancing: records are sorted by sequence length (descending) and
dealt round-robin, so every shard gets an ~equal sum of L^2 -- the trunk's
cost unit -- instead of whatever lengths happened to cluster in file order.

Must run under the opendde env (machine_config.yaml's env_opendde); set
OPENDDE_ROOT_DIR (checkpoint/ + common/) unless machine_config.yaml's
opendde_root already resolves on this machine.

``--dry-run`` reports cache warmth instead of computing anything (no GPU, no
checkpoint load): per shard, how many of its records are already cached vs
missing -- with the missing ones' sum of L^2, the trunk's cost unit -- plus a
total. Omit ``--shard`` to report every shard at once.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EPILORA = Path(__file__).resolve().parents[2]  # epilora/, for flat imports
if str(EPILORA) not in sys.path:
    sys.path.insert(0, str(EPILORA))

from data import default_cache_dir, parse_fasta  # noqa: E402

# train.py's default training set + shared benchmark, plus eval_final.py's
# held-out set -- everything any fold or the final eval will forward through.
FASTAS = [
    "data/train_test_eval/allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta",
    "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta",
    "data/train_test_eval/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
    "data/train_test_eval/eval/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta",
]

# Recycling cycles the trunk runs (and so the cache is keyed) with -- must
# match build_model_opendde's default, which the compute path below uses.
CYCLES = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def records_for(repo: Path):
    """Unique (header, seq, labels) records across the FASTAs, first-seen
    order (records repeat across the benchmark sets)."""
    entries, seen = [], set()
    for rel in FASTAS:
        for label, records in sorted(parse_fasta(repo / rel).items()):
            for h, s, l in records:
                if h not in seen:
                    seen.add(h)
                    entries.append((h, s, l))
    return entries


def dry_run(entries: list, args, emb_cache: Path, chunk_size: int) -> None:
    """Report cached vs missing records per shard, computing nothing (the
    cache keys come from models.opendde.trunk_cache_path, the same function
    the model's own cache reads go through)."""
    from models.opendde import trunk_cache_path

    shards = [args.shard] if args.shard is not None else range(args.nshards)
    log(f"dry run: {len(entries)} records, {args.nshards} shards, cache: {emb_cache}")
    tot_cached = tot_missing = tot_l2 = 0
    for shard in shards:
        mine = entries[shard::args.nshards]
        if args.limit is not None:
            mine = mine[:args.limit]
        missing_l2 = 0
        n_cached = 0
        for _, seq, _ in mine:
            path = trunk_cache_path(emb_cache, seq, cycles=CYCLES,
                                    chunk_size=chunk_size)
            if path.exists():
                n_cached += 1
            else:
                missing_l2 += len(seq) ** 2
        n_missing = len(mine) - n_cached
        tot_cached += n_cached
        tot_missing += n_missing
        tot_l2 += missing_l2
        extra = f" (missing sum L^2 = {missing_l2:,})" if n_missing else ""
        log(f"shard {shard}/{args.nshards}: {n_cached} cached, "
            f"{n_missing} missing{extra}")
    if args.shard is None:
        log(f"TOTAL: {tot_cached} cached, {tot_missing} missing "
            f"(missing sum L^2 = {tot_l2:,}) -> cache is "
            f"{'warm' if tot_missing == 0 else 'COLD'}; run without --dry-run "
            f"to compute the missing ones")



def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=EPILORA.parent,
                   help="the epiLoRA checkout (its data/train_test_eval/ is read)")
    p.add_argument("--emb-cache", type=Path, default=None,
                   help="trunk-feature cache dir (default: the coords cache dir "
                        "train.py's default --structures resolves to, so the array "
                        "feeds the fold trainings unchanged -- the path is derived, "
                        "no structure is read)")
    p.add_argument("--shard", type=int, default=None,
                   help="which shard to compute (required without --dry-run; with it, "
                        "the only shard reported)")
    p.add_argument("--nshards", type=int, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N records of this shard (testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="report cached vs missing records per shard without computing "
                        "anything (no GPU, no checkpoint load)")
    args = p.parse_args()

    if args.shard is None and not args.dry_run:
        p.error("--shard is required (or pass --dry-run to just report cache warmth)")

    repo = args.repo
    # Same dir train.py's default points its emb_cache at (the coords cache
    # next to its default --structures); derived for compatibility only --
    # this script never reads a structure.
    emb_cache = args.emb_cache or default_cache_dir(
        repo / "data/raw/all-structures-extracted")

    entries = records_for(repo)
    # longest first, ties by header -> deterministic identical assignment in
    # every task, and every shard's sum of L^2 (the trunk's cost unit) ~equal.
    entries.sort(key=lambda e: (-len(e[1]), e[0]))

    if args.dry_run:
        from models.opendde import trunk_chunk_size
        dry_run(entries, args, emb_cache, trunk_chunk_size())
        return

    mine = entries[args.shard::args.nshards]
    if args.limit is not None:
        mine = mine[:args.limit]
    log(f"shard {args.shard}/{args.nshards}: {len(mine)} of {len(entries)} records")

    import torch
    from models.opendde import build_model_opendde

    model = build_model_opendde(device="cuda", cycles=CYCLES, emb_cache=emb_cache)
    model.eval()
    t0, n_cached, n_fail = time.time(), 0, 0
    for i, (header, seq, _) in enumerate(mine):
        try:
            model._trunk_features_cached(seq)
            n_cached += 1
        except Exception as e:  # OOM, ... -- training skips these too
            n_fail += 1
            log(f"  [fail] {header.split()[0]}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(mine) - i - 1)
            log(f"shard {args.shard}: {i + 1}/{len(mine)}  {el / 60:.0f}m elapsed, "
                f"~{eta / 60:.0f}m left  ({n_cached} cached, {n_fail} failed)")
    log(f"shard {args.shard} DONE: {n_cached} cached, "
        f"{n_fail} failed in {(time.time() - t0) / 60:.0f}m -> {emb_cache}")


if __name__ == "__main__":
    main()
