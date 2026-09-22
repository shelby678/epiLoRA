"""Shard the OpenDDE trunk-feature precompute across Slurm array tasks.

    python precompute_shards.py --shard $SLURM_ARRAY_TASK_ID --nshards 32

Each task takes a disjoint slice of the records every training/eval run
needs (the champion training FASTA + the two shared-benchmark eval FASTAs +
the final held-out eval set), runs the frozen sequence-only trunk once per
record, and writes the per-residue features to the shared embedding cache.
The cache (atomic writes) makes shards safe to run concurrently and the whole
thing resumable: re-running a shard skips every entry already on disk, so a
preempted (--requeue) or resubmitted array loses at most the one in-flight
record.

Load balancing: records are sorted by sequence length (descending) and
dealt round-robin, so every shard gets an ~equal sum of L^2 -- the trunk's
cost unit -- instead of whatever lengths happened to cluster in file order.

Must run under the opendde env (see models/opendde/ and the hpc/ README);
set OPENDDE_ROOT_DIR (checkpoint/ + common/) unless machine_config.yaml's
opendde_root already resolves on this machine.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EPILORA = Path(__file__).resolve().parents[3]  # epilora/, for flat imports
if str(EPILORA) not in sys.path:
    sys.path.insert(0, str(EPILORA))

from data import default_cache_dir, load_samples, parse_fasta  # noqa: E402

# train.py's default training set + shared benchmark, plus eval_final.py's
# held-out set -- everything any fold or the final eval will forward through.
FASTAS = [
    "data/train_test_eval/allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta",
    "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta",
    "data/train_test_eval/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
    "data/train_test_eval/eval/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta",
]


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=EPILORA.parent,
                   help="the epiLoRA checkout (its data/train_test_eval/ is read)")
    p.add_argument("--structures", type=Path, default=None,
                   help="structures root (default: <repo>/data/raw/all-structures-extracted)")
    p.add_argument("--emb-cache", type=Path, default=None,
                   help="trunk-feature cache dir (default: the coords cache next "
                        "to --structures, matching train.py)")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N records of this shard (testing)")
    args = p.parse_args()

    repo = args.repo
    structures = args.structures or (repo / "data/raw/all-structures-extracted")
    emb_cache = args.emb_cache or default_cache_dir(structures)

    entries = records_for(repo)
    # longest first, ties by header -> deterministic identical assignment in
    # every task, and every shard's sum of L^2 (the trunk's cost unit) ~equal.
    entries.sort(key=lambda e: (-len(e[1]), e[0]))
    mine = entries[args.shard::args.nshards]
    if args.limit is not None:
        mine = mine[:args.limit]
    log(f"shard {args.shard}/{args.nshards}: {len(mine)} of {len(entries)} records")

    samples = load_samples(mine, structures)
    log(f"shard {args.shard}: {len(samples)} with structure "
        f"({sum(1 for s in samples if s[3] is None)} without)")

    import torch
    from models.opendde import build_model_opendde

    model = build_model_opendde(device="cuda", emb_cache=emb_cache)
    model.eval()
    t0, n_cached, n_skip, n_fail = time.time(), 0, 0, 0
    for i, (header, seq, _, coords, _) in enumerate(samples):
        if coords is None:
            n_skip += 1  # unusable in training either way (usable() skips them)
            continue
        try:
            model._trunk_features_cached(seq)
            n_cached += 1
        except Exception as e:  # bad CA, OOM, ... -- training skips these too
            n_fail += 1
            log(f"  [fail] {header.split()[0]}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(samples) - i - 1)
            log(f"shard {args.shard}: {i + 1}/{len(samples)}  {el / 60:.0f}m elapsed, "
                f"~{eta / 60:.0f}m left  ({n_cached} cached, {n_fail} failed)")
    log(f"shard {args.shard} DONE: {n_cached} cached, {n_skip} no-structure, "
        f"{n_fail} failed in {(time.time() - t0) / 60:.0f}m -> {emb_cache}")


if __name__ == "__main__":
    main()
