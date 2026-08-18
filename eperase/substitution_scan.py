"""Substitution scan on the eval-set epitopes.

For every epitope site in the held-out eval fasta, keep the backbone fixed
and swap the residue identity to each of the other 19 amino acids in turn,
then read the model's epitope probability at that position. The per-site
delta = sub_prob - orig_prob is pooled across the whole eval set into a 20x20
matrix indexed by (substitution, original) and rendered as a heatmap:

  - x-axis: original residue
  - y-axis: substitution
  - white = 0 (no change), red = increase, blue = decrease

Backbone stays fixed across substitutions -- ESM-IF1 is an inverse-folding
model, so this scans "if this backbone position held a different residue,
would the head still call it epitope?". The 5-fold ESM-IF1 ensemble
(``allowed_species_homo_sapiens_esmif1_fold[1-5].pt``) is averaged, the
same scheme eval_final.py / predict_ensemble.py use.

Outputs (in --out-dir, default this file's dir/results):
  - delta_matrix.csv     20x20 mean delta (rows=sub, cols=orig) + counts
  - delta_matrix.npy     same matrix as a float array (NaN on the diagonal)
  - delta_count.npy      integer counts per (sub, orig) cell
  - per_substitution.csv one row per (instance, pos, orig, sub) with raw probs
  - substitution_heatmap.png  the heatmap

Must run in the fair-esm environment: epilora/env/bin/python
"""
from __future__ import annotations

# fair-esm 2.0 predates biotite 1.0, which renamed filter_backbone ->
# filter_peptide_backbone. Alias it back before esm.inverse_folding is imported
# (importing predict -> model triggers it), mirroring predict_ensemble.py.
import biotite.structure as _bs

if not hasattr(_bs, "filter_backbone"):
    _bs.filter_backbone = _bs.filter_peptide_backbone

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "epilora"))

from data import parse_fasta, load_samples  # noqa: E402
from predict import load_model  # noqa: E402

# 20 standard amino acids, alphabetical -- this fixed order defines the
# matrix layout (rows = substitutions, cols = originals) and is used
# everywhere downstream (CSV header, heatmap ticks, .npy indices).
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}

EVAL_FASTA = REPO_ROOT / "data/train_test_eval/eval/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta"
STRUCTURES = REPO_ROOT / "data/raw/all-structures-extracted"
WEIGHTS = sorted(
    (REPO_ROOT / "weights/ablation").glob("allowed_species_homo_sapiens_fold?_esm2.pt")
)


@torch.no_grad()
def ensemble_probs_batched(models, coords, seq_batch, batch_size: int) -> np.ndarray:
    """Run each model on a batch of sequences that all share the same backbone
    coords (single-site substitutions of one parent), averaging sigmoid
    probabilities across the ensemble.

    Returns ``(len(seq_batch), L)`` -- per-residue epitope probability, where
    L is the parent sequence length.
    """
    n = len(seq_batch)
    out = np.zeros((n, len(seq_batch[0])), dtype=np.float32)
    for start in range(0, n, batch_size):
        chunk_seqs = list(seq_batch[start:start + batch_size])
        coords_batch = [coords] * len(chunk_seqs)
        per_model = []
        for m in models:
            logits_list = m(coords_batch, chunk_seqs)
            probs = np.stack(
                [1.0 / (1.0 + np.exp(-lg.cpu().numpy())) for lg in logits_list]
            )
            per_model.append(probs)
        out[start:start + len(chunk_seqs)] = np.mean(per_model, axis=0)
    return out


def checkpoint(delta_sum: np.ndarray, delta_count: np.ndarray,
               per_sub_rows: list, n_done: int, n_total: int, out_dir: Path) -> float:
    """Write delta_matrix.{npy,csv}, per_substitution.csv, and the heatmap PNG
    from the current (partial) accumulator state. Called after every sequence
    so a usable plot/canvas exists on disk at all times -- the user can watch
    the heatmap update as the scan progresses, and a crash mid-run still
    leaves the latest state on disk.

    Returns the current max |Δ| (for the final summary line).
    """
    delta_mean = np.full((20, 20), np.nan, dtype=np.float64)
    mask = delta_count > 0
    delta_mean[mask] = delta_sum[mask] / delta_count[mask]

    np.save(out_dir / "delta_matrix.npy", delta_mean)
    np.save(out_dir / "delta_count.npy", delta_count)

    with open(out_dir / "delta_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["substitution"] + list(AA) + ["n_substitutions"])
        for i, sub in enumerate(AA):
            row = [sub]
            for j in range(20):
                if i == j:
                    row.append("")
                else:
                    row.append(f"{delta_mean[i, j]:.6f}"
                               if not np.isnan(delta_mean[i, j]) else "")
            row.append(int(delta_count[i].sum()))
            w.writerow(row)

    with open(out_dir / "per_substitution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "position", "orig_aa", "sub_aa",
                    "orig_prob", "sub_prob", "delta"])
        w.writerows(per_sub_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_max = np.nanmax(np.abs(delta_mean))
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 1.0
    masked = np.ma.masked_invalid(delta_mean)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                   interpolation="nearest")
    ax.set_xticks(range(20)); ax.set_xticklabels(list(AA))
    ax.set_yticks(range(20)); ax.set_yticklabels(list(AA))
    ax.set_xlabel("Original residue")
    ax.set_ylabel("Substitution")
    status = "FINAL" if n_done == n_total else f"PARTIAL {n_done}/{n_total}"
    ax.set_title(
        "Mean Δ epitope probability on eval-set epitope sites\n"
        f"n = {int(delta_count.sum())} substitutions across {n_done} antigens "
        f"({status}, 5-fold ESM-IF1 ensemble)"
    )
    for i in range(20):
        for j in range(20):
            v = delta_mean[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(v) < abs_max * 0.5 else "white")
    for i in range(20):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="white", lw=1))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Δ epitope probability (sub − orig)")
    fig.tight_layout()
    fig.savefig(out_dir / "substitution_heatmap.png", dpi=200)
    plt.close(fig)
    return abs_max


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--eval-fasta", type=Path, default=EVAL_FASTA)
    p.add_argument("--structures", type=Path, default=STRUCTURES)
    p.add_argument("--weights", type=Path, nargs="+", default=WEIGHTS,
                   help="checkpoints to average (default: 5 ESM-IF1 folds)")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results")
    p.add_argument("--batch-size", type=int, default=16,
                   help="mutations per forward pass (all share the parent coords)")
    p.add_argument("--plot-every", type=int, default=1,
                   help="checkpoint (save matrix + redraw heatmap) every N sequences")
    args = p.parse_args()

    if not args.weights:
        p.error("no checkpoints given and no ESM-IF1 folds found under weights/ablation/")
    missing = [w for w in args.weights if not w.exists()]
    if missing:
        p.error("weights not found: " + ", ".join(str(m) for m in missing))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[scan] device={device}  n_models={len(args.weights)}  "
          f"batch_size={args.batch_size}", file=sys.stderr)
    for w in args.weights:
        print(f"[scan]   {w.name}", file=sys.stderr)

    print("[scan] loading models...", file=sys.stderr)
    models = [load_model(w, device) for w in args.weights]

    print(f"[scan] parsing {args.eval_fasta.name}...", file=sys.stderr)
    by_part = parse_fasta(args.eval_fasta)
    entries = [e for v in by_part.values() for e in v]
    samples = load_samples(entries, args.structures)
    n_struct = sum(1 for *_, c in samples if c is not None)
    print(f"[scan] {len(samples)} eval sequences, {n_struct} with usable structure",
          file=sys.stderr)

    # 20x20 accumulators (rows=sub, cols=orig); diagonal stays empty since
    # X->X is not a substitution.
    delta_sum = np.zeros((20, 20), dtype=np.float64)
    delta_count = np.zeros((20, 20), dtype=np.int64)

    per_sub_rows = []  # for per_substitution.csv
    total_subs = 0

    for si, (header, seq, labels, coords) in enumerate(samples, start=1):
        instance = header.split()[0]
        if coords is None:
            raise ValueError(
                f"no usable backbone coordinates for {header!r} -- every eval-set "
                f"antigen must be scorable for a fair comparison"
            )
        seq = seq.upper()
        L = len(seq)

        # Per-residue epitope probability on the parent (un-mutated) sequence,
        # averaged across the ensemble. Used as the baseline for every delta.
        base_probs = ensemble_probs_batched(models, coords, [seq], args.batch_size)[0]

        epi_sites = [i for i, lab in enumerate(labels) if lab == 1]
        # Skip sites whose original residue is non-standard (X, B, Z, U, O, *) --
        # they have no row/column in the 20x20 matrix.
        epi_sites = [i for i in epi_sites if seq[i] in AA_TO_IDX]
        if not epi_sites:
            print(f"[scan] {si}/{len(samples)} {instance}: no scorable epitope sites",
                  file=sys.stderr)
            continue

        # Build the flat mutation list: (pos, sub_aa) for every epitope site x
        # every non-identity amino acid. Then run them all through the ensemble
        # in chunks of batch_size, sharing the parent coords.
        mutations = [(pos, sub_aa)
                     for pos in epi_sites
                     for sub_aa in AA
                     if sub_aa != seq[pos]]

        sub_seqs = []
        for pos, sub_aa in mutations:
            s = list(seq)
            s[pos] = sub_aa
            sub_seqs.append("".join(s))

        sub_probs = ensemble_probs_batched(models, coords, sub_seqs, args.batch_size)

        for k, (pos, sub_aa) in enumerate(mutations):
            orig_aa = seq[pos]
            delta = float(sub_probs[k, pos] - base_probs[pos])
            i_sub = AA_TO_IDX[sub_aa]
            j_orig = AA_TO_IDX[orig_aa]
            delta_sum[i_sub, j_orig] += delta
            delta_count[i_sub, j_orig] += 1
            per_sub_rows.append((instance, pos, orig_aa, sub_aa,
                                 float(base_probs[pos]),
                                 float(sub_probs[k, pos]),
                                 delta))

        total_subs += len(mutations)
        print(f"[scan] {si}/{len(samples)} {instance}: {L} residues, "
              f"{len(epi_sites)} epi sites, {len(mutations)} substitutions "
              f"(running total {total_subs})", file=sys.stderr)

        # Checkpoint after each sequence (or every --plot-every) so a usable
        # heatmap + matrix are always on disk -- survives crashes and lets
        # the user watch the plot update as the scan progresses.
        if si % args.plot_every == 0 or si == len(samples):
            checkpoint(delta_sum, delta_count, per_sub_rows, si, len(samples), out_dir)
            print(f"[scan] checkpoint at {si}/{len(samples)} "
                  f"({int(delta_count.sum())} subs)", file=sys.stderr)

    # ---- final save + heatmap (in case len(samples) isn't a multiple of plot_every) --
    abs_max = checkpoint(delta_sum, delta_count, per_sub_rows,
                         len(samples), len(samples), out_dir)
    print(f"[scan] wrote delta_matrix.{{npy,csv}}, per_substitution.csv, "
          f"substitution_heatmap.png into {out_dir}", file=sys.stderr)
    print(f"[scan] done. {int(delta_count.sum())} substitutions scored. "
          f"max |Δ| = {abs_max:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
