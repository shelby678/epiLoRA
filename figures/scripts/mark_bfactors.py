"""Write two mmCIF structures per Ebola antigen instance: one with the
B-factor column set from the real (ground-truth) antibody-contact epitope
call, one from epiLoRA's predicted per-residue epitope probability.

Ground truth uses the exact same 4A heavy-atom contact rule as
data/scripts/get_epitopes.py (Hchain+Lchain heavy atoms vs. each antigen
residue's heavy atoms), reusing data/scripts/structures.py's helpers.

Predictions reuse epilora/predict.py's load_model/predict (5-checkpoint
"champion" ensemble, allowed_species_homo_sapiens folds 1-5 -- the
README-documented best config) and epilora/data.py's load_backbone_coords,
which builds (seq_len, 3, 3) N/CA/C coords directly from the mmCIF via plain
Bio.PDB and already supports concatenated multi-chain antigens -- unlike
predict.py's own --pdb/--chain path (single chain, via biotite), which can't
express a multi-chain antigen or a >1-character chain id (both occur in this
dataset). Output is mmCIF (not .pdb, unlike epilora/color_by_prediction.py)
for the same reason: some antigen_chain ids here are 2 characters (e.g.
"AA"), which fixed-column PDB can't represent.

Must run in the fair-esm environment (/root/epilora_envs/env/bin/python3).

    python mark_bfactors.py ebola_summary.tsv structures_dir out_dir log_path [fold]

By default the prediction B-factor is the mean over the 5-checkpoint "champion"
ensemble (allowed_species_homo_sapiens folds 1-5). Pass ``fold`` (e.g. ``5``) to
use a single fold's checkpoint instead -- necessary when the antigen of
interest is itself (or is homologous to) a record that was in that fold's
training set, so the ensemble would be partially predicting on training data.
Pick the fold whose holdout set (clusters labelled ``<fold>.0``/``<fold>.1`` in
the ablation FASTA) contains the antigen's homologs; that fold's model never saw
them during training.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

FIGURES_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = FIGURES_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "data/scripts"))
sys.path.insert(0, str(REPO_ROOT / "epilora"))

from structures import (  # noqa: E402
    chain_residues, chain_sequence, heavy_atoms, cif_path,
    load_model as load_structure_model,
)
from data import load_backbone_coords  # noqa: E402
from predict import load_model as load_checkpoint, predict as predict_probs  # noqa: E402

from Bio.PDB import MMCIFIO, Select  # noqa: E402

CONTACT_DIST = 4.0
CHAMPION_WEIGHTS = sorted((REPO_ROOT / "weights/ablation").glob("allowed_species_homo_sapiens_fold?.pt"))

in_tsv, structures_dir, out_dir, log_path = sys.argv[1:5]
fold = sys.argv[5] if len(sys.argv) > 5 else None
if fold is not None:
    CHAMPION_WEIGHTS = [w for w in CHAMPION_WEIGHTS if w.name == f"allowed_species_homo_sapiens_fold{fold}.pt"]
    if not CHAMPION_WEIGHTS:
        sys.exit(f"no champion weight matched fold {fold} "
                 f"(looked for allowed_species_homo_sapiens_fold{fold}.pt)")
out_dir = Path(out_dir)
gt_dir = out_dir / "structures_groundtruth"
pred_dir = out_dir / "structures_epilora"
csv_dir = out_dir / "epilora_predictions"
for d in (gt_dir, pred_dir, csv_dir):
    d.mkdir(parents=True, exist_ok=True)
cache_dir = out_dir / "coords_cache"
cache_dir.mkdir(parents=True, exist_ok=True)


class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids


def write_cif(model, antigen_chains, bfactors_by_residue, out_path):
    """bfactors_by_residue: dict keyed by id(residue) -> float; residues not
    present get 0.0 (unmatched, same convention as color_by_prediction.py)."""
    for chain_id in antigen_chains:
        if chain_id not in model:
            continue
        for res in model[chain_id]:
            b = bfactors_by_residue.get(id(res), 0.0)
            for atom in res.get_atoms():
                atom.set_bfactor(b)
    io = MMCIFIO()
    io.set_structure(model.get_parent())  # Structure, so MMCIFIO can find the model id
    io.save(str(out_path), select=ChainSelect(antigen_chains))


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[mark_bfactors] device={device}  n_models={len(CHAMPION_WEIGHTS)}", file=sys.stderr)
models = [load_checkpoint(w, device) for w in CHAMPION_WEIGHTS]

with open(in_tsv, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

n_ok = n_skipped = 0
for row in rows:
    instance = row["INSTANCE"]
    antigen_chains = row["antigen_chain"].split("|")
    try:
        cp = cif_path(structures_dir, row["PDB"])
        model = load_structure_model(structures_dir, row["PDB"])

        ab_residues = chain_residues(model, row["Hchain"]) + chain_residues(model, row["Lchain"])
        ab_coords = np.array([a.coord for res in ab_residues for a in heavy_atoms(res)])
        tree = cKDTree(ab_coords)

        antigen_residues = []
        for chain_id in antigen_chains:
            antigen_residues.extend(chain_residues(model, chain_id))

        gt_bfactors = {}
        for res in antigen_residues:
            atoms = heavy_atoms(res)
            coords = np.array([a.coord for a in atoms])
            dists, _ = tree.query(coords)
            gt_bfactors[id(res)] = 100.0 if dists.min() <= CONTACT_DIST else 0.0

        seq = "".join(chain_sequence(model, c) for c in antigen_chains)
        if len(seq) != len(antigen_residues):
            raise ValueError(f"seq length {len(seq)} != residue count {len(antigen_residues)}")

        coords = load_backbone_coords(cp, antigen_chains, len(seq), cache_dir)
        if coords is None:
            raise ValueError("load_backbone_coords failed (parse error / chain missing / length mismatch)")

        probs_stack = np.stack([predict_probs(m, coords, seq) for m in models])
        mean_probs = probs_stack.mean(axis=0)
        if len(mean_probs) != len(antigen_residues):
            raise ValueError(f"prob length {len(mean_probs)} != residue count {len(antigen_residues)}")

        pred_bfactors = {id(res): 100.0 * float(p) for res, p in zip(antigen_residues, mean_probs)}

        write_cif(model, antigen_chains, gt_bfactors, gt_dir / f"{instance}.cif")
        write_cif(model, antigen_chains, pred_bfactors, pred_dir / f"{instance}.cif")

        with open(csv_dir / f"{instance}.csv", "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["pos", "aa", "prob", "epitope", "ground_truth_epitope"])
            for i, (aa, p, res) in enumerate(zip(seq, mean_probs, antigen_residues), start=1):
                w.writerow([i, aa, f"{p:.4f}", int(p >= 0.5), int(gt_bfactors[id(res)] == 100.0)])

        n_ok += 1
        print(f"[mark_bfactors] {instance}: {len(seq)} residues", file=sys.stderr)
    except Exception as e:
        n_skipped += 1
        print(f"[mark_bfactors] SKIP {instance}: {e}", file=sys.stderr)

with open(log_path, "w") as log:
    log.write(f"input rows: {len(rows)}\n")
    log.write(f"records written: {n_ok}\n")
    log.write(f"records skipped: {n_skipped}\n")
    log.write(f"mode: {'single fold ' + fold if fold is not None else '5-fold ensemble'}\n")
    log.write(f"champion weights used: {[str(w) for w in CHAMPION_WEIGHTS]}\n")
