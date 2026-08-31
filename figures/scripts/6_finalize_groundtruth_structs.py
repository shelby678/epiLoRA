"""Build the conjoined Ebola ground-truth figure: one reference GP trimer
plus every surviving human antibody's Fv superposed onto it, with the union
of all antibody contacts marked in the antigen's B-factor column.

Filters, in order: (1) fully human antibody; (2) placeable on the 7kfe GP
trimer reference without the antibody clashing through it (rejects cryptic
epitopes and non-spike antigens like sGP/NP); (3) non-redundant (same SAbDab
antibody id or near-identical binding pose); (4) antigen sequence >= ~97%
identical to the reference antigen.

Usage:
    python 6_finalize_groundtruth_structs.py ebola_summary.tsv \\
        ../data/raw/all-structures-extracted results/conjoined_groundtruth log \\
        [--ref_pdb pdb_00007kfe] [--ref_chains A,B,C,D,E,F] \\
        [--ref_antigen_pdb pdb_00006qd8]
"""
from __future__ import annotations

import argparse
import csv
import functools
import itertools
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from Bio import Align
from Bio.PDB import (MMCIFParser, Superimposer, MMCIFIO, Select,
                     Structure, Chain, Atom, Residue)
from scipy.spatial import cKDTree

FIGURES_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = FIGURES_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "data/scripts"))

from structures import chain_residues, chain_sequence  # noqa: E402

CONTACT_DIST = 4.0  # heavy-atom contact rule, same as data/scripts/get_epitopes.py

# The variable region (Fv) ends right after the J segment, which begins with
# a conserved motif -- a robust V/C boundary without Kabat numbering.
_JH_MOTIF = "WGQG"
_JL_MOTIFS = ["FGQGTK", "FGSGTK", "FGGGTK", "FGQGTL", "FGSGTL", "FGGGTL",
              "FGQGTI", "FGSGTI", "FGGGTI", "FGQGTM", "FGSGTM", "FGGGTM"]


def fv_length(seq, is_heavy):
    """Residues in a chain's variable region (Fv); falls back to 130/115 if
    the J motif isn't found."""
    if is_heavy:
        idx = seq.find(_JH_MOTIF)
        if idx >= 0:
            return min(idx + 12, len(seq))
        return min(130, len(seq))
    for motif in _JL_MOTIFS:
        idx = seq.find(motif)
        if idx >= 0:
            return min(idx + 11, len(seq))
    return min(115, len(seq))

_parser = MMCIFParser(QUIET=True)
_cache: dict[str, object] = {}
REF_PDB = "pdb_00007kfe"


def load_model(structures_dir: str, pdb_id: str):
    if pdb_id not in _cache:
        path = Path(structures_dir) / pdb_id / f"{pdb_id}_sabdab.cif"
        _cache[pdb_id] = next(iter(_parser.get_structure(pdb_id, str(path))))
        if len(_cache) > 40:  # bounded; never evict the reference trimer
            for k in list(_cache):
                if k != REF_PDB:
                    del _cache[k]
                    break
    return _cache[pdb_id]


_aligner = Align.PairwiseAligner()
_aligner.mode = "global"
_aligner.open_gap_score = -10
_aligner.extend_gap_score = -0.5
_aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")


@functools.lru_cache(maxsize=None)
def matched_pairs(seq1: str, seq2: str):
    """(0-indexed i, j) aligned non-gap position pairs, and the alignment
    score. Cached: the same GP1/GP2 sequences align thousands of times."""
    aln = next(_aligner.align(seq1, seq2))
    pairs = []
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        for k in range(e1 - s1):
            pairs.append((s1 + k, s2 + k))
    return pairs, aln.score


def heavy_atoms(res):
    return [a for a in res.get_atoms() if a.element != "H"]


def chain_com(model, chain_id):
    res = chain_residues(model, chain_id)
    if not res:
        return None
    coords = np.array([a.coord for r in res for a in heavy_atoms(r)], dtype=float)
    return coords.mean(axis=0)


def classify_gp(model, chain_id, gp1_seq, gp2_seq, min_matched=20, min_ident=0.4):
    """'gp1' / 'gp2' if the chain matches that reference domain, else None.
    For single-chain-per-protomer antigens (e.g. SARS-CoV-2 spike), gp2_seq
    is empty and every matching chain is 'gp1'."""
    seq = chain_sequence(model, chain_id)
    if not seq or len(seq) < len(gp1_seq) * 0.4:  # antibodies etc., skip
        return None
    p1, _ = matched_pairs(seq, gp1_seq)
    id1 = (sum(1 for i, j in p1 if seq[i] == gp1_seq[j]) / len(p1)) if p1 else 0.0
    if gp2_seq:
        p2, _ = matched_pairs(seq, gp2_seq)
        id2 = (sum(1 for i, j in p2 if seq[i] == gp2_seq[j]) / len(p2)) if p2 else 0.0
        if id1 >= min_ident and len(p1) >= min_matched and id1 >= id2:
            return "gp1"
        if id2 >= min_ident and len(p2) >= min_matched and id2 >= id1:
            return "gp2"
        return None
    return "gp1" if (id1 >= min_ident and len(p1) >= min_matched) else None


def form_protomers(model, chain_ids, gp1_seq, gp2_seq):
    """Group antigen chains into GP1+GP2 protomers, pairing each GP1 chain
    with its center-of-mass-nearest GP2 chain. Returns a list of
    {'gp1': chain_id_or_None, 'gp2': chain_id_or_None}."""
    gp1s, gp2s = [], []
    for c in chain_ids:
        t = classify_gp(model, c, gp1_seq, gp2_seq)
        if t == "gp1":
            gp1s.append(c)
        elif t == "gp2":
            gp2s.append(c)
    protomers, used = [], set()
    for g1 in gp1s:
        c1 = chain_com(model, g1)
        best_g2, best_d = None, 1e9
        for g2 in gp2s:
            if g2 in used:
                continue
            c2 = chain_com(model, g2)
            d = float(np.linalg.norm(c1 - c2))
            if d < best_d:
                best_d, best_g2 = d, g2
        if best_g2 is not None:
            used.add(best_g2)
        protomers.append({"gp1": g1, "gp2": best_g2})
    for g2 in gp2s:
        if g2 not in used:
            protomers.append({"gp1": None, "gp2": g2})
    return protomers


def all_gp_protomers(model, gp1_seq, gp2_seq):
    """Protomers from EVERY GP chain in the structure (not just the ones
    SAbDab lists in antigen_chain, which are only the contacted ones)."""
    chains = [c.id for c in model if chain_residues(model, c.id)]
    return form_protomers(model, chains, gp1_seq, gp2_seq)


def contacted_protomers(model, contact_chains, gp1_seq, gp2_seq):
    """The protomer(s) the antibody actually binds (GP1 AND its GP2 partner,
    even when antigen_chain lists only the GP1 -- aligning the full protomer
    constrains the fit better than the contacted chains alone, and ignores
    the other protomers' conformationally-variable GP2 stalks). Falls back
    to all protomers if no contact chain matches."""
    all_prots = all_gp_protomers(model, gp1_seq, gp2_seq)
    cset = set(contact_chains)
    sel = [p for p in all_prots
           if (p["gp1"] is not None and p["gp1"] in cset)
           or (p["gp2"] is not None and p["gp2"] in cset)]
    return sel or all_prots


def _protomer_assignments(k_moving, k_fixed):
    """Every way to assign moving protomers to distinct fixed protomers (or
    vice versa if more moving than fixed). At most 6 permutations."""
    if k_moving == 0 or k_fixed == 0:
        return
    if k_moving <= k_fixed:
        for perm in itertools.permutations(range(k_fixed), k_moving):
            yield [(mi, perm[mi]) for mi in range(k_moving)]
    else:
        for perm in itertools.permutations(range(k_moving), k_fixed):
            yield [(perm[fi], fi) for fi in range(k_fixed)]


def superpose_protomers(moving_model, moving_protomers, fixed_model, fixed_protomers,
                        min_matched=50):
    """Try every protomer assignment, collect GP1->GP1 and GP2->GP2 CA pairs,
    and return (rot, tran, n_matched, rmsd) for the lowest-RMSD fit, or None
    if no assignment reaches min_matched CA atoms."""
    best = None
    for pairs_idx in _protomer_assignments(len(moving_protomers), len(fixed_protomers)):
        moving_atoms, fixed_atoms = [], []
        for mi, fi in pairs_idx:
            mp, fp = moving_protomers[mi], fixed_protomers[fi]
            for key in ("gp1", "gp2"):
                mc, fc = mp[key], fp[key]
                if mc is None or fc is None:
                    continue
                mseq = chain_sequence(moving_model, mc)
                rseq = chain_sequence(fixed_model, fc)
                pairs, _ = matched_pairs(mseq, rseq)
                mres = chain_residues(moving_model, mc)
                rres = chain_residues(fixed_model, fc)
                for mi2, fi2 in pairs:
                    if (mi2 < len(mres) and fi2 < len(rres)
                            and mres[mi2].has_id("CA") and rres[fi2].has_id("CA")):
                        moving_atoms.append(mres[mi2]["CA"])
                        fixed_atoms.append(rres[fi2]["CA"])
        if len(moving_atoms) < min_matched:
            continue
        sup = Superimposer()
        sup.set_atoms(fixed_atoms, moving_atoms)
        if best is None or sup.rms < best[3]:
            best = (sup.rotran[0], sup.rotran[1], len(moving_atoms), float(sup.rms))
    return best


def species_filter(row) -> str:
    """'' if the antibody is fully human, else a short reason string.
    SAbDab records the expression host in *_species but the variable-region
    origin in the subclass tag, so e.g. 'IGHV8 (Musmus)' on a homo-sapiens
    row is a mouse-variable / human-Fc chimera."""
    if row["heavy_species"] != "homo sapiens":
        return f"heavy_species={row['heavy_species']}"
    if row["light_species"] != "homo sapiens":
        return f"light_species={row['light_species']}"
    for col in ("heavy_subclass", "light_subclass"):
        val = row[col] or ""
        m = re.search(r"\((\w+)\)", val)
        if m and m.group(1) != "Homsap":
            return f"{col}={val}"
    return ""


def transform_coords(coords, rot, tran):
    return np.asarray(coords, dtype=float) @ rot + tran


def best_chain_identity(model, chain_ids, ref_seq):
    """Best sequence identity of any chain in chain_ids vs ref_seq, over the
    aligned region only (gaps don't count against identity)."""
    best = 0.0
    for ch in chain_ids:
        seq = chain_sequence(model, ch)
        if not seq:
            continue
        pairs, _ = matched_pairs(seq, ref_seq)
        if not pairs:
            continue
        ident = sum(1 for i, j in pairs if seq[i] == ref_seq[j]) / len(pairs)
        best = max(best, ident)
    return best


def find_ab_copies(model, hchain, lchain, gp_chain_set, min_ident=0.9):
    """All copies of the annotated antibody in the structure (a trimeric GP
    complex usually holds 3, one per protomer; SAbDab annotates only one).
    Matches heavy-like/light-like chains by sequence identity, pairs each
    heavy with its COM-nearest light. Returns (H, L) chain-id tuples, the
    annotated pair first."""
    h_seq = chain_sequence(model, hchain) if hchain in model else ""
    l_seq = chain_sequence(model, lchain) if lchain in model else ""

    heavy_chains, light_chains = [], []
    for c in model:
        if c.id in gp_chain_set:
            continue
        seq = chain_sequence(model, c.id)
        if not seq or not chain_residues(model, c.id):
            continue
        if h_seq:
            pairs, _ = matched_pairs(seq, h_seq)
            if pairs:
                ident = sum(1 for i, j in pairs if seq[i] == h_seq[j]) / len(pairs)
                if ident >= min_ident:
                    heavy_chains.append(c.id)
                    continue
        if l_seq:
            pairs, _ = matched_pairs(seq, l_seq)
            if pairs:
                ident = sum(1 for i, j in pairs if seq[i] == l_seq[j]) / len(pairs)
                if ident >= min_ident:
                    light_chains.append(c.id)

    copies = []
    used_l = set()
    if hchain in heavy_chains:  # annotated pair first
        heavy_chains.remove(hchain)
        heavy_chains.insert(0, hchain)
    for h in heavy_chains:
        hc = chain_com(model, h)
        if hc is None:
            continue
        best_l, best_d = None, 1e9
        for l in light_chains:
            if l in used_l:
                continue
            lc = chain_com(model, l)
            if lc is None:
                continue
            d = float(np.linalg.norm(hc - lc))
            if d < best_d:
                best_d, best_l = d, l
        if best_l is not None:
            used_l.add(best_l)
        copies.append((h, best_l))
    return copies


class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids


def write_combined_cif(ref_model, ref_antigen_chains, survivors,
                       gp1_seq, gp2_seq, ref_protomers, out_path, log):
    """Write one CIF with the reference antigen plus every surviving
    antibody's Fv (VH+VL only) superposed onto it: each survivor's full
    trimer is superposed onto the reference so each antibody copy lands on a
    different protomer; copies whose Fv doesn't contact the reference
    antigen within 4 A are skipped. Antigen B-factors mark the UNION of all
    antibody contacts. Returns (ab_mapping, n_epitope)."""
    combined = deepcopy(ref_model)
    combined.parent = None
    for ch in list(combined):
        if ch.id not in ref_antigen_chains:
            combined.detach_child(ch.id)
    combined_struct = Structure.Structure("combined")
    combined_struct.add(combined)

    ref_ag_coords = np.array([
        a.coord for ch_id in ref_antigen_chains if ch_id in combined
        for res in chain_residues(combined, ch_id) for a in heavy_atoms(res)
    ], dtype=float)
    ref_ag_tree = cKDTree(ref_ag_coords) if len(ref_ag_coords) else None

    all_ab_coords = []
    ab_mapping = []

    for i, surv in enumerate(survivors):
        inst = surv["row"]["INSTANCE"]
        sabdab_id = surv["row"]["SABDAB_ID"]
        pdb_id = surv["row"]["PDB"]
        model = surv["model"]

        all_prots = all_gp_protomers(model, gp1_seq, gp2_seq)
        res = superpose_protomers(model, all_prots, ref_model, ref_protomers,
                                  min_matched=50)
        if res is None:
            log(f"  SKIP {inst}: cannot superpose onto reference antigen")
            continue
        rot, tran = res[0], res[1]

        gp_chain_set = set()
        for p in all_prots:
            for key in ("gp1", "gp2"):
                if p[key] is not None:
                    gp_chain_set.add(p[key])
        copies = find_ab_copies(model, surv["ab_chains"][0],
                                surv["ab_chains"][1], gp_chain_set)

        n_added = 0
        for copy_idx, (h_ch, l_ch) in enumerate(copies):
            suffix = f"c{copy_idx}" if copy_idx > 0 else ""
            copy_fv_coords = []
            for j, ab_ch in enumerate([h_ch, l_ch]):
                if ab_ch is None or ab_ch not in model:
                    continue
                seq = chain_sequence(model, ab_ch)
                n_fv = fv_length(seq, is_heavy=(j == 0))
                new_id = f"Ab{i:02d}{'H' if j == 0 else 'L'}{suffix}"
                new_chain = Chain.Chain(new_id)
                for res_obj in chain_residues(model, ab_ch)[:n_fv]:
                    new_res = Residue.Residue(res_obj.id, res_obj.resname,
                                             res_obj.segid)
                    for atom in res_obj.get_atoms():
                        coord = np.asarray(atom.coord, dtype=float) @ rot + tran
                        new_atom = Atom.Atom(
                            atom.get_name(), coord, 0.0,
                            atom.get_occupancy(), atom.get_altloc(),
                            atom.get_fullname(), atom.get_serial_number(),
                            element=atom.element)
                        new_res.add(new_atom)
                        if atom.element != "H":
                            copy_fv_coords.append(coord)
                    new_chain.add(new_res)
                combined.add(new_chain)
                ab_mapping.append((new_id, inst, sabdab_id, pdb_id, copy_idx))

            if ref_ag_tree is not None and copy_fv_coords:
                d, _ = ref_ag_tree.query(np.array(copy_fv_coords, dtype=float))
                if d.min() > CONTACT_DIST:
                    for j in range(2):
                        suffix_ch = f"Ab{i:02d}{'H' if j == 0 else 'L'}{suffix}"
                        if suffix_ch in combined:
                            combined.detach_child(suffix_ch)
                    ab_mapping = [m for m in ab_mapping if m[0] !=
                                  f"Ab{i:02d}H{suffix}" and m[0] !=
                                  f"Ab{i:02d}L{suffix}"]
                    log(f"    skip copy {copy_idx} (Fv doesn't contact antigen)")
                    continue
            all_ab_coords.extend(copy_fv_coords)
            n_added += 1

        if n_added:
            log(f"  ADD {inst} -> {n_added}/{len(copies)} Fv copies ({sabdab_id})")
        else:
            log(f"  SKIP {inst}: no Fv copy contacts the antigen")

    n_epitope = 0
    if all_ab_coords:
        ab_tree = cKDTree(np.array(all_ab_coords, dtype=float))
        for ch_id in ref_antigen_chains:
            if ch_id not in combined:
                continue
            for res in chain_residues(combined, ch_id):
                atoms = heavy_atoms(res)
                if not atoms:
                    continue
                coords = np.array([a.coord for a in atoms])
                dists, _ = ab_tree.query(coords)
                is_epi = dists.min() <= CONTACT_DIST
                for atom in res.get_atoms():
                    atom.set_bfactor(100.0 if is_epi else 0.0)
                if is_epi:
                    n_epitope += 1

    all_chain_ids = list(ref_antigen_chains) + [m[0] for m in ab_mapping]
    io = MMCIFIO()
    io.set_structure(combined_struct)
    io.save(str(out_path), select=ChainSelect(all_chain_ids))
    return ab_mapping, n_epitope


def main():
    global REF_PDB
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_tsv")
    p.add_argument("structures_dir", help="raw SAbDab structure dir (<pdb>/<pdb>_sabdab.cif)")
    p.add_argument("out_dir")
    p.add_argument("log_path")
    p.add_argument("--ref_pdb", default="pdb_00007kfe",
                   help="reference trimer to place antibodies against")
    p.add_argument("--ref_chains", default="A,B,C,D,E,F")
    p.add_argument("--max_clash_atoms", type=int, default=10)
    p.add_argument("--clash_radius", type=float, default=2.0)
    p.add_argument("--cover_radius", type=float, default=4.0)
    p.add_argument("--max_place_rmsd", type=float, default=4.0)
    p.add_argument("--min_place_matched", type=int, default=50)
    p.add_argument("--dup_radius", type=float, default=2.0)
    p.add_argument("--dup_frac", type=float, default=0.95,
                   help="antibody-CA overlap fraction (both directions) that, "
                        "OR an identical SAbDab antibody id, marks a pair as "
                        "duplicate. 0.95 = truly identical binding pose; lower "
                        "over-merges distinct antibodies that merely bind a "
                        "similar epitope")
    p.add_argument("--ref_antigen_pdb", default="pdb_00006qd8",
                   help="PDB whose GP trimer is the single antigen in the "
                        "combined output; all survivors' antigens are "
                        "superposed onto it (then discarded, leaving only "
                        "this antigen + all antibodies)")
    p.add_argument("--min_antigen_identity", type=float, default=0.97,
                   help="minimum GP1 AND GP2 sequence identity to the "
                        "reference antigen, over the aligned region only")
    p.add_argument("--single_chain_protomer", action="store_true",
                   help="single-chain-per-protomer antigen (e.g. SARS-CoV-2 "
                        "spike): no GP1/GP2 split, no GP2 requirement")
    args = p.parse_args()

    REF_PDB = args.ref_pdb
    ref_chain_ids = args.ref_chains.split(",")
    final_dir = Path(args.out_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    SINGLE_CHAIN = args.single_chain_protomer

    with open(args.in_tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"input rows: {len(rows)}")

    # ---- filter 1: human-only ----
    human_rows, dropped_species = [], []
    for row in rows:
        reason = species_filter(row)
        if reason:
            dropped_species.append((row["INSTANCE"], reason))
        else:
            human_rows.append(row)
    log(f"filter 1 (human-only): kept {len(human_rows)}, dropped {len(dropped_species)}")
    for inst, reason in dropped_species:
        log(f"  DROP species {inst}: {reason}")

    # ---- load reference trimer ----
    ref_model = load_model(args.structures_dir, REF_PDB)
    ref_heavy = [a for ch in ref_chain_ids for res in chain_residues(ref_model, ch)
                 for a in heavy_atoms(res)]
    ref_coords = np.array([a.coord for a in ref_heavy], dtype=float)
    if SINGLE_CHAIN:
        gp1_seq = chain_sequence(ref_model, ref_chain_ids[0])
        gp2_seq = ""
    else:
        gp1_seq = chain_sequence(ref_model, ref_chain_ids[0])  # A = GP1
        gp2_seq = chain_sequence(ref_model, ref_chain_ids[3])  # D = GP2
    ref_protomers = form_protomers(ref_model, ref_chain_ids, gp1_seq, gp2_seq)
    log(f"reference {REF_PDB} chains {ref_chain_ids}: {len(ref_heavy)} heavy atoms, "
        f"{len(ref_protomers)} protomers: {ref_protomers}")

    # ---- filter 2: place on reference, reject cryptic/clashing ----
    # Structures without a full trimer (3 protomers) are skipped up front --
    # they'd fail filter 4 anyway and the superposition is wasted on them.
    placed = []
    n_place_fail = n_no_trimer = 0
    for row in human_rows:
        inst = row["INSTANCE"]
        try:
            model = load_model(args.structures_dir, row["PDB"])
            all_prots = all_gp_protomers(model, gp1_seq, gp2_seq)
            if len(all_prots) < 3:
                n_no_trimer += 1
                raise ValueError(f"only {len(all_prots)} protomers (need 3 for full trimer)")
            contact_chains = [c for c in row["antigen_chain"].split("|")
                              if chain_residues(model, c)]
            ag_protomers = contacted_protomers(model, contact_chains, gp1_seq, gp2_seq)
            if not ag_protomers or not any(p["gp1"] or p["gp2"] for p in ag_protomers):
                raise ValueError("no antigen-like chains (not a spike/GP antigen)")
            # GP2 required to exclude sGP (shares GP1's N-terminus but folds
            # differently) and nucleoprotein.
            if not SINGLE_CHAIN:
                if not any(p["gp2"] for p in all_prots):
                    raise ValueError("no GP2 chain (not the virion spike: sGP / NP)")

            res = superpose_protomers(model, ag_protomers, ref_model, ref_protomers,
                                      min_matched=args.min_place_matched)
            if res is None:
                raise ValueError(f"fewer than {args.min_place_matched} matched CA atoms")
            rot, tran, n_matched, rmsd = res
            if rmsd > args.max_place_rmsd:
                raise ValueError(f"placement rmsd {rmsd:.2f} > {args.max_place_rmsd}")

            # "covered" = where the contacted antigen sits on the trimer (the
            # antibody is allowed there); clashes with the REST of the
            # reference are what make an epitope cryptic.
            ag_heavy = [a for ch in contact_chains for res in chain_residues(model, ch)
                        for a in heavy_atoms(res)]
            ag_coords_t = transform_coords([a.coord for a in ag_heavy], rot, tran)
            ab_chains = [c for c in (row["Hchain"], row["Lchain"]) if c]
            ab_heavy = [a for ch in ab_chains for res in chain_residues(model, ch)
                        for a in heavy_atoms(res)]
            ab_coords_t = transform_coords([a.coord for a in ab_heavy], rot, tran)

            ant_tree = cKDTree(ag_coords_t)
            d_cov_ref, _ = ant_tree.query(ref_coords)
            covered = d_cov_ref < args.cover_radius
            unc_coords = ref_coords[~covered]
            if len(unc_coords):
                unc_tree = cKDTree(unc_coords)
                d_clash, _ = unc_tree.query(ab_coords_t)
                n_clash = int((d_clash < args.clash_radius).sum())
            else:
                n_clash = 0

            reason = "cryptic (antibody clashes with trimer)" if n_clash > args.max_clash_atoms else ""
            placed.append({
                "row": row, "n_matched": n_matched, "rmsd": rmsd,
                "n_covered": int(covered.sum()), "n_clash": n_clash,
                "contact_chains": contact_chains, "ab_chains": ab_chains,
                "model": model, "rot": rot, "tran": tran,
                "drop_reason": reason,
            })
            log(f"  place {inst}: matched={n_matched} rmsd={rmsd:.2f} "
                f"covered={int(covered.sum())} clash_atoms={n_clash}"
                + (f" -> DROP {reason}" if reason else " -> OK"))
        except Exception as e:
            n_place_fail += 1
            log(f"  place {inst}: DROP cannot place on reference: {e}")

    n_cryptic = sum(1 for p in placed if p["drop_reason"])
    survivors2 = [p for p in placed if not p["drop_reason"]]
    log(f"filter 2 (placeable + no cryptic): "
        f"kept {len(survivors2)}, dropped {n_no_trimer} (no trimer) + "
        f"{n_place_fail - n_no_trimer} (can't place) + {n_cryptic} (cryptic)")

    # ---- filter 3: dedup by antibody overlap ----
    # Every survivor already has a rigid-body transform into the reference
    # frame, so transform each antibody's CAs there once and compare directly
    # instead of re-superposing every pair.
    kept: list[dict] = []
    kept_ab_ref: list[np.ndarray] = []  # parallel to kept: ab CAs in reference frame
    for new in survivors2:
        new_row = new["row"]
        new_model = new["model"]
        new_ab_ca = np.array([
            a.coord for ch in new["ab_chains"]
            for res in chain_residues(new_model, ch) if res.has_id("CA")
            for a in [res["CA"]]
        ], dtype=float)
        new_ab_ref = transform_coords(new_ab_ca, new["rot"], new["tran"])
        is_dup = False
        for k_idx, k in enumerate(kept):
            if new_row["SABDAB_ID"] == k["row"]["SABDAB_ID"]:
                is_dup = True
                log(f"  DEDUP {new_row['INSTANCE']} is duplicate of "
                    f"{k['row']['INSTANCE']} (same SAbDab antibody id "
                    f"{new_row['SABDAB_ID']})")
                break
            k_ab_ref = kept_ab_ref[k_idx]
            if len(k_ab_ref) == 0 or len(new_ab_ref) == 0:
                continue
            k_tree = cKDTree(k_ab_ref)
            d, _ = k_tree.query(new_ab_ref)
            frac_new = float((d < args.dup_radius).mean())
            d2, _ = cKDTree(new_ab_ref).query(k_ab_ref)
            frac_kept = float((d2 < args.dup_radius).mean())
            if frac_new >= args.dup_frac and frac_kept >= args.dup_frac:
                is_dup = True
                log(f"  DEDUP {new_row['INSTANCE']} is duplicate of "
                    f"{k['row']['INSTANCE']} (overlap frac new={frac_new:.2f} "
                    f"kept={frac_kept:.2f})")
                break
        if not is_dup:
            kept.append(new)
            kept_ab_ref.append(new_ab_ref)

    log(f"filter 3 (dedup <{args.dup_radius}A): kept {len(kept)}, "
        f"dropped {len(survivors2) - len(kept)}")

    # ---- filter 4: antigen sequence identity to the reference antigen ----
    ref_ag_model = load_model(args.structures_dir, args.ref_antigen_pdb)
    ref_ag_prots = all_gp_protomers(ref_ag_model, gp1_seq, gp2_seq)
    ref_ag_gp1_seq = chain_sequence(ref_ag_model, ref_ag_prots[0]["gp1"])
    ref_ag_gp2_seq = (chain_sequence(ref_ag_model, ref_ag_prots[0]["gp2"])
                      if ref_ag_prots[0]["gp2"] and not SINGLE_CHAIN else "")
    ref_antigen_chains = []
    for p in ref_ag_prots:
        for key in ("gp1", "gp2"):
            if p[key] is not None and p[key] not in ref_antigen_chains:
                ref_antigen_chains.append(p[key])
    log(f"filter 4: reference antigen {args.ref_antigen_pdb} chains {ref_antigen_chains}")

    survivors4 = []
    prot_cache: dict[str, list] = {}  # PDB -> prots (same PDB = same structure)
    ident_cache: dict[str, tuple[float, float]] = {}  # PDB -> (id1, id2)
    for k in kept:
        pdb_id = k["row"]["PDB"]
        if pdb_id not in prot_cache:
            model = k["model"]
            prots = all_gp_protomers(model, gp1_seq, gp2_seq)
            prot_cache[pdb_id] = prots
            gp1_chains = [p["gp1"] for p in prots if p["gp1"]]
            gp2_chains = [p["gp2"] for p in prots if p["gp2"]]
            id1 = best_chain_identity(model, gp1_chains, ref_ag_gp1_seq)
            id2 = (best_chain_identity(model, gp2_chains, ref_ag_gp2_seq)
                   if not SINGLE_CHAIN and ref_ag_gp2_seq else 1.0)
            ident_cache[pdb_id] = (id1, id2)
        prots = prot_cache[pdb_id]
        id1, id2 = ident_cache[pdb_id]
        gp1_chains = [p["gp1"] for p in prots if p["gp1"]]
        gp2_chains = [p["gp2"] for p in prots if p["gp2"]]
        n_needed_gp2 = 3 if not SINGLE_CHAIN else 0
        if len(prots) < 3 or len(gp1_chains) < 3 or len(gp2_chains) < n_needed_gp2:
            log(f"  DROP {k['row']['INSTANCE']}: only {len(prots)} protomers "
                f"({len(gp1_chains)} S/GP1 + {len(gp2_chains)} GP2), need full trimer")
            continue
        if SINGLE_CHAIN:
            ok = id1 >= args.min_antigen_identity
            log_detail = f"S {id1:.1%}"
        else:
            ok = id1 >= args.min_antigen_identity and id2 >= args.min_antigen_identity
            log_detail = f"GP1 {id1:.1%} GP2 {id2:.1%}"
        if ok:
            survivors4.append(k)
        else:
            log(f"  DROP {k['row']['INSTANCE']}: {log_detail} "
                f"< {args.min_antigen_identity:.0%}")
    log(f"filter 4 (antigen identity >= {args.min_antigen_identity:.0%}): "
        f"kept {len(survivors4)}, dropped {len(kept) - len(survivors4)}")

    # ---- write one combined CIF: reference antigen + all antibodies ----
    for old in list(final_dir.glob("*.cif")) + list(final_dir.glob("*.tsv")):
        old.unlink()
    out_cif = final_dir / f"combined_on_{args.ref_antigen_pdb}.cif"
    out_tsv = final_dir / f"combined_on_{args.ref_antigen_pdb}_mapping.tsv"

    ab_mapping, n_epitope = write_combined_cif(
        ref_ag_model, ref_antigen_chains, survivors4,
        gp1_seq, gp2_seq, ref_ag_prots, out_cif, log)

    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chain_id", "instance", "sabdab_id", "pdb", "copy"])
        for chain_id, inst, sabdab_id, pdb_id, copy_idx in ab_mapping:
            w.writerow([chain_id, inst, sabdab_id, pdb_id, copy_idx])

    log(f"wrote combined CIF: {out_cif.name} "
        f"({len(ab_mapping) // 2} antibody chains, {n_epitope} union epitope residues)")
    log(f"wrote mapping TSV: {out_tsv.name}")

    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
