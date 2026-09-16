"""Build the opendde epitope-dist ablation datasets from an opendde
``epitope_dist`` export.

    python3 scripts/make_opendde_dataset.py \
        --epitope-dist ../../opendde_coreweave/data/epitope_dist

Source data: 250 campaign antibodies docked (5 samples each) against 352
human-secreted-protein antigens (contact_dist A, see meta.json); per residue,
``epitope_prob`` is the fraction of antibodies whose docked structure contacts
it -- a CONTINUOUS label, unlike the binary SAbDab epitope calls every other
dataset under train_test_eval/ carries. Two flavours of it exist: ``prob_best``
(each antibody counted through its best-ranked sample) and ``prob_all`` (every
sample pooled).

Per flavour this writes:

    train_test_eval/opendde_<flavour>_epitopes.fasta
        one record per antigen, header in the combine_epitopes.py shape
        (instance/date/resolution/chains/species/n=/fold) but with an extra
        ``labels=soft`` field and an UPPERCASE sequence: the record's labels
        are not in the casing (there is no binary call to encode) but in the
        companion file below. ``chains`` is the antigen chain of the source
        CIF (the docked complex also carries the antibody chains, which the
        pipeline never reads).
    train_test_eval/opendde_<flavour>_epitopes_soft_labels.tsv
        ``<header>\\t<p,p,p,...>`` per record -- the exact probability strings
        from the source TSV. train.py substitutes these for the casing-derived
        labels, so the BCE loss simply trains each residue's probability
        toward its contact fraction (BCE accepts any target in [0, 1]).

plus the champion dataset with the opendde records appended:

    train_test_eval/allowed_species_homo_sapiens_min_resolution_10_plus_opendde_<flavour>_epitopes.fasta
        (and its own _soft_labels.tsv, opendde records only -- the SAbDab
        records keep their casing labels, so one training set can mix binary
        and soft targets)

Antigens >=40% identical (mmseqs2 -- the repo's standing leakage threshold,
see split_eval_clusters.py) to any antigen of the shared benchmark or the
temporal eval holdout are dropped, so no opendde training antigen can be a
near-duplicate of a held-out epitope. The surviving antigens are 40%-identity
fold-grouped (mmseqs2 connected components, cluster_fasta.py's fold-group
tier), load-balanced into the 5 CV groups, and each takes a fold label i.j
like every other dataset. There is no 95% epitope-defining tier here: every
opendde antigen is already a single record (one per UniProt), not a cluster of
SAbDab entries to merge.

Each source CIF (a full predicted antibody-antigen complex) is copied
verbatim to <structures-root>/pdb_<uniprot>/pdb_<uniprot>.cif -- the pipeline
reads only the chain(s) named in the FASTA header, and data.structure_path
falls back from the SAbDab <pdb_id>_sabdab.cif layout to this plain name.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # so defaults don't depend on cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))       # structures.py (sibling)
sys.path.insert(0, str(REPO_ROOT / "epilora"))                 # data.py (marker + companion naming)

from Bio.PDB import MMCIFParser  # noqa: E402  (after the sys.path setup above)
from structures import chain_sequence  # noqa: E402
from data import SOFT_LABEL_MARKER, soft_labels_path  # noqa: E402

MMSEQS = "mmseqs"  # assumed on PATH in the active env
LEAKAGE_SEQ_ID = 0.40  # split_eval_clusters.py's MIN_SIMILARITY
FOLD_GROUP_SEQ_ID = 0.40  # cluster_fasta.py's fold-group tier
# See cluster_fasta.py's MIN_COV/COV_MODE comment: require only the shorter
# sequence of a pair to be covered, so fragment/full-length near-duplicates are
# caught too.
MIN_COV = 0.3
COV_MODE = 5
N_CV_GROUPS = 5
FLAVOURS = ("best", "all")
CHAMPION_DATASET = "allowed_species_homo_sapiens_min_resolution_10_epitopes.fasta"


def parse_plain_fasta(path: Path) -> list[tuple[str, str]]:
    """(header, sequence) pairs, header with the leading '>' stripped."""
    out, header, seq = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(seq)))
            header, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if header is not None:
        out.append((header, "".join(seq)))
    return out


def read_epitope_dist(dist_dir: Path) -> dict[str, dict]:
    """{uniprot: {sequence, best: [str], all: [str]}} from the two flavour TSVs.

    Probabilities are kept as their exact source strings -- written through to
    the companion files verbatim, so no rounding decision is ever made here.
    """
    by_flavour = {}
    for flavour in FLAVOURS:
        tsv = dist_dir / f"epitope_{'best_sample' if flavour == 'best' else 'all_samples'}.tsv"
        rows = {}
        with open(tsv) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                uid = row["antigen_id"]
                if uid in rows:
                    raise ValueError(f"{tsv}: duplicate antigen {uid}")
                probs = row["epitope_prob"].split(",")
                seq = row["sequence"]
                if len(probs) != len(seq):
                    raise ValueError(f"{tsv} {uid}: {len(probs)} probabilities for a "
                                     f"{len(seq)}-residue sequence")
                vals = [float(p) for p in probs]
                if min(vals) < 0.0 or max(vals) > 1.0:
                    raise ValueError(f"{tsv} {uid}: probabilities outside [0, 1]")
                rows[uid] = {"sequence": seq, flavour: probs}
        by_flavour[flavour] = rows
        print(f"{tsv.name}: {len(rows)} antigens")

    if set(by_flavour["best"]) != set(by_flavour["all"]):
        raise ValueError("the two flavour TSVs cover different antigen sets")
    out = {}
    for uid in by_flavour["best"]:
        best, alls = by_flavour["best"][uid], by_flavour["all"][uid]
        if best["sequence"] != alls["sequence"]:
            raise ValueError(f"{uid}: sequences differ between the flavour TSVs")
        out[uid] = {"sequence": best["sequence"], "best": best["best"], "all": alls["all"]}
    return out


def antigen_chains(dist_dir: Path, antigens: dict[str, dict]) -> dict[str, str]:
    """The chain of each source CIF whose standard residues match the antigen
    sequence -- the chain the FASTA header names (the rest of the complex is
    the antibody, never read). Exactly one chain must match."""
    parser = MMCIFParser(QUIET=True)
    chains = {}
    for i, (uid, rec) in enumerate(sorted(antigens.items())):
        model = next(iter(parser.get_structure(uid, str(dist_dir / "structures" / f"{uid}.cif"))))
        matches = [c.id for c in model if chain_sequence(model, c.id) == rec["sequence"]]
        if len(matches) != 1:
            raise ValueError(f"{uid}: expected exactly one chain matching the antigen "
                             f"sequence, found {matches}")
        chains[uid] = matches[0]
        if (i + 1) % 100 == 0:
            print(f"  matched antigen chains: {i + 1}/{len(antigens)}")
    return chains


def mmseqs_easy_search_hits(query: Path, target: Path, tmp: Path,
                            min_seq_id: float) -> set[str]:
    """Query ids with at least one >=min_seq_id target hit (the same search
    split_eval_clusters.py uses to scrub eval leakage)."""
    result = tmp / "result.m8"
    subprocess.run(
        [MMSEQS, "easy-search", str(query), str(target), str(result), str(tmp / "search_work"),
         "--min-seq-id", str(min_seq_id), "-c", str(MIN_COV), "--cov-mode", str(COV_MODE)],
        check=True, capture_output=True, text=True,
    )
    return {line.split("\t", 1)[0] for line in result.read_text().splitlines() if line.strip()}


def mmseqs_fold_groups(fasta: Path, tmp: Path) -> dict[str, str]:
    """{sequence id -> fold-group representative id} via connected-component
    clustering at FOLD_GROUP_SEQ_ID (cluster_fasta.py's fold-group tier:
    near-duplicate antigens always share a CV group, so they can never be
    split across train/eval)."""
    db, clu, tsv = tmp / "db", tmp / "db_clu", tmp / "clu.tsv"
    subprocess.run([MMSEQS, "createdb", str(fasta), str(db)],
                   check=True, capture_output=True, text=True)
    subprocess.run(
        [MMSEQS, "cluster", str(db), str(clu), str(tmp / "cluster_work"),
         "--min-seq-id", str(FOLD_GROUP_SEQ_ID), "-c", str(MIN_COV),
         "--cov-mode", str(COV_MODE), "--cluster-mode", "1"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([MMSEQS, "createtsv", str(db), str(db), str(clu), str(tsv)],
                   check=True, capture_output=True, text=True)
    return {member: rep for rep, member in
            (line.strip().split("\t") for line in tsv.read_text().splitlines() if line.strip())}


def fold_labels(antigens: dict[str, dict], tmp: Path, seed: int) -> dict[str, str]:
    """{uniprot: i.j} -- fold groups load-balanced into the 5 CV groups
    (largest-group-first into the smallest running total, cluster_fasta.py's
    LPT heuristic), then a seeded i.j draw per antigen."""
    fasta = tmp / "antigens.fasta"
    uids = sorted(antigens)
    with open(fasta, "w") as f:
        for i, uid in enumerate(uids):
            f.write(f">o{i}\n{antigens[uid]['sequence']}\n")
    group_of = mmseqs_fold_groups(fasta, tmp)

    members_by_group = defaultdict(list)
    for i, uid in enumerate(uids):
        members_by_group[group_of[f"o{i}"]].append(uid)
    # largest group first, ties broken by representative id for determinism
    groups = sorted(members_by_group.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    totals = [0] * N_CV_GROUPS
    cv_group_of = {}
    for _, members in groups:
        g = min(range(N_CV_GROUPS), key=lambda g: (totals[g], g))
        totals[g] += len(members)
        for uid in members:
            cv_group_of[uid] = g

    random.seed(seed)
    return {uid: f"{cv_group_of[uid] + 1}.{random.choice((0, 1))}" for uid in uids}


def record_header(uid: str, chain: str, fold_label: str, date: str) -> str:
    """combine_epitopes.py's header shape (instance/date/resolution/chains/
    species/n=/fold) plus the labels=soft marker; sequence casing carries no
    label here, the companion file does."""
    return (f"pdb_{uid} {date} -1.0 {chain} opendde opendde n=1 "
            f"{SOFT_LABEL_MARKER} {fold_label}")


def write_dataset(fasta: Path, records: list[tuple[str, str, str | None]],
                  soft_probs: dict[str, list[str]] | None) -> None:
    """Write a labelled FASTA (and its soft-labels companion, for the opendde
    records) -- ``records`` are (header, sequence, soft-probs key|None), and
    the companion's rows are keyed by the full FASTA header (the same
    convention as the surface FASTAs, see data.load_surface_masks)."""
    with open(fasta, "w") as f:
        for header, seq, _ in records:
            f.write(f">{header}\n{seq}\n")
    if soft_probs is None:
        return
    with open(soft_labels_path(fasta), "w") as f:
        for header, _, key in records:
            if key is not None:
                f.write(f"{header}\t{','.join(soft_probs[key])}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epitope-dist", type=Path,
                   default=REPO_ROOT.parent / "opendde_coreweave/data/epitope_dist",
                   help="opendde epitope_dist export (epitope_*.tsv, meta.json, structures/)")
    p.add_argument("--tte-dir", type=Path, default=REPO_ROOT / "data/train_test_eval")
    p.add_argument("--structures-root", type=Path,
                   default=REPO_ROOT / "data/raw/all-structures-extracted",
                   help="where source CIFs are staged as pdb_<uniprot>/pdb_<uniprot>.cif")
    p.add_argument("--champion-fasta", type=Path,
                   default=REPO_ROOT / "data/train_test_eval" / CHAMPION_DATASET,
                   help=f"the dataset the plus-opendde variants append to "
                        f"(default: {CHAMPION_DATASET})")
    p.add_argument("--eval-fastas", type=Path, nargs="+", default=[
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_epitopes.fasta",
        REPO_ROOT / "data/train_test_eval/allowed_species_homo_sapiens_mus_musculus_epitopes.fasta",
        REPO_ROOT / "data/train_test_eval/eval/allowed_species_homo_sapiens_min_resolution_5_epitopes.fasta",
    ], help="antigens of these sets (the shared benchmark + temporal holdout) are "
            "what the leakage guard protects")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the i.j role draw (cluster_fasta.py's convention)")
    args = p.parse_args()

    meta = json.loads((args.epitope_dist / "meta.json").read_text())
    date = meta["generated"][:10].replace("-", "/")
    print(f"epitope_dist: {meta['n_antibodies']} antibodies, {meta['antigens_total']} "
          f"antigens, contact_dist {meta['contact_dist']}A, generated {meta['generated']}")

    antigens = read_epitope_dist(args.epitope_dist)
    chains = antigen_chains(args.epitope_dist, antigens)
    print(f"antigen chain matched in all {len(chains)} structures "
          f"({ {c: list(chains.values()).count(c) for c in set(chains.values())} })")

    # -- leakage guard: drop antigens near-duplicating a benchmark/eval antigen
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        query = tmp / "query.fasta"
        with open(query, "w") as f:
            for i, uid in enumerate(sorted(antigens)):
                f.write(f">o{i}\n{antigens[uid]['sequence']}\n")
        target = tmp / "target.fasta"
        with open(target, "w") as f:
            for ef in args.eval_fastas:
                for header, seq in parse_plain_fasta(ef):
                    f.write(f">{header.split()[0]}\n{seq.upper().replace('-', '')}\n")
        leaking = mmseqs_easy_search_hits(query, target, tmp, LEAKAGE_SEQ_ID)
        dropped = set()
        for i, uid in enumerate(sorted(antigens)):
            if f"o{i}" in leaking:
                dropped.add(uid)
        for uid in sorted(dropped):
            print(f"  dropped {uid}: >= {LEAKAGE_SEQ_ID:.0%} identical to a "
                  f"benchmark/eval antigen")
        kept = {uid: rec for uid, rec in antigens.items() if uid not in dropped}
        print(f"leakage guard: kept {len(kept)}/{len(antigens)} "
              f"(dropped {len(dropped)} at >={LEAKAGE_SEQ_ID:.0%} identity)")

        labels = fold_labels(kept, tmp, args.seed)
        by_fold = defaultdict(int)
        for label in labels.values():
            by_fold[label.split(".")[0]] += 1
        print(f"fold groups -> CV groups: {dict(sorted(by_fold.items()))}")

    # -- stage the source CIFs where the pipeline looks for them
    n_staged = 0
    for uid in sorted(antigens):
        src = args.epitope_dist / "structures" / f"{uid}.cif"
        dst_dir = args.structures_root / f"pdb_{uid}"
        dst = dst_dir / f"pdb_{uid}.cif"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_staged += 1
    print(f"structures: staged {n_staged} CIFs -> {args.structures_root}/pdb_<uniprot>/")

    # -- the two opendde-only datasets
    for flavour in FLAVOURS:
        records = [(record_header(uid, chains[uid], labels[uid], date),
                    antigens[uid]["sequence"], uid) for uid in sorted(kept)]
        fasta = args.tte_dir / f"opendde_{flavour}_epitopes.fasta"
        write_dataset(fasta, records, {uid: antigens[uid][flavour] for uid in kept})
        print(f"wrote {fasta} ({len(records)} records) + "
              f"{soft_labels_path(fasta).name}")

    # -- the champion-plus-opendde datasets: champion records verbatim (their
    #    casing labels and fold labels must not be touched) + opendde records
    champion = parse_plain_fasta(args.champion_fasta)
    instances = {h.split()[0] for h, _ in champion}
    for uid in kept:
        if f"pdb_{uid}" in instances:
            raise ValueError(f"instance pdb_{uid} collides with a champion record")
    for flavour in FLAVOURS:
        # champion records verbatim: their casing labels and fold labels are
        # exactly the champion dataset's, and must stay that way
        records = [(header, seq, None) for header, seq in champion]
        records += [(record_header(uid, chains[uid], labels[uid], date),
                     antigens[uid]["sequence"], uid) for uid in sorted(kept)]
        fasta = args.tte_dir / f"{args.champion_fasta.stem.removesuffix('_epitopes')}_plus_opendde_{flavour}_epitopes.fasta"
        write_dataset(fasta, records, {uid: antigens[uid][flavour] for uid in kept})
        print(f"wrote {fasta} ({len(champion)} champion + {len(kept)} opendde records) + "
              f"{soft_labels_path(fasta).name}")


if __name__ == "__main__":
    main()
