"""Cluster antigen seqs with mmseqs2 (95% identity by default) to define each epitope's
backbone, then assign fold labels using a SEPARATE, looser 40%-identity connected-component
grouping over the cluster representatives -- so two 95%-clusters that are themselves distinct
epitope entries, but are near-duplicates of each other (e.g. many independently-deposited
structures of the same antigen that didn't quite hit 95% identity pairwise), still always end
up in the same CV fold group and can never be split across train/eval. This is deliberately two
tiers: 95% identity decides what one epitope IS (kept strict, so epitope calls aren't diluted
across distinct antigens); 40% identity only decides which of the 5 CV groups a cluster's fold
label draws its `i` from (see FOLD_LABELS below and the module-level grouping code).

Output format (fasta-like, one block per cluster)::

    >>CLUSTER {rep_instance} {fold_label}
    >{instance} {date} {resolution} {antigen_chains} {heavy_species} {light_species}
    {seq aligned to the cluster's shared frame, '-' where this member has no residue at that column}
    >{instance2} ...
    {seq2...}
    >>CLUSTER ...

Each member carries its own antigen_chains (rather than only the cluster header
carrying the representative's) so that combine_epitopes.py can elect a different
member as backbone without stranding it with the wrong PDB entry's chain IDs.
"""
import argparse
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("in_fasta")
p.add_argument("out_fasta")
p.add_argument("log_path")
p.add_argument("--min_seq_id", type=float, default=0.95,
                help="mmseqs2 identity threshold for the clusters that define each epitope's backbone "
                     "(range 0.0-1.0)")
p.add_argument("--cluster_mode", type=int, default=0,
                help="mmseqs2 --cluster-mode for the epitope-defining clustering above: 0 Set-Cover "
                     "(greedy, default), 1 Connected component. Set-Cover is not guaranteed to merge "
                     "every pairwise-similar sequence into one cluster (it can split a mutually-similar "
                     "group depending on greedy peel order); Connected component guarantees transitive "
                     "closure, at the cost of chaining unrelated sequences together through intermediate "
                     "homologs -- more of a risk the lower --min_seq_id is set.")
p.add_argument("--fold_group_seq_id", type=float, default=0.40,
                help="separate, looser mmseqs2 identity threshold used ONLY to decide which of the 5 CV "
                     "groups a cluster's fold label draws its `i` from -- distinct clusters that are "
                     "near-duplicates at this threshold always share an `i`, so they can never be split "
                     "across train/eval, without lowering the identity threshold that defines an epitope.")
p.add_argument("--fold_group_cluster_mode", type=int, default=1,
                help="mmseqs2 --cluster-mode for the fold-grouping step above (default: 1, connected "
                     "component -- this grouping exists specifically to guarantee transitive closure, "
                     "so Set-Cover's not-guaranteed merging would defeat the point).")
args = p.parse_args()
in_fasta, out_fasta, log_path = args.in_fasta, args.out_fasta, args.log_path
MIN_SEQ_ID = args.min_seq_id  # mmseqs2 clustering identity threshold
# Default mmseqs2 coverage (--cov-mode 0) requires BOTH sequences in a pair to have >=80% of
# their length covered by the alignment. That fails for a short fragment of a much longer
# sequence (its own coverage is ~100%, but it covers only a small fraction of the long one),
# so fragments and their full-length parents were landing in separate clusters -- and since
# fold labels are assigned per cluster, near-duplicate fragment/full-length pairs could end up
# in different folds. --cov-mode 5 instead requires only the SHORTER sequence of the pair to
# meet MIN_COV, which is mmseqs2's documented setting for clustering fragments into full-length
# representatives. Used for both the epitope-defining clustering and the fold-grouping step.
MIN_COV = 0.3
COV_MODE = 5
CLUSTER_MODE = args.cluster_mode
FOLD_GROUP_SEQ_ID = args.fold_group_seq_id
FOLD_GROUP_CLUSTER_MODE = args.fold_group_cluster_mode
MMSEQS = "mmseqs"  # assumed on PATH in the active env
N_CV_GROUPS = 5

random.seed(0)


def parse_fasta(path):
    """Yield (header, sequence) pairs from a fasta file, header with the leading '>' stripped."""
    header, seq = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq)
            header, seq = line[1:], []
        else:
            seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def parse_member_row(aligned_row, member_seq):
    """Split an mmseqs A3M row into per-rep-column chars (None = deletion) and insertion
    runs (case-preserved), keyed by the rep column each insertion precedes (rep_len itself
    keys a trailing insertion after the last column). mmseqs2's alignment is local, so it can
    stop short of member_seq's own end; whatever it doesn't cover is kept as part of that
    trailing insertion instead of silently dropped."""
    chars_by_col = []
    insertions = {}
    rep_col = mem_pos = 0
    run_start = None  # mem_pos where the current insertion run started, if any
    for ch in aligned_row:
        if ch.islower():
            # insertion: consumes a member residue but no rep column
            if run_start is None:
                run_start = mem_pos
            mem_pos += 1
            continue
        if run_start is not None:
            # run just ended -- file it under the rep column it precedes
            insertions[rep_col] = member_seq[run_start:mem_pos]
            run_start = None
        if ch == "-":
            # deletion: rep has this column, member doesn't
            chars_by_col.append(None)
            rep_col += 1
        else:
            # match/mismatch: both have this column
            chars_by_col.append(member_seq[mem_pos])
            rep_col += 1
            mem_pos += 1
    tail_start = run_start if run_start is not None else mem_pos
    if tail_start < len(member_seq):
        insertions[rep_col] = member_seq[tail_start:]
    return chars_by_col, insertions


def render_row(rep_len, chars_by_col, insertions, max_len):
    """Lay out one member across the cluster's shared frame: `rep_len` rep-derived columns
    plus `max_len[point]` extra columns wherever any cluster member has an insertion at
    that point. Members without a residue in a given slot get '-' there."""
    parts = []
    for point in range(rep_len + 1):  # rep_len+1 insertion points: before/between/after columns
        width = max_len.get(point, 0)
        if width:
            # pad this member's run (possibly empty) out to the cluster-wide width for this point
            ins = insertions.get(point, "")
            parts.append(ins + "-" * (width - len(ins)))
        if point < rep_len:
            char = chars_by_col[point]
            parts.append(char if char is not None else "-")
    return "".join(parts)


records = list(parse_fasta(in_fasta))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    query_fasta = tmp / "query.fasta"
    with open(query_fasta, "w") as f:
        # mmseqs needs uppercase sequences; headers are just the record's index into `records`
        for i, (_, seq) in enumerate(records):
            f.write(f">seq{i}\n{seq.upper()}\n")

    query_db = tmp / "queryDB"
    clu_db = tmp / "queryDB_clu"
    msa_db = tmp / "queryDB_clu_msa"
    unpacked = tmp / "unpacked"
    unpacked.mkdir()

    # build an mmseqs sequence database from the input fasta
    subprocess.run([MMSEQS, "createdb", str(query_fasta), str(query_db)], check=True, capture_output=True, text=True)
    # cluster sequences at MIN_SEQ_ID identity
    subprocess.run(
        [MMSEQS, "cluster", str(query_db), str(clu_db), str(tmp / "work"),
         "--min-seq-id", str(MIN_SEQ_ID), "-c", str(MIN_COV), "--cov-mode", str(COV_MODE),
         "--cluster-mode", str(CLUSTER_MODE)],
        check=True, capture_output=True, text=True,
    )
    # align every cluster member to its representative, producing one A3M MSA per cluster
    subprocess.run(
        [MMSEQS, "result2msa", str(query_db), str(query_db), str(clu_db), str(msa_db), "--msa-format-mode", "5"],
        check=True, capture_output=True, text=True,
    )
    # split the packed MSA database into individual per-cluster A3M files
    subprocess.run(
        [MMSEQS, "unpackdb", str(msa_db), str(unpacked), "--unpack-name-mode", "0"],
        check=True, capture_output=True, text=True,
    )

    # sort by rep_index so fold-label assignment (below) is reproducible across
    # runs/machines given the same random.seed, instead of depending on directory order.
    parsed_clusters = []
    for cluster_file in unpacked.iterdir():
        entries = list(parse_fasta(cluster_file))
        rep_id, _ = entries[0]
        rep_index = int(rep_id[3:])  # strip the "seq" prefix added above
        parsed_clusters.append((rep_index, entries))
    parsed_clusters.sort(key=lambda x: x[0])

    # --- fold grouping: cluster the (95%-cluster) representatives again at the much looser
    # FOLD_GROUP_SEQ_ID, purely to decide which CV group (`i`) each cluster's fold label gets.
    # This never touches which sequences got merged into which epitope above -- only which of
    # the 5 CV groups a whole cluster is confined to.
    fold_rep_fasta = tmp / "fold_reps.fasta"
    with open(fold_rep_fasta, "w") as f:
        for k, (rep_index, _) in enumerate(parsed_clusters):
            f.write(f">c{k}\n{records[rep_index][1].upper()}\n")
    fold_rep_db = tmp / "fold_repDB"
    fold_clu_db = tmp / "fold_repDB_clu"
    subprocess.run([MMSEQS, "createdb", str(fold_rep_fasta), str(fold_rep_db)],
                    check=True, capture_output=True, text=True)
    subprocess.run(
        [MMSEQS, "cluster", str(fold_rep_db), str(fold_clu_db), str(tmp / "fold_work"),
         "--min-seq-id", str(FOLD_GROUP_SEQ_ID), "-c", str(MIN_COV), "--cov-mode", str(COV_MODE),
         "--cluster-mode", str(FOLD_GROUP_CLUSTER_MODE)],
        check=True, capture_output=True, text=True,
    )
    fold_clu_tsv = tmp / "fold_clu.tsv"
    subprocess.run(
        [MMSEQS, "createtsv", str(fold_rep_db), str(fold_rep_db), str(fold_clu_db), str(fold_clu_tsv)],
        check=True, capture_output=True, text=True,
    )
    fold_group_label = {}  # k -> the sub-clustering's group label (its own representative's id)
    for line in open(fold_clu_tsv):
        group_rep, member = line.strip().split("\t")
        fold_group_label[int(member[1:])] = group_rep  # strip the "c" prefix added above

    members_by_group = defaultdict(list)
    for k in range(len(parsed_clusters)):
        members_by_group[fold_group_label[k]].append(k)

    # Load-balance the 5 CV groups by total member count (not cluster count) so a fold-group
    # that swallowed many small clusters -- or one big one -- doesn't skew CV-run dataset sizes.
    # Largest-group-first (LPT heuristic) into whichever CV group has the smallest running total;
    # both the grouping and this assignment are fully determined by sizes/labels, not randomized,
    # since the point is guaranteed balance rather than arbitrary balance.
    group_sizes = sorted(
        ((sum(len(parsed_clusters[k][1]) for k in ks), group_rep, ks)
         for group_rep, ks in members_by_group.items()),
        key=lambda x: (-x[0], x[1]),
    )
    cv_group_totals = [0] * N_CV_GROUPS
    cv_group_of_cluster = [None] * len(parsed_clusters)
    for size, group_rep, ks in group_sizes:
        cv_group = min(range(N_CV_GROUPS), key=lambda g: (cv_group_totals[g], g))
        cv_group_totals[cv_group] += size
        for k in ks:
            cv_group_of_cluster[k] = cv_group

    with open(out_fasta, "w") as out:
        n_clusters = 0
        for k, (rep_index, entries) in enumerate(parsed_clusters):
            # the rep's original header carries the metadata written on the >>CLUSTER line
            rep_header, rep_seq = records[rep_index]
            rep_instance = rep_header.split()[0]
            i = cv_group_of_cluster[k] + 1
            j = random.choice((0, 1))  # role within the CV group: 0 eval, 1 test
            fold_label = f"{i}.{j}"
            rep_len = len(rep_seq)

            # align each member against the rep and track the widest insertion at each point,
            # so every member can later be padded out to one shared cluster-wide frame
            parsed_members = []
            max_len = {}
            for member_id, aligned_row in entries:
                member_index = int(member_id[3:])
                member_header, member_seq = records[member_index]
                chars_by_col, insertions = parse_member_row(aligned_row, member_seq)
                parsed_members.append((member_header, chars_by_col, insertions))
                for point, ins in insertions.items():
                    if len(ins) > max_len.get(point, 0):
                        max_len[point] = len(ins)

            # write the cluster header followed by each member's header + aligned sequence
            out.write(f">>CLUSTER {rep_instance} {fold_label}\n")
            for member_header, chars_by_col, insertions in parsed_members:
                m_instance, m_date, m_resolution, m_chains, m_heavy, m_light = member_header.split()[:6]
                aligned_seq = render_row(rep_len, chars_by_col, insertions, max_len)
                out.write(f">{m_instance} {m_date} {m_resolution} {m_chains} {m_heavy} {m_light}\n")
                out.write(aligned_seq + "\n")
            n_clusters += 1

with open(log_path, "w") as log:
    # basic run stats for sanity-checking the pipeline
    log.write(f"min_seq_id: {MIN_SEQ_ID}\n")
    log.write(f"cluster_mode: {CLUSTER_MODE}\n")
    log.write(f"input records: {len(records)}\n")
    log.write(f"clusters: {n_clusters}\n")
    log.write(f"fold_group_seq_id: {FOLD_GROUP_SEQ_ID}\n")
    log.write(f"fold_group_cluster_mode: {FOLD_GROUP_CLUSTER_MODE}\n")
    log.write(f"fold groups (looser sub-clusters used only for CV group assignment): {len(members_by_group)}\n")
    log.write(f"members per CV group (i=1..{N_CV_GROUPS}): {cv_group_totals}\n")
