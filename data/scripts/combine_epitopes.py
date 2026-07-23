"""Combine per-cluster member epitope calls onto a representative sequence.

Reads the per-cluster block format written by cluster_fasta.py. A member contributes its
epitope (lowercase) calls to the combined sequence -- and counts toward --min_num_clusters --
only if it passes --allowed_species and --min_resolution. If the cluster's original
representative passes, it's used as the coordinate backbone as before. If it doesn't, a new
representative is elected from the qualifying members: whichever has the fewest MSA gaps
('-') in its column-aligned sequence (i.e. the most complete coverage), ties broken by
first appearance. The cluster keeps its original fold-bin label regardless of which member
ends up as backbone.

Output filename is derived automatically from whichever ablation args were set (non-default):
{out_dir}/{ablation_name}_{ablation_param}[_..._..]_epitopes.fasta, or {out_dir}/all_epitopes.fasta
if every arg is left at its default.
"""
import argparse
from pathlib import Path


def parse_clusters(path):
    cluster_header, members = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">>CLUSTER"):
            if cluster_header is not None:
                yield cluster_header, members
            cluster_header = line[len(">>CLUSTER "):]
            members = []
        elif line.startswith(">"):
            members.append([line[1:], ""])
        else:
            members[-1][1] += line
    if cluster_header is not None:
        yield cluster_header, members


def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_fasta")
    p.add_argument("out_dir")
    p.add_argument("log_path")
    p.add_argument("--allowed_species", default=None,
                    help='pipe-separated list, e.g. "homo sapiens|mus musculus" (default: all species allowed)')
    p.add_argument("--min_resolution", type=float, default=5.0,
                    help="keep only members with resolution <= this value (Angstrom)")
    p.add_argument("--min_num_clusters", type=int, default=1,
                    help="minimum number of qualifying members required to keep a cluster")
    args = p.parse_args()

    ablation_parts = []
    allowed_species = None
    if args.allowed_species is not None:
        allowed_species = {s.strip().replace(" ", "_") for s in args.allowed_species.split("|")}
        ablation_parts.append("allowed_species_" + "_".join(sorted(allowed_species)))
    if args.min_resolution != 5.0:
        val = args.min_resolution
        param = str(int(val)) if val == int(val) else str(val)
        ablation_parts.append(f"min_resolution_{param}")
    if args.min_num_clusters != 1:
        ablation_parts.append(f"min_num_clusters_{args.min_num_clusters}")

    name = "_".join(ablation_parts) if ablation_parts else "all"
    out_path = Path(args.out_dir) / f"{name}_epitopes.fasta"

    n_clusters_in = n_clusters_out = 0
    with open(out_path, "w") as out:
        for cluster_header, members in parse_clusters(args.in_fasta):
            n_clusters_in += 1
            rep_instance, antigen_chains, fold_label = cluster_header.split()

            parsed = []
            rep = None
            for header, seq in members:
                instance, date, resolution, heavy_species, light_species = header.split()
                species_ok = allowed_species is None or (
                    heavy_species in allowed_species and light_species in allowed_species
                )
                resolution_ok = resolution not in ("", "NA") and float(resolution) <= args.min_resolution
                m = dict(instance=instance, date=date, resolution=resolution, heavy_species=heavy_species,
                         light_species=light_species, seq=seq, qualifies=species_ok and resolution_ok)
                parsed.append(m)
                if instance == rep_instance:
                    rep = m

            qualifying = [m for m in parsed if m["qualifies"]]
            if len(qualifying) < args.min_num_clusters:
                continue

            # Prefer the original representative as backbone; if it doesn't pass the filters,
            # elect the qualifying member with the fewest MSA gaps (most complete coverage).
            backbone = rep if rep is not None and rep["qualifies"] else min(qualifying, key=lambda m: m["seq"].count("-"))

            backbone_cols = [i for i, c in enumerate(backbone["seq"]) if c != "-"]
            col_to_pos = {col: pos for pos, col in enumerate(backbone_cols)}
            backbone_seq = "".join(backbone["seq"][c] for c in backbone_cols)

            epitope = [False] * len(backbone_seq)
            for m in qualifying:
                seq = m["seq"]
                for col, pos in col_to_pos.items():
                    if seq[col].islower():
                        epitope[pos] = True

            combined_seq = "".join(c.lower() if ep else c.upper() for c, ep in zip(backbone_seq, epitope))
            oldest_date = min(m["date"] for m in qualifying)
            out.write(
                f">{backbone['instance']} {oldest_date} {backbone['resolution']} {antigen_chains} "
                f"{backbone['heavy_species']} {backbone['light_species']} n={len(qualifying)} {fold_label}\n"
            )
            out.write(combined_seq + "\n")
            n_clusters_out += 1

    with open(args.log_path, "w") as log:
        log.write(f"output: {out_path}\n")
        log.write(f"clusters in: {n_clusters_in}\n")
        log.write(f"clusters kept: {n_clusters_out}\n")


if __name__ == "__main__":
    main()
