#!/opt/conda/bin/python
"""
Prepare antibody and antigen PDBs for HADDOCK docking.
  antibody: merge all chains → chain B, sequential renumber, protein ATOM only
  antigen:  merge all chains → chain A, protein ATOM only
"""
import argparse, os

AMINO_ACIDS = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}


def merge_to_chain(src, dst, target_chain):
    """Merge all chains of src into target_chain with sequential renumbering.
    Keeps only ATOM records with standard amino acid residue names.
    Returns count of removed lines."""
    offset, last_resseq, prev_chain = 0, 0, None
    lines_out = []
    n_removed = 0
    with open(src) as f:
        for line in f:
            if line.startswith("ATOM"):
                resname = line[17:20].strip()
                if resname not in AMINO_ACIDS:
                    n_removed += 1
                    continue
            elif line.startswith("HETATM"):
                n_removed += 1
                continue
            else:
                continue
            ch = line[21]
            resseq = int(line[22:26])
            if ch != prev_chain:
                if prev_chain is not None:
                    lines_out.append("TER\n")
                    offset = last_resseq + 10
                prev_chain = ch
            new_resseq = resseq + offset
            lines_out.append(line[:21] + target_chain + f"{new_resseq:4d}" + line[26:])
            last_resseq = new_resseq
    lines_out.append("TER\nEND\n")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        f.writelines(lines_out)
    return n_removed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--antibody", required=True)
    p.add_argument("--antigen",  required=True)
    p.add_argument("--out-ab",   required=True)
    p.add_argument("--out-ag",   required=True)
    args = p.parse_args()

    n_ab = merge_to_chain(args.antibody, args.out_ab, "B")
    n_ag = merge_to_chain(args.antigen,  args.out_ag, "A")

    if n_ab:
        print(f"antibody: removed {n_ab} non-protein records")
    if n_ag:
        print(f"antigen:  removed {n_ag} non-protein records")


if __name__ == "__main__":
    main()
