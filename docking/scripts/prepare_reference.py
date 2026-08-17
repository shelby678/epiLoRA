#!/opt/conda/bin/python
"""
Build a caprieval-ready reference PDB from the crystal-structure complex.

Uses fv_ab to identify which chains are antibody vs antigen (since reference
PDBs use inconsistent chain naming: H/L, A/B, etc.).

Output matches the docked-model convention:
  antibody chains → chain B (trimmed to Fv residues, sequential renumber)
  antigen  chains → chain A
"""
import argparse, os

AMINO_ACIDS = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}


def read_fv_residues(fv_ab_path):
    """Return {chain_id: set(resseqs)} for all ATOM chains in the Fv antibody."""
    fv_res = {}
    with open(fv_ab_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch = line[21]
            resname = line[17:20].strip()
            if resname not in AMINO_ACIDS:
                continue
            resseq = int(line[22:26])
            fv_res.setdefault(ch, set()).add(resseq)
    return fv_res


def ref_chain_order(ref_path):
    """Return ordered unique chain IDs from ATOM records in ref."""
    seen, chains = set(), []
    with open(ref_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                ch = line[21]
                if ch not in seen:
                    seen.add(ch)
                    chains.append(ch)
    return chains


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fv-ab", required=True)
    p.add_argument("--ref",   required=True)
    p.add_argument("--out",   required=True)
    args = p.parse_args()

    fv_res   = read_fv_residues(args.fv_ab)
    ab_chains = set(fv_res.keys())
    all_ref_chains = ref_chain_order(args.ref)
    ag_chains = [c for c in all_ref_chains if c not in ab_chains]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    offset, last_resseq, prev_chain = 0, 0, None
    n_ab_stripped = 0

    with open(args.out, "w") as fout:
        # Antibody: trim to Fv residues, merge into chain B
        with open(args.ref) as fin:
            for line in fin:
                if not line.startswith("ATOM"):
                    continue
                ch = line[21]
                if ch not in ab_chains:
                    continue
                resname = line[17:20].strip()
                if resname not in AMINO_ACIDS:
                    n_ab_stripped += 1
                    continue
                resseq = int(line[22:26])
                if resseq not in fv_res[ch]:
                    n_ab_stripped += 1
                    continue
                if ch != prev_chain:
                    if prev_chain is not None:
                        fout.write("TER\n")
                        offset = last_resseq + 10
                    prev_chain = ch
                new_resseq = resseq + offset
                fout.write(line[:21] + "B" + f"{new_resseq:4d}" + line[26:])
                last_resseq = new_resseq
        fout.write("TER\n")

        # Antigen: merge into chain A
        ag_offset, ag_last, ag_prev = 0, 0, None
        with open(args.ref) as fin:
            for line in fin:
                if not line.startswith("ATOM"):
                    continue
                ch = line[21]
                if ch not in ag_chains:
                    continue
                resname = line[17:20].strip()
                if resname not in AMINO_ACIDS:
                    continue
                resseq = int(line[22:26])
                if ch != ag_prev:
                    if ag_prev is not None:
                        fout.write("TER\n")
                        ag_offset = ag_last + 10
                    ag_prev = ch
                new_resseq = resseq + ag_offset
                fout.write(line[:21] + "A" + f"{new_resseq:4d}" + line[26:])
                ag_last = new_resseq
        fout.write("TER\nEND\n")

    if n_ab_stripped:
        print(f"reference: stripped {n_ab_stripped} ab atoms not in Fv")


if __name__ == "__main__":
    main()
