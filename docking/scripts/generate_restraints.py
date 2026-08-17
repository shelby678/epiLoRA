#!/opt/conda/bin/python
"""
Generate CDR-based ambiguous and VH-VL unambiguous restraints.

Expects prepared PDBs:
  ab: all chain B (VH then VL, separated by a renumbering gap)
  ag: all chain A

Outputs:
  ambig.tbl   - CDR residues (B) → antigen surface (A), distance 2.0 2.0 0.0
  unambig.tbl - two VH-VL CA-CA distance restraints to keep Fv rigid
"""
import argparse, os, subprocess, tempfile, warnings

HADDOCK3_RESTRAINTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env", "bin", "haddock3-restraints"
)

AMINO_ACIDS = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}
_3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def read_chain_residues(pdb_path, chain_id):
    """Return [(resseq, resname), ...] for chain, protein ATOM only, deduplicated."""
    residues, seen = [], set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            resname = line[17:20].strip()
            if resname not in AMINO_ACIDS:
                continue
            resseq = int(line[22:26])
            if resseq not in seen:
                seen.add(resseq)
                residues.append((resseq, resname))
    return residues


def split_vh_vl(residues):
    """Split by largest resseq gap. Returns (vh_residues, vl_residues)."""
    if len(residues) < 4:
        return residues, []
    max_gap, gap_idx = 0, len(residues)
    for i in range(1, len(residues)):
        gap = residues[i][0] - residues[i-1][0]
        if gap > max_gap:
            max_gap, gap_idx = gap, i
    return residues[:gap_idx], residues[gap_idx:]


def get_cdr_resseqs(residues):
    """Run abnumber on residues list, return CDR resseqs. Returns [] on failure."""
    from abnumber import Chain as AbChain
    seq = "".join(_3TO1.get(r, "X") for _, r in residues)
    if len(seq) < 50:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            chain = AbChain(seq, scheme="chothia", use_anarcii=True)
        except Exception:
            return []
    cdr_resseqs = []
    for i, (pos, _) in enumerate(chain):
        if "CDR" in str(pos.get_region()) and i < len(residues):
            cdr_resseqs.append(residues[i][0])
    return cdr_resseqs


def calc_surface_residues(pdb_path, chain_id):
    """Run haddock3-restraints calc_accessibility, return surface resseqs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_pdb = os.path.join(tmpdir, "chain.pdb")
        with open(chain_pdb, "w") as out, open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")) and line[21] == chain_id:
                    out.write(line)
            out.write("END\n")
        try:
            subprocess.run(
                [HADDOCK3_RESTRAINTS, "calc_accessibility",
                 "--export_to_actpass", chain_pdb],
                check=True, capture_output=True, cwd=tmpdir, timeout=120,
            )
            actpass = os.path.join(tmpdir, f"chain_passive_{chain_id}.actpass")
            if os.path.exists(actpass):
                with open(actpass) as f:
                    return [int(x) for x in f.read().split() if x.strip().isdigit()]
        except Exception as e:
            print(f"WARNING: calc_accessibility failed for chain {chain_id}: {e}")
    return []


def get_passive_from_active(pdb_path, chain_id, active_resseqs):
    """Run haddock3-restraints passive_from_active, return passive resseqs."""
    if not active_resseqs:
        return []
    active_str = ",".join(str(r) for r in active_resseqs)
    try:
        result = subprocess.run(
            [HADDOCK3_RESTRAINTS, "passive_from_active",
             "--chain-id", chain_id, pdb_path, active_str],
            capture_output=True, text=True, timeout=60,
        )
        passive = []
        for line in result.stdout.splitlines():
            for tok in line.split():
                try:
                    passive.append(int(tok))
                except ValueError:
                    pass
        return sorted(set(passive) - set(active_resseqs))
    except Exception as e:
        print(f"WARNING: passive_from_active failed: {e}")
        return []


def write_ambig_tbl(ab_active, ab_passive, ag_passive, out_path):
    """Write AIR TBL. CDR residues in chain B restrained to antigen surface in chain A."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for rn in ab_active:
            f.write(f"assign (resi {rn} and segid B)\n(\n")
            for j, ag_rn in enumerate(ag_passive):
                sep = "       or" if j < len(ag_passive) - 1 else ""
                f.write(f"       (resi {ag_rn} and segid A){sep}\n")
            f.write(") 2.0 2.0 0.0\n\n")


def write_unambig_tbl(ab_pdb, out_path):
    """Write two VH-VL CA-CA distance restraints from merged chain B."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("ab", ab_pdb)[0]

    residues = read_chain_residues(ab_pdb, "B")
    vh_res, vl_res = split_vh_vl(residues)

    if not vh_res or not vl_res:
        open(out_path, "w").close()
        print("WARNING: could not split VH/VL for unambig restraints")
        return

    ca_coords = {}
    for ch in struct.get_chains():
        for res in ch.get_residues():
            if res.get_id()[0] == " " and "CA" in res:
                ca_coords[res.get_id()[1]] = res["CA"].get_vector()

    vh_rseqs = [r[0] for r in vh_res]
    vl_rseqs = [r[0] for r in vl_res]

    def pick(lst, frac):
        return lst[int(len(lst) * frac)]

    pairs = [
        (pick(vh_rseqs, 0.20), pick(vl_rseqs, 0.80)),
        (pick(vh_rseqs, 0.80), pick(vl_rseqs, 0.20)),
    ]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("! VH-VL unambiguous distance restraints\n")
        for r1, r2 in pairs:
            if r1 not in ca_coords or r2 not in ca_coords:
                print(f"WARNING: CA not found for resseq {r1} or {r2}")
                continue
            dist = (ca_coords[r1] - ca_coords[r2]).norm()
            f.write(
                f"assign (resid {r1} and name CA and segid B)"
                f" (resid {r2} and name CA and segid B)"
                f" {dist:.3f} 0.000 0.000\n"
            )


def load_epitope_residues(epitope_csv, threshold):
    """Return {res_id, ...} with epiLoRA epitope probability > threshold,
    from a cache/epitope/{stem}.csv file (columns: res_id,prob) built by
    build_epitope_cache.py."""
    import csv
    residues = set()
    with open(epitope_csv) as f:
        for row in csv.DictReader(f):
            if float(row["prob"]) > threshold:
                residues.add(int(row["res_id"]))
    return residues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ab",     required=True)
    p.add_argument("--ag",     required=True)
    p.add_argument("--ambig",  required=True)
    p.add_argument("--unambig",required=True)
    p.add_argument("--epitope-csv", default=None,
                    help="cache/epitope/{stem}.csv (res_id,prob) from build_epitope_cache.py; "
                         "if given, the antigen passive set is surface ∩ (prob > --epitope-threshold) "
                         "instead of the full surface")
    p.add_argument("--epitope-threshold", type=float, default=0.20)
    args = p.parse_args()

    # --- CDR residues from chain B (VH + VL merged) ---
    ab_residues = read_chain_residues(args.ab, "B")
    vh_res, vl_res = split_vh_vl(ab_residues)

    cdr_resseqs = get_cdr_resseqs(vh_res) + get_cdr_resseqs(vl_res)
    if not cdr_resseqs:
        print("WARNING: CDR detection failed, using all ab residues as active")
        cdr_resseqs = [r[0] for r in ab_residues]

    print(f"CDR residues (active): {len(cdr_resseqs)}")

    ab_passive = get_passive_from_active(args.ab, "B", cdr_resseqs)
    print(f"Ab passive residues: {len(ab_passive)}")

    # --- Antigen surface from chain A ---
    ag_surface = calc_surface_residues(args.ag, "A")
    if not ag_surface:
        print("WARNING: surface detection failed, using all ag residues as passive")
        ag_residues = read_chain_residues(args.ag, "A")
        ag_surface = [r[0] for r in ag_residues]

    print(f"Ag surface residues (passive): {len(ag_surface)}")

    ag_passive = ag_surface
    if args.epitope_csv:
        epitope_residues = load_epitope_residues(args.epitope_csv, args.epitope_threshold)
        constrained = sorted(set(ag_surface) & epitope_residues)
        print(f"Ag epitope residues (prob > {args.epitope_threshold}): {len(epitope_residues)}")
        print(f"Ag surface ∩ epitope (passive): {len(constrained)}")
        if constrained:
            ag_passive = constrained
        else:
            print("WARNING: surface ∩ epitope is empty, falling back to full surface as passive")

    write_ambig_tbl(cdr_resseqs, ab_passive, ag_passive, args.ambig)
    write_unambig_tbl(args.ab, args.unambig)

    print(f"Wrote {args.ambig}")
    print(f"Wrote {args.unambig}")


if __name__ == "__main__":
    main()
