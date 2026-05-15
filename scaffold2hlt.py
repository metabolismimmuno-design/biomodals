#!/usr/bin/env python3
"""
scaffold2hlt.py — Convert VHH scaffold PDB + YAML CDR definitions to RFantibody HLT format.

HLT format requirements (per RFantibody/scripts/util/chothia2HLT.py):
  - VHH chain renamed to H
  - Residues renumbered sequentially from 1
  - CDR loops annotated with REMARK PDBinfo-LABEL at end of file
  - Protein residues only (no HETATM)

CDR boundaries are read from the YAML `design` block (not Chothia hardcoded 95-102),
which was validated against ANARCI IMGT numbering for each specific scaffold PDB.

Usage:
    python scaffold2hlt.py \\
        --pdb inputs/scaffolds/7a6o_B_clean.pdb \\
        --yaml inputs/scaffolds/7a6o_B.yaml \\
        --output inputs/rfantibody/7a6o_B.hlt.pdb

    # batch (all 4 scaffolds):
    for name in 7a6o_B 7dv4_B 7xl0_B 8coh_A; do
        python scaffold2hlt.py \\
            --pdb inputs/scaffolds/${name}_clean.pdb \\
            --yaml inputs/scaffolds/${name}.yaml \\
            --output inputs/rfantibody/${name}.hlt.pdb
    done
"""

import argparse
import sys
from pathlib import Path

import yaml


PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


def parse_res_index(res_index_str: str) -> list[tuple[int, int]]:
    """Parse BoltzGen res_index string into list of (start, end) ranges.

    '26..35,50..65,98..118' → [(26,35), (50,65), (98,118)]
    """
    ranges = []
    for part in res_index_str.split(","):
        part = part.strip()
        if ".." in part:
            start, end = part.split("..")
            ranges.append((int(start), int(end)))
        else:
            n = int(part)
            ranges.append((n, n))
    return ranges


def load_yaml_cdr(yaml_path: Path) -> tuple[str, tuple[int,int], tuple[int,int], tuple[int,int]]:
    """Read scaffold YAML and return (chain_id, cdr1_range, cdr2_range, cdr3_range).

    Expects exactly 3 comma-separated ranges in design[0].chain.res_index,
    ordered CDR1, CDR2, CDR3.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    chain_id = cfg["include"][0]["chain"]["id"]
    res_index_str = cfg["design"][0]["chain"]["res_index"]
    ranges = parse_res_index(res_index_str)

    if len(ranges) != 3:
        raise ValueError(
            f"{yaml_path}: design block must have exactly 3 ranges (CDR1,CDR2,CDR3), "
            f"got {len(ranges)}: '{res_index_str}'"
        )

    return chain_id, ranges[0], ranges[1], ranges[2]


def build_cdr_set(cdr_range: tuple[int,int]) -> set[int]:
    return set(range(cdr_range[0], cdr_range[1] + 1))


def scaffold_to_hlt(pdb_path: Path, yaml_path: Path, output_path: Path) -> None:
    """Main conversion: scaffold PDB + YAML → HLT PDB."""

    chain_id, cdr1_range, cdr2_range, cdr3_range = load_yaml_cdr(yaml_path)
    cdr1_set = build_cdr_set(cdr1_range)
    cdr2_set = build_cdr_set(cdr2_range)
    cdr3_set = build_cdr_set(cdr3_range)

    # --- Read scaffold PDB, keep only ATOM records for the VHH chain ---
    raw_lines = Path(pdb_path).read_text().splitlines()
    chain_atoms = []  # list of (orig_resnum, icode, line)

    for line in raw_lines:
        rec = line[:6].strip()
        if rec != "ATOM":
            continue
        line_chain = line[21]
        if line_chain != chain_id:
            continue
        resname = line[17:20].strip()
        if resname not in PROTEIN_RESIDUES:
            continue
        orig_resnum = int(line[22:26])
        icode = line[26].strip()
        chain_atoms.append((orig_resnum, icode, line))

    if not chain_atoms:
        raise ValueError(f"No ATOM records found for chain {chain_id} in {pdb_path}")

    # --- Build residue groups (same orig_resnum+icode = same residue) ---
    residue_groups: list[tuple[int, str, list[str]]] = []
    current_key = None
    current_lines: list[str] = []

    for orig_resnum, icode, line in chain_atoms:
        key = (orig_resnum, icode)
        if key != current_key:
            if current_key is not None:
                residue_groups.append((current_key[0], current_key[1], current_lines))
            current_key = key
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_key is not None:
        residue_groups.append((current_key[0], current_key[1], current_lines))

    # --- Generate HLT output ---
    output_lines: list[str] = []
    remark_lines: list[str] = []
    atom_serial = 1

    for seq_num, (orig_resnum, icode, atom_lines) in enumerate(residue_groups, start=1):

        # Determine CDR label for this residue
        if orig_resnum in cdr1_set:
            cdr_label = "H1"
        elif orig_resnum in cdr2_set:
            cdr_label = "H2"
        elif orig_resnum in cdr3_set:
            cdr_label = "H3"
        else:
            cdr_label = None

        if cdr_label:
            remark_lines.append(f"REMARK PDBinfo-LABEL: {seq_num:4d} {cdr_label}\n")

        # Rewrite each ATOM line: chain→H, resnum→seq_num, serial→atom_serial
        for line in atom_lines:
            new_line = (
                line[:6]                    # "ATOM  "
                + f"{atom_serial:5d}"       # serial
                + line[11]                  # space
                + line[12:16]               # atom name (preserved exactly)
                + line[16]                  # altloc
                + line[17:21]               # resname + space
                + "H"                       # chain → H
                + f"{seq_num:4d}"           # sequential resnum
                + " "                       # insertion code (cleared)
                + line[27:]                 # coordinates + rest (preserve exactly)
            )
            output_lines.append(new_line if new_line.endswith("\n") else new_line + "\n")
            atom_serial += 1

    output_lines.append("END\n")
    output_lines.extend(remark_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(output_lines))

    # --- Summary ---
    n_h1 = sum(1 for r, _, _ in residue_groups if r in cdr1_set)
    n_h2 = sum(1 for r, _, _ in residue_groups if r in cdr2_set)
    n_h3 = sum(1 for r, _, _ in residue_groups if r in cdr3_set)
    n_total = len(residue_groups)

    print(f"[scaffold2hlt] {yaml_path.stem}")
    print(f"  chain {chain_id} → H | {n_total} residues renumbered 1–{n_total}")
    print(f"  H1: orig {cdr1_range[0]}–{cdr1_range[1]} ({n_h1} res) | "
          f"H2: orig {cdr2_range[0]}–{cdr2_range[1]} ({n_h2} res) | "
          f"H3: orig {cdr3_range[0]}–{cdr3_range[1]} ({n_h3} res)")
    print(f"  output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VHH scaffold PDB + YAML CDR definitions to RFantibody HLT format"
    )
    parser.add_argument("--pdb", required=True, help="Scaffold clean PDB (single VHH chain)")
    parser.add_argument("--yaml", required=True, help="Scaffold YAML with design block CDR ranges")
    parser.add_argument("--output", required=True, help="Output HLT PDB path")
    args = parser.parse_args()

    scaffold_to_hlt(
        pdb_path=Path(args.pdb),
        yaml_path=Path(args.yaml),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
