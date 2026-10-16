#!/usr/bin/env python3
"""
bond_energies.py - Bond (dissociation) energies from Gaussian 16 .out files.

For a parent molecule P that breaks into fragments A + B:

    BDE = (E_A + E_B) - E_P

Folder layout
-------------
    my_folder/
        parent.out              <- the intact molecule (same for every pair)
        CC_A.out   CC_B.out     <- pair "CC"
        CH_1.out   CH_2.out     <- pair "CH"
        CN_frag1.out CN_frag2.out

Fragment files must be named  <pair-label><sep><tag>.out  where <sep> is "_" or "-"
and <tag> is one of: A/B, 1/2, frag1/frag2, f1/f2 (case-insensitive).
Pair label can be anything (e.g. C1-C2, CH3_loss, bond7).

The parent is found automatically if exactly one file is named parent*/reactant*/
precursor*/molecule*/intact*.out, or you can pass it with --parent.

Energies reported (Hartree values converted to kcal/mol and kJ/mol):
    De  : electronic energy (last "SCF Done")        -> no thermal corrections
    D0  : electronic + zero-point energy             -> needs freq jobs
    H298: electronic + thermal enthalpy              -> needs freq jobs
    G298: electronic + thermal free energy           -> needs freq jobs

Usage
-----
    python bond_energies.py my_folder
    python bond_energies.py my_folder --parent my_folder/ethane.out --csv results.csv
    python bond_energies.py my_folder --units kj
"""

import argparse
import csv
import re
import sys
from pathlib import Path

HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_KJ = 2625.4996394799

FRAG_RE = re.compile(
    r"^(?P<pair>.+?)[_-](?P<tag>frag[12]|f[12]|[12]|[ab])$", re.IGNORECASE
)
PARENT_RE = re.compile(
    r"^(parent|reactant|precursor|molecule|intact)", re.IGNORECASE
)

FLOAT = r"(-?\d+\.\d+(?:[DEde][+-]?\d+)?)"


def to_float(s):
    return float(s.replace("D", "E").replace("d", "e"))


def parse_gaussian(path):
    """Extract energies and status flags from a Gaussian 16 output file."""
    text = Path(path).read_text(errors="replace")
    info = {
        "file": Path(path).name,
        "scf": None,
        "zpe_total": None,   # electronic + ZPE
        "h_total": None,     # electronic + thermal enthalpy
        "g_total": None,     # electronic + thermal free energy
        "normal_term": "Normal termination" in text,
        "n_imag": None,
        "multiplicity": None,
        "warnings": [],
    }

    # Last SCF energy (works for HF, DFT; method name inside E(...) varies)
    scf = re.findall(r"SCF Done:\s+E\([^)]*\)\s*=\s*" + FLOAT, text)
    if scf:
        info["scf"] = to_float(scf[-1])

    # Thermochemistry (last frequency job in file)
    for key, label in [
        ("zpe_total", "Sum of electronic and zero-point Energies="),
        ("h_total", "Sum of electronic and thermal Enthalpies="),
        ("g_total", "Sum of electronic and thermal Free Energies="),
    ]:
        m = re.findall(re.escape(label) + r"\s*" + FLOAT, text)
        if m:
            info[key] = to_float(m[-1])

    # Imaginary frequencies (from the last frequency block)
    freq_lines = re.findall(r"Frequencies --\s+(.*)", text)
    if freq_lines:
        vals = [float(x) for line in freq_lines for x in line.split()]
        info["n_imag"] = sum(v < 0 for v in vals)

    m = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", text)
    if m:
        info["multiplicity"] = int(m.group(2))

    # Sanity warnings
    if not info["normal_term"]:
        info["warnings"].append("no 'Normal termination' (job incomplete/failed)")
    if info["scf"] is None:
        info["warnings"].append("no SCF energy found")
    if info["n_imag"]:
        info["warnings"].append(f"{info['n_imag']} imaginary frequenc"
                                f"{'y' if info['n_imag'] == 1 else 'ies'}")
    return info


def find_parent(folder, explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"Parent file not found: {p}")
        return p
    cands = [f for f in folder.glob("*.out") if PARENT_RE.match(f.stem)]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        sys.exit("Could not find a parent file. Name it parent.out (or reactant.out, "
                 "etc.) or pass --parent FILE.")
    sys.exit("Multiple parent candidates found: "
             + ", ".join(c.name for c in cands) + ". Use --parent FILE.")


def find_pairs(folder, parent_path):
    """Group fragment files into pairs by naming convention."""
    groups = {}
    for f in sorted(folder.glob("*.out")):
        if f.resolve() == parent_path.resolve():
            continue
        m = FRAG_RE.match(f.stem)
        if not m:
            print(f"  [skip] {f.name}: name doesn't match <pair>_<A|B|1|2>.out",
                  file=sys.stderr)
            continue
        tag = m.group("tag").lower()
        slot = 0 if tag in ("a", "1", "f1", "frag1") else 1
        groups.setdefault(m.group("pair"), {})
        if slot in groups[m.group("pair")]:
            sys.exit(f"Pair '{m.group('pair')}' has two files for the same "
                     f"fragment slot: check {f.name}")
        groups[m.group("pair")][slot] = f

    pairs = {}
    for label, slots in groups.items():
        if len(slots) != 2:
            print(f"  [skip] pair '{label}': only one fragment file found "
                  f"({next(iter(slots.values())).name})", file=sys.stderr)
            continue
        pairs[label] = (slots[0], slots[1])
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder containing Gaussian 16 .out files")
    ap.add_argument("--parent", help="path to the parent molecule .out file")
    ap.add_argument("--units", choices=["kcal", "kj", "hartree"], default="kcal",
                    help="output units (default: kcal/mol)")
    ap.add_argument("--csv", help="also write results to this CSV file")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    factor = {"kcal": HARTREE_TO_KCAL, "kj": HARTREE_TO_KJ, "hartree": 1.0}[args.units]
    unit_label = {"kcal": "kcal/mol", "kj": "kJ/mol", "hartree": "Eh"}[args.units]

    parent_path = find_parent(folder, args.parent)
    parent = parse_gaussian(parent_path)
    print(f"Parent: {parent['file']}")
    for w in parent["warnings"]:
        print(f"  WARNING (parent): {w}")

    pairs = find_pairs(folder, parent_path)
    if not pairs:
        sys.exit("No complete fragment pairs found.")

    quantities = [
        ("De", "scf", "electronic"),
        ("D0", "zpe_total", "+ZPE"),
        ("H298", "h_total", "enthalpy"),
        ("G298", "g_total", "free energy"),
    ]
    # Only report columns for which the parent and every fragment have data
    all_data = {label: (parse_gaussian(a), parse_gaussian(b))
                for label, (a, b) in pairs.items()}
    usable = [q for q in quantities
              if parent[q[1]] is not None
              and all(f1[q[1]] is not None and f2[q[1]] is not None
                      for f1, f2 in all_data.values())]
    if not usable:
        sys.exit("No energy type is available for the parent and all fragments.")

    rows = []
    for label, (f1, f2) in all_data.items():
        row = {"pair": label, "frag1": f1["file"], "frag2": f2["file"]}
        for name, key, _ in usable:
            row[name] = ((f1[key] + f2[key]) - parent[key]) * factor
        warns = [f"{f['file']}: {w}" for f in (f1, f2) for w in f["warnings"]]
        row["warnings"] = "; ".join(warns)
        rows.append(row)

    # Console table
    cols = [q[0] for q in usable]
    head = f"{'Pair':<20}" + "".join(f"{c + ' (' + unit_label + ')':>20}" for c in cols)
    print("\nBond energy = (E_frag1 + E_frag2) - E_parent\n")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['pair']:<20}" + "".join(f"{r[c]:>20.2f}" for c in cols)
              + ("   <-- see warnings" if r["warnings"] else ""))
    flagged = [r for r in rows if r["warnings"]]
    if flagged:
        print("\nWarnings:")
        for r in flagged:
            print(f"  {r['pair']}: {r['warnings']}")
    print("\nNotes: De uses the last 'SCF Done' energy. D0/H298/G298 use Gaussian's "
          "thermochemistry block\n(requires freq jobs on parent and all fragments, "
          "at consistent levels of theory).")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "frag1_file", "frag2_file"]
                       + [f"{c}_{unit_label}" for c in cols] + ["warnings"])
            for r in rows:
                w.writerow([r["pair"], r["frag1"], r["frag2"]]
                           + [f"{r[c]:.4f}" for c in cols] + [r["warnings"]])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()