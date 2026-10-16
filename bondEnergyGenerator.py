#!/usr/bin/env python3
"""Generate Gaussian inputs for two-fragment bond dissociation calculations.

Reads the final Standard orientation and charge/multiplicity from a Gaussian
16 output, infers covalent connectivity, and writes one A/B input pair for
each bond whose removal divides the molecular graph into exactly two pieces.
The generated names match bondEnergyAnalysisB3LYP.py after Gaussian runs:
<atom>-<atom>_A.out and <atom>-<atom>_B.out.

Examples:
	python bondEnergyGenerator.py optimized.out
	python bondEnergyGenerator.py optimized.out --bond 2-5 --bond 5-8
	python bondEnergyGenerator.py ion.out --fragment-charges 0,-1

Fragment charge/spin states are chemical choices, not reliably determined by
geometry. Neutral fragments default to neutral singlet/doublet states based on
electron parity. For an ionic parent, explicitly supply --fragment-charges.
Review multiplicities for the intended dissociation channel before running.
"""

import argparse
import math
import re
import sys
from collections import deque
from pathlib import Path


ATOMIC_SYMBOLS = {
	1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
	8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al",
	14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
	20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
	26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga",
	32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb",
	38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
	44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
	50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
	56: "Ba", 57: "La", 72: "Hf", 73: "Ta", 74: "W", 75: "Re",
	76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
	82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
}

# Cordero covalent radii in angstroms for common organic/main-group elements
# and selected metals. Connectivity remains a geometric estimate.
COVALENT_RADII = {
	"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
	"F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
	"Br": 1.20, "I": 1.39, "Li": 1.28, "Na": 1.66, "Mg": 1.41,
	"Al": 1.21, "K": 2.03, "Ca": 1.76, "Fe": 1.32, "Cu": 1.32,
	"Zn": 1.22, "Ag": 1.45, "Au": 1.36,
}

FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[DEde][-+]?\d+)?"
CHARGE_MULT_RE = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")


def as_float(value):
	return float(value.replace("D", "E").replace("d", "e"))


def parse_output(path):
	text = path.read_text(errors="replace")
	lines = text.splitlines()

	orientations = []
	for index, line in enumerate(lines):
		if "Standard orientation:" not in line:
			continue
		atoms = []
		cursor = index + 1
		separators = 0
		while cursor < len(lines):
			row = lines[cursor].split()
			cursor += 1
			if not row:
				continue
			if all(char == "-" for char in row[0]):
				separators += 1
				if separators == 3:
					break
				continue
			if separators == 2 and len(row) >= 6:
				try:
					atomic_number = int(row[1])
					xyz = tuple(as_float(value) for value in row[3:6])
				except (ValueError, IndexError):
					continue
				if atomic_number not in ATOMIC_SYMBOLS:
					raise ValueError(f"Unsupported atomic number in output: {atomic_number}")
				atoms.append((ATOMIC_SYMBOLS[atomic_number], xyz))
		if atoms:
			orientations.append(atoms)

	if not orientations:
		raise ValueError("No atom coordinates found in a Standard orientation block.")

	charge_mults = CHARGE_MULT_RE.findall(text)
	if not charge_mults:
		raise ValueError("Could not find Gaussian 'Charge = ... Multiplicity = ...'.")

	return orientations[-1], int(charge_mults[-1][0]), int(charge_mults[-1][1])


def build_graph(atoms, scale):
	graph = [set() for _ in atoms]
	for first in range(len(atoms)):
		symbol_a, xyz_a = atoms[first]
		if symbol_a not in COVALENT_RADII:
			raise ValueError(f"No covalent radius for element {symbol_a}.")
		for second in range(first + 1, len(atoms)):
			symbol_b, xyz_b = atoms[second]
			if symbol_b not in COVALENT_RADII:
				raise ValueError(f"No covalent radius for element {symbol_b}.")
			distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz_a, xyz_b)))
			threshold = scale * (COVALENT_RADII[symbol_a] + COVALENT_RADII[symbol_b])
			if 0.1 < distance <= threshold:
				graph[first].add(second)
				graph[second].add(first)
	return graph


def components(graph, removed_edge=None):
	unseen = set(range(len(graph)))
	result = []
	while unseen:
		start = min(unseen)
		found = {start}
		queue = deque([start])
		unseen.remove(start)
		while queue:
			current = queue.popleft()
			for neighbor in graph[current]:
				if removed_edge and {current, neighbor} == set(removed_edge):
					continue
				if neighbor in unseen:
					unseen.remove(neighbor)
					found.add(neighbor)
					queue.append(neighbor)
		result.append(sorted(found))
	return result


def candidate_cuts(graph, requested_bonds):
	edges = [(a, b) for a in range(len(graph)) for b in graph[a] if a < b]
	if requested_bonds:
		requested = {tuple(sorted(edge)) for edge in requested_bonds}
		edges = [edge for edge in edges if edge in requested]
		missing = requested - set(edges)
		if missing:
			missing_text = ", ".join(f"{a + 1}-{b + 1}" for a, b in sorted(missing))
			raise ValueError(f"Requested bond(s) not inferred from geometry: {missing_text}")

	cuts = []
	for edge in edges:
		pieces = components(graph, edge)
		if len(pieces) == 2:
			cuts.append((edge, pieces))
		elif requested_bonds:
			raise ValueError(
				f"Bond {edge[0] + 1}-{edge[1] + 1} does not split the structure into two fragments."
			)
	return cuts


def parse_pair_values(value, option_name):
	try:
		left, right = (int(part.strip()) for part in value.split(","))
	except (ValueError, TypeError):
		raise ValueError(f"{option_name} must be two comma-separated integers, e.g. 0,0.")
	return left, right


def default_multiplicity(atoms, charge):
	# Atomic numbers are the electron counts for neutral atoms.
	symbol_to_number = {symbol: number for number, symbol in ATOMIC_SYMBOLS.items()}
	electrons = sum(symbol_to_number[symbol] for symbol, _ in atoms) - charge
	return 1 if electrons % 2 == 0 else 2


def write_input(path, atoms, charge, multiplicity, route, title, processors, memory):
	with path.open("w", newline="\n") as output:
		output.write(f"%nprocshared={processors}\n%mem={memory}\n")
		output.write(f"{route}\n\n{title}\n\n{charge} {multiplicity}\n")
		for symbol, (x, y, z) in atoms:
			output.write(f"{symbol:<3} {x: .8f} {y: .8f} {z: .8f}\n")
		output.write("\n")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("input", type=Path, help="optimized Gaussian 16 .out file")
	parser.add_argument("--output-dir", type=Path, help="directory for generated .gjf files (default: input directory)")
	parser.add_argument("--bond", action="append", default=[], metavar="I-J",
						help="generate only this 1-based atom-index bond; repeatable (default: all inferred cuts)")
	parser.add_argument("--scale", type=float, default=1.25,
						help="covalent-radius connectivity multiplier (default: 1.25)")
	parser.add_argument("--fragment-charges", metavar="QA,QB",
						help="integer fragment charges; required for charged parent structures")
	parser.add_argument("--multiplicities", metavar="MA,MB",
						help="fragment spin multiplicities (default: singlet/doublet from electron parity)")
	parser.add_argument("--processors", type=int, default=16, help="processors in each generated input (default: 16)")
	parser.add_argument("--memory", default="32GB", help="Gaussian memory per job (default: 32GB)")
	args = parser.parse_args()

	try:
		if not args.input.is_file():
			raise ValueError(f"Input file not found: {args.input}")
		if args.scale <= 0 or args.processors < 1:
			raise ValueError("--scale and --processors must be positive.")

		atoms, parent_charge, parent_multiplicity = parse_output(args.input)
		route = "# opt=(calcall,tight) freq b3lyp/6-31g(d,p)"

		if args.fragment_charges:
			charges = parse_pair_values(args.fragment_charges, "--fragment-charges")
			if sum(charges) != parent_charge:
				raise ValueError("Fragment charges must sum to the parent charge.")
		elif parent_charge == 0:
			charges = (0, 0)
		else:
			raise ValueError("Parent is charged; supply --fragment-charges QA,QB for the desired channel.")

		multiplicities = (
			parse_pair_values(args.multiplicities, "--multiplicities")
			if args.multiplicities else None
		)
		if multiplicities and min(multiplicities) < 1:
			raise ValueError("Multiplicities must be positive integers.")

		requested = []
		for value in args.bond:
			match = re.fullmatch(r"(\d+)-(\d+)", value)
			if not match:
				raise ValueError(f"Invalid bond {value!r}; use 1-based atom indices like 2-5.")
			first, second = (int(part) - 1 for part in match.groups())
			if first < 0 or second < 0 or first >= len(atoms) or second >= len(atoms) or first == second:
				raise ValueError(f"Bond indices out of range: {value}")
			requested.append((first, second))

		graph = build_graph(atoms, args.scale)
		if len(components(graph)) != 1:
			raise ValueError("Inferred structure is disconnected; review geometry/connectivity scale.")
		cuts = candidate_cuts(graph, requested)
		if not cuts:
			raise ValueError("No bonds split the inferred structure into exactly two fragments.")

		output_dir = args.output_dir or args.input.parent
		output_dir.mkdir(parents=True, exist_ok=True)
		for (first, second), pieces in cuts:
			label = f"{atoms[first][0]}{first + 1}-{atoms[second][0]}{second + 1}"
			print(f"\n{label}: bond {first + 1}-{second + 1}")
			for slot, atom_indices in enumerate(pieces):
				fragment_atoms = [atoms[index] for index in atom_indices]
				charge = charges[slot]
				mult = multiplicities[slot] if multiplicities else default_multiplicity(fragment_atoms, charge)
				# Validate the spin parity using each atom's actual atomic number.
				symbol_to_number = {symbol: number for number, symbol in ATOMIC_SYMBOLS.items()}
				electron_count = sum(symbol_to_number[symbol] for symbol, _ in fragment_atoms) - charge
				if (electron_count + mult) % 2 != 1:
					raise ValueError(f"Multiplicity {mult} is incompatible with electron count for {label} fragment {slot + 1}.")
				suffix = "A" if slot == 0 else "B"
				destination = output_dir / f"{label}_{suffix}.gjf"
				write_input(destination, fragment_atoms, charge, mult, route,
							f"{label} fragment {suffix} from {args.input.name}",
							args.processors, args.memory)
				atom_list = ",".join(str(index + 1) for index in atom_indices)
				print(f"  {suffix}: atoms {atom_list}; charge {charge}; multiplicity {mult}; {destination}")

		print(f"\nGenerated {len(cuts)} fragment pair(s) using route: {route}")
		print(f"Parent charge/multiplicity: {parent_charge} {parent_multiplicity}")
		print("Review inferred connectivity and fragment spin states before submitting jobs.")
		print("After Gaussian runs, analyze the resulting .out files with bondEnergyAnalysisB3LYP.py.")
	except (OSError, ValueError) as error:
		sys.exit(f"Error: {error}")


if __name__ == "__main__":
	main()
