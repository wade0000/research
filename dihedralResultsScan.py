#!/usr/bin/env python3
"""Analyzes results of Gaussian dihedral scans and produces corresponding .gjfs of the lowest energies found, based on group conventions.
See the input and selection settings below to modify.
File format: "XXX_scanN.out"
If multiple scans of the same molecule find different lower energy states, new .gjfs will be created that re-scan these lowest energies together to find the true global minimum.

Example:
    python dihedralResultsScan.py
    python dihedralResultsScan.py --scan-dir ./scans --out-dir ./gjf_out

PM6 scans produce up to three ranked, distinct geometries per species. All
generated inputs retain the configured B3LYP optimization/frequency route.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Sequence

"""ADJUST GAUSSIAN SETTINGS HERE- this pertains to the new files being generated"""
# User-adjustable Gaussian input and selection settings.
MEM = "32GB"
NPROC = 16
ROUTE_LINE = "# opt=(calcall,tight) freq=hinderedrotor ub3lyp/6-31g(d,p) guess=save"
SCAN_ROUTE_LINE = "# opt=modredundant ub3lyp/6-31g(d,p)"
TITLE_TEMPLATE = "{species} global min from scan{scan_number} point {point_index}, E={energy} Eh"
USE_CHK = False
TIE_TOL_HARTREE = 0.0
DISAGREE_TOL_KCAL = 0.0
KCAL_PER_HARTREE = Decimal("627.5095")
PM6_CANDIDATE_COUNT = 3
# Duplicate PM6 geometries are identified by RMS differences in atom-pair distances.
GEOMETRY_DUPLICATE_RMS_DISTANCE_A = 0.1

ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ELEMENT_BY_ATOMIC_NUMBER = {number: symbol for number, symbol in enumerate(ELEMENTS, start=1)}

FILENAME_RE = re.compile(r"(SP\d+)_scan(\d+)", re.IGNORECASE)
CHARGE_RE = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")
ENERGY_RE = re.compile(r"SCF Done:\s+E\(.*?\)\s*=\s*(-?\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)")
SCAN_POINT_RE = re.compile(r"scan point\s+(\d+)\s+out of\s+(\d+)", re.IGNORECASE)
SCANNED_COORDINATE_RE = re.compile(r"!\s*(\S+)\s+(D\([^)]+\))\s+.*?\bScan\s*!?\s*$")
MODREDUNDANT_SCAN_RE = re.compile(
    r"^\s*D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+S\s+(\d+)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DihedralScan:
    atom_indices: tuple[int, int, int, int]
    steps: int
    step_degrees: float

    def input_line(self) -> str:
        atoms = " ".join(str(index) for index in self.atom_indices)
        return f"D {atoms} S {self.steps} {self.step_degrees:.6f}"


@dataclass(frozen=True)
class ScanPoint:
    energy: Decimal
    geometry: tuple[tuple[str, float, float, float], ...]
    point_index: int
    expected_points: int | None


@dataclass
class ScanResult:
    path: Path
    species: str
    scan_number: int
    points: list[ScanPoint] = field(default_factory=list)
    charge: int | None = None
    multiplicity: int | None = None
    expected_points: int | None = None
    scanned_coordinate: str | None = None
    scan_directive: DihedralScan | None = None
    route_method: str | None = None
    normal_termination: bool = False
    n_stationary_markers: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_filename(path: Path) -> tuple[str, int] | None:
    """Return species and scan number from a recognized scan filename."""
    match = FILENAME_RE.search(path.stem)
    if match is None:
        return None
    return match.group(1).upper(), int(match.group(2))


def _parse_energy(line: str) -> Decimal | None:
    match = ENERGY_RE.search(line)
    if match is None:
        return None
    return Decimal(match.group(1).replace("D", "E").replace("d", "e"))


def parse_scan_file(path: Path, species: str, scan_number: int) -> ScanResult:
    """Stream one Gaussian output, retaining only the latest geometry before commits."""
    result = ScanResult(path=path, species=species, scan_number=scan_number)
    latest_energy: Decimal | None = None
    latest_geometry: tuple[tuple[str, float, float, float], ...] | None = None
    current_scan_point: int | None = None
    current_expected: int | None = None
    stationary_count = 0

    reading_orientation = False
    orientation_headers_left = 0
    orientation_rows: list[tuple[str, float, float, float]] = []
    orientation_invalid = False
    route_card_lines: list[str] | None = None

    def record_route_card() -> None:
        if route_card_lines is None or result.route_method is not None:
            return
        route_card = " ".join(route_card_lines)
        if re.search(r"\bPM6\b", route_card, re.IGNORECASE):
            result.route_method = "PM6"
        elif re.search(r"\bU?B3LYP\b", route_card, re.IGNORECASE):
            result.route_method = "B3LYP"
        else:
            result.route_method = "OTHER"

    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            if line.lstrip().startswith("#"):
                if route_card_lines is not None:
                    record_route_card()
                route_card_lines = [line.strip()]
            elif route_card_lines is not None:
                if not line.strip():
                    record_route_card()
                    route_card_lines = None
                else:
                    route_card_lines.append(line.strip())

            if reading_orientation:
                if orientation_headers_left:
                    orientation_headers_left -= 1
                    continue
                if line.lstrip().startswith("---"):
                    if orientation_rows and not orientation_invalid:
                        latest_geometry = tuple(orientation_rows)
                    reading_orientation = False
                    continue
                tokens = line.split()
                if len(tokens) != 6:
                    result.warnings.append(
                        f"Input orientation block ending near line {line_number} is malformed."
                    )
                    orientation_invalid = True
                    continue
                try:
                    atomic_number = int(tokens[1])
                    symbol = ELEMENT_BY_ATOMIC_NUMBER[atomic_number]
                    xyz = (float(tokens[3]), float(tokens[4]), float(tokens[5]))
                except (ValueError, KeyError) as exc:
                    result.warnings.append(
                        f"Input orientation block near line {line_number} has an invalid "
                        f"atomic number or coordinate ({exc})."
                    )
                    orientation_invalid = True
                    continue
                orientation_rows.append((symbol, *xyz))
                continue

            charge_match = CHARGE_RE.search(line)
            if charge_match and result.charge is None:
                result.charge = int(charge_match.group(1))
                result.multiplicity = int(charge_match.group(2))

            energy = _parse_energy(line)
            if energy is not None:
                latest_energy = energy

            scan_match = SCAN_POINT_RE.search(line)
            if scan_match:
                current_scan_point = int(scan_match.group(1))
                current_expected = int(scan_match.group(2))
                result.expected_points = current_expected

            if result.scanned_coordinate is None:
                coordinate_match = SCANNED_COORDINATE_RE.search(line)
                if coordinate_match:
                    result.scanned_coordinate = (
                        f"{coordinate_match.group(1)} {coordinate_match.group(2)}"
                    )

            if result.scan_directive is None:
                directive_match = MODREDUNDANT_SCAN_RE.match(line)
                if directive_match:
                    result.scan_directive = DihedralScan(
                        atom_indices=tuple(
                            int(directive_match.group(index)) for index in range(1, 5)
                        ),
                        steps=int(directive_match.group(5)),
                        step_degrees=float(directive_match.group(6)),
                    )

            if "Normal termination of Gaussian 16" in line:
                result.normal_termination = True

            if "Input orientation:" in line:
                reading_orientation = True
                orientation_headers_left = 4
                orientation_rows = []
                orientation_invalid = False
                latest_geometry = None
                continue

            if "-- Stationary point found." in line:
                stationary_count += 1
                result.n_stationary_markers += 1
                if latest_energy is None or latest_geometry is None:
                    missing = []
                    if latest_energy is None:
                        missing.append("SCF energy")
                    if latest_geometry is None:
                        missing.append("complete Input orientation geometry")
                    result.warnings.append(
                        f"Stationary point {stationary_count} near line {line_number} "
                        f"has no {' or '.join(missing)} and was not committed."
                    )
                    continue
                point_index = current_scan_point or stationary_count
                result.points.append(
                    ScanPoint(
                        energy=latest_energy,
                        geometry=latest_geometry,
                        point_index=point_index,
                        expected_points=current_expected,
                    )
                )

    record_route_card()
    if reading_orientation:
        result.warnings.append("The final Input orientation block is incomplete.")
    if result.charge is None or result.multiplicity is None:
        raise ValueError("no Charge/Multiplicity line was found")
    if result.multiplicity < 1:
        raise ValueError(f"invalid multiplicity {result.multiplicity}")
    return result


def select_global_minimum(
    scans: Sequence[ScanResult], tie_tolerance: float
) -> tuple[ScanResult, ScanPoint]:
    """Select the numeric minimum, resolving near-ties by file then point order."""
    candidates = [(scan, point) for scan in scans for point in scan.points]
    if not candidates:
        raise ValueError("cannot select a minimum without committed scan points")
    numeric_minimum = min(point.energy for _, point in candidates)
    decimal_tolerance = Decimal(str(tie_tolerance))
    tied = [
        (scan, point)
        for scan, point in candidates
        if point.energy <= numeric_minimum + decimal_tolerance
    ]
    return min(tied, key=lambda item: (item[0].scan_number, item[0].path.name.casefold(), item[1].point_index))


def _geometry_rms_distance_difference(
    first: tuple[tuple[str, float, float, float], ...],
    second: tuple[tuple[str, float, float, float], ...],
) -> float:
    """Compare internal pair distances, independent of whole-molecule rotation/translation."""
    pair_distance_differences: list[float] = []
    for first_index in range(len(first)):
        for second_index in range(first_index + 1, len(first)):
            first_distance = math.dist(first[first_index][1:], first[second_index][1:])
            second_distance = math.dist(second[first_index][1:], second[second_index][1:])
            pair_distance_differences.append(first_distance - second_distance)
    if not pair_distance_differences:
        return 0.0
    return math.sqrt(
        sum(difference * difference for difference in pair_distance_differences)
        / len(pair_distance_differences)
    )


def select_lowest_distinct_points(
    scans: Sequence[ScanResult],
    tie_tolerance: float,
    duplicate_tolerance: float,
    limit: int = PM6_CANDIDATE_COUNT,
) -> list[tuple[ScanResult, ScanPoint]]:
    """Rank points by energy and retain at most limit geometrically distinct structures."""
    candidates = [(scan, point) for scan in scans for point in scan.points]
    if not candidates:
        return []
    ranked_candidates: list[tuple[ScanResult, ScanPoint]] = []
    remaining = list(candidates)
    while remaining:
        numeric_minimum = min(point.energy for _, point in remaining)
        decimal_tolerance = Decimal(str(tie_tolerance))
        tied_candidates = [
            item for item in remaining if item[1].energy <= numeric_minimum + decimal_tolerance
        ]
        chosen = min(
            tied_candidates,
            key=lambda item: (
                item[0].scan_number,
                item[0].path.name.casefold(),
                item[1].point_index,
            ),
        )
        ranked_candidates.append(chosen)
        remaining.remove(chosen)

    selected: list[tuple[ScanResult, ScanPoint]] = []
    for scan, point in ranked_candidates:
        duplicate = any(
            _geometry_rms_distance_difference(point.geometry, selected_point.geometry)
            <= duplicate_tolerance
            for _, selected_point in selected
        )
        if not duplicate:
            selected.append((scan, point))
            if len(selected) == limit:
                break
    return selected


def check_minima_disagreement(
    per_scan_minima: Sequence[tuple[str, Decimal | float]], tolerance_kcal: float
) -> tuple[bool, Decimal]:
    """Return whether per-scan minima span more than the requested kcal/mol."""
    if len(per_scan_minima) < 2:
        return False, Decimal(0)
    energies = [energy if isinstance(energy, Decimal) else Decimal(str(energy)) for _, energy in per_scan_minima]
    spread_kcal = (max(energies) - min(energies)) * KCAL_PER_HARTREE
    exceeds_tolerance = spread_kcal > Decimal(str(tolerance_kcal))
    return exceeds_tolerance, spread_kcal


def check_minima_geometry_disagreement(
    scan_minima: Sequence[tuple[ScanResult, ScanPoint]],
    duplicate_tolerance: float,
) -> tuple[bool, int]:
    """Report whether scan minima contain multiple geometrically distinct structures."""
    representatives: list[tuple[ScanResult, ScanPoint]] = []
    for candidate in scan_minima:
        candidate_geometry = candidate[1].geometry
        if not any(
            _geometry_rms_distance_difference(candidate_geometry, representative[1].geometry)
            <= duplicate_tolerance
            for representative in representatives
        ):
            representatives.append(candidate)
    return len(representatives) > 1, len(representatives)


def write_gjf(
    path: Path,
    species: str,
    scan: ScanResult,
    point: ScanPoint,
    charge: int,
    multiplicity: int,
    candidate_rank: int | None = None,
) -> None:
    if candidate_rank is None:
        title = TITLE_TEMPLATE.format(
            species=species,
            scan_number=scan.scan_number,
            point_index=point.point_index,
            energy=point.energy,
        )
    else:
        title = (
            f"{species} PM6 candidate {candidate_rank} from scan{scan.scan_number} "
            f"point {point.point_index}, E={point.energy} Eh"
        )
    lines = [f"%mem={MEM}", f"%nprocshared={NPROC}"]
    if USE_CHK:
        lines.append(f"%chk={species}.chk")
    lines.extend((ROUTE_LINE, "", title, "", f"{charge} {multiplicity}"))
    for symbol, x, y, z in point.geometry:
        lines.append(f" {symbol:<18}{x: .8f}   {y: .8f}   {z: .8f}")
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8", newline="\n")


def write_dihedral_scan_gjf(
    path: Path,
    species: str,
    source_scan: ScanResult,
    source_point: ScanPoint,
    target_scan: ScanResult,
    charge: int,
    multiplicity: int,
) -> None:
    if target_scan.scan_directive is None:
        raise ValueError(f"scan directive is unavailable for {target_scan.path.name}")
    atom_count = len(source_point.geometry)
    invalid_indices = [
        index
        for index in target_scan.scan_directive.atom_indices
        if index < 1 or index > atom_count
    ]
    if invalid_indices:
        raise ValueError(
            f"target scan {target_scan.path.name} refers to atom indices "
            f"{invalid_indices}, but the starting geometry has {atom_count} atoms"
        )
    title = (
        f"{species} scan{target_scan.scan_number} started from scan"
        f"{source_scan.scan_number} minimum point {source_point.point_index}, "
        f"E={source_point.energy} Eh"
    )
    lines = [f"%mem={MEM}", f"%nprocshared={NPROC}"]
    if USE_CHK:
        lines.append(f"%chk={path.stem}.chk")
    lines.extend(
        (
            SCAN_ROUTE_LINE,
            "",
            title,
            "",
            f"{charge} {multiplicity}",
        )
    )
    for symbol, x, y, z in source_point.geometry:
        lines.append(f" {symbol:<18}{x: .8f}   {y: .8f}   {z: .8f}")
    lines.extend(("", target_scan.scan_directive.input_line()))
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8", newline="\n")


def _warning(code: str, species: str, path: Path | None, detail: str) -> str:
    file_part = f" file {path.name}" if path is not None else ""
    return f"WARNING [{code}] {species}:{file_part} {detail}"


def _scan_minimum(scan: ScanResult, tie_tolerance: float) -> ScanPoint:
    minimum = min(point.energy for point in scan.points)
    decimal_tolerance = Decimal(str(tie_tolerance))
    return min(
        (point for point in scan.points if point.energy <= minimum + decimal_tolerance),
        key=lambda point: point.point_index,
    )


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "species",
        "n_scan_files",
        "best_scan_file",
        "best_scan_point",
        "n_points_total",
        "energy_Eh",
        "n_gjf_files",
        "gjf_files",
        "per_scan_minimum_relative_energies_kcal_mol",
        "minima_disagree",
        "warnings",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _self_test(path: Path) -> None:
    parsed_name = parse_filename(path)
    if parsed_name is None:
        species, scan_number = "SELFTEST", 1
    else:
        species, scan_number = parsed_name
    scan = parse_scan_file(path, species, scan_number)
    assert scan.points, f"No committed scan points found in {path}"
    assert len(scan.points) == scan.n_stationary_markers, (
        f"Found {scan.n_stationary_markers} stationary-point markers but "
        f"committed {len(scan.points)} points"
    )
    selected_scan, selected_point = select_global_minimum([scan], TIE_TOL_HARTREE)
    with tempfile.TemporaryDirectory(prefix="scan_to_gjf_test_") as temp_dir:
        gjf_path = Path(temp_dir) / f"{species}.gjf"
        assert scan.charge is not None and scan.multiplicity is not None
        write_gjf(
            gjf_path,
            species,
            selected_scan,
            selected_point,
            scan.charge,
            scan.multiplicity,
        )
        lines = gjf_path.read_text(encoding="utf-8").splitlines()
        expected_headers = [f"%mem={MEM}", f"%nprocshared={NPROC}"]
        if USE_CHK:
            expected_headers.append(f"%chk={species}.chk")
        expected_headers.append(ROUTE_LINE)
        assert lines[: len(expected_headers)] == expected_headers, "Unexpected Gaussian headers"
        charge_index = len(expected_headers) + 3
        assert lines[charge_index] == f"{scan.charge} {scan.multiplicity}"
        expected_title = TITLE_TEMPLATE.format(
            species=species,
            scan_number=selected_scan.scan_number,
            point_index=selected_point.point_index,
            energy=selected_point.energy,
        )
        assert lines[len(expected_headers) + 1] == expected_title
        assert selected_point.energy == min(point.energy for point in scan.points)
        atom_lines = lines[charge_index + 1 : charge_index + 1 + len(selected_point.geometry)]
        assert len(atom_lines) == len(selected_point.geometry)
        assert all(len(line.split()) == 4 for line in atom_lines)

    under_tolerance = [("scan1", Decimal("-100.0")), ("scan2", Decimal("-99.9994"))]
    over_tolerance = [("scan1", Decimal("-100.0")), ("scan2", Decimal("-99.9990"))]
    assert not check_minima_disagreement(under_tolerance, 0.5)[0]
    assert check_minima_disagreement(over_tolerance, 0.5)[0]
    assert check_minima_disagreement(
        [("scan1", Decimal("-100.000000000")), ("scan2", Decimal("-99.999999999"))],
        0.0,
    )[0]
    print(
        f"Self-test passed: {len(scan.points)} committed points, "
        f"{len(selected_point.geometry)} atoms, charge/multiplicity "
        f"{scan.charge}/{scan.multiplicity}."
    )


def _positive_or_zero_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than or equal to zero")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find each species' lowest committed Gaussian scan geometry and write an optimization/frequency .gjf.",
        epilog=(
            "By default, scans .out files in the folder containing this script. "
            "Examples:\n"
            "  python scan_to_gjf.py\n"
            "  python scan_to_gjf.py --scan-dir ./scans --out-dir ./gjf_out"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="folder containing Gaussian .out scan files (default: folder containing this script)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./gjf_out"), help="output folder (default: ./gjf_out)")
    parser.add_argument("--glob", default="*.out", help="file pattern (default: *.out)")
    parser.add_argument(
        "--tie-tol",
        type=_positive_or_zero_float,
        default=TIE_TOL_HARTREE,
        help="energy tie tolerance in Hartree (default: 0; compare reported energies exactly)",
    )
    parser.add_argument(
        "--disagree-tol",
        type=_positive_or_zero_float,
        default=DISAGREE_TOL_KCAL,
        help="scan-minimum energy spread tolerance in kcal/mol (default: 0; flag any reported energy difference)",
    )
    parser.add_argument("--recursive", action="store_true", help="search subfolders recursively")
    parser.add_argument("--dry-run", action="store_true", help="parse and report without writing .gjf files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing species .gjf files")
    parser.add_argument("--self-test", type=Path, metavar="OUT_FILE", help="run parser and disagreement self-tests on one Gaussian output")
    args = parser.parse_args(argv)

    if args.self_test is not None:
        try:
            _self_test(args.self_test)
        except (AssertionError, OSError, ValueError) as exc:
            print(f"Self-test failed for {args.self_test}: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.scan_dir.is_dir():
        parser.error(f"--scan-dir is not a folder: {args.scan_dir}")

    files = list(args.scan_dir.rglob(args.glob) if args.recursive else args.scan_dir.glob(args.glob))
    files = sorted((path for path in files if path.is_file()), key=lambda path: str(path).casefold())
    grouped: dict[str, list[tuple[int, Path]]] = {}
    global_warnings: list[str] = []
    for path in files:
        parsed_name = parse_filename(path)
        if parsed_name is None:
            global_warnings.append(
                f"WARNING [FILENAME_UNMATCHED] file {path.name}: expected a name containing "
                "'SP<number>_scan<number>'; this file was not included."
            )
            continue
        species, scan_number = parsed_name
        grouped.setdefault(species, []).append((scan_number, path))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    flagged_species: list[str] = []
    species_names = sorted(grouped)
    all_warnings = list(global_warnings)

    for species_index, species in enumerate(species_names, start=1):
        discovered = sorted(grouped[species], key=lambda item: (item[0], item[1].name.casefold()))
        local_warnings: list[str] = []
        scan_numbers = sorted({number for number, _ in discovered})
        if scan_numbers:
            missing_numbers = sorted(set(range(scan_numbers[0], scan_numbers[-1] + 1)) - set(scan_numbers))
            if missing_numbers:
                local_warnings.append(
                    _warning(
                        "SCAN_NUMBER_GAP",
                        species,
                        None,
                        f"scan numbers {', '.join(map(str, missing_numbers))} are missing; check the input folder.",
                    )
                )

        parsed_scans: list[ScanResult] = []
        for scan_number, path in discovered:
            try:
                scan = parse_scan_file(path, species, scan_number)
            except (OSError, UnicodeError, ValueError, KeyError) as exc:
                local_warnings.append(
                    _warning(
                        "PARSE_ERROR",
                        species,
                        path,
                        f"could not parse this scan ({exc}); check that it is a complete Gaussian output.",
                    )
                )
                continue
            parsed_scans.append(scan)
            for detail in scan.warnings:
                local_warnings.append(
                    _warning("PARSE_WARNING", species, path, f"{detail} Check the Gaussian output around this point.")
                )
            if not scan.points:
                local_warnings.append(
                    _warning(
                        "NO_CONVERGED_POINTS",
                        species,
                        path,
                        "no stationary-point geometry/energy pairs were committed; this file will be skipped.",
                    )
                )
            if not scan.normal_termination:
                local_warnings.append(
                    _warning(
                        "ABNORMAL_TERMINATION",
                        species,
                        path,
                        "no normal Gaussian 16 termination was found; inspect the end of the output. "
                        "Any committed points will still be used.",
                    )
                )
            if scan.expected_points is not None and len(scan.points) < scan.expected_points:
                local_warnings.append(
                    _warning(
                        "INCOMPLETE_SCAN",
                        species,
                        path,
                        f"found {len(scan.points)} committed points but the log reports "
                        f"{scan.expected_points} expected; inspect convergence failures. Available points will be used.",
                    )
                )

        scans_with_points = [scan for scan in parsed_scans if scan.points]
        reference_signature: tuple[str, ...] | None = None
        usable_scans: list[ScanResult] = []
        known_methods = [scan.route_method for scan in scans_with_points if scan.route_method is not None]
        reference_method = known_methods[0] if known_methods else None
        for scan in scans_with_points:
            if reference_method is None:
                local_warnings.append(
                    _warning(
                        "ROUTE_METHOD_UNKNOWN",
                        species,
                        scan.path,
                        "no Gaussian route card method was found; legacy single-candidate behavior will be used. "
                        "Check the route card if this scan was run with PM6.",
                    )
                )
            elif scan.route_method is None:
                local_warnings.append(
                    _warning(
                        "ROUTE_METHOD_UNKNOWN",
                        species,
                        scan.path,
                        f"no method was found to compare with {reference_method}; this scan is skipped to avoid "
                        "mixing energies from different methods. Check the Gaussian route card.",
                    )
                )
                continue
            elif scan.route_method != reference_method:
                local_warnings.append(
                    _warning(
                        "ROUTE_METHOD_MISMATCH",
                        species,
                        scan.path,
                        f"route method {scan.route_method} differs from {reference_method}; this scan is skipped "
                        "to avoid comparing energies from different methods.",
                    )
                )
                continue
            first_signature = tuple(atom[0] for atom in scan.points[0].geometry)
            if any(tuple(atom[0] for atom in point.geometry) != first_signature for point in scan.points):
                local_warnings.append(
                    _warning(
                        "INTRA_SCAN_ATOM_MISMATCH",
                        species,
                        scan.path,
                        "atom count/order changes within this file; all its points are skipped. Check the output geometry blocks.",
                    )
                )
                continue
            if reference_signature is None:
                reference_signature = first_signature
            elif first_signature != reference_signature:
                local_warnings.append(
                    _warning(
                        "ATOM_MISMATCH",
                        species,
                        scan.path,
                        "atom count/order differs from the first usable scan; all its points are skipped. Check atom ordering and input files.",
                    )
                )
                continue
            usable_scans.append(scan)

        charged_scans = [scan for scan in parsed_scans if scan.charge is not None]
        reference_charge_scan = charged_scans[0] if charged_scans else None
        if reference_charge_scan is not None:
            for scan in charged_scans[1:]:
                if (scan.charge, scan.multiplicity) != (
                    reference_charge_scan.charge,
                    reference_charge_scan.multiplicity,
                ):
                    local_warnings.append(
                        _warning(
                            "CHARGE_MULTIPLICITY_MISMATCH",
                            species,
                            scan.path,
                            f"charge/multiplicity {scan.charge}/{scan.multiplicity} differs from "
                            f"the first scan {reference_charge_scan.path.name} "
                            f"({reference_charge_scan.charge}/{reference_charge_scan.multiplicity}); "
                            "the first scan's values will be used.",
                        )
                    )

        total_points = sum(len(scan.points) for scan in usable_scans)
        best_scan: ScanResult | None = None
        best_point: ScanPoint | None = None
        selected_candidates: list[tuple[ScanResult, ScanPoint]] = []
        candidate_output_names: list[str] = []
        scan_handoffs: list[tuple[ScanResult, ScanPoint, ScanResult]] = []
        minima_disagree: bool | None = False if len(usable_scans) == 1 else None
        relative_minima: list[str] = []
        energy_cell = ""
        best_scan_file = ""
        best_scan_point = ""

        output_destination: str | None = None
        if usable_scans:
            best_scan, best_point = select_global_minimum(usable_scans, args.tie_tol)
            energy_cell = str(best_point.energy)
            best_scan_file = best_scan.path.name
            best_scan_point = str(best_point.point_index)
            scan_minima = [(scan, _scan_minimum(scan, args.tie_tol)) for scan in usable_scans]
            for scan, point in scan_minima:
                relative = (point.energy - best_point.energy) * KCAL_PER_HARTREE
                relative_minima.append(f"scan{scan.scan_number}={relative:.6f}")

            if len(usable_scans) >= 2:
                energy_disagree, spread = check_minima_disagreement(
                    [(f"scan{scan.scan_number}", point.energy) for scan, point in scan_minima],
                    args.disagree_tol,
                )
                geometry_disagree, distinct_geometry_count = check_minima_geometry_disagreement(
                    scan_minima,
                    GEOMETRY_DUPLICATE_RMS_DISTANCE_A,
                )
                minima_disagree = energy_disagree or geometry_disagree
                if minima_disagree:
                    flagged_species.append(species)
                    detail = ", ".join(
                        f"scan{scan.scan_number} min {point.energy} Eh (pt {point.point_index})"
                        for scan, point in scan_minima
                    )
                    reasons = []
                    if energy_disagree:
                        reasons.append(f"energy spread = {spread:.2f} kcal/mol")
                    if geometry_disagree:
                        reasons.append(
                            f"{distinct_geometry_count} distinct minimum geometries "
                            f"(duplicate threshold {GEOMETRY_DUPLICATE_RMS_DISTANCE_A:.2f} A)"
                        )
                    local_warnings.append(
                        f"WARNING [MINIMA_DISAGREE] {species}: {detail}; "
                        f"{'; '.join(reasons)}; sequential B3LYP dihedral scans will be generated "
                        "instead of optimization/frequency inputs."
                    )

            if minima_disagree:
                for (source_scan, source_point), (target_scan, _) in zip(
                    scan_minima,
                    scan_minima[1:],
                ):
                    if target_scan.scan_directive is None:
                        local_warnings.append(
                            _warning(
                                "SCAN_HANDOFF_UNAVAILABLE",
                                species,
                                target_scan.path,
                                "the target dihedral scan directive could not be read; no handoff input was "
                                f"created from scan{source_scan.scan_number}. Check its ModRedundant input section.",
                            )
                        )
                        continue
                    atom_count = len(source_point.geometry)
                    invalid_indices = [
                        index
                        for index in target_scan.scan_directive.atom_indices
                        if index < 1 or index > atom_count
                    ]
                    if invalid_indices:
                        local_warnings.append(
                            _warning(
                                "SCAN_HANDOFF_INVALID",
                                species,
                                target_scan.path,
                                f"the target dihedral refers to atom indices {invalid_indices}, but the "
                                f"starting geometry from scan{source_scan.scan_number} has {atom_count} atoms; "
                                "this handoff was not written. Check atom ordering and the scan directive.",
                            )
                        )
                        continue
                    scan_handoffs.append((source_scan, source_point, target_scan))
                candidate_output_names = [
                    f"{species}_scan{target_scan.scan_number}_from_scan{source_scan.scan_number}.gjf"
                    for source_scan, _, target_scan in scan_handoffs
                ]
                planned_outputs = set(candidate_output_names)
                stale_outputs = [
                    path
                    for path in (
                        args.out_dir / f"{species}.gjf",
                        *args.out_dir.glob(f"{species}_min*.gjf"),
                    )
                    if path.exists() and path.name not in planned_outputs
                ]
                if stale_outputs:
                    local_warnings.append(
                        _warning(
                            "STALE_OPTIMIZATION_INPUTS",
                            species,
                            None,
                            f"previous optimization/frequency input(s) remain in the output folder "
                            f"({', '.join(path.name for path in stale_outputs)}); they were not deleted. "
                            "Review and remove obsolete files manually before submitting jobs.",
                        )
                    )
            elif reference_method == "PM6":
                selected_candidates = select_lowest_distinct_points(
                    usable_scans,
                    args.tie_tol,
                    GEOMETRY_DUPLICATE_RMS_DISTANCE_A,
                )
                if len(selected_candidates) < PM6_CANDIDATE_COUNT:
                    local_warnings.append(
                        _warning(
                            "PM6_FEWER_DISTINCT_GEOMETRIES",
                            species,
                            None,
                            f"only {len(selected_candidates)} distinct geometry/geometries were found; "
                            f"expected up to {PM6_CANDIDATE_COUNT}. Check the scan coverage or duplicate-distance "
                            f"tolerance ({GEOMETRY_DUPLICATE_RMS_DISTANCE_A:.2f} A).",
                        )
                    )
                candidate_output_names = [
                    f"{species}_min{rank}.gjf"
                    for rank in range(1, len(selected_candidates) + 1)
                ]
            else:
                selected_candidates = [(best_scan, best_point)]
                candidate_output_names = [f"{species}.gjf"]
            if best_point.expected_points is not None and best_point.point_index in (
                1,
                best_point.expected_points,
            ):
                local_warnings.append(
                    _warning(
                        "ENDPOINT_MINIMUM",
                        species,
                        best_scan.path,
                        f"global minimum is at scan point {best_point.point_index} of "
                        f"{best_point.expected_points}; for a non-circular scan the minimum may lie outside "
                        "the scanned range. For full-circle scans, the endpoint is noted only.",
                    )
                )

            if reference_charge_scan is None:
                local_warnings.append(
                    _warning(
                        "MISSING_CHARGE",
                        species,
                        best_scan.path,
                        "no parsed charge/multiplicity is available; no .gjf can be written.",
                    )
                )
                output_destination = "no .gjf written (charge/multiplicity unavailable)"
            else:
                if minima_disagree:
                    output_jobs = [
                        (name, source_scan, source_point, target_scan, None)
                        for name, (source_scan, source_point, target_scan) in zip(
                            candidate_output_names,
                            scan_handoffs,
                        )
                    ]
                else:
                    output_jobs = [
                        (
                            name,
                            candidate_scan,
                            candidate_point,
                            None,
                            rank if reference_method == "PM6" else None,
                        )
                        for rank, (name, (candidate_scan, candidate_point)) in enumerate(
                            zip(candidate_output_names, selected_candidates),
                            start=1,
                        )
                    ]
                output_status: list[str] = []
                for (
                    output_name,
                    candidate_scan,
                    candidate_point,
                    target_scan,
                    candidate_rank,
                ) in output_jobs:
                    output_path = args.out_dir / output_name
                    if args.dry_run:
                        output_status.append(f"{output_name} (dry run)")
                        continue
                    if output_path.exists() and not args.overwrite:
                        local_warnings.append(
                            _warning(
                                "OUTPUT_EXISTS",
                                species,
                                output_path,
                                "existing .gjf was not overwritten; pass --overwrite to replace it.",
                            )
                        )
                        output_status.append(f"{output_name} skipped (exists)")
                        continue
                    try:
                        if target_scan is not None:
                            write_dihedral_scan_gjf(
                                output_path,
                                species,
                                candidate_scan,
                                candidate_point,
                                target_scan,
                                reference_charge_scan.charge,
                                reference_charge_scan.multiplicity,
                            )
                        else:
                            write_gjf(
                                output_path,
                                species,
                                candidate_scan,
                                candidate_point,
                                reference_charge_scan.charge,
                                reference_charge_scan.multiplicity,
                                candidate_rank=candidate_rank,
                            )
                        output_status.append(output_name)
                    except ValueError as exc:
                            local_warnings.append(
                                _warning(
                                    "SCAN_HANDOFF_INVALID",
                                    species,
                                    target_scan.path if target_scan is not None else candidate_scan.path,
                                    f"could not create this scan input ({exc}); check its geometry and directive.",
                                )
                            )
                            output_status.append(f"{output_name} not written")
                    except OSError as exc:
                        local_warnings.append(
                            _warning(
                                "WRITE_ERROR",
                                species,
                                output_path,
                                f"could not write this .gjf ({exc}); check output-folder permissions and available space.",
                            )
                        )
                        output_status.append(f"{output_name} not written")
                output_destination = (
                    ", ".join(output_status)
                    if output_status
                    else "no scan handoffs could be created"
                    if minima_disagree
                    else "no distinct geometries to write"
                )

        all_warnings.extend(local_warnings)

        if usable_scans and best_scan is not None and best_point is not None:
            destination = output_destination or "no .gjf written (see warning)"
            minimum_label = f"min = {best_point.energy} Eh (scan{best_scan.scan_number} pt {best_point.point_index})"
        else:
            destination = "no .gjf (no usable scan points)"
            minimum_label = "no usable scan points"

        print(
            f"[{species_index}/{len(species_names)}] {species}: {len(usable_scans)} scans, "
            f"{total_points} points, {minimum_label} -> {destination}"
        )
        summary_rows.append(
            {
                "species": species,
                "n_scan_files": str(len(discovered)),
                "best_scan_file": best_scan_file,
                "best_scan_point": best_scan_point,
                "n_points_total": str(total_points),
                "energy_Eh": energy_cell,
                "n_gjf_files": str(len(candidate_output_names)),
                "gjf_files": "; ".join(candidate_output_names),
                "per_scan_minimum_relative_energies_kcal_mol": "; ".join(relative_minima),
                "minima_disagree": "" if minima_disagree is None else str(minima_disagree),
                "warnings": " | ".join(local_warnings),
            }
        )

    _write_summary(args.out_dir / "summary.csv", summary_rows)

    print("Species summary:")
    print(
        f"{'Species':<12} {'Scan files':>10} {'Points':>8} {'Minimum (Eh)':>18} "
        f"{'GJFs':>6} {'Disagree':>10}"
    )
    for row in summary_rows:
        print(
            f"{row['species']:<12} {row['n_scan_files']:>10} {row['n_points_total']:>8} "
            f"{row['energy_Eh'] or 'n/a':>18} {row['n_gjf_files']:>6} "
            f"{row['minima_disagree'] or 'n/a':>10}"
        )
    print("Warnings:")
    if all_warnings:
        for warning in all_warnings:
            print(warning)
    else:
        print("None")
    print(
        f"Summary: {len(species_names)} species, {sum(1 for row in summary_rows if row['energy_Eh'])} "
        f"with usable minima, {len(flagged_species)} minima disagreement(s). "
        f"CSV: {args.out_dir / 'summary.csv'}"
    )
    print("Species needing review:")
    print(", ".join(flagged_species) if flagged_species else "None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
