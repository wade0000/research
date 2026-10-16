"""
check_hinderedrotor.py

Standalone checker for Gaussian .log files: determines whether
freq=hinderedrotor was (a) requested in the route section and
(b) actually completed successfully.

NOTE: Gaussian's exact output wording for the hindered-rotor treatment
can vary slightly by version/revision. HINDERED_ROTOR_MARKERS below is
a best-effort list -- open ONE log you know completed hindered-rotor
successfully, search it manually, and add/adjust markers if needed
before trusting results across a full batch.

Usage:
    python3 check_hinderedrotor.py
    python3 check_hinderedrotor.py --pattern "*.log" --csv summary.csv
"""

import glob
import argparse
import csv
from collections import Counter

# Route-section text that indicates hinderedrotor was requested
ROUTE_MARKERS = ["hinderedrotor", "hindrot"]

# Output text that indicates the hindered-rotor treatment actually ran
HINDERED_ROTOR_MARKERS = [
    "Hindered Internal Rotation",
    "Hindered Rotor",
    "HinderedRotor",
    "Internal Rotation Analysis",
    "Treatment of Hindered Internal Rotation",
]


def check_log(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    lines = text.splitlines()

    result = {
        "file": path,
        "route_requested_hinderedrotor": False,
        "normal_termination": False,
        "hindered_rotor_output_found": False,
        "matched_marker": "",
        "status": "UNKNOWN",
    }

    # Route section is echoed near the top; check first ~200 lines, case-insensitive
    header_blob = " ".join(lines[:200]).lower()
    for marker in ROUTE_MARKERS:
        if marker in header_blob:
            result["route_requested_hinderedrotor"] = True
            break

    # Normal termination appears at/near the very end of the file
    if "Normal termination of Gaussian" in text:
        result["normal_termination"] = True

    # Search full file for hindered-rotor output evidence
    for marker in HINDERED_ROTOR_MARKERS:
        if marker.lower() in text.lower():
            result["hindered_rotor_output_found"] = True
            result["matched_marker"] = marker
            break

    # Classify
    if not result["route_requested_hinderedrotor"]:
        result["status"] = "NOT_REQUESTED"
    elif not result["normal_termination"]:
        result["status"] = "REQUESTED_BUT_JOB_DID_NOT_FINISH"
    elif result["hindered_rotor_output_found"]:
        result["status"] = "COMPLETED_SUCCESSFULLY"
    else:
        result["status"] = "REQUESTED_AND_FINISHED_BUT_NO_HR_OUTPUT_FOUND"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check Gaussian .log files for hindered-rotor freq completion"
    )
    parser.add_argument("--pattern", default="*.log", help="glob pattern for log files")
    parser.add_argument(
        "--csv",
        default="hinderedrotor_summary.csv",
        help="path to write the CSV summary (default: hinderedrotor_summary.csv)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched pattern: {args.pattern}")
        return

    results = [check_log(f) for f in files]

    for r in results:
        print(f"{r['file']:40s} {r['status']}")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWritten summary to {args.csv}")

    counts = Counter(r["status"] for r in results)
    print("\nSummary:")
    for status, count in counts.items():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()