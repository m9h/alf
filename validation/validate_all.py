#!/usr/bin/env python3
"""Run all ALF validation scripts and produce a summary report.

Usage:
    python validate_all.py

This script runs validate_hgf.py, validate_ddm.py, and
validate_metacognition.py in sequence, capturing their exit codes.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

VALIDATIONS = [
    ("HGF", "validate_hgf.py"),
    ("DDM", "validate_ddm.py"),
    ("Metacognition", "validate_metacognition.py"),
]


def main():
    print("=" * 60)
    print("ALF Full Validation Suite")
    print("=" * 60)
    print()

    results = {}

    for name, script in VALIDATIONS:
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            print(f"SKIP: {script} not found")
            results[name] = None
            continue

        print(f"Running {script}...")
        print("-" * 60)

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            timeout=600,
        )
        results[name] = proc.returncode == 0

        print()

    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {status}: {name}")

    n_fail = sum(1 for v in results.values() if v is False)
    n_pass = sum(1 for v in results.values() if v is True)
    n_skip = sum(1 for v in results.values() if v is None)

    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")

    if n_fail > 0:
        print("\n  Some validations FAILED. Check output above for details.")
        sys.exit(1)
    else:
        print("\n  All validations passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
