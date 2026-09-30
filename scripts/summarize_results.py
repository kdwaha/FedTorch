#!/usr/bin/env python3
"""Collect the final server test accuracy from a FedTorch baseline batch."""

import argparse
import csv
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--prefix", required=True,
                        help="The --tag value passed to run_baselines.sh")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.logs)
    rows = []
    for log_file in sorted(root.glob("*_{}*/experiment_summary.log".format(args.prefix))):
        content = log_file.read_text()
        if "Global iteration finished successfully." not in content:
            continue
        matches = re.findall(r"Test Accuracy: ([0-9.]+)", content)
        if matches:
            rows.append({
                "run": log_file.parent.name,
                "final_test_accuracy": "{:.6f}".format(float(matches[-1])),
            })

    if not rows:
        raise SystemExit("No completed baseline logs found for prefix '{}'".format(args.prefix))

    output = Path(args.output) if args.output else root / "{}_summary.csv".format(args.prefix)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "final_test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
