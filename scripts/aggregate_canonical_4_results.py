#!/usr/bin/env python3
"""Aggregate multiple canonical_4_results.json files into a single table.

Usage:
    python scripts/aggregate_canonical_4_results.py \
        --pattern 'outputs/no_vpp_fixed_s*/canonical_4_results.json' \
        --output outputs/no_vpp_fixed_canonical4_summary.csv
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched pattern: {args.pattern}")
        return

    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        row = {"run": Path(f).parent.name}
        for scen in ["favorable", "neutral", "disadvantage", "challenging"]:
            row[f"{scen}_success"] = data["per_scenario"][scen]["success"]
            row[f"{scen}_crash"] = data["per_scenario"][scen]["crash"]
            row[f"{scen}_oob"] = data["per_scenario"][scen]["oob"]
            row[f"{scen}_timeout"] = data["per_scenario"][scen]["timeout"]
        row["overall_success"] = data["overall_success"]
        rows.append(row)

    # Compute mean and std
    if len(rows) > 1:
        mean_row = {"run": "mean"}
        std_row = {"run": "std"}
        for key in rows[0]:
            if key == "run":
                continue
            vals = [r[key] for r in rows]
            mean_row[key] = sum(vals) / len(vals)
            std_row[key] = (
                sum((v - mean_row[key]) ** 2 for v in vals) / len(vals)
            ) ** 0.5
        rows.append(mean_row)
        rows.append(std_row)

    # Print table
    print(f"\nAggregated results from {len(files)} runs:\n")
    headers = ["run", "overall_success"] + [
        f"{scen}_success" for scen in ["favorable", "neutral", "disadvantage", "challenging"]
    ]
    print(", ".join(headers))
    for row in rows:
        print(", ".join(f"{row[h]:.1f}" if isinstance(row[h], float) else str(row[h]) for h in headers))

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved aggregated results to {args.output}")


if __name__ == "__main__":
    main()
