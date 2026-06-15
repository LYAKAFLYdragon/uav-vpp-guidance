#!/usr/bin/env python3
"""Compare two Monte-Carlo red-controller result files produced by monte_carlo_dogfight.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


def _fmt_cell(name: str, v1: float, v2: float) -> str:
    if name.endswith("_rate"):
        return f"{v1*100:>6.1f}% | {v2*100:>6.1f}% | {((v2-v1)*100):>+6.1f}%"
    if name.endswith("_m"):
        return f"{v1:>7.0f} | {v2:>7.0f} | {v2-v1:>+7.0f}"
    return f"{v1:>7.2f} | {v2:>7.2f} | {v2-v1:>+7.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a", type=str, help="First MC result JSON")
    parser.add_argument("file_b", type=str, help="Second MC result JSON")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    args = parser.parse_args()

    a = _load(args.file_a)
    b = _load(args.file_b)

    cells_a = a["summary"]["cells"]
    cells_b = b["summary"]["cells"]

    print("=" * 110)
    print(f"{'Scenario':<14} {'Policy':<10} {'Metric':<22} {args.label_a:<10} | {args.label_b:<10} | Diff")
    print("-" * 110)

    metrics = [
        ("success_rate", "Success rate"),
        ("own_crash_rate", "Own crash rate"),
        ("bandit_crash_rate", "Bandit crash rate"),
        ("mean_min_range_m", "Mean min range [m]"),
        ("mean_final_range_m", "Mean final range [m]"),
        ("mean_steps", "Mean steps"),
    ]

    for cell_key in sorted(cells_a.keys()):
        s_a = cells_a[cell_key]
        s_b = cells_b.get(cell_key)
        if s_b is None:
            continue
        scenario, policy = cell_key.rsplit("_", 1)
        for m, label in metrics:
            v1 = s_a[m]
            v2 = s_b[m]
            print(
                f"{scenario:<14} {policy:<10} {label:<22} {_fmt_cell(m, v1, v2)}"
            )

    print("=" * 110)
    oa = a["summary"]["overall"]
    ob = b["summary"]["overall"]
    for m, label in metrics:
        print(f"{'OVERALL':<14} {'ALL':<10} {label:<22} {_fmt_cell(m, oa[m], ob[m])}")


if __name__ == "__main__":
    main()
