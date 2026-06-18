#!/usr/bin/env python3
"""Aggregate VPP / No-VPP / E2E maneuver-target pilot results.

Reads outputs/pilot_maneuver_target/{vpp,no_vpp,e2e}_s{seed}/logs/eval_log.csv
and writes:
  - outputs/pilot_maneuver_target/summary.json
  - outputs/pilot_maneuver_target/comparison_table.md
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("outputs/pilot_maneuver_target")
METHODS = ["vpp", "no_vpp", "e2e"]
SEEDS = [0, 1, 2]


def read_last_eval_row(method: str, seed: int):
    path = ROOT / f"{method}_s{seed}" / "logs" / "eval_log.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return None
    return reader[-1]


def main():
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_seeds": len(SEEDS),
        "config_root": "config/experiment/maneuver_target_*_pilot.yaml",
        "methods": {},
    }

    md_lines = [
        "# Maneuver-Target Pilot: VPP vs No-VPP vs E2E\n",
        f"**Date**: {summary['timestamp']}\n",
        f"**Seeds per method**: {len(SEEDS)}\n",
        f"**Eval episodes per seed**: 30\n\n",
        "## Aggregate Comparison\n\n",
        "| Method | Mean SR | Std SR | Mean Return | Std Return | Mean Final Range (m) | Mean Final ATA (°) | Crash Rate |\n",
        "|--------|---------|--------|-------------|------------|----------------------|--------------------|------------|\n",
    ]

    for method in METHODS:
        rows = []
        for seed in SEEDS:
            row = read_last_eval_row(method, seed)
            if row is not None:
                rows.append({k: float(v) for k, v in row.items()})

        if not rows:
            continue

        srs = [r["success_rate"] for r in rows]
        returns = [r["mean_return"] for r in rows]
        ranges = [r["mean_final_range_m"] for r in rows]
        atas = [r["mean_final_ata_deg"] for r in rows]
        crashes = [r["crash_rate"] for r in rows]

        summary["methods"][method] = {
            "seeds": SEEDS,
            "success_rate_per_seed": srs,
            "mean_success_rate": float(np.mean(srs)),
            "std_success_rate": float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "mean_final_range_m": float(np.mean(ranges)),
            "mean_final_ata_deg": float(np.mean(atas)),
            "mean_crash_rate": float(np.mean(crashes)),
        }

        md_lines.append(
            f"| {method.upper()} | "
            f"{summary['methods'][method]['mean_success_rate']:.1%} | "
            f"{summary['methods'][method]['std_success_rate']:.1%} | "
            f"{summary['methods'][method]['mean_return']:.1f} | "
            f"{summary['methods'][method]['std_return']:.1f} | "
            f"{summary['methods'][method]['mean_final_range_m']:.1f} | "
            f"{summary['methods'][method]['mean_final_ata_deg']:.1f} | "
            f"{summary['methods'][method]['mean_crash_rate']:.1%} |\n"
        )

    # Interpretation
    vpp_sr = summary["methods"].get("vpp", {}).get("mean_success_rate", 0.0)
    no_vpp_sr = summary["methods"].get("no_vpp", {}).get("mean_success_rate", 0.0)
    e2e_sr = summary["methods"].get("e2e", {}).get("mean_success_rate", 0.0)

    if abs(vpp_sr - no_vpp_sr) < 0.05:
        interpretation = (
            "**Interpretation**: The existing maneuver-target pilot configurations do NOT differentiate "
            "VPP and No-VPP (success-rate gap < 5 percentage points). All methods saturate the easy "
            "scenarios. The next step is to increase target difficulty (e.g. curriculum v2 Stage 3) "
            "or tighten success criteria before running the full 10-seed study.\n"
        )
    else:
        interpretation = (
            "**Interpretation**: VPP shows a meaningful advantage over No-VPP in the pilot. "
            "Proceed to the full 10-seed curriculum + adversarial experiments.\n"
        )

    md_lines.extend(["\n", interpretation, "\n"])
    md_lines.append("## Per-Seed Details\n\n")
    md_lines.append("| Method | Seed | Success Rate | Crash Rate | Final Range (m) | Final ATA (°) |\n")
    md_lines.append("|--------|------|--------------|------------|-----------------|---------------|\n")

    for method in METHODS:
        for seed in SEEDS:
            row = read_last_eval_row(method, seed)
            if row is None:
                continue
            md_lines.append(
                f"| {method.upper()} | {seed} | "
                f"{float(row['success_rate']):.1%} | "
                f"{float(row['crash_rate']):.1%} | "
                f"{float(row['mean_final_range_m']):.1f} | "
                f"{float(row['mean_final_ata_deg']):.1f} |\n"
            )

    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(ROOT / "comparison_table.md", "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    print(f"Saved {ROOT / 'summary.json'}")
    print(f"Saved {ROOT / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
