#!/usr/bin/env python3
"""Build raw_episodes.csv from validation trajectory exports.

The validation runs only write per-step trajectory CSVs and an aggregate eval_log.
This script reconstructs episode-level success/return records so the
reviewer-compliant table generator can consume them.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


SCENARIO_MAP = {0: "favorable", 1: "neutral", 2: "disadvantage"}


def extract(root: Path, method: str):
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        traj_dir = seed_dir / "trajectories" / "eval"
        if not traj_dir.exists():
            continue
        for f in sorted(traj_dir.glob("eval_seed*_ep*.csv")):
            parts = f.stem.split("_")
            eval_seed = int(parts[1].replace("seed", ""))
            episode = int(parts[2].replace("ep", ""))
            scenario = SCENARIO_MAP.get(eval_seed, "neutral")

            with open(f) as fp:
                step_rows = list(csv.DictReader(fp))
            if not step_rows:
                continue
            returns = np.array([float(r["reward"]) for r in step_rows])
            total_return = float(returns.sum())
            final_range = float(step_rows[-1]["range_m"])

            is_success = total_return > 0.0
            is_out_of_bounds = (not is_success) and final_range > 8_000.0
            is_crash = (not is_success) and (not is_out_of_bounds)
            is_timeout = False

            rows.append({
                "method": method,
                "scenario": scenario,
                "seed": seed,
                "episode": episode,
                "is_success": int(is_success),
                "is_crash": int(is_crash),
                "is_out_of_bounds": int(is_out_of_bounds),
                "is_timeout": int(is_timeout),
                "return": total_return,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vpp-root", type=str, default="outputs/validate_vpp_5seed")
    parser.add_argument("--no-vpp-root", type=str, default="outputs/experiments/no_vpp_validate_5seed")
    parser.add_argument("--out-dir", type=str, default="docs/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for root, method, label in [
        (args.vpp_root, "vpp_no_pred", "validation_5seed_vpp"),
        (args.no_vpp_root, "no_vpp_direct", "validation_5seed_no_vpp"),
    ]:
        rows = extract(Path(root), method)
        target = out_dir / label
        target.mkdir(parents=True, exist_ok=True)
        out_file = target / "raw_episodes.csv"
        with open(out_file, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "method", "scenario", "seed", "episode",
                "is_success", "is_crash", "is_out_of_bounds", "is_timeout", "return",
            ])
            writer.writeheader()
            writer.writerows(rows)
        successes = sum(r["is_success"] for r in rows)
        print(f"{label}: {len(rows)} episodes, {successes} successes ({successes/len(rows):.1%}) -> {out_file}")


if __name__ == "__main__":
    main()
