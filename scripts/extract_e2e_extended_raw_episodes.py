#!/usr/bin/env python3
"""Build raw_episodes.csv for the E2E extended-training study."""

import argparse
import csv
from pathlib import Path

import numpy as np


SCENARIO_MAP = {0: "favorable", 1: "neutral", 2: "disadvantage"}


def extract(root: Path, method: str = "end_to_end"):
    rows = []
    for budget_dir in sorted(root.glob("e2e_steps_*")):
        budget = int(budget_dir.name.split("_")[-1])
        for seed_dir in sorted(budget_dir.glob("seed_*")):
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
                total_return = float(np.sum([float(r["reward"]) for r in step_rows]))
                final_range = float(step_rows[-1]["range_m"])
                is_success = total_return > 0.0
                is_out_of_bounds = (not is_success) and final_range > 8_000.0
                is_crash = (not is_success) and (not is_out_of_bounds)
                rows.append({
                    "method": method,
                    "budget": budget,
                    "scenario": scenario,
                    "seed": seed,
                    "episode": episode,
                    "is_success": int(is_success),
                    "is_crash": int(is_crash),
                    "is_out_of_bounds": int(is_out_of_bounds),
                    "is_timeout": 0,
                    "return": total_return,
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/validate_e2e_extended")
    parser.add_argument("--out-dir", type=str, default="docs/results/e2e_extended_aligned")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = extract(Path(args.root))
    out_file = out_dir / "raw_episodes.csv"
    with open(out_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "method", "budget", "scenario", "seed", "episode",
            "is_success", "is_crash", "is_out_of_bounds", "is_timeout", "return",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.groupby(["budget", "seed"])["is_success"].mean().reset_index().pivot(index="budget", columns="seed", values="is_success"))
    print(f"\nWrote {len(rows)} episodes to {out_file}")


if __name__ == "__main__":
    main()
