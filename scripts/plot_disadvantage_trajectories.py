#!/usr/bin/env python3
"""Plot median disadvantage trajectory geometry for Dense vs Sparse-Gaussian."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"matplotlib unavailable: {exc}")
    sys.exit(1)


def load_series(trajectories_dir: Path, key: str, max_len: int = 400):
    rows_by_step = {}
    count_by_step = {}
    for traj_path in sorted(trajectories_dir.glob("*.csv")):
        with open(traj_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["step"])
                if step >= max_len:
                    continue
                val = float(row[key])
                if not np.isfinite(val):
                    continue
                rows_by_step.setdefault(step, []).append(val)
                count_by_step[step] = count_by_step.get(step, 0) + 1
    steps = sorted(rows_by_step.keys())
    median = [float(np.median(rows_by_step[s])) for s in steps]
    p25 = [float(np.percentile(rows_by_step[s], 25)) for s in steps]
    p75 = [float(np.percentile(rows_by_step[s], 75)) for s in steps]
    n = [count_by_step[s] for s in steps]
    return steps, median, p25, p75, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="outputs/disadvantage_dense_vs_sparse_analysis")
    parser.add_argument("--output", default="outputs/disadvantage_dense_vs_sparse_analysis/trajectory_timeseries.png")
    args = parser.parse_args()

    base = Path(args.analysis_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"dense": "black", "sparse_gaussian": "green"}

    for name, label in [("dense", "Dense"), ("sparse_gaussian", "Sparse-Gaussian")]:
        traj_dir = base / "trajectories" / name
        if not traj_dir.exists():
            print(f"Trajectory dir not found: {traj_dir}")
            continue
        for ax, key, title, ylabel in [
            (axes[0, 0], "range_m", "Range to target", "range_m"),
            (axes[0, 1], "ata_deg", "Antenna Train Angle", "deg"),
            (axes[1, 0], "altitude_m", "Ego altitude", "m"),
            (axes[1, 1], "relative_speed_mps", "Ego speed", "m/s"),
        ]:
            steps, median, p25, p75, n = load_series(traj_dir, key)
            time_s = [s * 0.2 for s in steps]
            ax.plot(time_s, median, label=label, color=colors[name])
            ax.fill_between(time_s, p25, p75, alpha=0.2, color=colors[name])
            ax.set_title(title)
            ax.set_xlabel("time_s")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved time-series plot to {args.output}")


if __name__ == "__main__":
    main()
