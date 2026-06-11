#!/usr/bin/env python3
"""
Collect results from multiple distributed machines into a central location.

Assumes each machine has produced:
- outputs/experiments/<exp_name>/checkpoints/best.pt
- outputs/experiments/<exp_name>/eval_*.json or eval_*.csv
- outputs/distributed_runs/<exp_id>_run_manifest.json
- docs/results/<group>/summary.md and raw_results.json

Usage:
    python scripts/collect_distributed_results.py \
        --machine-dirs /path/to/machine1 /path/to/machine2 /path/to/machine3 \
        --output-dir outputs/aggregated
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def collect_experiments(machine_dirs, output_dir):
    """Copy experiment outputs from all machines into output_dir/experiments/."""
    exp_dst = Path(output_dir) / "experiments"
    exp_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for machine_dir in machine_dirs:
        machine_path = Path(machine_dir)
        exp_src = machine_path / "outputs" / "experiments"
        if not exp_src.exists():
            print(f"[WARN] No experiments dir in {machine_dir}")
            continue

        for exp_dir in exp_src.iterdir():
            if not exp_dir.is_dir():
                continue
            dst = exp_dst / exp_dir.name
            if dst.exists():
                print(f"[SKIP] {exp_dir.name} already exists")
                skipped += 1
                continue
            print(f"[COPY] {exp_dir.name} from {machine_dir}")
            shutil.copytree(exp_dir, dst)
            copied += 1

    print(f"\nCopied {copied} experiment dirs, skipped {skipped}")
    return copied, skipped


def collect_results_docs(machine_dirs, output_dir):
    """Merge docs/results from all machines into output_dir/results/."""
    results_dst = Path(output_dir) / "results"
    results_dst.mkdir(parents=True, exist_ok=True)

    for machine_dir in machine_dirs:
        machine_path = Path(machine_dir)
        results_src = machine_path / "docs" / "results"
        if not results_src.exists():
            continue

        for subdir in results_src.iterdir():
            if not subdir.is_dir():
                continue
            dst = results_dst / subdir.name
            if dst.exists():
                # Merge files; if same filename exists, rename with machine suffix
                for f in subdir.iterdir():
                    target = dst / f.name
                    if target.exists():
                        target = dst / f"{f.stem}_{Path(machine_dir).name}{f.suffix}"
                    if f.is_file():
                        shutil.copy2(f, target)
                    elif f.is_dir():
                        shutil.copytree(f, target, dirs_exist_ok=True)
            else:
                shutil.copytree(subdir, dst)

    print(f"Collected docs/results into {results_dst}")


def collect_run_manifests(machine_dirs, output_dir):
    """Copy run manifests from outputs/distributed_runs."""
    manifests_dst = Path(output_dir) / "run_manifests"
    manifests_dst.mkdir(parents=True, exist_ok=True)

    for machine_dir in machine_dirs:
        machine_path = Path(machine_dir)
        runs_src = machine_path / "outputs" / "distributed_runs"
        if not runs_src.exists():
            continue
        for f in runs_src.glob("*.json"):
            dst = manifests_dst / f"{Path(machine_dir).name}_{f.name}"
            shutil.copy2(f, dst)

    print(f"Collected run manifests into {manifests_dst}")


def write_collection_report(machine_dirs, output_dir, copied, skipped):
    report = {
        "machine_dirs": [str(d) for d in machine_dirs],
        "output_dir": str(output_dir),
        "experiments_copied": copied,
        "experiments_skipped": skipped,
    }
    report_path = Path(output_dir) / "collection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Collection report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect distributed experiment results")
    parser.add_argument("--machine-dirs", nargs="+", required=True, help="Directories from each machine")
    parser.add_argument("--output-dir", type=str, required=True, help="Central aggregation directory")
    args = parser.parse_args()

    print("=" * 70)
    print("COLLECTING DISTRIBUTED RESULTS")
    print("=" * 70)

    copied, skipped = collect_experiments(args.machine_dirs, args.output_dir)
    collect_results_docs(args.machine_dirs, args.output_dir)
    collect_run_manifests(args.machine_dirs, args.output_dir)
    write_collection_report(args.machine_dirs, args.output_dir, copied, skipped)

    print("\nCollection complete. Run aggregation scripts next.")
    print(f"  Central directory: {args.output_dir}")


if __name__ == "__main__":
    main()
