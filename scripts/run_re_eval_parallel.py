#!/usr/bin/env python3
"""Launch scenario-stratified re-evaluation across all 50 checkpoints in parallel.

Splits checkpoints into N concurrent batches to maximize CPU/GPU utilization
on high-core hosts.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path


def run_batch(args):
    batch_id, checkpoints, output_dir, n_seeds, episodes_per_scenario, device = args
    script = Path(__file__).parent / "re_evaluate_stratified.py"
    log_file = Path(output_dir) / f"batch_{batch_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--output-dir", str(Path(output_dir) / f"batch_{batch_id}"),
        "--n-seeds", str(n_seeds),
        "--episodes-per-scenario", str(episodes_per_scenario),
        "--device", device,
    ]
    for cp in checkpoints:
        cmd.extend(["--checkpoint", cp])

    with open(log_file, "w") as f:
        f.write(f"Batch {batch_id}: {len(checkpoints)} checkpoints\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\nExit code: {proc.returncode}\n")

    return batch_id, proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent evaluation batches")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/jsbsim_re_eval_stratified")
    args = parser.parse_args()

    patterns = [
        "outputs/jsbsim_10seed_matrix/vpp_s*/checkpoints/best.pt",
        "outputs/jsbsim_10seed_matrix/no_vpp_s*/checkpoints/best.pt",
        "outputs/jsbsim_10seed_matrix/e2e_s*/checkpoints/best.pt",
        "outputs/jsbsim_ablations/ablation_no_safety_penalty_jsbsim_s*/checkpoints/best.pt",
        "outputs/jsbsim_ablations/ablation_with_gain_obs_jsbsim_s*/checkpoints/best.pt",
    ]

    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))
    checkpoints = sorted(set(checkpoints))
    print(f"Total checkpoints: {len(checkpoints)}")

    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    # Split into batches
    n_batches = min(args.workers, len(checkpoints))
    batch_size = (len(checkpoints) + n_batches - 1) // n_batches
    batches = [
        checkpoints[i:i + batch_size]
        for i in range(0, len(checkpoints), batch_size)
    ]
    print(f"Split into {len(batches)} batches, ~{batch_size} checkpoints each")

    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()

    pool_args = [
        (i, batch, args.output_dir, args.n_seeds, args.episodes_per_scenario, args.device)
        for i, batch in enumerate(batches)
    ]

    with Pool(n_batches) as pool:
        results = pool.map(run_batch, pool_args)

    elapsed = time.time() - start
    print(f"\nAll batches finished in {elapsed/3600:.2f}h")
    for batch_id, rc in results:
        print(f"  Batch {batch_id}: exit code {rc}")


if __name__ == "__main__":
    main()
