#!/usr/bin/env python3
"""Train one seed at a time to avoid memory exhaustion."""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_one(config_path, seed, output_dir):
    cmd = [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", config_path,
        "--device", "cpu", "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    print(f"[{time.strftime('%H:%M:%S')}] Training {output_dir}...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {output_dir}: {status} ({elapsed:.0f}s)", flush=True)
    return result.returncode == 0

# Train baseline seeds 3-5 one at a time
for seed in [3, 4, 5]:
    ok = train_one("config/experiment/train_no_prediction_vpp_ppo.yaml", seed, f"outputs/experiments/baseline_10seed_s{seed}")
    if not ok:
        print(f"WARNING: baseline_s{seed} failed, continuing...", flush=True)

# Train constrained seeds 3-5 one at a time
for seed in [3, 4, 5]:
    ok = train_one("config/experiment/stage6f5_feasible_geometry_constrained.yaml", seed, f"outputs/experiments/constrained_10seed_s{seed}")
    if not ok:
        print(f"WARNING: constrained_s{seed} failed, continuing...", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] All single-seed training complete.", flush=True)
