#!/usr/bin/env python3
"""Train baseline and constrained models with 10 seeds for statistical power."""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "logs", "10seed_train")
os.makedirs(LOGDIR, exist_ok=True)

jobs = []
# Baseline (no domain rand, standard config)
for seed in range(10):
    out_dir = f"outputs/experiments/baseline_10seed_s{seed}"
    log_path = os.path.join(LOGDIR, f"baseline_s{seed}.log")
    cmd = [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "--device", "cpu", "--seed", str(seed),
        "--output-dir", out_dir,
    ]
    jobs.append((f"baseline_s{seed}", cmd, log_path))

# Constrained (dynamics-aware, max_heading_rate=0.2)
for seed in range(10):
    out_dir = f"outputs/experiments/constrained_10seed_s{seed}"
    log_path = os.path.join(LOGDIR, f"constrained_s{seed}.log")
    cmd = [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/stage6f5_feasible_geometry_constrained.yaml",
        "--device", "cpu", "--seed", str(seed),
        "--output-dir", out_dir,
    ]
    jobs.append((f"constrained_s{seed}", cmd, log_path))

processes = []
for name, cmd, log_path in jobs:
    f = open(log_path, "w")
    print(f"[{time.strftime('%H:%M:%S')}] Launching {name}")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    processes.append((name, p, f))

print(f"[{time.strftime('%H:%M:%S')}] All {len(processes)} jobs launched. Waiting...")

for name, p, f in processes:
    p.wait()
    f.close()
    status = "OK" if p.returncode == 0 else f"FAILED({p.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {status}")

print(f"[{time.strftime('%H:%M:%S')}] All 10-seed training complete.")
