#!/usr/bin/env python3
"""Orchestrate crossing breakthrough experiments (3 methods x 3 seeds)."""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "logs", "crossing_train")
os.makedirs(LOGDIR, exist_ok=True)

METHODS = [
    ("curriculum", "config/experiment/train_curriculum_ppo.yaml", "scripts/train_curriculum_ppo.py"),
    ("constrained", "config/experiment/stage6f5_feasible_geometry_constrained.yaml",
     "uav_vpp_guidance.training.train_no_prediction_vpp_ppo"),
    ("hybrid", "config/experiment/train_hybrid_mode_switch.yaml",
     "uav_vpp_guidance.training.train_no_prediction_vpp_ppo"),
]

jobs = []
for method, cfg, module in METHODS:
    for seed in [0, 1, 2]:
        out_dir = f"outputs/experiments/crossing_{method}_s{seed}"
        log_path = os.path.join(LOGDIR, f"{method}_s{seed}.log")
        if module.startswith("scripts/"):
            cmd = [
                sys.executable, "-u", "-m", module.replace("/", ".").replace(".py", ""),
                "--config", cfg, "--device", "cpu", "--seed", str(seed),
                "--output-dir", out_dir,
            ]
        else:
            cmd = [
                sys.executable, "-u", "-m", module,
                "--config", cfg, "--device", "cpu", "--seed", str(seed),
                "--output-dir", out_dir,
            ]
        jobs.append((f"{method}_s{seed}", cmd, log_path))

processes = []
for name, cmd, log_path in jobs:
    f = open(log_path, "w")
    print(f"[{time.strftime('%H:%M:%S')}] Launching {name} -> {log_path}")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    processes.append((name, p, f))

print(f"[{time.strftime('%H:%M:%S')}] All {len(processes)} jobs launched. Waiting...")

for name, p, f in processes:
    p.wait()
    f.close()
    status = "OK" if p.returncode == 0 else f"FAILED({p.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {status}")

print(f"[{time.strftime('%H:%M:%S')}] All crossing experiments complete.")
