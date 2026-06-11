#!/usr/bin/env python3
"""Re-run domain randomization training with corrected curriculum (3 seeds)."""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "logs", "domain_rand_v2")
os.makedirs(LOGDIR, exist_ok=True)

jobs = []
for seed in [0, 1, 2]:
    out_dir = f"outputs/experiments/no_prediction_vpp_ppo_domain_rand_v2_s{seed}"
    log_path = os.path.join(LOGDIR, f"s{seed}.log")
    cmd = [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml",
        "--device", "cpu", "--seed", str(seed),
        "--output-dir", out_dir,
    ]
    jobs.append((f"s{seed}", cmd, log_path))

processes = []
for name, cmd, log_path in jobs:
    f = open(log_path, "w")
    print(f"[{time.strftime('%H:%M:%S')}] Launching {name} -> {log_path}")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    processes.append((name, p, f))

for name, p, f in processes:
    p.wait()
    f.close()
    status = "OK" if p.returncode == 0 else f"FAILED({p.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {status}")
