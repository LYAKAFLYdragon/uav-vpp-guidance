#!/usr/bin/env python3
"""Train remaining seeds (2-9) for baseline and constrained models."""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "logs", "10seed_train")
os.makedirs(LOGDIR, exist_ok=True)

for seed in range(2, 10):
    print(f"\n[{time.strftime('%H:%M:%S')}] === Seed {seed} ===", flush=True)
    
    # Baseline
    baseline_out = f"outputs/experiments/baseline_10seed_s{seed}"
    baseline_log = os.path.join(LOGDIR, f"baseline_s{seed}.log")
    baseline_cmd = [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "--device", "cpu", "--seed", str(seed),
        "--output-dir", baseline_out,
    ]
    
    # Constrained
    constrained_out = f"outputs/experiments/constrained_10seed_s{seed}"
    constrained_log = os.path.join(LOGDIR, f"constrained_s{seed}.log")
    constrained_cmd = [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/stage6f5_feasible_geometry_constrained.yaml",
        "--device", "cpu", "--seed", str(seed),
        "--output-dir", constrained_out,
    ]
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting baseline_s{seed}...", flush=True)
    f1 = open(baseline_log, "w")
    p1 = subprocess.Popen(baseline_cmd, cwd=ROOT, stdout=f1, stderr=subprocess.STDOUT)
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting constrained_s{seed}...", flush=True)
    f2 = open(constrained_log, "w")
    p2 = subprocess.Popen(constrained_cmd, cwd=ROOT, stdout=f2, stderr=subprocess.STDOUT)
    
    p1.wait()
    p2.wait()
    f1.close()
    f2.close()
    
    s1 = "OK" if p1.returncode == 0 else f"FAIL({p1.returncode})"
    s2 = "OK" if p2.returncode == 0 else f"FAIL({p2.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] baseline_s{seed}: {s1}, constrained_s{seed}: {s2}", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] All remaining seeds (2-9) complete.", flush=True)
