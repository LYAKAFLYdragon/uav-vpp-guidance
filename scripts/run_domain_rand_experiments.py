#!/usr/bin/env python3
"""
Orchestrate domain randomization vs control group training (6 parallel jobs).
"""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "logs", "domain_rand_train")
os.makedirs(LOGDIR, exist_ok=True)

jobs = [
    # Domain randomization (3 seeds)
    ("domain_rand_s0", [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml",
        "--device", "cpu", "--seed", "0",
        "--output-dir", "outputs/experiments/no_prediction_vpp_ppo_domain_rand_s0",
    ]),
    ("domain_rand_s1", [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml",
        "--device", "cpu", "--seed", "1",
        "--output-dir", "outputs/experiments/no_prediction_vpp_ppo_domain_rand_s1",
    ]),
    ("domain_rand_s2", [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml",
        "--device", "cpu", "--seed", "2",
        "--output-dir", "outputs/experiments/no_prediction_vpp_ppo_domain_rand_s2",
    ]),
    # Control group (3 seeds)
    ("control_s0", [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "--device", "cpu", "--seed", "0",
        "--output-dir", "outputs/experiments/no_prediction_vpp_ppo_control_s0",
    ]),
    ("control_s1", [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "--device", "cpu", "--seed", "1",
        "--output-dir", "outputs/experiments/no_prediction_vpp_ppo_control_s1",
    ]),
    ("control_s2", [
        sys.executable, "-u", "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "--device", "cpu", "--seed", "2",
        "--output-dir", "outputs/experiments/no_prediction_vpp_ppo_control_s2",
    ]),
]

processes = []
for name, cmd in jobs:
    log_path = os.path.join(LOGDIR, f"{name}.log")
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

print(f"[{time.strftime('%H:%M:%S')}] All jobs complete.")
