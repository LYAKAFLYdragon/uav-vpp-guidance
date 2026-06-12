#!/usr/bin/env python3
"""
Launch Machine 2 tasks on a host with CPU + NVIDIA GPU.

Runs CPU-only seeds in parallel and GPU seeds sequentially.
Training processes are detached from the launcher session so they survive
SSH disconnects, and per-seed logs are line-buffered so crashes are captured.
"""
import argparse
import os
import subprocess
import sys

# Resolve project root relative to this script (scripts/launch_machine2_ebcloud.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = f"PYTHONPATH={ROOT}/src python3 -u"

# CPU tasks: no_pred, cv, ca, default_gains
cpu_tasks = [
    ("D1_no_pred", f"{PYTHON} -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo --config {ROOT}/config/experiment/train_no_prediction_vpp_ppo.yaml --seed {{seed}} --output-dir {ROOT}/outputs/experiments/maneuver_no_pred_s{{seed}} --device cpu"),
    ("D2_cv", f"{PYTHON} -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo --config {ROOT}/config/experiment/train_vpp_ppo_cv.yaml --seed {{seed}} --output-dir {ROOT}/outputs/experiments/maneuver_cv_s{{seed}} --device cpu"),
    ("D3_ca", f"{PYTHON} -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo --config {ROOT}/config/experiment/train_vpp_ppo_ca.yaml --seed {{seed}} --output-dir {ROOT}/outputs/experiments/maneuver_ca_s{{seed}} --device cpu"),
    ("C2_default_gains", f"{PYTHON} -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo --config {ROOT}/config/experiment/train_no_prediction_vpp_ppo.yaml --seed {{seed}} --output-dir {ROOT}/outputs/experiments/default_gains_seed{{seed}} --device cpu"),
]

# GPU tasks: lstm, gru
gpu_tasks = [
    ("D4_lstm", f"{PYTHON} -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo --config {ROOT}/config/experiment/train_vpp_ppo_lstm_frozen.yaml --seed {{seed}} --output-dir {ROOT}/outputs/experiments/maneuver_lstm_s{{seed}} --device cuda"),
    ("D5_gru", f"{PYTHON} -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo --config {ROOT}/config/experiment/train_vpp_ppo_gru_frozen.yaml --seed {{seed}} --output-dir {ROOT}/outputs/experiments/maneuver_gru_s{{seed}} --device cuda"),
]


def _now():
    return subprocess.check_output(["date"]).decode().strip()


def run_cmd(name, cmd, log_dir=f"{ROOT}/logs"):
    """Run a single training command with line-buffered logging and session detach."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name.replace(' ', '_')}.log")
    with open(log_path, "a", buffering=1, encoding="utf-8") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"Started: {_now()}\n\n")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            start_new_session=True,
        )
        proc.wait()
        f.write(f"\nFinished: {_now()}\n")
        f.write(f"Exit code: {proc.returncode}\n")
        f.write("=" * 70 + "\n\n")
    return proc.returncode == 0


def run_cmd_and_wait(args):
    """Helper for ThreadPoolExecutor."""
    return run_cmd(args[0], args[1])


def run_parallel_cpu(tasks, seeds, workers=4):
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for task_name, task_template in tasks:
        for seed in seeds:
            cmd = task_template.format(seed=seed)
            jobs.append((f"{task_name}_s{seed}", cmd))

    print(f"Running {len(jobs)} CPU jobs with {workers} parallel workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_cmd_and_wait, jobs))

    success = sum(results)
    print(f"CPU jobs: {success}/{len(results)} succeeded", flush=True)
    return results


def run_sequential_gpu(tasks, seeds):
    results = []
    for task_name, task_template in tasks:
        for seed in seeds:
            cmd = task_template.format(seed=seed)
            name = f"{task_name}_s{seed}"
            print(f"\nStarting {name}...", flush=True)
            ok = run_cmd(name, cmd)
            results.append(ok)
            print(f"{name}: {'OK' if ok else 'FAIL'}", flush=True)

    success = sum(results)
    print(f"GPU jobs: {success}/{len(results)} succeeded", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Launch Machine 2 experiments.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per experiment.")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel CPU training jobs.")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU tasks.")
    parser.add_argument("--gpu-only", action="store_true", help="Run only GPU tasks.")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    results = {}

    if not args.gpu_only:
        print("=" * 70, flush=True)
        print("PHASE 1: CPU tasks (no_pred, cv, ca, default_gains)", flush=True)
        print("=" * 70, flush=True)
        results["cpu"] = run_parallel_cpu(cpu_tasks, seeds, workers=args.workers)

    if not args.cpu_only:
        print("\n" + "=" * 70, flush=True)
        print("PHASE 2: GPU tasks (lstm, gru)", flush=True)
        print("=" * 70, flush=True)
        results["gpu"] = run_sequential_gpu(gpu_tasks, seeds)

    print("\n" + "=" * 70, flush=True)
    print("MACHINE 2 SUMMARY", flush=True)
    print("=" * 70, flush=True)
    total_ok = sum(sum(v) for v in results.values())
    total = sum(len(v) for v in results.values())
    print(f"Total: {total_ok}/{total} succeeded", flush=True)


if __name__ == "__main__":
    main()
