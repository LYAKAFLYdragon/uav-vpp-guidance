#!/usr/bin/env python3
"""Evaluate nz_step_response task across multiple seeds."""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.nz_step_response_env import NzStepResponseEnv


def merge_config(base: dict, override: dict) -> dict:
    import copy
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_experiment_config(path: str) -> dict:
    cfg = load_yaml_config(path)
    includes = cfg.pop("includes", [])
    merged = {}
    d = os.path.dirname(path)
    for inc in includes:
        fp = os.path.join(d, inc)
        if not os.path.exists(fp):
            fp = os.path.join(d, "..", os.path.basename(inc))
        if os.path.exists(fp):
            merged = merge_config(merged, load_experiment_config(fp))
    return merge_config(merged, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/nz_step_eval/results.json")
    args = parser.parse_args()

    base_path = str(REPO_ROOT / "config" / "experiment" / "compare_flight_control_base.yaml")
    task_path = str(REPO_ROOT / "config" / "experiment" / "task_nz_step_response.yaml")

    base = load_experiment_config(base_path)
    task = load_experiment_config(task_path)
    config = merge_config(base, task)

    # Use bank76 protection profile
    config["low_level_controller"] = {
        "type": "enhanced",
        "Kp_nz": 0.50, "Ki_nz": 0.05, "Kd_nz": 0.10,
        "enable_bank_angle_protection": True,
        "max_bank_rad": 1.3264502315156905,
        "bank_violation_threshold": 25,
        "bank_protection_nz_increment": 0.5,
        "altitude_hold_gain": 0.01,
    }
    config["guidance"] = {
        "mode": "los_rate",
        "post_process": {
            "enabled": True,
            "enable_lift_compensation": True,
            "lift_compensation_factor": 1.0,
            "lift_compensation_min_cos": 0.5,
        },
    }
    config["virtual_point"] = {"enabled": True, "mode": "zero_offset", "action_dim": 4}
    config["policy"] = {"action_dim": 4}
    config["backend"] = "jsbsim"
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = "jsbsim"
    config["env"]["use_jsbsim"] = True

    results = []
    t0 = time.time()

    for seed in range(args.seeds):
        try:
            env = NzStepResponseEnv(config)
            env.reset(seed=seed)

            nz_cmd_hist = []
            nz_actual_hist = []
            terminated, truncated = False, False
            steps = 0

            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step()
                nz_cmd_hist.append(float(info.get("nz_cmd", 1.0)))
                nz_actual_hist.append(float(info.get("actual_nz", 1.0)))
                steps += 1
                if steps >= env.max_steps:
                    break

            summary = env.get_step_response_summary()
            is_crash = bool(info.get("is_crash"))
            results.append({
                "seed": seed,
                "crashed": is_crash,
                "n_steps": steps,
                "phase_metrics": summary["phase_metrics"],
                "nz_cmd_hist": summary["nz_cmd_hist"],
                "nz_actual_hist": summary["actual_nz_hist"],
            })
            print(f"  seed={seed}: crash={is_crash} steps={steps}")
        except Exception:
            traceback.print_exc()
            results.append({"seed": seed, "crashed": True, "error": traceback.format_exc()})

    elapsed = time.time() - t0
    crashes = sum(1 for r in results if r.get("crashed"))

    # Aggregate phase metrics
    phase_keys = ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]
    agg = {}
    for pk in phase_keys:
        delays = []
        overshoots = []
        for r in results:
            pm = r.get("phase_metrics", {}).get(pk)
            if pm:
                delays.append(pm.get("recovery_delay_steps", 0))
                overshoots.append(pm.get("overshoot_pct", 0))
        agg[pk] = {
            "mean_recovery_delay": float(np.mean(delays)) if delays else None,
            "mean_overshoot_pct": float(np.mean(overshoots)) if overshoots else None,
        }

    output = {
        "metadata": {"seeds": args.seeds, "elapsed_s": round(elapsed, 1)},
        "summary": {
            "total": args.seeds,
            "crashes": crashes,
            "crash_rate": crashes / args.seeds,
        },
        "phase_aggregates": agg,
        "acceptance": {
            "crash_eq_0": crashes == 0,
            "recovery_delay_lt_5": all(
                (v.get("mean_recovery_delay") or 999) < 5
                for v in agg.values() if v.get("mean_recovery_delay") is not None
            ),
        },
        "per_seed": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone in {elapsed:.0f}s. Crashes: {crashes}/{args.seeds}")
    for pk, v in agg.items():
        print(f"  {pk}: delay={v['mean_recovery_delay']:.1f}steps overshoot={v['mean_overshoot_pct']:.1f}%")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
