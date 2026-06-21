"""Diagnose VPP Stage 3 training issues by comparing logs across stages/conditions."""
import os
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def load_pursuer_log(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "step" else int(v) for k, v in row.items()})
    return rows


def load_target_log(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "step" else int(v) for k, v in row.items()})
    return rows


def summarize(rows, keys):
    if not rows:
        return {}
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        out[k] = {
            "first": vals[0],
            "last": vals[-1],
            "min": min(vals),
            "max": max(vals),
            "mean": np.mean(vals),
        }
    return out


def print_summary(name, rows, is_target=False):
    print(f"\n=== {name} ===")
    if not rows:
        print("  (no data)")
        return
    if is_target:
        keys = ["policy_loss", "value_loss", "entropy", "eval_return", "eval_survival_rate", "eval_capture_rate", "eval_crash_rate", "eval_timeout_rate"]
    else:
        keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "capture_rate", "mean_return"]
    stats = summarize(rows, keys)
    for k, v in stats.items():
        print(f"  {k:20s}: first={v['first']:.4f}, last={v['last']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}, mean={v['mean']:.4f}")


def main():
    base = Path("outputs/adversarial_curriculum_pilot")
    conditions = {
        "vpp_s2_pursuer": base / "vpp_full_gate25_s0/stage2/pursuer/logs/adversarial_training_log.csv",
        "vpp_s3_pursuer": base / "vpp_full_gate25_s0/stage3/pursuer/logs/adversarial_training_log.csv",
        "vpp_s2_target": base / "vpp_full_gate25_s0/stage2/target/logs/target_training_log.csv",
        "vpp_s3_target": base / "vpp_full_gate25_s0/stage3/target/logs/target_training_log.csv",
        "no_vpp_s2_pursuer": base / "no_vpp_full_gate10_s0/stage2/pursuer/logs/adversarial_training_log.csv",
        "no_vpp_s3_pursuer": base / "no_vpp_full_gate10_s0/stage3/pursuer/logs/adversarial_training_log.csv",
        "no_vpp_s2_target": base / "no_vpp_full_gate10_s0/stage2/target/logs/target_training_log.csv",
        "no_vpp_s3_target": base / "no_vpp_full_gate10_s0/stage3/target/logs/target_training_log.csv",
    }

    data = {}
    for name, path in conditions.items():
        if path.exists():
            data[name] = load_pursuer_log(path) if "pursuer" in name else load_target_log(path)
        else:
            print(f"Missing: {path}")
            data[name] = []

    for name, rows in data.items():
        print_summary(name, rows, is_target="target" in name)

    # Compute target eval survival rate progression
    print("\n=== Target eval survival rate progression ===")
    for cond in ["vpp_s2_target", "vpp_s3_target", "no_vpp_s2_target", "no_vpp_s3_target"]:
        rows = data[cond]
        if not rows:
            continue
        print(f"\n{cond}")
        for r in rows[::len(rows)//10 or 1]:
            print(f"  step={r['step']:6d}: survival={r.get('eval_survival_rate', 0):.3f}, capture={r.get('eval_capture_rate', 0):.3f}, crash={r.get('eval_crash_rate', 0):.3f}, timeout={r.get('eval_timeout_rate', 0):.3f}")

    # Compute pursuer capture rate progression
    print("\n=== Pursuer capture rate progression ===")
    for cond in ["vpp_s2_pursuer", "vpp_s3_pursuer", "no_vpp_s2_pursuer", "no_vpp_s3_pursuer"]:
        rows = data[cond]
        if not rows:
            continue
        print(f"\n{cond}")
        for r in rows[::len(rows)//10 or 1]:
            print(f"  step={r['step']:6d}: capture={r.get('capture_rate', 0):.3f}, return={r.get('mean_return', 0):.1f}, value_loss={r.get('value_loss', 0):.1f}, entropy={r.get('entropy', 0):.3f}")


if __name__ == "__main__":
    main()
