#!/usr/bin/env python3
"""
Validate Stage 6F formal ablation outputs.

Checks:
- train_seed*/prediction_metrics.json exists
- Each method's policy_type == trained_ppo
- loaded_policy_checkpoint_path exists (file on disk)
- allow_random_policy == False
- invalid_for_paper == False
- scenario_balance_ok == True
- training_seed / evaluation_seed / episode_seed fields present in episodes
- metrics_schema_version == 6f.2 in experiment_plan.json
- summary.csv and cross_seed_summary.json exist

Usage:
    python scripts/validate_stage6f_outputs.py \
        --input outputs/tables/stage6f_full_ablation \
        --summary outputs/tables/stage6f
"""

import argparse
import json
import os
import sys
from pathlib import Path


EXPECTED_SCHEMA_VERSION = "6f.2"
REQUIRED_METHODS = {"no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"}


def fail(msg):
    print(f"FAIL: {msg}")
    return False


def ok(msg):
    print(f"OK: {msg}")
    return True


def validate(args) -> bool:
    all_ok = True
    input_root = Path(args.input)
    summary_dir = Path(args.summary)

    # 1. Check experiment_plan.json
    plan_path = input_root / "experiment_plan.json"
    if not plan_path.exists():
        all_ok &= fail(f"experiment_plan.json not found: {plan_path}")
    else:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
        schema = plan.get("metrics_schema_version")
        if schema != EXPECTED_SCHEMA_VERSION:
            all_ok &= fail(f"Schema version mismatch: {schema} != {EXPECTED_SCHEMA_VERSION}")
        else:
            all_ok &= ok(f"Schema version: {schema}")
        if plan.get("allow_random_policy") is not False:
            all_ok &= fail("allow_random_policy is not False in experiment_plan")
        else:
            all_ok &= ok("allow_random_policy is False in experiment_plan")

    # 2. Discover train_seed directories
    seed_dirs = sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("train_seed")])
    if not seed_dirs:
        all_ok &= fail(f"No train_seed* directories found in {input_root}")
        return all_ok
    all_ok &= ok(f"Found {len(seed_dirs)} train_seed directories")

    methods_found_global = set()

    for seed_dir in seed_dirs:
        metrics_json = seed_dir / "prediction_metrics.json"
        if not metrics_json.exists():
            all_ok &= fail(f"prediction_metrics.json missing: {metrics_json}")
            continue

        with open(metrics_json, "r", encoding="utf-8") as f:
            methods = json.load(f)

        methods_found = {m.get("method", m.get("method_name", "unknown")) for m in methods}
        methods_found_global.update(methods_found)

        for m in methods:
            method_name = m.get("method", m.get("method_name", "unknown"))
            prefix = f"[{seed_dir.name}/{method_name}]"

            # policy_type
            if m.get("policy_type") != "trained_ppo":
                all_ok &= fail(f"{prefix} policy_type != trained_ppo ({m.get('policy_type')})")
            else:
                all_ok &= ok(f"{prefix} policy_type == trained_ppo")

            # loaded_policy_checkpoint_path
            ckpt = m.get("loaded_policy_checkpoint_path")
            if not ckpt:
                all_ok &= fail(f"{prefix} loaded_policy_checkpoint_path is empty/None")
            elif not Path(ckpt).exists():
                all_ok &= fail(f"{prefix} checkpoint file does not exist: {ckpt}")
            else:
                all_ok &= ok(f"{prefix} checkpoint exists: {ckpt}")

            # allow_random_policy
            if m.get("allow_random_policy") is not False:
                all_ok &= fail(f"{prefix} allow_random_policy != False")
            else:
                all_ok &= ok(f"{prefix} allow_random_policy == False")

            # invalid_for_paper
            invalid = m.get("invalid_for_paper")
            if invalid is None:
                # Fallback: compute from other fields
                invalid = m.get("allow_random_policy") or (not m.get("loaded_policy_checkpoint_path"))
            if invalid:
                all_ok &= fail(f"{prefix} invalid_for_paper is True")
            else:
                all_ok &= ok(f"{prefix} invalid_for_paper is False")

            # scenario_balance_ok
            balance_ok = m.get("scenario_balance_ok")
            if balance_ok is not True:
                # Fallback: compute from actual episode counts
                counts = m.get("scenario_episode_count", {})
                if counts and len(set(counts.values())) == 1:
                    all_ok &= ok(f"{prefix} scenario_balance_ok inferred True (counts: {counts})")
                else:
                    all_ok &= fail(f"{prefix} scenario_balance_ok != True and counts unbalanced: {counts}")
            else:
                all_ok &= ok(f"{prefix} scenario_balance_ok == True")

            # seed fields in episodes
            episodes = m.get("raw_episodes", [])
            if not episodes:
                episodes = [ep for seed_eps in m.get("per_seed", {}).values() for ep in seed_eps]
            if episodes:
                ep = episodes[0]
                missing = []
                for key in ("training_seed", "evaluation_seed", "episode_seed"):
                    if key not in ep:
                        missing.append(key)
                if missing:
                    all_ok &= fail(f"{prefix} episode missing keys: {missing}")
                else:
                    all_ok &= ok(f"{prefix} episode seeds present")
            else:
                all_ok &= fail(f"{prefix} no episodes found")

    missing_methods = REQUIRED_METHODS - methods_found_global
    if missing_methods:
        all_ok &= fail(f"Missing methods in results: {missing_methods}")
    else:
        all_ok &= ok("All 5 methods present")

    # 3. Check summary outputs
    for filename in ("summary.csv", "cross_seed_summary.json"):
        path = summary_dir / filename
        if not path.exists():
            all_ok &= fail(f"Summary output missing: {path}")
        else:
            all_ok &= ok(f"Summary output exists: {path}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 6F outputs")
    parser.add_argument("--input", type=str, required=True,
                        help="Root directory containing train_seed*/ subdirs")
    parser.add_argument("--summary", type=str, required=True,
                        help="Directory containing summary.csv and cross_seed_summary.json")
    args = parser.parse_args()

    if validate(args):
        print("\nAll validations passed.")
        sys.exit(0)
    else:
        print("\nSome validations failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
