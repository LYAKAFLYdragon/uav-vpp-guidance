#!/usr/bin/env python3
"""Fixed-offset ablation for the No-VPP architecture on JSBSim.

The learned VPP offset on the disadvantage scenario has a mean longitudinal
shift of ~2987~m. This ablation tests whether a constant offset can replicate
that behavior, and whether smaller lead offsets (e.g. +500~m) can rescue the
No-VPP timeout/crash pattern.

Usage:
    python scripts/run_fixed_offset_ablation.py --episodes-per-scenario 20
"""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.evaluation.statistical_comparison import bootstrap_success_rate_ci
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.virtual_point.fixed_offset_guidance import FixedOffsetGuidance


SCENARIO_NAMES = ["favorable", "neutral", "disadvantage", "challenging"]
CONFIG_PATH = _PROJECT_ROOT / "config/experiment/train_no_vpp_ppo.yaml"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs/fixed_offset_ablation"


# Offsets to test: [longitudinal, lateral, vertical] in meters.
OFFSET_CONDITIONS = {
    "zero_offset": [0.0, 0.0, 0.0],
    "lead_500m": [500.0, 0.0, 0.0],
    "lead_1000m": [1000.0, 0.0, 0.0],
    "lead_2987m": [2987.0, 0.0, 0.0],
}


class DummyAgent:
    """Agent that always returns a zero action (ignored by FixedOffsetGuidance)."""

    def __init__(self, action_dim=3):
        self.action_dim = action_dim

    def get_deterministic_action(self, _obs):
        return np.zeros(self.action_dim, dtype=np.float64)


def _resolve_config(config_path: Path) -> dict:
    cfg = load_yaml_config(str(config_path))
    includes = cfg.pop("includes", [])
    merged: dict = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, cfg)


def _make_env_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["backend"] = "jsbsim"
    if "env" not in cfg:
        cfg["env"] = {}
    cfg["env"]["backend"] = "jsbsim"
    cfg["env"]["use_jsbsim"] = True
    if "guidance" not in cfg:
        cfg["guidance"] = {}
    if "mode_switch" not in cfg["guidance"]:
        cfg["guidance"]["mode_switch"] = {}
    cfg["guidance"]["mode_switch"]["enabled"] = False
    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    cfg["evaluation"]["domain_rand_scale"] = 0.0
    return cfg


def evaluate_offset(env, agent, offset_name, offset_m, episodes_per_scenario, output_dir):
    """Evaluate one fixed offset across all scenarios."""
    print(f"\n[Offset: {offset_name} = {offset_m}]")
    env.virtual_point_generator = FixedOffsetGuidance(offset_m=offset_m)
    episodes = []
    scenarios = {name: env.config["scenarios"][name] for name in SCENARIO_NAMES}

    for scen_idx, (scen_name, scenario) in enumerate(scenarios.items()):
        for ep_idx in range(episodes_per_scenario):
            episode_seed = scen_idx * 100000 + ep_idx * 1000
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=env.config,
                scenario=scenario,
                seed=episode_seed,
                save_trajectory=False,
                method_name=offset_name,
            )
            result["method"] = offset_name
            result["scenario"] = scen_name
            result["eval_seed"] = ep_idx
            result["training_seed"] = 0
            episodes.append(result)
    return episodes


def _safe_mean(values):
    clean = [v for v in values if np.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


def aggregate(episodes):
    by_scenario = {}
    for ep in episodes:
        by_scenario.setdefault(ep["scenario"], []).append(ep)
    result = {}
    for scen_name, eps in by_scenario.items():
        outcomes = [1 if e["is_success"] else 0 for e in eps]
        sr, sr_lo, sr_hi = bootstrap_success_rate_ci(outcomes, n_bootstrap=2000)
        result[scen_name] = {
            "n_episodes": len(eps),
            "success_rate": sr,
            "success_rate_ci_lo": sr_lo,
            "success_rate_ci_hi": sr_hi,
            "crash_rate": _safe_mean([1.0 if e["is_crash"] else 0.0 for e in eps]),
            "timeout_rate": _safe_mean([1.0 if e["is_timeout"] else 0.0 for e in eps]),
            "mean_return": _safe_mean([e["return"] for e in eps]),
            "mean_final_range_m": _safe_mean([e["final_range_m"] for e in eps]),
            "mean_virtual_point_shift_m": _safe_mean(
                [e.get("mean_virtual_point_shift_m", float("nan")) for e in eps]
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-scenario", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _resolve_config(CONFIG_PATH)
    env_cfg = _make_env_config(config)
    env = CloseRangeTrackingEnv(env_cfg)
    agent = DummyAgent(action_dim=env_cfg.get("policy", {}).get("action_dim", 3))

    all_episodes = []
    per_offset = {}
    for offset_name, offset_m in OFFSET_CONDITIONS.items():
        episodes = evaluate_offset(
            env, agent, offset_name, offset_m, args.episodes_per_scenario, output_dir
        )
        all_episodes.extend(episodes)
        per_offset[offset_name] = aggregate(episodes)

    env.close()

    # Write raw episodes CSV
    keys = [
        "method", "scenario", "eval_seed", "return", "length",
        "is_success", "is_crash", "is_timeout", "is_out_of_bounds",
        "final_range_m", "final_ata_deg", "min_range_m", "min_ata_deg",
        "mean_virtual_point_shift_m", "reason",
    ]
    raw_path = output_dir / "raw_episodes.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for ep in all_episodes:
            writer.writerow({k: ep.get(k, "") for k in keys})
    print(f"[CSV] {raw_path}")

    # Write summary JSON
    summary = {
        "meta": {
            "episodes_per_scenario": args.episodes_per_scenario,
            "backend": "jsbsim",
            "offsets": OFFSET_CONDITIONS,
        },
        "per_offset": per_offset,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[JSON] {summary_path}")

    # Print quick table
    print("\n=== Fixed-Offset Ablation Summary ===")
    print(f"{'Offset':<15} {'Scenario':<12} {'SR %':>6} {'Crash %':>7} {'Timeout %':>9}")
    for offset_name, agg in per_offset.items():
        for scen in SCENARIO_NAMES:
            s = agg.get(scen, {})
            print(
                f"{offset_name:<15} {scen:<12} "
                f"{s.get('success_rate', 0)*100:6.1f} "
                f"{s.get('crash_rate', 0)*100:7.1f} "
                f"{s.get('timeout_rate', 0)*100:9.1f}"
            )


if __name__ == "__main__":
    main()
