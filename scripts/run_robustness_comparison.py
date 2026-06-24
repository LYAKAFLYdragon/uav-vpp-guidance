#!/usr/bin/env python3
"""Robustness comparison runner for the simple backend.

Evaluates PPO-based methods under sensor noise, communication delay, and
Dryden wind gusts.  Produces a CSV and Markdown comparison table.
"""

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.guidance.gain_config import GuidanceGains
from uav_vpp_guidance.utils.config import merge_config


METHODS = {
    "no_prediction": {
        "name": "No-Prediction",
        "checkpoint": "outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt",
        "config_method": "no_prediction",
    },
    "gain_only": {
        "name": "Gain-Only",
        "checkpoint": "outputs/audit_no_pred_final/checkpoints/best.pt",
        "config_method": "no_prediction",
        "gains_path": "outputs/gain_only_cem/cem_results.json",
        "note": "Same policy as no_prediction but with CEM-optimized gains",
    },
}


def load_method_config(config_path: str, method_name: str) -> dict:
    """Load base config merged with the method-specific override block."""
    full_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get(method_name, {})
    base_config = copy.deepcopy(full_config)
    for k, v in method_override.items():
        if isinstance(v, dict) and k in base_config and isinstance(base_config[k], dict):
            base_config[k].update(copy.deepcopy(v))
        else:
            base_config[k] = copy.deepcopy(v)
    return base_config


def build_env_config(
    config_path: str,
    method_name: str,
    backend: str,
    robustness: Optional[dict],
    wind: Optional[dict],
) -> dict:
    """Build an environment config with robustness/wind overrides."""
    config = load_method_config(config_path, method_name)
    config["backend"] = backend
    config.setdefault("env", {})
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"
    config.setdefault("guidance", {}).setdefault("mode_switch", {})["enabled"] = False

    # Load CEM gains for gain_only.
    method_cfg = METHODS[method_name]
    if "gains_path" in method_cfg:
        gains_data = json.loads(Path(method_cfg["gains_path"]).read_text(encoding="utf-8"))
        loaded_gains = gains_data.get("best_gains", gains_data)
        config.setdefault("guidance", {}).setdefault("gains", {}).update(loaded_gains)

    if robustness is not None:
        config["robustness"] = copy.deepcopy(robustness)
    else:
        config.pop("robustness", None)

    if wind is not None:
        # The simple backend reads wind from the env block.
        config.setdefault("env", {})["wind"] = copy.deepcopy(wind)
        config["wind"] = copy.deepcopy(wind)
    else:
        config.pop("wind", None)
        config.setdefault("env", {}).pop("wind", None)

    return config


def run_episodes(
    env: CloseRangeTrackingEnv,
    agent: PPOAgent,
    scenarios: List[dict],
    seeds: List[int],
) -> List[dict]:
    """Run all scenario/seed combinations and return per-episode metrics."""
    episodes = []
    for scen in scenarios:
        for seed in seeds:
            obs = env.reset(scenario=scen, seed=seed)
            terminated = False
            truncated = False
            min_range_m = float("inf")
            min_ata_deg = float("inf")
            final_range_m = math.nan
            final_ata_deg = math.nan
            reason = "timeout"
            length = 0

            while not (terminated or truncated):
                action, _log_prob, _value = agent.select_action(
                    obs["observation_vector"], deterministic=True, store=False
                )
                obs, _reward, terminated, truncated, info = env.step(action)
                length += 1
                rel = info.get("relative_state", {})
                rng = float(rel.get("range_m", math.nan))
                ata = float(math.degrees(rel.get("ata_rad", math.nan)))
                if not math.isnan(rng):
                    min_range_m = min(min_range_m, rng)
                    final_range_m = rng
                if not math.isnan(ata):
                    min_ata_deg = min(min_ata_deg, ata)
                    final_ata_deg = ata
                if terminated or truncated:
                    reason = info.get("reason", "unknown")

            episodes.append(
                {
                    "scenario": scen.get("name", "unknown"),
                    "seed": seed,
                    "success": reason == "success",
                    "reason": reason,
                    "length": length,
                    "min_range_m": min_range_m,
                    "final_range_m": final_range_m,
                    "min_ata_deg": min_ata_deg,
                    "final_ata_deg": final_ata_deg,
                }
            )
    return episodes


def aggregate(episodes: List[dict]) -> dict:
    """Aggregate episode metrics into a summary dict."""
    n = len(episodes)
    successes = sum(1 for ep in episodes if ep["success"])
    min_ranges = [ep["min_range_m"] for ep in episodes if not math.isnan(ep["min_range_m"])]
    final_ranges = [ep["final_range_m"] for ep in episodes if not math.isnan(ep["final_range_m"])]
    min_atas = [ep["min_ata_deg"] for ep in episodes if not math.isnan(ep["min_ata_deg"])]
    return {
        "n": n,
        "success_rate": successes / n if n else math.nan,
        "successes": successes,
        "mean_min_range_m": float(np.mean(min_ranges)) if min_ranges else math.nan,
        "std_min_range_m": float(np.std(min_ranges)) if min_ranges else math.nan,
        "mean_final_range_m": float(np.mean(final_ranges)) if final_ranges else math.nan,
        "mean_min_ata_deg": float(np.mean(min_atas)) if min_atas else math.nan,
    }


def make_agent(config: dict) -> PPOAgent:
    """Create a PPOAgent and infer obs_dim from a reset environment."""
    tmp_env = CloseRangeTrackingEnv(config)
    obs = tmp_env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    tmp_env.close()
    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    return agent


def load_agent(method_name: str, config: dict) -> PPOAgent:
    """Build and load checkpoint/gains for a method."""
    agent = make_agent(config)
    ckpt_path = METHODS[method_name]["checkpoint"]
    agent.load(ckpt_path)
    return agent


CONDITIONS: List[Tuple[str, Optional[dict], Optional[dict]]] = [
    (
        "baseline",
        None,
        None,
    ),
    (
        "imu_low",
        {
            "enabled": True,
            "sensor_noise": {
                "enabled": True,
                "imu_noise_std_deg_s": 0.1,
                "gps_position_noise_std_m": 0.0,
            },
        },
        None,
    ),
    (
        "imu_high",
        {
            "enabled": True,
            "sensor_noise": {
                "enabled": True,
                "imu_noise_std_deg_s": 1.0,
                "gps_position_noise_std_m": 0.0,
            },
        },
        None,
    ),
    (
        "gps_low",
        {
            "enabled": True,
            "sensor_noise": {
                "enabled": True,
                "imu_noise_std_deg_s": 0.0,
                "gps_position_noise_std_m": 1.0,
            },
        },
        None,
    ),
    (
        "gps_high",
        {
            "enabled": True,
            "sensor_noise": {
                "enabled": True,
                "imu_noise_std_deg_s": 0.0,
                "gps_position_noise_std_m": 10.0,
            },
        },
        None,
    ),
    (
        "delay_low",
        {
            "enabled": True,
            "communication_delay": {"enabled": True, "delay_s": 0.01},
        },
        None,
    ),
    (
        "delay_high",
        {
            "enabled": True,
            "communication_delay": {"enabled": True, "delay_s": 0.1},
        },
        None,
    ),
    (
        "wind_light",
        None,
        {"enabled": True, "model": "dryden", "intensity": "light"},
    ),
    (
        "wind_medium",
        None,
        {"enabled": True, "model": "dryden", "intensity": "medium"},
    ),
    (
        "wind_severe",
        None,
        {"enabled": True, "model": "dryden", "intensity": "severe"},
    ),
    (
        "combined",
        {
            "enabled": True,
            "sensor_noise": {
                "enabled": True,
                "imu_noise_std_deg_s": 0.5,
                "gps_position_noise_std_m": 5.0,
            },
            "communication_delay": {"enabled": True, "delay_s": 0.05},
        },
        {"enabled": True, "model": "dryden", "intensity": "medium"},
    ),
]


def format_condition(condition: str, robustness: Optional[dict], wind: Optional[dict]) -> str:
    """Human-readable condition description for the table."""
    if condition == "baseline":
        return "Baseline (clean)"
    if condition.startswith("imu_"):
        level = robustness["sensor_noise"]["imu_noise_std_deg_s"]
        return f"IMU noise {level} deg/s"
    if condition.startswith("gps_"):
        level = robustness["sensor_noise"]["gps_position_noise_std_m"]
        return f"GPS noise {level} m"
    if condition.startswith("delay_"):
        level = robustness["communication_delay"]["delay_s"] * 1000
        return f"Comm delay {level:.0f} ms"
    if condition.startswith("wind_"):
        return f"Wind {wind['intensity']}"
    if condition == "combined":
        return "Combined noise"
    return condition


def main():
    parser = argparse.ArgumentParser(description="Robustness comparison on simple backend")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
    )
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"])
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["no_prediction", "gain_only"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="regression",
        choices=["regression", "candidate", "all"],
    )
    parser.add_argument("--output-dir", type=str, default="outputs/robustness_comparison")
    args = parser.parse_args()

    initialize_canonical_scenarios()
    if args.scenarios == "regression":
        scenarios = ScenarioRegistry.get_regression_suite()
    elif args.scenarios == "candidate":
        scenarios = ScenarioRegistry.get_candidate_suite()
    else:
        scenarios = (
            ScenarioRegistry.get_regression_suite()
            + ScenarioRegistry.get_candidate_suite()
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition, robustness, wind in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")
        for method_name in args.methods:
            if method_name not in METHODS:
                raise ValueError(f"Unknown method: {method_name}")
            config = build_env_config(
                args.config, method_name, args.backend, robustness, wind
            )
            env = CloseRangeTrackingEnv(config)
            agent = load_agent(method_name, config)
            episodes = run_episodes(env, agent, scenarios, args.seeds)
            env.close()
            stats = aggregate(episodes)
            row = {
                "condition": condition,
                "condition_label": format_condition(condition, robustness, wind),
                "method": method_name,
                "n": stats["n"],
                "success_rate": stats["success_rate"],
                "successes": stats["successes"],
                "mean_min_range_m": stats["mean_min_range_m"],
                "std_min_range_m": stats["std_min_range_m"],
                "mean_final_range_m": stats["mean_final_range_m"],
                "mean_min_ata_deg": stats["mean_min_ata_deg"],
            }
            rows.append(row)
            print(
                f"{method_name:15s}: success={stats['success_rate']:.2%} "
                f"({stats['successes']}/{stats['n']})  "
                f"min_range={stats['mean_min_range_m']:.1f}±{stats['std_min_range_m']:.1f} m"
            )

    # Write CSV
    csv_path = output_dir / "robustness_comparison.csv"
    fieldnames = [
        "condition",
        "condition_label",
        "method",
        "n",
        "success_rate",
        "successes",
        "mean_min_range_m",
        "std_min_range_m",
        "mean_final_range_m",
        "mean_min_ata_deg",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")

    # Write Markdown table per method
    md_path = output_dir / "robustness_comparison.md"
    md_lines = [
        "# Robustness Comparison (Simple Backend)",
        "",
        f"- Backend: `{args.backend}`",
        f"- Scenarios: `{args.scenarios}`",
        f"- Seeds: {args.seeds}",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for method_name in args.methods:
        md_lines += [
            f"## Method: {method_name}",
            "",
            "| Condition | Success Rate | Mean min range (m) | Std min range (m) | Mean final range (m) | Mean min ATA (deg) |",
            "|-----------|-------------:|-------------------:|------------------:|---------------------:|-------------------:|",
        ]
        for row in rows:
            if row["method"] != method_name:
                continue
            md_lines.append(
                f"| {row['condition_label']} | "
                f"{row['success_rate']:.2%} ({row['successes']}/{row['n']}) | "
                f"{row['mean_min_range_m']:.1f} | "
                f"{row['std_min_range_m']:.1f} | "
                f"{row['mean_final_range_m']:.1f} | "
                f"{row['mean_min_ata_deg']:.1f} |"
            )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
