#!/usr/bin/env python3
"""Baseline comparison runner.

Evaluates PPO-based methods, rule-based pursuit baselines, and MPC guidance on
the regression suite (or any scenario suite) using the simple backend.

Outputs a CSV and Markdown comparison table.
"""

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.baselines.rule_based_pursuit import RuleBasedPursuitPolicy
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


PPO_METHODS = {
    "no_prediction": {
        "checkpoint": "outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt",
        "config_method": "no_prediction",
    },
    "gain_only": {
        "checkpoint": "outputs/audit_no_pred_final/checkpoints/best.pt",
        "config_method": "no_prediction",
        "gains_path": "outputs/gain_only_cem/cem_results.json",
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


def build_ppo_config(config_path: str, method_name: str, backend: str) -> dict:
    """Build config for a PPO-based method."""
    method_cfg = PPO_METHODS[method_name]
    config = load_method_config(config_path, method_cfg["config_method"])
    config["backend"] = backend
    config.setdefault("env", {})
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"
    config.setdefault("guidance", {}).setdefault("mode_switch", {})["enabled"] = False

    if "gains_path" in method_cfg:
        gains_data = json.loads(Path(method_cfg["gains_path"]).read_text(encoding="utf-8"))
        loaded_gains = gains_data.get("best_gains", gains_data)
        config.setdefault("guidance", {}).setdefault("gains", {}).update(loaded_gains)
    return config


def build_rule_config(config_path: str, backend: str, mode: str) -> dict:
    """Build config for a rule-based pursuit baseline."""
    config = load_method_config(config_path, "no_prediction")
    config["backend"] = backend
    config.setdefault("env", {})
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"
    config.setdefault("guidance", {}).setdefault("mode_switch", {})["enabled"] = False
    # Force zero-offset VPP so the policy's action directly becomes the VP offset.
    config.setdefault("virtual_point", {})["mode"] = "zero_offset"
    return config, RuleBasedPursuitPolicy(mode=mode)


def build_mpc_config(config_path: str, backend: str) -> dict:
    """Build config for MPC guidance baseline."""
    config = load_method_config(config_path, "no_prediction")
    config["backend"] = backend
    config.setdefault("env", {})
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"
    config.setdefault("guidance", {}).setdefault("mode_switch", {})["enabled"] = False
    config["guidance"]["mode"] = "mpc"
    config["guidance"].setdefault("params", {}).update({
        "horizon_steps": 10,
        "dt": config.get("env", {}).get("high_level_dt", 0.2),
        "w_range": 1.0,
        "w_ata": 0.01,
        "w_effort": 1e-4,
        "w_speed": 0.1,
    })
    # Zero-offset VPP so MPC tracks the target directly.
    config.setdefault("virtual_point", {})["mode"] = "zero_offset"
    return config


def make_ppo_agent(config: dict, checkpoint: str) -> PPOAgent:
    """Create a PPOAgent and load a checkpoint."""
    tmp_env = CloseRangeTrackingEnv(config)
    obs = tmp_env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    tmp_env.close()
    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    agent.load(checkpoint)
    return agent


def run_episodes(
    env: CloseRangeTrackingEnv,
    scenarios: List[dict],
    seeds: List[int],
    agent: Optional[PPOAgent] = None,
    rule_policy: Optional[RuleBasedPursuitPolicy] = None,
    zero_action: bool = False,
) -> List[Dict[str, Any]]:
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
                if agent is not None:
                    action, _log_prob, _value = agent.select_action(
                        obs["observation_vector"], deterministic=True, store=False
                    )
                elif rule_policy is not None:
                    action = rule_policy.get_action(
                        obs.get("own_state", {}),
                        obs.get("target_state", {}),
                        obs.get("relative_state", {}),
                    )
                elif zero_action:
                    action = np.zeros(3, dtype=np.float64)
                else:
                    raise ValueError("No controller provided")

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


def aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
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


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison on simple backend")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
    )
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"])
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["no_prediction", "gain_only", "pure_pursuit", "mpc"],
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
    parser.add_argument("--output-dir", type=str, default="outputs/baseline_comparison")
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
    for baseline in args.baselines:
        print(f"\n{'='*60}")
        print(f"Baseline: {baseline}")
        print(f"{'='*60}")

        if baseline in PPO_METHODS:
            config = build_ppo_config(args.config, baseline, args.backend)
            env = CloseRangeTrackingEnv(config)
            agent = make_ppo_agent(config, PPO_METHODS[baseline]["checkpoint"])
            episodes = run_episodes(env, scenarios, args.seeds, agent=agent)
            env.close()
        elif baseline.startswith("rule_") or baseline in ("pure_pursuit", "lag_pursuit", "lead_pursuit"):
            mode = baseline.replace("rule_", "")
            config, policy = build_rule_config(args.config, args.backend, mode)
            env = CloseRangeTrackingEnv(config)
            episodes = run_episodes(env, scenarios, args.seeds, rule_policy=policy)
            env.close()
        elif baseline == "mpc":
            config = build_mpc_config(args.config, args.backend)
            env = CloseRangeTrackingEnv(config)
            episodes = run_episodes(env, scenarios, args.seeds, zero_action=True)
            env.close()
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        stats = aggregate(episodes)
        rows.append(
            {
                "baseline": baseline,
                "n": stats["n"],
                "success_rate": stats["success_rate"],
                "successes": stats["successes"],
                "mean_min_range_m": stats["mean_min_range_m"],
                "std_min_range_m": stats["std_min_range_m"],
                "mean_final_range_m": stats["mean_final_range_m"],
                "mean_min_ata_deg": stats["mean_min_ata_deg"],
            }
        )
        print(
            f"{baseline:20s}: success={stats['success_rate']:.2%} "
            f"({stats['successes']}/{stats['n']})  "
            f"min_range={stats['mean_min_range_m']:.1f}±{stats['std_min_range_m']:.1f} m"
        )

    # Write CSV
    csv_path = output_dir / "baseline_comparison.csv"
    fieldnames = [
        "baseline",
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

    # Write Markdown
    md_path = output_dir / "baseline_comparison.md"
    md_lines = [
        "# Baseline Comparison (Simple Backend)",
        "",
        f"- Backend: `{args.backend}`",
        f"- Scenarios: `{args.scenarios}`",
        f"- Seeds: {args.seeds}",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Baseline | Success Rate | Mean min range (m) | Std min range (m) | Mean final range (m) | Mean min ATA (deg) |",
        "|----------|-------------:|-------------------:|------------------:|---------------------:|-------------------:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['baseline']} | "
            f"{row['success_rate']:.2%} ({row['successes']}/{row['n']}) | "
            f"{row['mean_min_range_m']:.1f} | "
            f"{row['std_min_range_m']:.1f} | "
            f"{row['mean_final_range_m']:.1f} | "
            f"{row['mean_min_ata_deg']:.1f} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
