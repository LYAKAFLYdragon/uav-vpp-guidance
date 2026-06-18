#!/usr/bin/env python3
"""
Adversarial Evaluation: Compare all pursuer-target pairings.

Evaluates 4 configurations:
1. Baseline pursuer vs baseline target (constant-velocity, via CloseRangeTrackingEnv)
2. Baseline pursuer vs adversarial target (RL evader)
3. Adversarial pursuer vs baseline target
4. Adversarial pursuer vs adversarial target (full adversarial pair)

Outputs JSON results and generates comparison plots.

Usage:
    python scripts/evaluate_adversarial.py \\
        --config config/adversarial/evaluate.yaml \\
        --num-episodes 30 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluate_adversarial")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load and merge experiment config."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: Dict[str, Any] = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(os.path.dirname(config_path), "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------

def make_default_scenario(range_m: float = 2000.0, heading_offset_deg: float = 0.0) -> Dict[str, Any]:
    """Create a simple tail-chase scenario."""
    import math
    hdg_rad = math.radians(heading_offset_deg)
    return {
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [range_m, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": heading_offset_deg,
        },
    }


def make_head_on_scenario(range_m: float = 4000.0) -> Dict[str, Any]:
    """Create a head-on scenario."""
    return {
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [range_m, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 180.0,
        },
    }


SCENARIO_BUILDERS = {
    "tail_chase_2000m": lambda: make_default_scenario(2000.0, 0.0),
    "tail_chase_4000m": lambda: make_default_scenario(4000.0, 0.0),
    "head_on_4000m": lambda: make_head_on_scenario(4000.0),
    "crossing_90deg_2000m": lambda: make_default_scenario(2000.0, 90.0),
    "offset_45deg_2000m": lambda: make_default_scenario(2000.0, 45.0),
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_adversarial_episode(
    env: AdversarialJSBSimEnv,
    pursuer_agent: PPOAgent,
    target_agent: AdversarialTargetAgent,
    scenario: Any,
    seed: int,
) -> Dict[str, Any]:
    """Run one episode in the adversarial environment."""
    p_obs, t_obs = env.reset(scenario=scenario, seed=seed)

    p_return = 0.0
    t_return = 0.0
    ep_length = 0
    min_range = float("inf")
    final_range = 0.0
    final_ata = 0.0
    reason = "timeout"

    for step in range(env.max_steps):
        p_action = pursuer_agent.get_deterministic_action(p_obs["observation_vector"])
        t_action = target_agent.get_deterministic_action(t_obs["observation_vector"])

        (p_obs, t_obs, p_rew, t_rew, terminated, truncated, info) = env.step(p_action, t_action)
        p_return += p_rew
        t_return += t_rew
        ep_length += 1

        rel = p_obs.get("relative_state", {})
        range_m = float(rel.get("range_m", 0.0))
        ata_deg = float(np.rad2deg(rel.get("ata_rad", 0.0)))
        min_range = min(min_range, range_m)
        final_range = range_m
        final_ata = ata_deg

        if terminated or truncated:
            reason = info.get("termination", {}).get("reason", "unknown")
            break

    return {
        "pursuer_return": float(p_return),
        "target_return": float(t_return),
        "length": ep_length,
        "min_range_m": float(min_range),
        "final_range_m": float(final_range),
        "final_ata_deg": float(final_ata),
        "reason": reason,
        "captured": reason == "success",
        "survived": reason in ("timeout", "out_of_bounds"),
    }


def run_baseline_episode(
    env: CloseRangeTrackingEnv,
    pursuer_agent: PPOAgent,
    scenario: Any,
    seed: int,
) -> Dict[str, Any]:
    """Run one episode in the baseline (single-agent) environment."""
    obs = env.reset(scenario=scenario, seed=seed)

    ep_return = 0.0
    ep_length = 0
    min_range = float("inf")
    final_range = 0.0
    final_ata = 0.0
    reason = "timeout"

    for step in range(env.max_steps):
        action = pursuer_agent.get_deterministic_action(obs["observation_vector"])
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_length += 1

        rel = obs.get("relative_state", {})
        range_m = float(rel.get("range_m", 0.0))
        ata_deg = float(np.rad2deg(rel.get("ata_rad", 0.0)))
        min_range = min(min_range, range_m)
        final_range = range_m
        final_ata = ata_deg

        if terminated or truncated:
            reason = info.get("reason", "unknown")
            break

    return {
        "pursuer_return": float(ep_return),
        "target_return": 0.0,
        "length": ep_length,
        "min_range_m": float(min_range),
        "final_range_m": float(final_range),
        "final_ata_deg": float(final_ata),
        "reason": reason,
        "captured": reason == "success",
        "survived": reason == "timeout",
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_pairing(
    pairing_name: str,
    config: Dict[str, Any],
    pursuer_ckpt: Optional[str],
    target_ckpt: Optional[str],
    scenarios: List[str],
    num_episodes: int,
    seeds: List[int],
    use_adversarial_env: bool,
    pursuer_policy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one pursuer-target pairing across scenarios and seeds."""
    logger.info("=== Evaluating: %s ===", pairing_name)

    results: Dict[str, List[Dict[str, Any]]] = {}

    # Load agents and environment
    ppo_cfg = config.get("ppo", {})
    policy_cfg = config.get("policy", {})
    if pursuer_policy_config is None:
        pursuer_policy_config = policy_cfg

    sampler = None
    if config.get("scenario_sampler"):
        sampler = make_scenario_sampler(config["scenario_sampler"])

    # Local copy of config with the correct pursuer policy architecture.
    pursuer_cfg = dict(config)
    pursuer_cfg["policy"] = pursuer_policy_config

    if use_adversarial_env:
        env = AdversarialJSBSimEnv(config)
        pursuer_agent = PPOAgent(16, 3, pursuer_cfg)
        if pursuer_ckpt and os.path.exists(pursuer_ckpt):
            pursuer_agent.load(pursuer_ckpt)
        pursuer_agent.network.eval()

        target_cfg = dict(config)
        target_cfg["ppo"] = config.get("target_ppo", ppo_cfg)
        target_cfg["policy"] = config.get("target_policy", policy_cfg)
        target_agent = AdversarialTargetAgent(config=target_cfg)
        if target_ckpt and os.path.exists(target_ckpt):
            target_agent.load(target_ckpt)
        target_agent.eval()
    else:
        env = CloseRangeTrackingEnv(config)
        pursuer_agent = PPOAgent(16, 3, pursuer_cfg)
        if pursuer_ckpt and os.path.exists(pursuer_ckpt):
            pursuer_agent.load(pursuer_ckpt)
        pursuer_agent.network.eval()

    for sc_name in scenarios:
        builder = SCENARIO_BUILDERS.get(sc_name)
        if builder is None:
            logger.warning("Unknown scenario: %s, skipping", sc_name)
            continue
        scenario = builder()
        sc_results: List[Dict[str, Any]] = []

        for seed in seeds:
            for ep in range(num_episodes):
                ep_seed = seed * 10000 + ep
                episode_scenario = sampler.sample() if sampler else scenario
                if use_adversarial_env:
                    ep_result = run_adversarial_episode(
                        env, pursuer_agent, target_agent, episode_scenario, ep_seed
                    )
                else:
                    ep_result = run_baseline_episode(
                        env, pursuer_agent, episode_scenario, ep_seed
                    )
                ep_result["seed"] = seed
                ep_result["episode"] = ep
                sc_results.append(ep_result)

        results[sc_name] = sc_results

    if use_adversarial_env:
        env.close()
    else:
        env.close()

    # Aggregate statistics
    all_eps = [e for sc_eps in results.values() for e in sc_eps]
    captures = sum(1 for e in all_eps if e["captured"])
    survives = sum(1 for e in all_eps if e["survived"])

    return {
        "pairing": pairing_name,
        "num_episodes": len(all_eps),
        "capture_rate": captures / max(1, len(all_eps)),
        "survival_rate": survives / max(1, len(all_eps)),
        "mean_final_range_m": float(np.mean([e["final_range_m"] for e in all_eps])),
        "mean_min_range_m": float(np.mean([e["min_range_m"] for e in all_eps])),
        "mean_pursuer_return": float(np.mean([e["pursuer_return"] for e in all_eps])),
        "mean_target_return": float(np.mean([e["target_return"] for e in all_eps])),
        "mean_length": float(np.mean([e["length"] for e in all_eps])),
        "per_scenario": {
            sc: {
                "capture_rate": sum(1 for e in eps if e["captured"]) / max(1, len(eps)),
                "survival_rate": sum(1 for e in eps if e["survived"]) / max(1, len(eps)),
                "mean_final_range_m": float(np.mean([e["final_range_m"] for e in eps])),
                "mean_pursuer_return": float(np.mean([e["pursuer_return"] for e in eps])),
            }
            for sc, eps in results.items()
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(all_results: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pairings = [r["pairing"] for r in all_results]
        capture_rates = [r["capture_rate"] for r in all_results]

        # Bar chart: capture rate comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(pairings, capture_rates, color=["#2196F3", "#FF9800", "#4CAF50", "#F44336"])
        ax.set_ylabel("Capture Rate")
        ax.set_title("Pursuer Capture Rate by Agent Pairing")
        ax.set_ylim(0, 1.1)
        for bar, rate in zip(bars, capture_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{rate:.2%}", ha="center", va="bottom", fontweight="bold")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "capture_rate_comparison.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "capture_rate_comparison.pdf"))
        plt.close()
        logger.info("Saved capture rate comparison plot.")

    except ImportError:
        logger.warning("matplotlib not available; skipping plots.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adversarial agent pairings.")
    parser.add_argument("--config", type=str, default="config/adversarial/evaluate.yaml")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--pursuer-baseline-ckpt", type=str, default=None)
    parser.add_argument("--pursuer-adversarial-ckpt", type=str, default=None)
    parser.add_argument("--target-adversarial-ckpt", type=str, default=None)
    parser.add_argument("--scenarios", type=str, nargs="+", default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    ckpt_cfg = config.get("checkpoints", {})
    pursuer_baseline = args.pursuer_baseline_ckpt or ckpt_cfg.get(
        "pursuer_baseline", "outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt")
    pursuer_adversarial = args.pursuer_adversarial_ckpt or ckpt_cfg.get(
        "pursuer_adversarial", "outputs/adversarial/training/pursuer_adversarial/checkpoints/best.pt")
    target_adversarial = args.target_adversarial_ckpt or ckpt_cfg.get(
        "target_adversarial", "outputs/adversarial/training/target_agent_ppo/checkpoints/best.pt")

    scenarios = args.scenarios or config.get("scenarios", [
        "tail_chase_2000m", "tail_chase_4000m", "head_on_4000m",
    ])

    output_dir = args.output_dir or config.get("output_dir", "outputs/adversarial/evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Define pairings
    policies_cfg = config.get("policies", {})
    baseline_policy = policies_cfg.get("baseline_pursuer", config.get("policy", {}))
    adversarial_policy = policies_cfg.get("adversarial_pursuer", config.get("policy", {}))

    pairings = [
        ("BaselinePursuer_vs_BaselineTarget", pursuer_baseline, None, False, baseline_policy),
        ("BaselinePursuer_vs_AdversarialTarget", pursuer_baseline, target_adversarial, True, baseline_policy),
        ("AdversarialPursuer_vs_BaselineTarget", pursuer_adversarial, None, False, adversarial_policy),
        ("AdversarialPursuer_vs_AdversarialTarget", pursuer_adversarial, target_adversarial, True, adversarial_policy),
    ]

    all_results: List[Dict[str, Any]] = []
    for name, p_ckpt, t_ckpt, use_adv, p_policy in pairings:
        result = evaluate_pairing(
            pairing_name=name,
            config=config,
            pursuer_ckpt=p_ckpt,
            target_ckpt=t_ckpt,
            scenarios=scenarios,
            num_episodes=args.num_episodes,
            seeds=args.seeds,
            use_adversarial_env=use_adv,
            pursuer_policy_config=p_policy,
        )
        all_results.append(result)
        logger.info(
            "%s: capture_rate=%.2f%% survival_rate=%.2f%% mean_range=%.1fm",
            name,
            result["capture_rate"] * 100,
            result["survival_rate"] * 100,
            result["mean_final_range_m"],
        )

    # Save results
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "pairings": all_results,
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "num_episodes": args.num_episodes,
                "seeds": args.seeds,
                "scenarios": scenarios,
                "config": args.config,
            },
        }, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Generate plots
    fig_dir = os.path.join(output_dir, "figures")
    generate_plots(all_results, fig_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL EVALUATION SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['pairing']}:")
        print(f"    Capture Rate: {r['capture_rate']:.1%}")
        print(f"    Survival Rate: {r['survival_rate']:.1%}")
        print(f"    Mean Final Range: {r['mean_final_range_m']:.0f} m")
        print(f"    Mean Pursuer Return: {r['mean_pursuer_return']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
