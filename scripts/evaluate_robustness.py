#!/usr/bin/env python3
"""
Robustness Evaluation: Test agent performance under perturbations.

Sweeps perturbation levels and evaluates agent pairings under
wind gusts, sensor noise, and actuator delay.

Outputs:
- robustness_results.json: per-level per-pairing statistics
- figures/robustness_heatmap.png: capture rate vs perturbation level

Usage:
    python scripts/evaluate_robustness.py \\
        --config config/adversarial/robustness_eval.yaml \\
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
from uav_vpp_guidance.envs.perturbed_jsbsim_env import PerturbedJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluate_robustness")


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


def make_default_scenario(range_m: float = 2000.0) -> Dict[str, Any]:
    """Simple tail-chase scenario."""
    return {
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [range_m, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 0.0,
        },
    }


def evaluate_at_level(
    pairing_name: str,
    config: Dict[str, Any],
    perturbation_level: float,
    pursuer_ckpt: str,
    target_ckpt: Optional[str],
    num_episodes: int,
    seeds: List[int],
    use_adversarial: bool,
    pursuer_policy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a pairing at a specific perturbation level."""
    default_scenario = make_default_scenario(2000.0)
    sampler = make_scenario_sampler(config.get("scenario_sampler", {}))
    pert_cfg = config.get("wind", {})  # pass perturbation config

    ppo_cfg = config.get("ppo", {})
    policy_cfg = config.get("policy", {})
    if pursuer_policy_config is None:
        pursuer_policy_config = policy_cfg
    pursuer_cfg = dict(config)
    pursuer_cfg["policy"] = pursuer_policy_config

    target_cfg = dict(config)
    target_cfg["ppo"] = config.get("target_ppo", ppo_cfg)
    target_cfg["policy"] = config.get("target_policy", policy_cfg)

    if use_adversarial:
        base_env = AdversarialJSBSimEnv(config)
        pert_env = PerturbedJSBSimEnv(
            base_env,
            perturbation_config=config,
            scale=perturbation_level,
        )

        pursuer = PPOAgent(16, 3, pursuer_cfg)
        if os.path.exists(pursuer_ckpt):
            pursuer.load(pursuer_ckpt)
        pursuer.network.eval()

        target = AdversarialTargetAgent(config=target_cfg)
        if target_ckpt and os.path.exists(target_ckpt):
            target.load(target_ckpt)
        target.eval()
    else:
        base_env = CloseRangeTrackingEnv(config)
        pert_env = PerturbedJSBSimEnv(
            base_env,
            perturbation_config=config,
            scale=perturbation_level,
        )
        pursuer = PPOAgent(16, 3, pursuer_cfg)
        if os.path.exists(pursuer_ckpt):
            pursuer.load(pursuer_ckpt)
        pursuer.network.eval()

    episodes: List[Dict[str, Any]] = []
    for seed in seeds:
        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep

            if use_adversarial:
                scenario = sampler.sample() if sampler else default_scenario
                p_obs, t_obs = pert_env.reset(scenario=scenario, seed=ep_seed)
                ep_return = 0.0
                length = 0
                min_range = float("inf")
                reason = "timeout"

                for _ in range(getattr(pert_env, "max_steps", 512)):
                    p_action = pursuer.get_deterministic_action(p_obs["observation_vector"])
                    t_action = target.get_deterministic_action(t_obs["observation_vector"])
                    (p_obs, t_obs, p_rew, t_rew, terminated, truncated, info) = pert_env.step(
                        (p_action, t_action)
                    )
                    ep_return += p_rew
                    length += 1
                    rel = p_obs.get("relative_state", {})
                    range_m = float(rel.get("range_m", 0.0))
                    min_range = min(min_range, range_m)
                    if terminated or truncated:
                        reason = info.get("termination", {}).get("reason", "unknown")
                        break
            else:
                scenario = sampler.sample() if sampler else default_scenario
                obs = pert_env.reset(scenario=scenario, seed=ep_seed)
                ep_return = 0.0
                length = 0
                min_range = float("inf")
                reason = "timeout"

                for _ in range(getattr(pert_env, "max_steps", 512)):
                    action = pursuer.get_deterministic_action(obs["observation_vector"])
                    obs, reward, terminated, truncated, info = pert_env.step(action)
                    ep_return += reward
                    length += 1
                    rel = obs.get("relative_state", {})
                    range_m = float(rel.get("range_m", 0.0))
                    min_range = min(min_range, range_m)
                    if terminated or truncated:
                        reason = info.get("termination", {}).get("reason",
                                     info.get("reason", "unknown"))
                        break

            episodes.append({
                "seed": seed, "episode": ep,
                "return": float(ep_return), "length": length,
                "min_range_m": float(min_range), "reason": reason,
                "captured": reason == "success",
            })

    pert_env.close()

    captures = sum(1 for e in episodes if e["captured"])
    crashes = sum(1 for e in episodes if e["reason"] == "crash")
    returns = [e["return"] for e in episodes]

    return {
        "pairing": pairing_name,
        "perturbation_level": perturbation_level,
        "num_episodes": len(episodes),
        "capture_rate": captures / max(1, len(episodes)),
        "crash_rate": crashes / max(1, len(episodes)),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_min_range_m": float(np.mean([e["min_range_m"] for e in episodes])),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
    }


def generate_heatmap(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate robustness heatmap plot."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Extract unique pairings and levels
        pairings = sorted(set(r["pairing"] for r in results))
        levels = sorted(set(r["perturbation_level"] for r in results))

        # Build data matrix
        data = np.zeros((len(pairings), len(levels)))
        for r in results:
            i = pairings.index(r["pairing"])
            j = levels.index(r["perturbation_level"])
            data[i, j] = r["capture_rate"]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{l:.0%}" for l in levels])
        ax.set_yticks(range(len(pairings)))
        ax.set_yticklabels(pairings)
        ax.set_xlabel("Perturbation Level")
        ax.set_title("Capture Rate vs Perturbation Level")
        plt.colorbar(im, ax=ax, label="Capture Rate")

        for i in range(len(pairings)):
            for j in range(len(levels)):
                ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center",
                        color="black" if data[i, j] > 0.5 else "white", fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "robustness_heatmap.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "robustness_heatmap.pdf"))
        plt.close()
        logger.info("Saved robustness heatmap.")

    except ImportError:
        logger.warning("matplotlib not available; skipping plots.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation across perturbation levels.")
    parser.add_argument("--config", type=str, default="config/adversarial/robustness_eval.yaml")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--pursuer-baseline-ckpt", type=str, default=None)
    parser.add_argument("--pursuer-adversarial-ckpt", type=str, default=None)
    parser.add_argument("--target-adversarial-ckpt", type=str, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    ckpt_cfg = config.get("checkpoints", {})
    pursuer_baseline = args.pursuer_baseline_ckpt or ckpt_cfg.get(
        "pursuer_baseline", "outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt")
    pursuer_adversarial = args.pursuer_adversarial_ckpt or ckpt_cfg.get(
        "pursuer_adversarial", "outputs/adversarial/training/pursuer_adversarial/checkpoints/best.pt")
    target_adversarial = args.target_adversarial_ckpt or ckpt_cfg.get(
        "target_adversarial", "outputs/adversarial/training/target_agent_ppo/checkpoints/best.pt")

    levels = config.get("evaluation", {}).get("perturbation_levels", [0.0, 0.25, 0.5, 0.75, 1.0])
    output_dir = args.output_dir or config.get("output_dir", "outputs/adversarial/robustness")
    os.makedirs(output_dir, exist_ok=True)

    # Define configurations to evaluate
    policies_cfg = config.get("policies", {})
    baseline_policy = policies_cfg.get("baseline_pursuer", config.get("policy", {}))
    adversarial_policy = policies_cfg.get("adversarial_pursuer", config.get("policy", {}))

    configs_to_eval = [
        ("BaselinePursuer_BaselineTarget", pursuer_baseline, None, False, baseline_policy),
        ("BaselinePursuer_AdversarialTarget", pursuer_baseline, target_adversarial, True, baseline_policy),
        ("AdversarialPursuer_BaselineTarget", pursuer_adversarial, None, False, adversarial_policy),
        ("AdversarialPursuer_AdversarialTarget", pursuer_adversarial, target_adversarial, True, adversarial_policy),
    ]

    all_results: List[Dict[str, Any]] = []
    for pairing_name, p_ckpt, t_ckpt, use_adv, p_policy in configs_to_eval:
        for level in levels:
            logger.info("Evaluating %s at perturbation level %.2f", pairing_name, level)
            result = evaluate_at_level(
                pairing_name, config, level, p_ckpt, t_ckpt,
                args.num_episodes, args.seeds, use_adv,
                pursuer_policy_config=p_policy,
            )
            all_results.append(result)
            logger.info(
                "  Level %.2f: capture_rate=%.2f%% crash_rate=%.2f%% mean_return=%.1f",
                level, result["capture_rate"] * 100, result["crash_rate"] * 100,
                result["mean_return"],
            )

    # Save results
    results_path = os.path.join(output_dir, "robustness_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "num_episodes": args.num_episodes,
                "seeds": args.seeds,
                "perturbation_levels": levels,
                "config": args.config,
            },
        }, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Generate plots
    fig_dir = os.path.join(output_dir, "figures")
    generate_heatmap(all_results, fig_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['pairing']} @ level={r['perturbation_level']:.0%}: "
              f"capture={r['capture_rate']:.1%} crash={r['crash_rate']:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
