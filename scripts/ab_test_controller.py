#!/usr/bin/env python3
"""
A/B test script for Enhanced vs Baseline low-level controller.

Usage:
    # Test with a trained policy:
    python scripts/ab_test_controller.py \
        --policy outputs/ablation_matrix/training/vpp_ca_constant_s0/checkpoints/best.pt \
        --config config/canonical/guidance.yaml \
        --env-config config/env.yaml \
        --n-episodes 50 \
        --output results/controller_ab_test.json

    # Test without policy (zero VPP offset, direct LOS tracking):
    python scripts/ab_test_controller.py \
        --fixed-action "0,0,0" \
        --n-episodes 50

    # Quick smoke test:
    python scripts/ab_test_controller.py --n-episodes 5 --quick

Requires:
    - PyTorch (for policy loading)
    - JSBSim data in data/jsbsim/
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import merge_config

CANONICAL_DIR = PROJECT_ROOT / "config" / "canonical"


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_config(controller: str, policy_path: str = None, fixed_action: str = None,
                 env_config_path: str = None, extra: dict = None) -> dict:
    """Build environment config for A/B test.
    
    Args:
        controller (str): One of "baseline", "enhanced", "gain_scheduled"
    """
    config = {}
    for name in ("guidance", "reward", "virtual_point", "gain_space"):
        config = merge_config(config, load_yaml(CANONICAL_DIR / f"{name}.yaml"))

    env_path = Path(env_config_path) if env_config_path else PROJECT_ROOT / "config" / "env.yaml"
    config = merge_config(config, load_yaml(env_path))

    success_criteria = PROJECT_ROOT / "config" / "success_criteria" / "medium.yaml"
    if success_criteria.exists():
        config = merge_config(config, load_yaml(success_criteria))

    # JSBSim backend
    config["backend"] = "jsbsim"
    config["env"]["backend"] = "jsbsim"
    config["env"]["use_jsbsim"] = True
    config["env"]["jsbsim_data_dir"] = str(PROJECT_ROOT / "data" / "jsbsim")

    # Disable mode switch for pure guidance testing
    config.setdefault("guidance", {})
    config["guidance"].setdefault("mode_switch", {})
    config["guidance"]["mode_switch"]["enabled"] = False

    # Low-level controller selection
    config["low_level_controller"] = controller
    if controller in ("enhanced", "gain_scheduled"):
        # Relaxed limits for enhanced/gain_scheduled controllers
        config["guidance"]["limits"]["nz_max"] = 9.0
        config["guidance"]["limits"]["roll_rate_max"] = 3.0
        config["guidance"]["limits"]["roll_rate_min"] = -3.0
        if controller == "gain_scheduled":
            # Enable aggressive mode for gain_scheduled controller
            config["guidance"]["gains"]["use_gain_scheduling"] = True
            config["guidance"]["gains"]["use_conditional_integration"] = True
            config["guidance"]["gains"]["use_descent_protection"] = True
    else:
        config["low_level_controller"] = "baseline"

    # Policy or fixed action
    if policy_path:
        config["policy_path"] = policy_path
    if fixed_action:
        config["fixed_action"] = [float(x) for x in fixed_action.split(",")]

    if extra:
        config = merge_config(config, extra)

    return config


def load_policy(config: dict, device: str = "cpu"):
    """Load PPO policy if available."""
    policy_path = config.get("policy_path")
    if not policy_path:
        return None

    try:
        from uav_vpp_guidance.agents.ppo_agent import PPOAgent
        import torch

        agent = PPOAgent(config, device=device)
        agent.load(policy_path)
        agent.network.eval()
        return agent
    except Exception as exc:
        print(f"Warning: Could not load policy from {policy_path}: {exc}")
        return None


def run_episode(env: CloseRangeTrackingEnv, agent, fixed_action: List[float] = None,
                seed: int = None, max_steps: int = 512) -> Dict[str, Any]:
    """Run a single episode and collect metrics."""
    if seed is not None:
        np.random.seed(seed)

    obs = env.reset(seed=seed)
    if hasattr(obs, "flatten"):
        obs = obs.flatten()

    done = False
    step_count = 0

    # Metrics collection
    nz_commands = []
    nz_actuals = []
    roll_rate_commands = []
    roll_rate_actuals = []
    beta_values = []
    alpha_values = []
    speed_values = []
    saturation_flags = []
    reward_values = []
    range_values = []

    while not done and step_count < max_steps:
        if fixed_action is not None:
            action = np.array(fixed_action, dtype=np.float32)
        elif agent is not None:
            action = agent.get_deterministic_action(obs)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        if hasattr(obs, "flatten"):
            obs = obs.flatten()

        if "actuator_info" in info:
            actuator = info["actuator_info"]
            saturation_flags.append(actuator.get("saturation_flag", False))

        # Collect aircraft state metrics
        own_state = info.get("own_state", {})
        if own_state:
            nz_commands.append(info.get("guidance_command", {}).get("nz_cmd", 0.0))
            nz_actuals.append(own_state.get("nz_g", 0.0))
            roll_rate_commands.append(info.get("guidance_command", {}).get("roll_rate_cmd", 0.0))
            roll_rate_actuals.append(own_state.get("p_rps", 0.0))
            beta_values.append(abs(own_state.get("beta_rad", 0.0)))
            alpha_values.append(abs(own_state.get("alpha_rad", 0.0)))
            speed_values.append(own_state.get("speed_mps", 0.0))

        # Track range
        rel_state = info.get("relative_state", {})
        if rel_state:
            range_values.append(rel_state.get("range_m", 0.0))

        reward_values.append(reward)

    # Compute summary metrics
    result = {
        "success": info.get("termination_reason") == "success",
        "terminated": terminated,
        "truncated": truncated,
        "steps": step_count,
        "total_reward": sum(reward_values),
        "mean_reward": np.mean(reward_values) if reward_values else 0.0,
    }

    # Controller tracking metrics
    if nz_commands and nz_actuals:
        nz_errors = np.array(nz_commands) - np.array(nz_actuals)
        result["nz_rmse"] = float(np.sqrt(np.mean(nz_errors ** 2)))
        result["nz_mae"] = float(np.mean(np.abs(nz_errors)))
    else:
        result["nz_rmse"] = 0.0
        result["nz_mae"] = 0.0

    if roll_rate_commands and roll_rate_actuals:
        roll_errors = np.array(roll_rate_commands) - np.array(roll_rate_actuals)
        result["roll_rate_rmse"] = float(np.sqrt(np.mean(roll_errors ** 2)))
        result["roll_rate_mae"] = float(np.mean(np.abs(roll_errors)))
    else:
        result["roll_rate_rmse"] = 0.0
        result["roll_rate_mae"] = 0.0

    # Flight quality metrics
    result["mean_beta_deg"] = float(np.degrees(np.mean(beta_values))) if beta_values else 0.0
    result["max_beta_deg"] = float(np.degrees(np.max(beta_values))) if beta_values else 0.0
    result["mean_alpha_deg"] = float(np.degrees(np.mean(alpha_values))) if alpha_values else 0.0
    result["max_alpha_deg"] = float(np.degrees(np.max(alpha_values))) if alpha_values else 0.0
    result["mean_speed_mps"] = float(np.mean(speed_values)) if speed_values else 0.0
    result["min_speed_mps"] = float(np.min(speed_values)) if speed_values else 0.0
    result["saturation_rate"] = float(np.mean(saturation_flags)) if saturation_flags else 0.0

    # Range metrics
    if range_values:
        result["final_range_m"] = float(range_values[-1])
        result["min_range_m"] = float(np.min(range_values))
    else:
        result["final_range_m"] = 0.0
        result["min_range_m"] = 0.0

    return result


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across episodes."""
    if not results:
        return {}

    def _mean(key):
        vals = [r[key] for r in results if key in r]
        return float(np.mean(vals)) if vals else 0.0

    def _std(key):
        vals = [r[key] for r in results if key in r]
        return float(np.std(vals)) if vals else 0.0

    successes = [r for r in results if r.get("success", False)]

    return {
        "n_episodes": len(results),
        "success_count": len(successes),
        "success_rate": len(successes) / len(results) if results else 0.0,
        "mean_steps": _mean("steps"),
        "std_steps": _std("steps"),
        "mean_total_reward": _mean("total_reward"),
        "mean_reward": _mean("mean_reward"),
        "nz_rmse_mean": _mean("nz_rmse"),
        "nz_rmse_std": _std("nz_rmse"),
        "nz_mae_mean": _mean("nz_mae"),
        "roll_rate_rmse_mean": _mean("roll_rate_rmse"),
        "roll_rate_rmse_std": _std("roll_rate_rmse"),
        "roll_rate_mae_mean": _mean("roll_rate_mae"),
        "mean_beta_deg": _mean("mean_beta_deg"),
        "max_beta_deg_max": _mean("max_beta_deg"),
        "mean_alpha_deg": _mean("mean_alpha_deg"),
        "max_alpha_deg_max": _mean("max_alpha_deg"),
        "mean_speed_mps": _mean("mean_speed_mps"),
        "min_speed_mps_mean": _mean("min_speed_mps"),
        "saturation_rate_mean": _mean("saturation_rate"),
        "final_range_m_mean": _mean("final_range_m"),
        "min_range_m_mean": _mean("min_range_m"),
    }


def run_ab_test(args):
    """Run A/B test comparing baseline and enhanced controllers."""
    seeds = list(range(args.seed, args.seed + args.n_episodes))

    print(f"\n{'='*60}")
    print(f"A/B Test: Baseline vs Enhanced Low-Level Controller")
    print(f"Episodes: {args.n_episodes}, Seeds: {seeds[0]}..{seeds[-1]}")
    print(f"{'='*60}\n")

    # --- Baseline ---
    print("[1/2] Running BASELINE controller...")
    baseline_config = build_config(
        use_enhanced=False,
        policy_path=args.policy,
        fixed_action=args.fixed_action,
        env_config_path=args.env_config,
    )

    baseline_env = CloseRangeTrackingEnv(baseline_config)
    baseline_agent = load_policy(baseline_config, device=args.device)
    baseline_fixed = baseline_config.get("fixed_action")

    baseline_results = []
    for i, seed in enumerate(seeds):
        try:
            result = run_episode(baseline_env, baseline_agent, baseline_fixed, seed=seed,
                                 max_steps=args.max_steps)
            baseline_results.append(result)
            if (i + 1) % 10 == 0 or args.n_episodes <= 10:
                print(f"  Baseline {i+1}/{args.n_episodes}: success={result['success']}, "
                      f"steps={result['steps']}, nz_rmse={result['nz_rmse']:.2f}, "
                      f"roll_rmse={result['roll_rate_rmse']:.2f}")
        except Exception as exc:
            print(f"  Baseline episode {i+1} failed: {exc}")
            traceback.print_exc()

    baseline_agg = aggregate_results(baseline_results)

    # --- Enhanced ---
    print("\n[2/2] Running ENHANCED controller...")
    enhanced_config = build_config(
        use_enhanced=True,
        policy_path=args.policy,
        fixed_action=args.fixed_action,
        env_config_path=args.env_config,
    )

    enhanced_env = CloseRangeTrackingEnv(enhanced_config)
    enhanced_agent = load_policy(enhanced_config, device=args.device)
    enhanced_fixed = enhanced_config.get("fixed_action")

    enhanced_results = []
    for i, seed in enumerate(seeds):
        try:
            result = run_episode(enhanced_env, enhanced_agent, enhanced_fixed, seed=seed,
                                 max_steps=args.max_steps)
            enhanced_results.append(result)
            if (i + 1) % 10 == 0 or args.n_episodes <= 10:
                print(f"  Enhanced {i+1}/{args.n_episodes}: success={result['success']}, "
                      f"steps={result['steps']}, nz_rmse={result['nz_rmse']:.2f}, "
                      f"roll_rmse={result['roll_rate_rmse']:.2f}")
        except Exception as exc:
            print(f"  Enhanced episode {i+1} failed: {exc}")
            traceback.print_exc()

    enhanced_agg = aggregate_results(enhanced_results)

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    def fmt(val, fmt=".3f"):
        return f"{val:{fmt}}"

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Enhanced':>15} {'Delta':>15}")
    print("-" * 80)

    metrics = [
        ("Success Rate", "success_rate", ".1%"),
        ("Mean Steps", "mean_steps", ".1f"),
        ("NZ RMSE (g)", "nz_rmse_mean", ".3f"),
        ("NZ RMSE std", "nz_rmse_std", ".3f"),
        ("Roll Rate RMSE (rad/s)", "roll_rate_rmse_mean", ".3f"),
        ("Roll Rate RMSE std", "roll_rate_rmse_std", ".3f"),
        ("Mean Beta (deg)", "mean_beta_deg", ".2f"),
        ("Max Beta (deg)", "max_beta_deg_max", ".2f"),
        ("Mean Alpha (deg)", "mean_alpha_deg", ".2f"),
        ("Max Alpha (deg)", "max_alpha_deg_max", ".2f"),
        ("Mean Speed (m/s)", "mean_speed_mps", ".1f"),
        ("Min Speed (m/s)", "min_speed_mps_mean", ".1f"),
        ("Saturation Rate", "saturation_rate_mean", ".1%"),
        ("Final Range (m)", "final_range_m_mean", ".1f"),
        ("Min Range (m)", "min_range_m_mean", ".1f"),
    ]

    comparison = {}
    for label, key, fmt_str in metrics:
        b = baseline_agg.get(key, 0.0)
        e = enhanced_agg.get(key, 0.0)
        if b != 0:
            delta = (e - b) / abs(b) * 100
        else:
            delta = 0.0 if e == 0 else float('inf')
        print(f"{label:<30} {fmt(b, fmt_str):>15} {fmt(e, fmt_str):>15} {delta:>+14.1f}%")
        comparison[key] = {"baseline": b, "enhanced": e, "delta_pct": delta}

    # --- Decision ---
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    success_improvement = comparison.get("success_rate", {}).get("delta_pct", 0)
    nz_improvement = comparison.get("nz_rmse_mean", {}).get("delta_pct", 0)
    roll_improvement = comparison.get("roll_rate_rmse_mean", {}).get("delta_pct", 0)
    beta_improvement = comparison.get("mean_beta_deg", {}).get("delta_pct", 0)

    if success_improvement > 5 or (nz_improvement < -10 and roll_improvement < -10):
        print("[PASS] ENHANCED controller shows SIGNIFICANT improvement.")
        print("   Recommendation: ADOPT enhanced controller.")
    elif success_improvement > 0 or nz_improvement < -5 or roll_improvement < -5:
        print("[PASS] ENHANCED controller shows MODERATE improvement.")
        print("   Recommendation: ADOPT enhanced controller with minor tuning.")
    elif abs(success_improvement) < 2 and abs(nz_improvement) < 5 and abs(roll_improvement) < 5:
        print("[WARN] No significant difference between controllers.")
        print("   Recommendation: Keep baseline (simpler, less risk).")
        print("   Or consider PPO training for further gains.")
    else:
        print("[FAIL] ENHANCED controller performs WORSE.")
        print("   Recommendation: KEEP baseline. Debug enhanced PID gains.")

    # --- Save results ---
    output_data = {
        "config": {
            "n_episodes": args.n_episodes,
            "seed_start": args.seed,
            "policy": args.policy,
            "fixed_action": args.fixed_action,
            "max_steps": args.max_steps,
        },
        "baseline": {
            "aggregate": baseline_agg,
            "episodes": baseline_results,
        },
        "enhanced": {
            "aggregate": enhanced_agg,
            "episodes": enhanced_results,
        },
        "comparison": comparison,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_data, indent=2, default=str), encoding="utf-8")
        print(f"\n[SAVE] Results saved to: {out_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="A/B test for low-level controller")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to trained PPO policy (.pt)")
    parser.add_argument("--fixed-action", type=str, default="0,0,0",
                        help="Fixed VPP offset action (no policy needed)")
    parser.add_argument("--config", type=str, default="config/canonical/guidance.yaml",
                        help="Guidance config path")
    parser.add_argument("--env-config", type=str, default="config/env.yaml",
                        help="Environment config path")
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Number of episodes per controller")
    parser.add_argument("--max-steps", type=int, default=512,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="results/controller_ab_test.json",
                        help="Output JSON path for results")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for policy inference")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (5 episodes)")

    args = parser.parse_args()

    if args.quick:
        args.n_episodes = 5
        args.max_steps = 200

    run_ab_test(args)


if __name__ == "__main__":
    main()
