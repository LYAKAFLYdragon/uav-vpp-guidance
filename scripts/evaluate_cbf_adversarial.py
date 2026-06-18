#!/usr/bin/env python3
"""
Evaluate a pursuer-target pair in AdversarialJSBSimEnv with optional CBF safety filter.

This script is a focused adversarial counterpart to evaluate_cbf_safety.py:
- Loads one pursuer checkpoint and one adversarial target checkpoint.
- Runs fixed scenarios (tail-chase, head-on, crossing, etc.).
- Records per-step CBF telemetry when enabled.
- Outputs JSON compatible with plot_cbf_adversarial.py.

Example:
    python scripts/evaluate_cbf_adversarial.py \
        --pursuer-ckpt outputs/jsbsim_10seed_matrix/vpp_s0/checkpoints/best.pt \
        --target-ckpt outputs/adversarial/training/target_agent_ppo_v2/checkpoints/best.pt \
        --use-cbf --cbf-config config/safety/cbf_default.yaml \
        --num-episodes 30 --seeds 0 1 2 \
        --output-dir outputs/cbf_adversarial/cbf_default
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluate_cbf_adversarial")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: Dict[str, Any] = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if not inc_full.exists():
            inc_full = Path(config_path).parent / ".." / os.path.basename(inc_path)
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, base_config)


def load_checkpoint_policy_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load policy config from the checkpoint's config_snapshot.yaml if present."""
    snapshot = Path(checkpoint_path).parent.parent / "config_snapshot.yaml"
    if snapshot.exists():
        cfg = load_yaml_config(str(snapshot))
        return cfg.get("policy")
    return None


def make_default_scenario(range_m: float = 2000.0, heading_offset_deg: float = 0.0) -> Dict[str, Any]:
    import math
    hdg_rad = math.radians(heading_offset_deg)
    return {
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [range_m * math.cos(hdg_rad), range_m * math.sin(hdg_rad), 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": heading_offset_deg,
        },
    }


def make_head_on_scenario(range_m: float = 4000.0) -> Dict[str, Any]:
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


def run_adversarial_episode(
    env: AdversarialJSBSimEnv,
    pursuer_agent: PPOAgent,
    target_agent: AdversarialTargetAgent,
    scenario: Any,
    seed: int,
    d_min: float,
) -> Dict[str, Any]:
    """Run one adversarial episode and record per-step telemetry."""
    p_obs, t_obs = env.reset(scenario=scenario, seed=seed)

    p_return = 0.0
    t_return = 0.0
    ep_length = 0
    min_range = float("inf")
    min_h = float("inf")
    final_range = 0.0
    final_ata = 0.0
    reason = "timeout"

    cbf_active_steps = 0
    cbf_solve_times_ms: List[float] = []
    cbf_fallback_count = 0
    trajectory: List[Dict[str, Any]] = []

    for step in range(env.max_steps):
        p_action = pursuer_agent.get_deterministic_action(p_obs["observation_vector"])
        t_action = target_agent.get_deterministic_action(t_obs["observation_vector"])

        p_obs, t_obs, p_rew, t_rew, terminated, truncated, info = env.step(p_action, t_action)
        p_return += p_rew
        t_return += t_rew
        ep_length += 1

        rel = p_obs.get("relative_state", {})
        range_m = float(rel.get("range_m", 0.0))
        ata_deg = float(np.rad2deg(rel.get("ata_rad", 0.0)))
        h = range_m - d_min
        min_range = min(min_range, range_m)
        min_h = min(min_h, h)
        final_range = range_m
        final_ata = ata_deg

        cbf_info = info.get("cbf")
        step_record: Dict[str, Any] = {
            "step": step,
            "range_m": range_m,
            "h": h,
            "range_rate_mps": float(rel.get("range_rate_mps", np.nan)),
        }
        if cbf_info:
            step_record["cbf_active"] = bool(cbf_info.get("active", False))
            step_record["cbf_solve_time_ms"] = float(cbf_info.get("solve_time_ms", 0.0))
            step_record["cbf_fallback"] = bool(cbf_info.get("fallback", False))
            step_record["cbf_constraint_value"] = float(cbf_info.get("constraint_value", np.nan))
            if cbf_info.get("active", False):
                cbf_active_steps += 1
            if cbf_info.get("fallback", False):
                cbf_fallback_count += 1
            solve_time = float(cbf_info.get("solve_time_ms", 0.0))
            if solve_time > 0:
                cbf_solve_times_ms.append(solve_time)
        trajectory.append(step_record)

        if terminated or truncated:
            reason = info.get("termination", {}).get("reason", "unknown")
            break

    return {
        "pursuer_return": float(p_return),
        "target_return": float(t_return),
        "length": ep_length,
        "min_range_m": float(min_range),
        "min_h": float(min_h),
        "final_range_m": float(final_range),
        "final_ata_deg": float(final_ata),
        "reason": reason,
        "captured": reason == "success",
        "survived": reason == "timeout",
        "crash": reason == "crash",
        "out_of_bounds": reason == "out_of_bounds",
        "cbf_active_steps": cbf_active_steps,
        "cbf_solve_times_ms": cbf_solve_times_ms,
        "cbf_fallback_count": cbf_fallback_count,
        "trajectory": trajectory,
    }


def aggregate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {}

    def _count(key: str) -> int:
        return sum(1 for ep in episodes if ep.get(key, False))

    min_ranges = [ep["min_range_m"] for ep in episodes if np.isfinite(ep["min_range_m"])]
    min_hs = [ep["min_h"] for ep in episodes if np.isfinite(ep["min_h"])]
    solve_times = [t for ep in episodes for t in ep.get("cbf_solve_times_ms", [])]
    total_steps = sum(ep["length"] for ep in episodes)

    capture_times = [ep["length"] for ep in episodes if ep["captured"]]

    return {
        "n_episodes": total,
        "success_rate": _count("captured") / total,
        "crash_rate": _count("crash") / total,
        "oob_rate": _count("out_of_bounds") / total,
        "timeout_rate": _count("survived") / total,
        "mean_min_range_m": float(np.mean(min_ranges)) if min_ranges else np.nan,
        "min_min_range_m": float(np.min(min_ranges)) if min_ranges else np.nan,
        "mean_min_h": float(np.mean(min_hs)) if min_hs else np.nan,
        "min_h": float(np.min(min_hs)) if min_hs else np.nan,
        "mean_capture_time": float(np.mean(capture_times)) if capture_times else np.nan,
        "cbf_intervention_rate": (
            sum(ep["cbf_active_steps"] for ep in episodes) / total_steps if total_steps else 0.0
        ),
        "mean_cbf_solve_time_ms": float(np.mean(solve_times)) if solve_times else 0.0,
        "max_cbf_solve_time_ms": float(np.max(solve_times)) if solve_times else 0.0,
        "cbf_fallback_count": sum(ep["cbf_fallback_count"] for ep in episodes),
    }


def evaluate(
    config: Dict[str, Any],
    pursuer_ckpt: str,
    target_ckpt: str,
    scenarios: List[str],
    num_episodes: int,
    seeds: List[int],
    use_cbf: bool,
    cbf_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ppo_cfg = config.get("ppo", {})
    policy_cfg = config.get("policy", {})

    pursuer_cfg = dict(config)
    pursuer_policy = load_checkpoint_policy_config(pursuer_ckpt)
    pursuer_cfg["policy"] = pursuer_policy if pursuer_policy is not None else policy_cfg

    target_cfg = dict(config)
    target_policy = load_checkpoint_policy_config(target_ckpt)
    target_cfg["ppo"] = config.get("target_ppo", ppo_cfg)
    target_cfg["policy"] = target_policy if target_policy is not None else config.get("target_policy", policy_cfg)

    # Inject CBF config into environment config.
    env_cfg = dict(config)
    if use_cbf:
        env_cfg["cbf"] = cbf_config or {"enabled": True}
    elif "cbf" in env_cfg:
        env_cfg["cbf"] = {"enabled": False}

    env = AdversarialJSBSimEnv(env_cfg)

    pursuer_agent = PPOAgent(16, 3, pursuer_cfg)
    pursuer_agent.load(pursuer_ckpt)
    pursuer_agent.network.eval()

    target_agent = AdversarialTargetAgent(config=target_cfg)
    target_agent.load(target_ckpt)
    target_agent.eval()

    d_min = 500.0
    if use_cbf and cbf_config and "params" in cbf_config:
        d_min = cbf_config["params"].get("d_min", 500.0)

    sampler = None
    if config.get("scenario_sampler"):
        sampler = make_scenario_sampler(config["scenario_sampler"])

    per_scenario: Dict[str, List[Dict[str, Any]]] = {}
    all_episodes: List[Dict[str, Any]] = []

    try:
        for sc_name in scenarios:
            builder = SCENARIO_BUILDERS.get(sc_name)
            if builder is None:
                logger.warning("Unknown scenario: %s, skipping", sc_name)
                continue
            base_scenario = builder()
            sc_results: List[Dict[str, Any]] = []

            for seed in seeds:
                for ep in range(num_episodes):
                    ep_seed = seed * 10000 + ep
                    scenario = sampler.sample() if sampler else base_scenario
                    ep_result = run_adversarial_episode(
                        env, pursuer_agent, target_agent, scenario, ep_seed, d_min
                    )
                    ep_result["scenario"] = sc_name
                    ep_result["seed"] = seed
                    ep_result["episode"] = ep
                    sc_results.append(ep_result)
                    all_episodes.append(ep_result)

            per_scenario[sc_name] = sc_results
    finally:
        env.close()

    return {
        "per_scenario": {s: aggregate_episodes(eps) for s, eps in per_scenario.items()},
        "overall": aggregate_episodes(all_episodes),
        "raw_episodes": all_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CBF in adversarial scenarios")
    parser.add_argument("--config", type=str, default="config/adversarial/evaluate.yaml")
    parser.add_argument("--pursuer-ckpt", type=str, required=True)
    parser.add_argument("--target-ckpt", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--scenarios", type=str, nargs="+", default=None)
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument("--cbf-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    scenarios = args.scenarios or config.get("scenarios", [
        "tail_chase_2000m", "head_on_4000m", "crossing_90deg_2000m", "offset_45deg_2000m"
    ])

    cbf_config = None
    if args.use_cbf:
        if args.cbf_config:
            cbf_config = load_yaml_config(args.cbf_config).get("cbf", {"enabled": True})
        else:
            cbf_config = {"enabled": True}

    t0 = time.time()
    results = evaluate(
        config=config,
        pursuer_ckpt=args.pursuer_ckpt,
        target_ckpt=args.target_ckpt,
        scenarios=scenarios,
        num_episodes=args.num_episodes,
        seeds=args.seeds,
        use_cbf=args.use_cbf,
        cbf_config=cbf_config,
    )
    elapsed = time.time() - t0

    summary = {
        "pursuer_ckpt": args.pursuer_ckpt,
        "target_ckpt": args.target_ckpt,
        "use_cbf": args.use_cbf,
        "cbf_config": cbf_config,
        "scenarios": scenarios,
        "seeds": args.seeds,
        "num_episodes": args.num_episodes,
        "elapsed_time_s": elapsed,
        "timestamp": datetime.now().isoformat(),
        **results,
    }

    print(f"\n=== Adversarial CBF evaluation: CBF={args.use_cbf} ===")
    print(f"{'Scenario':<20} {'Success':>8} {'Crash':>8} {'OOB':>8} {'Timeout':>8} {'MinRange':>10} {'CBF_Act':>10}")
    for scen, stat in summary["per_scenario"].items():
        print(
            f"{scen:<20} {stat['success_rate']*100:>7.1f}% {stat['crash_rate']*100:>7.1f}% "
            f"{stat['oob_rate']*100:>7.1f}% {stat['timeout_rate']*100:>7.1f}% "
            f"{stat['mean_min_range_m']:>9.1f}m {stat['cbf_intervention_rate']*100:>9.1f}%"
        )
    ov = summary["overall"]
    print(
        f"{'Overall':<20} {ov['success_rate']*100:>7.1f}% {ov['crash_rate']*100:>7.1f}% "
        f"{ov['oob_rate']*100:>7.1f}% {ov['timeout_rate']*100:>7.1f}% "
        f"{ov['mean_min_range_m']:>9.1f}m {ov['cbf_intervention_rate']*100:>9.1f}%"
    )
    if args.use_cbf:
        print(
            f"\nCBF mean solve time: {ov['mean_cbf_solve_time_ms']:.3f} ms  "
            f"max: {ov['max_cbf_solve_time_ms']:.3f} ms  fallbacks: {ov['cbf_fallback_count']}"
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
