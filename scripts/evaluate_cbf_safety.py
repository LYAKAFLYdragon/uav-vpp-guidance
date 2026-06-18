#!/usr/bin/env python3
"""
Evaluate a trained VPP/No-VPP checkpoint with and without the CBF safety filter.

Runs the four canonical scenarios and reports success/crash rates, minimum
separation distances, and CBF activation statistics.

Examples
--------
VPP baseline (no CBF):
    python scripts/evaluate_cbf_safety.py \
        --checkpoint outputs/experiments/p0a_vpp_s0/checkpoints/best.pt \
        --method vpp --backend jsbsim \
        --output-dir outputs/cbf_safety/vpp_baseline

VPP with CBF:
    python scripts/evaluate_cbf_safety.py \
        --checkpoint outputs/experiments/p0a_vpp_s0/checkpoints/best.pt \
        --method vpp --backend jsbsim --use-cbf \
        --cbf-config config/safety/cbf_default.yaml \
        --output-dir outputs/cbf_safety/vpp_cbf

No-VPP with CBF:
    python scripts/evaluate_cbf_safety.py \
        --method no_vpp --backend jsbsim --use-cbf \
        --cbf-config config/safety/cbf_default.yaml \
        --output-dir outputs/cbf_safety/no_vpp_cbf
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


SCENARIOS = ["favorable", "neutral", "challenging", "disadvantage"]


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    base = load_yaml_config(config_path)
    includes = base.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = config_path.parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base)


def get_scenario_dicts(base_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Load canonical scenario definitions from config."""
    if base_config is not None and "scenarios" in base_config:
        return base_config["scenarios"]
    default_path = PROJECT_ROOT / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml"
    default_cfg = load_experiment_config(default_path)
    return default_cfg.get("scenarios", {})


def load_checkpoint_config(checkpoint_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None:
        return None
    ckpt_path = Path(checkpoint_path)
    snapshot = ckpt_path.parent.parent / "config_snapshot.yaml"
    if snapshot.exists():
        return load_yaml_config(snapshot)
    return None


def make_env(
    method: str,
    backend: str = "jsbsim",
    use_cbf: bool = False,
    cbf_config: Optional[Dict[str, Any]] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> CloseRangeTrackingEnv:
    """Build the evaluation environment."""
    if base_config is None:
        config_path = PROJECT_ROOT / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml"
        config = load_experiment_config(config_path)
    else:
        config = base_config

    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"

    if method == "e2e":
        config["end_to_end"] = {"enabled": True}
        config["virtual_point"]["enabled"] = False
    elif method == "no_vpp":
        config["virtual_point"]["enabled"] = True
        config["virtual_point"]["mode"] = "zero_offset"
    else:  # vpp
        config["virtual_point"]["enabled"] = True
        config["virtual_point"].pop("mode", None)

    if use_cbf:
        config["cbf"] = cbf_config or {"enabled": True}
    elif "cbf" in config:
        # Ensure a baseline run does not accidentally inherit a CBF config.
        config["cbf"] = {"enabled": False}

    return CloseRangeTrackingEnv(config)


def load_agent(env: CloseRangeTrackingEnv, checkpoint_path: Optional[str]) -> Optional[PPOAgent]:
    if checkpoint_path is None:
        return None
    obs = env.reset(seed=0, scenario="favorable")
    obs_dim = obs["observation_vector"].shape[0]
    act_dim = 3
    agent = PPOAgent(obs_dim, act_dim, env.config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    agent.network.load_state_dict(ckpt["network_state_dict"])
    agent.network.eval()
    return agent


def run_episode(
    env: CloseRangeTrackingEnv,
    agent: Optional[PPOAgent],
    scenario: Dict[str, Any],
    seed: int,
    d_min: float,
) -> Dict[str, Any]:
    obs = env.reset(seed=seed, scenario=scenario)
    done = False

    min_range_m = float("inf")
    min_h = float("inf")
    max_h_dot = -float("inf")

    cbf_active_steps = 0
    cbf_solve_times_ms: List[float] = []
    cbf_fallback_count = 0
    cbf_min_constraint_value = float("inf")

    while not done:
        if agent is None:
            action = np.zeros(3, dtype=np.float64)
        else:
            action = agent.select_action(
                obs["observation_vector"], deterministic=True, store=False
            )[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rel_state = info.get("relative_state") or obs.get("relative_state", {})
        range_m = float(rel_state.get("range_m", np.nan))
        h = range_m - d_min
        h_dot = float(rel_state.get("range_rate_mps", np.nan))
        min_range_m = min(min_range_m, range_m)
        min_h = min(min_h, h)
        max_h_dot = max(max_h_dot, h_dot)

        cbf_info = info.get("cbf")
        if cbf_info:
            if cbf_info.get("active", False):
                cbf_active_steps += 1
            if cbf_info.get("fallback", False):
                cbf_fallback_count += 1
            solve_time = cbf_info.get("solve_time_ms", 0.0)
            if solve_time > 0:
                cbf_solve_times_ms.append(solve_time)
            cv = cbf_info.get("constraint_value", np.nan)
            if np.isfinite(cv):
                cbf_min_constraint_value = min(cbf_min_constraint_value, cv)

    term_info = info.get("termination_info", {})
    reason = term_info.get("reason", "unknown")

    return {
        "scenario": scenario,
        "seed": seed,
        "reason": reason,
        "length": env.current_step,
        "min_range_m": min_range_m,
        "final_range_m": float(rel_state.get("range_m", np.nan)),
        "min_h": min_h,
        "max_h_dot": max_h_dot,
        "cbf_active_steps": cbf_active_steps,
        "cbf_solve_times_ms": cbf_solve_times_ms,
        "cbf_fallback_count": cbf_fallback_count,
        "cbf_min_constraint_value": cbf_min_constraint_value,
    }


def aggregate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {}

    def _count(reason: str) -> int:
        return sum(1 for ep in episodes if ep["reason"] == reason)

    min_ranges = [ep["min_range_m"] for ep in episodes if np.isfinite(ep["min_range_m"])]
    min_hs = [ep["min_h"] for ep in episodes if np.isfinite(ep["min_h"])]
    solve_times = [
        t
        for ep in episodes
        for t in ep["cbf_solve_times_ms"]
    ]
    active_steps = sum(ep["cbf_active_steps"] for ep in episodes)
    total_steps = sum(ep["length"] for ep in episodes)

    return {
        "n_episodes": total,
        "success_rate": _count("success") / total,
        "crash_rate": _count("crash") / total,
        "oob_rate": _count("out_of_bounds") / total,
        "timeout_rate": _count("timeout") / total,
        "mean_min_range_m": float(np.mean(min_ranges)) if min_ranges else np.nan,
        "min_min_range_m": float(np.min(min_ranges)) if min_ranges else np.nan,
        "mean_min_h": float(np.mean(min_hs)) if min_hs else np.nan,
        "min_h": float(np.min(min_hs)) if min_hs else np.nan,
        "cbf_intervention_rate": active_steps / total_steps if total_steps else 0.0,
        "mean_cbf_solve_time_ms": float(np.mean(solve_times)) if solve_times else 0.0,
        "max_cbf_solve_time_ms": float(np.max(solve_times)) if solve_times else 0.0,
        "cbf_fallback_count": sum(ep["cbf_fallback_count"] for ep in episodes),
    }


def evaluate(
    checkpoint_path: Optional[str],
    method: str,
    backend: str,
    use_cbf: bool,
    cbf_config: Optional[Dict[str, Any]],
    scenarios: List[str],
    seeds: List[int],
    n_eps: int,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    env = make_env(method, backend, use_cbf, cbf_config, base_config)
    agent = load_agent(env, checkpoint_path)

    d_min = 500.0
    if use_cbf and cbf_config and "params" in cbf_config:
        d_min = cbf_config["params"].get("d_min", 500.0)

    per_scenario: Dict[str, List[Dict[str, Any]]] = {s: [] for s in scenarios}
    all_episodes: List[Dict[str, Any]] = []

    try:
        scenario_dicts = get_scenario_dicts(base_config)
        scenario_objs = {s: scenario_dicts.get(s, s) for s in scenarios}
        for scenario_name in scenarios:
            scenario_obj = scenario_objs[scenario_name]
            for seed in seeds:
                for ep_idx in range(n_eps):
                    ep_seed = seed * 1000 + ep_idx + 1
                    ep = run_episode(env, agent, scenario_obj, ep_seed, d_min)
                    ep["scenario_name"] = scenario_name
                    per_scenario[scenario_name].append(ep)
                    all_episodes.append(ep)
    finally:
        env.close()

    results = {
        "per_scenario": {s: aggregate_episodes(eps) for s, eps in per_scenario.items()},
        "overall": aggregate_episodes(all_episodes),
        "raw_episodes": all_episodes,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBF safety filter")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to PPO checkpoint.  Not required for method=no_vpp.",
    )
    parser.add_argument(
        "--method", type=str, default="vpp", choices=["vpp", "no_vpp", "e2e"],
    )
    parser.add_argument("--backend", type=str, default="jsbsim", choices=["simple", "jsbsim"])
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument("--cbf-config", type=str, default=None)
    parser.add_argument(
        "--scenarios", type=str, nargs="+", default=SCENARIOS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-eps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.method != "no_vpp" and args.checkpoint is None:
        parser.error("--checkpoint is required when method is not 'no_vpp'")

    cbf_config = None
    if args.use_cbf:
        if args.cbf_config:
            cbf_config = load_yaml_config(args.cbf_config).get("cbf", {"enabled": True})
        else:
            cbf_config = {"enabled": True}

    base_config = load_checkpoint_config(args.checkpoint)

    t0 = time.time()
    results = evaluate(
        checkpoint_path=args.checkpoint,
        method=args.method,
        backend=args.backend,
        use_cbf=args.use_cbf,
        cbf_config=cbf_config,
        base_config=base_config,
        scenarios=args.scenarios,
        seeds=args.seeds,
        n_eps=args.n_eps,
    )
    elapsed = time.time() - t0

    summary = {
        "checkpoint": args.checkpoint,
        "method": args.method,
        "backend": args.backend,
        "cbf_enabled": args.use_cbf,
        "cbf_config": cbf_config,
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "n_eps_per_scenario": args.n_eps,
        "elapsed_time_s": elapsed,
        "timestamp": datetime.now().isoformat(),
        **results,
    }

    print(f"\n=== CBF safety evaluation: {args.method} (CBF={args.use_cbf}) ===")
    print(f"{'Scenario':<14} {'Success':>8} {'Crash':>8} {'OOB':>8} {'Timeout':>8} {'MinRange':>10}")
    for scen in args.scenarios:
        stat = summary["per_scenario"][scen]
        print(
            f"{scen:<14} {stat['success_rate']*100:>7.1f}% {stat['crash_rate']*100:>7.1f}% "
            f"{stat['oob_rate']*100:>7.1f}% {stat['timeout_rate']*100:>7.1f}% "
            f"{stat['mean_min_range_m']:>9.1f}m"
        )
    ov = summary["overall"]
    print(
        f"{'Overall':<14} {ov['success_rate']*100:>7.1f}% {ov['crash_rate']*100:>7.1f}% "
        f"{ov['oob_rate']*100:>7.1f}% {ov['timeout_rate']*100:>7.1f}% "
        f"{ov['mean_min_range_m']:>9.1f}m"
    )
    if args.use_cbf:
        print(
            f"\nCBF intervention rate: {ov['cbf_intervention_rate']*100:.1f}%  "
            f"mean solve time: {ov['mean_cbf_solve_time_ms']:.3f} ms  "
            f"max solve time: {ov['max_cbf_solve_time_ms']:.3f} ms  "
            f"fallbacks: {ov['cbf_fallback_count']}"
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
