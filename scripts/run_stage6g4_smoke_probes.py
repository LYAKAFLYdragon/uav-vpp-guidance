#!/usr/bin/env python3
"""
Stage 6G.4 Smoke Probes Runner.

Executes minimal smoke evaluations to decompose tail-chase failure root cause:
1. Oracle VPP anchor (perfect prediction)
2. Rule-based pursuit (pure geometry, bypass policy)
3. Terminal control ablation (capture radius, post-process, terminal protection)
4. Geometry feasibility sweep (small grid of initial conditions)

Usage:
    python scripts/run_stage6g4_smoke_probes.py --all
    python scripts/run_stage6g4_smoke_probes.py --oracle --rule-based
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_method,
    load_experiment_config,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_env_agent(config, method_name, method_override, allow_random_policy=False):
    """Create env + agent for a method."""
    method_config = merge_config(copy.deepcopy(config), copy.deepcopy(method_override))
    env = CloseRangeTrackingEnv(method_config)
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(method_config.get("policy", {}).get("action_dim", 3))
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device="cpu")

    method_ckpt = method_override.get("checkpoint")
    if method_ckpt and os.path.exists(method_ckpt):
        try:
            agent.load(method_ckpt)
            print(f"  Loaded checkpoint: {method_ckpt}")
        except RuntimeError as exc:
            if allow_random_policy or "size mismatch" in str(exc) or "Missing key" in str(exc):
                print(f"  WARNING: checkpoint load failed ({exc}), using random policy")
            else:
                raise
    elif allow_random_policy:
        print("  WARNING: checkpoint missing, using random policy")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {method_ckpt}")
    return env, agent, method_config


def _save_smoke_summary(output_dir, name, summary_dict):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}_smoke_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Summary saved: {path}")


def _method_success_breakdown(metrics):
    """Extract per-scenario success rates from evaluate_method output."""
    breakdown = {}
    for sc_name, sc_metrics in metrics.get("per_scenario", {}).items():
        breakdown[sc_name] = {
            "success_rate": sc_metrics.get("success_rate", np.nan),
            "crash_rate": sc_metrics.get("crash_rate", np.nan),
            "out_of_bounds_rate": sc_metrics.get("out_of_bounds_rate", np.nan),
            "timeout_rate": sc_metrics.get("timeout_rate", np.nan),
            "mean_final_range_m": sc_metrics.get("mean_final_range_m", np.nan),
            "episodes": sc_metrics.get("num_episodes", 0),
        }
    return breakdown


# ------------------------------------------------------------------
# 1. Oracle VPP Anchor Smoke
# ------------------------------------------------------------------

def run_oracle_smoke(config_path, output_dir, episodes=2, seeds=None):
    if seeds is None:
        seeds = [0]
    print("\n=== Stage 6G.4A: Oracle VPP Anchor Smoke ===")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    methods_cfg = config.get("methods", {})
    if not methods_cfg:
        print("ERROR: No methods in config.")
        return False

    all_results = {}
    for method_name, method_override in methods_cfg.items():
        print(f"\n-- Method: {method_name} --")
        env, agent, method_config = _make_env_agent(config, method_name, method_override)
        metrics = evaluate_method(
            env, agent, method_config, method_name,
            num_episodes=episodes, seeds=seeds,
            scenarios=["favorable", "disadvantage"],
        )
        env.close()
        all_results[method_name] = {
            "overall_success_rate": metrics.get("success_rate"),
            "overall_crash_rate": metrics.get("crash_rate"),
            "per_scenario": _method_success_breakdown(metrics),
        }
        print(f"  Success: {metrics['success_rate']:.2%} | Crash: {metrics['crash_rate']:.2%}")

    summary = {
        "probe": "oracle_vpp_anchor",
        "config": str(config_path),
        "episodes": episodes,
        "seeds": seeds,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "results": all_results,
    }
    _save_smoke_summary(output_dir, "oracle_anchor", summary)
    return True


# ------------------------------------------------------------------
# 2. Rule-Based Pursuit Smoke
# ------------------------------------------------------------------

def run_rule_based_smoke(config_path, output_dir, episodes=2, seeds=None):
    if seeds is None:
        seeds = [0]
    print("\n=== Stage 6G.4B: Rule-Based Pursuit Smoke ===")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    methods_cfg = config.get("methods", {})
    all_results = {}

    # Geometric direction check (static, no env needed)
    print("\n-- Geometric Direction Check --")
    vpg = VirtualPointGenerator(config.get("methods", {}).get("rule_based_pursuit_500m", {}).get("virtual_point", {}))
    own_state = {"position_neu": [0.0, 0.0, 5000.0]}
    target_state = {"position_neu": [1000.0, 0.0, 5000.0], "velocity_vector_mps": [180.0, 0.0, 0.0]}
    vp, info = vpg.action_to_virtual_point(
        np.zeros(3), own_state, target_state,
        anchor_mode="rule_based_pursuit", return_info=True
    )
    anchor_pos = info["anchor_pos"]
    # LOS from own to target is +x. Anchor should be ahead of target (+x direction)
    direction_correct = anchor_pos[0] > target_state["position_neu"][0]
    print(f"  Own: {own_state['position_neu']}, Target: {target_state['position_neu']}")
    print(f"  Anchor: {anchor_pos}, Direction correct (ahead of target): {direction_correct}")
    geo_check = {
        "own_position": own_state["position_neu"],
        "target_position": target_state["position_neu"],
        "anchor_position": anchor_pos.tolist() if hasattr(anchor_pos, "tolist") else list(anchor_pos),
        "direction_correct": bool(direction_correct),
    }

    for method_name, method_override in methods_cfg.items():
        print(f"\n-- Method: {method_name} --")
        env, agent, method_config = _make_env_agent(config, method_name, method_override)
        metrics = evaluate_method(
            env, agent, method_config, method_name,
            num_episodes=episodes, seeds=seeds,
            scenarios=["favorable", "disadvantage"],
        )
        env.close()
        all_results[method_name] = {
            "overall_success_rate": metrics.get("success_rate"),
            "overall_crash_rate": metrics.get("crash_rate"),
            "per_scenario": _method_success_breakdown(metrics),
        }
        print(f"  Success: {metrics['success_rate']:.2%} | Crash: {metrics['crash_rate']:.2%}")

    summary = {
        "probe": "rule_based_pursuit",
        "config": str(config_path),
        "episodes": episodes,
        "seeds": seeds,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "geometric_direction_check": geo_check,
        "results": all_results,
    }
    _save_smoke_summary(output_dir, "rule_based_pursuit", summary)
    return geo_check["direction_correct"]


# ------------------------------------------------------------------
# 3. Terminal Control Ablation Smoke
# ------------------------------------------------------------------

def run_terminal_control_ablation(config_path, output_dir, episodes=2, seeds=None):
    if seeds is None:
        seeds = [0]
    print("\n=== Stage 6G.4C: Terminal Control Ablation Smoke ===")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    methods_cfg = config.get("methods", {})
    method_name = "no_prediction"
    method_override = methods_cfg.get(method_name, {})

    # Define variants as guidance param overrides
    variants = {
        "baseline": {},
        "no_capture_radius": {"guidance": {"params": {"capture_radius_m": 0.0}}},
        "no_terminal_protection": {"guidance": {"post_process": {"enable_terminal_protection": False}}},
        "no_post_process": {"guidance": {"post_process": {"enabled": False}}},
        "no_energy_comp": {"guidance": {"post_process": {"enable_energy_compensation": False}}},
        "no_load_roll_coord": {"guidance": {"post_process": {"enable_load_roll_coordination": False}}},
    }

    all_results = {}
    for variant_name, override in variants.items():
        print(f"\n-- Variant: {variant_name} --")
        variant_config = merge_config(copy.deepcopy(config), copy.deepcopy(override))
        env, agent, method_config = _make_env_agent(variant_config, method_name, method_override)

        # Extract effective runtime flags for verification
        effective_flags = {
            "guidance_mode": getattr(env.guidance, "mode", type(env.guidance).__name__),
            "capture_radius_m": getattr(env.guidance, "capture_radius_m", None),
            "enable_internal_clip": getattr(env.guidance, "enable_internal_clip", None),
            "enable_internal_filter": getattr(env.guidance, "enable_internal_filter", None),
        }
        if env.command_post_processor is not None:
            cpp = env.command_post_processor
            effective_flags.update({
                "post_process_enabled": True,
                "enable_terminal_protection": getattr(cpp, "enable_terminal_protection", None),
                "terminal_range_m": getattr(cpp, "terminal_range_m", None),
                "enable_energy_compensation": getattr(cpp, "enable_energy_comp", None),
                "enable_load_roll_coordination": getattr(cpp, "enable_load_roll_coord", None),
                "nz_min": getattr(cpp, "nz_min", None),
                "nz_max": getattr(cpp, "nz_max", None),
                "roll_rate_min": getattr(cpp, "roll_rate_min", None),
                "roll_rate_max": getattr(cpp, "roll_rate_max", None),
            })
        else:
            effective_flags["post_process_enabled"] = False

        # Log flags for auditability
        print(f"  Effective flags: {effective_flags}")

        metrics = evaluate_method(
            env, agent, method_config, method_name,
            num_episodes=episodes, seeds=seeds,
            scenarios=["favorable", "disadvantage"],
        )
        env.close()
        all_results[variant_name] = {
            "overall_success_rate": metrics.get("success_rate"),
            "overall_crash_rate": metrics.get("crash_rate"),
            "overall_oob_rate": metrics.get("out_of_bounds_rate"),
            "overall_timeout_rate": metrics.get("timeout_rate"),
            "per_scenario": _method_success_breakdown(metrics),
            "requested_config": {
                "guidance": variant_config.get("guidance", {}),
            },
            "effective_runtime_flags": effective_flags,
        }
        print(f"  Success: {metrics['success_rate']:.2%} | Crash: {metrics['crash_rate']:.2%} | OOB: {metrics['out_of_bounds_rate']:.2%}")

    summary = {
        "probe": "terminal_control_ablation",
        "config": str(config_path),
        "episodes": episodes,
        "seeds": seeds,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "results": all_results,
    }
    _save_smoke_summary(output_dir, "terminal_control_ablation", summary)
    return True


# ------------------------------------------------------------------
# 4. Geometry Feasibility Sweep (small grid)
# ------------------------------------------------------------------

def run_geometry_feasibility(config_path, output_dir, episodes=1, seeds=None):
    if seeds is None:
        seeds = [0]
    print("\n=== Stage 6G.4D: Geometry Feasibility Smoke ===")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    sweeps = config.get("geometry_sweeps", {})
    template = config.get("scenario_template", {})
    method_name = "no_prediction"
    method_override = config.get("methods", {}).get(method_name, {})

    # Build a small grid: sample 1 value from each sweep axis to keep smoke fast
    sweep_axes = list(sweeps.keys())
    grid_points = []
    # Cartesian product of first 2 values per axis (max 2^4 = 16 combos)
    from itertools import product
    value_lists = []
    for axis in sweep_axes:
        vals = sweeps[axis].get("values", [])
        value_lists.append(vals[:2] if len(vals) >= 2 else vals[:1])

    for combo in product(*value_lists):
        point = dict(zip(sweep_axes, combo))
        grid_points.append(point)

    print(f"  Sweep axes: {sweep_axes}")
    print(f"  Grid points to evaluate: {len(grid_points)}")

    env, agent, method_config = _make_env_agent(config, method_name, method_override)
    all_episodes = []

    for gp in grid_points:
        scenario = copy.deepcopy(template)
        scenario["name"] = "_".join(f"{k}={v}" for k, v in gp.items())
        own_init = scenario.setdefault("own_init", {})
        target_init = scenario.setdefault("target_init", {})

        for axis, val in gp.items():
            if axis == "initial_range_m":
                # Place target along +x from own at base_position
                base_pos = sweeps[axis].get("base_position", [0.0, 0.0, 5000.0])
                target_init["position_m"] = [base_pos[0] + float(val), base_pos[1], base_pos[2]]
                own_init["position_m"] = base_pos
            elif axis == "ego_speed_mps":
                own_init["velocity_mps"] = float(val)
            elif axis == "target_speed_mps":
                target_init["velocity_mps"] = float(val)
            elif axis == "altitude_diff_m":
                base_alt = own_init.get("position_m", [0.0, 0.0, 5000.0])[2]
                target_init.setdefault("position_m", [800.0, 0.0, base_alt])[2] = base_alt + float(val)

        for seed in seeds:
            set_seed(seed)
            from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode as eval_ep
            ep_result, traj = eval_ep(
                env, agent, method_config, scenario=scenario, seed=seed,
                save_trajectory=False, method_name=method_name,
            )
            ep_result["training_seed"] = None
            ep_result["evaluation_seed"] = seed
            ep_result["episode_seed"] = seed
            ep_result["geometry_params"] = gp
            all_episodes.append(ep_result)
            status = "SUCCESS" if ep_result["is_success"] else ep_result["reason"]
            print(f"  {scenario['name']} | seed={seed} | {status} | final_range={ep_result['final_range_m']:.1f}m")

    env.close()

    # Aggregate
    success_by_params = {}
    for ep in all_episodes:
        gp = ep["geometry_params"]
        key = "_".join(f"{k}={v}" for k, v in gp.items())
        success_by_params.setdefault(key, {"success": 0, "total": 0, "params": gp})
        success_by_params[key]["total"] += 1
        if ep["is_success"]:
            success_by_params[key]["success"] += 1

    summary = {
        "probe": "geometry_feasibility",
        "config": str(config_path),
        "episodes": episodes,
        "seeds": seeds,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "grid_size": len(grid_points),
        "success_by_params": {
            k: {
                "success_rate": v["success"] / v["total"],
                "success_count": v["success"],
                "total": v["total"],
                "params": v["params"],
            }
            for k, v in success_by_params.items()
        },
        "raw_episodes": all_episodes,
    }
    _save_smoke_summary(output_dir, "geometry_feasibility", summary)
    return True


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 6G.4 Smoke Probes")
    parser.add_argument("--all", action="store_true", help="Run all smoke probes")
    parser.add_argument("--oracle", action="store_true", help="Run oracle anchor smoke")
    parser.add_argument("--rule-based", action="store_true", help="Run rule-based pursuit smoke")
    parser.add_argument("--terminal", action="store_true", help="Run terminal control ablation smoke")
    parser.add_argument("--geometry", action="store_true", help="Run geometry feasibility smoke")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per scenario per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Evaluation seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g4_smoke", help="Output directory")
    args = parser.parse_args()

    if not any([args.all, args.oracle, args.rule_based, args.terminal, args.geometry]):
        print("No probe selected. Use --all or specific flags.")
        parser.print_help()
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ok = True
    if args.all or args.oracle:
        ok = run_oracle_smoke(
            "config/experiment/stage6g3_oracle_vpp_anchor.yaml",
            output_dir, episodes=args.episodes, seeds=args.seeds,
        ) and ok
    if args.all or args.rule_based:
        ok = run_rule_based_smoke(
            "config/experiment/stage6g3_rule_based_pursuit.yaml",
            output_dir, episodes=args.episodes, seeds=args.seeds,
        ) and ok
    if args.all or args.terminal:
        ok = run_terminal_control_ablation(
            "config/experiment/stage6g3_terminal_protection_ablation.yaml",
            output_dir, episodes=args.episodes, seeds=args.seeds,
        ) and ok
    if args.all or args.geometry:
        ok = run_geometry_feasibility(
            "config/experiment/stage6g3_geometry_feasibility.yaml",
            output_dir, episodes=args.episodes, seeds=args.seeds,
        ) and ok

    print(f"\n{'='*60}")
    print(f"Stage 6G.4 smoke complete. Output: {output_dir}")
    print(f"Status: {'ALL OK' if ok else 'SOME PROBES FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
