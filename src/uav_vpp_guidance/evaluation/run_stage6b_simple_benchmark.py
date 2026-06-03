"""
Stage 6B: Full Simple-Backend Benchmark Runner.

Runs no_prediction, cv_prediction, and ca_prediction on fixed scenarios
with multiple seeds, generates unified metrics CSV/JSON, and produces
a summary.md with statistical comparison.

Usage:
    # Smoke (minimal, for CI)
    python -m uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark \
        --config config/experiment/benchmark_simple_prediction_comparison.yaml --smoke

    # Small run
    python -m uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark \
        --config config/experiment/benchmark_simple_prediction_comparison.yaml \
        --episodes 3 --seeds 0 1 --scenarios favorable neutral

    # Full run (uses config defaults)
    python -m uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark \
        --config config/experiment/benchmark_simple_prediction_comparison.yaml
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.utils.reproducibility import get_run_metadata, save_run_metadata
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.evaluation.statistical_comparison import compare_methods


def load_experiment_config(config_path):
    """Load and merge experiment configuration with includes."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def sample_scenario(config, rng):
    """Sample a random scenario from config."""
    scenarios = config.get("scenarios", {})
    if not scenarios:
        return None
    name = rng.choice(list(scenarios.keys()))
    return scenarios[name]


def evaluate_single_episode(
    env, agent, config, scenario=None, seed=0, save_trajectory=False, method_name=""
):
    """Evaluate a single episode and return metrics + trajectory."""
    obs = env.reset(scenario=scenario, seed=seed)
    ep_reward = 0.0
    ep_length = 0
    min_range = float("inf")
    min_ata_deg = float("inf")
    final_range = 0.0
    final_ata = 0.0
    reason = "timeout"
    trajectory = []
    prediction_enabled_count = 0
    prediction_valid_count = 0
    prediction_fallback_count = 0
    prediction_errors = []
    virtual_point_shifts = []
    anchor_shifts = []
    ego_score_sum = 0.0
    target_score_sum = 0.0

    for step in range(env.max_steps):
        obs_vec = obs["observation_vector"]
        action = agent.get_deterministic_action(obs_vec)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1

        rel_state = obs.get("relative_state", {})
        range_m = rel_state.get("range_m", 0.0)
        ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
        min_range = min(min_range, range_m)
        min_ata_deg = min(min_ata_deg, ata_deg)
        final_range = range_m
        final_ata = ata_deg

        ego_score = info.get("ego_score", 0.0)
        target_score = info.get("target_score", 0.0)
        ego_score_sum += ego_score
        target_score_sum += target_score

        if info.get("prediction_enabled"):
            prediction_enabled_count += 1
            if info.get("prediction_valid"):
                prediction_valid_count += 1
            if info.get("prediction_fallback_reason"):
                prediction_fallback_count += 1

        pred_error = info.get("prediction_error_m", np.nan)
        if np.isfinite(pred_error):
            prediction_errors.append(pred_error)

        target_pos = info.get("target_state", {}).get("position_m")
        if target_pos is None:
            target_pos = info.get("target_state", {}).get("position_neu")
        vp_pos = info.get("virtual_point", {}).get("position")
        if target_pos is not None and vp_pos is not None:
            virtual_point_shifts.append(
                float(np.linalg.norm(np.asarray(vp_pos) - np.asarray(target_pos)))
            )

        pred_target_pos = info.get("predicted_target_position")
        if target_pos is not None and pred_target_pos is not None:
            anchor_shifts.append(
                float(
                    np.linalg.norm(np.asarray(pred_target_pos) - np.asarray(target_pos))
                )
            )

        if save_trajectory:
            own_s = info.get("own_state", {})
            target_s = info.get("target_state", {})
            own_pos = own_s.get(
                "position_m", own_s.get("position_neu", np.full(3, np.nan))
            )
            target_pos_arr = target_s.get(
                "position_m", target_s.get("position_neu", np.full(3, np.nan))
            )
            # target velocity available if needed for trajectory analysis
            pred_target = info.get(
                "predicted_target_position", [np.nan, np.nan, np.nan]
            )
            vp = info.get("virtual_point", {})
            vp_pos_arr = vp.get("position", np.full(3, np.nan))

            trajectory.append(
                {
                    "step": step,
                    "time": step * env.env_config.get("high_level_dt", 0.2),
                    "backend": env._backend,
                    "method": method_name,
                    "predictor_type": info.get("predictor_type", ""),
                    "prediction_enabled": int(info.get("prediction_enabled", False)),
                    "prediction_valid": int(info.get("prediction_valid", False)),
                    "prediction_fallback_reason": info.get(
                        "prediction_fallback_reason", ""
                    ),
                    "target_x": (
                        float(target_pos_arr[0]) if len(target_pos_arr) > 0 else np.nan
                    ),
                    "target_y": (
                        float(target_pos_arr[1]) if len(target_pos_arr) > 1 else np.nan
                    ),
                    "target_z": (
                        float(target_pos_arr[2]) if len(target_pos_arr) > 2 else np.nan
                    ),
                    "predicted_target_x": (
                        float(pred_target[0]) if pred_target is not None else np.nan
                    ),
                    "predicted_target_y": (
                        float(pred_target[1]) if pred_target is not None else np.nan
                    ),
                    "predicted_target_z": (
                        float(pred_target[2]) if pred_target is not None else np.nan
                    ),
                    "prediction_error_m": (
                        float(pred_error) if np.isfinite(pred_error) else np.nan
                    ),
                    "virtual_x": (
                        float(vp_pos_arr[0]) if len(vp_pos_arr) > 0 else np.nan
                    ),
                    "virtual_y": (
                        float(vp_pos_arr[1]) if len(vp_pos_arr) > 1 else np.nan
                    ),
                    "virtual_z": (
                        float(vp_pos_arr[2]) if len(vp_pos_arr) > 2 else np.nan
                    ),
                    "ego_x": float(own_pos[0]) if len(own_pos) > 0 else np.nan,
                    "ego_y": float(own_pos[1]) if len(own_pos) > 1 else np.nan,
                    "ego_z": float(own_pos[2]) if len(own_pos) > 2 else np.nan,
                    "range_m": range_m,
                    "ata_deg": ata_deg,
                    "nz_cmd": info.get("nz_cmd", np.nan),
                    "roll_rate_cmd": info.get("roll_rate_cmd", np.nan),
                    "throttle_cmd": info.get("throttle_cmd", np.nan),
                    "done": int(terminated or truncated),
                    "termination_reason": info.get("reason", ""),
                }
            )

        if terminated or truncated:
            reason = info.get("reason", "unknown")
            break

    return {
        "seed": seed,
        "scenario": (
            scenario.get("name", "random") if isinstance(scenario, dict) else "random"
        ),
        "return": ep_reward,
        "length": ep_length,
        "min_range_m": min_range,
        "min_ata_deg": min_ata_deg,
        "final_range_m": final_range,
        "final_ata_deg": final_ata,
        "reason": reason,
        "is_success": reason == "success",
        "is_crash": reason == "crash",
        "is_timeout": reason == "timeout",
        "is_out_of_bounds": reason == "out_of_bounds",
        "score_win": ego_score_sum > target_score_sum,
        "prediction_enabled_rate": prediction_enabled_count / max(1, ep_length),
        "prediction_valid_rate": prediction_valid_count / max(1, ep_length),
        "prediction_fallback_rate": prediction_fallback_count / max(1, ep_length),
        "mean_prediction_error_m": (
            float(np.mean(prediction_errors)) if prediction_errors else np.nan
        ),
        "mean_virtual_point_shift_m": (
            float(np.mean(virtual_point_shifts)) if virtual_point_shifts else np.nan
        ),
        "mean_anchor_shift_m": (
            float(np.mean(anchor_shifts)) if anchor_shifts else np.nan
        ),
    }, trajectory


def aggregate_metrics(episodes):
    """Aggregate metrics from a list of episode results."""
    if not episodes:
        return {}
    returns = [e["return"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    final_ranges = [e["final_range_m"] for e in episodes]
    final_atas = [e["final_ata_deg"] for e in episodes]
    min_ranges = [e["min_range_m"] for e in episodes]
    min_atas = [e["min_ata_deg"] for e in episodes]

    def safe_mean(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else np.nan

    result = {
        "num_episodes": len(episodes),
        "mean_return": safe_mean(returns),
        "std_return": float(np.std(returns)) if returns else np.nan,
        "mean_length": safe_mean(lengths),
        "success_rate": sum(1 for e in episodes if e["is_success"]) / len(episodes),
        "crash_rate": sum(1 for e in episodes if e["is_crash"]) / len(episodes),
        "out_of_bounds_rate": sum(1 for e in episodes if e["is_out_of_bounds"])
        / len(episodes),
        "timeout_rate": sum(1 for e in episodes if e["is_timeout"]) / len(episodes),
        "score_win_rate": sum(1 for e in episodes if e.get("score_win", False))
        / len(episodes),
        "mean_final_range_m": safe_mean(final_ranges),
        "mean_final_ata_deg": safe_mean(final_atas),
        "mean_min_range_m": safe_mean(min_ranges),
        "mean_min_ata_deg": safe_mean(min_atas),
        "mean_prediction_enabled_rate": safe_mean(
            [e["prediction_enabled_rate"] for e in episodes]
        ),
        "mean_prediction_valid_rate": safe_mean(
            [e["prediction_valid_rate"] for e in episodes]
        ),
        "mean_prediction_fallback_rate": safe_mean(
            [e["prediction_fallback_rate"] for e in episodes]
        ),
        "mean_prediction_error_m": safe_mean(
            [e["mean_prediction_error_m"] for e in episodes]
        ),
        "mean_virtual_point_shift_m": safe_mean(
            [e["mean_virtual_point_shift_m"] for e in episodes]
        ),
        "mean_anchor_shift_m": safe_mean([e["mean_anchor_shift_m"] for e in episodes]),
    }
    result["instant_success_rate"] = result["success_rate"]
    result["prediction_rmse_m"] = result["mean_prediction_error_m"]
    result["prediction_fallback_rate"] = result["mean_prediction_fallback_rate"]
    return result


def evaluate_method(
    env,
    agent,
    config,
    method_name,
    num_episodes=10,
    seeds=None,
    scenarios=None,
    save_trajectories=False,
    output_dir=None,
):
    """Evaluate a single method across multiple seeds and optional fixed scenarios."""
    if seeds is None:
        seeds = [0, 1, 2]

    all_episodes = []
    per_scenario_episodes = {}
    per_seed_results = {}

    for seed in seeds:
        set_seed(seed)
        seed_episodes = []

        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)

            if scenarios:
                scenario_name = scenarios[ep % len(scenarios)]
                scenario = config.get("scenarios", {}).get(scenario_name)
            else:
                scenario = sample_scenario(config, rng)
                scenario_name = (
                    scenario.get("name", "random")
                    if isinstance(scenario, dict)
                    else "random"
                )

            ep_result, trajectory = evaluate_single_episode(
                env,
                agent,
                config,
                scenario=scenario,
                seed=ep_seed,
                save_trajectory=save_trajectories,
                method_name=method_name,
            )
            all_episodes.append(ep_result)
            seed_episodes.append(ep_result)
            per_scenario_episodes.setdefault(scenario_name, []).append(ep_result)

            if save_trajectories and output_dir is not None and trajectory:
                traj_dir = os.path.join(output_dir, "trajectories", method_name)
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"seed{seed}_ep{ep}.csv")
                with open(traj_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                    writer.writeheader()
                    writer.writerows(trajectory)

        per_seed_results[f"seed_{seed}"] = seed_episodes

    overall = aggregate_metrics(all_episodes)
    overall["method"] = method_name
    overall["scenario"] = "all"
    overall["seed"] = "all"
    overall["episodes"] = len(all_episodes)
    overall["per_scenario"] = {
        name: aggregate_metrics(eps) for name, eps in per_scenario_episodes.items()
    }
    overall["raw_episodes"] = all_episodes
    overall["per_seed"] = per_seed_results
    return overall


def run_benchmark(
    config_path,
    smoke=False,
    episodes=None,
    seeds=None,
    scenarios=None,
    output_dir=None,
    backend=None,
    force=False,
):
    """Run the full Stage 6B benchmark.

    Args:
        config_path (str): Path to benchmark config YAML.
        smoke (bool): Run minimal CI smoke benchmark.
        episodes (int, optional): Override episodes per seed.
        seeds (list[int], optional): Override random seeds.
        scenarios (list[str], optional): Override scenario names.
        output_dir (str, optional): Override output directory.
        backend (str, optional): Override backend type ('simple' or 'jsbsim').
        force (bool): If True, allow overwriting existing output directory.
    """
    config = load_experiment_config(config_path)

    bench_cfg = config.get("benchmark", {})
    if episodes is None:
        episodes = bench_cfg.get("episodes", 20)
    if seeds is None:
        seeds = bench_cfg.get("seeds", [0, 1, 2, 3, 4])
    if scenarios is None:
        scenarios = bench_cfg.get(
            "scenarios", ["favorable", "neutral", "disadvantage", "challenging"]
        )

    if smoke:
        episodes = 2
        seeds = [0]
        scenarios = ["favorable", "neutral"]
        print("[SMOKE] Running minimal benchmark for CI validation")

    # Backend override
    if backend is not None:
        config["backend"] = backend
        config.setdefault("env", {})
        config["env"]["use_jsbsim"] = backend == "jsbsim"
        print(f"[CONFIG] Backend overridden to: {backend}")

    if output_dir is None:
        output_dir = os.path.join(
            config.get("experiment", {}).get("output_root", "outputs/benchmark"),
            config.get("experiment", {}).get("name", "stage6b_simple_prediction"),
        )

    if not force and os.path.exists(output_dir) and os.listdir(output_dir):
        print(
            f"WARNING: Output directory '{output_dir}' already exists and is not empty.\n"
            f"         Use --force to overwrite, or specify a different --output-dir."
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("=== Stage 6B Simple-Backend Benchmark ===")
    print(f"Config: {config_path}")
    print(f"Output dir: {output_dir}")
    print(f"Episodes: {episodes}")
    print(f"Seeds: {seeds}")
    print(f"Scenarios: {scenarios}")
    if backend:
        print(f"Backend override: {backend}")

    methods_cfg = config.get("methods", {})
    if not methods_cfg:
        print("ERROR: No methods defined in config.")
        sys.exit(1)

    all_method_metrics = []
    start_time = time.time()

    for method_name, method_override in methods_cfg.items():
        print(f"\n--- Evaluating method: {method_name} ---")
        method_config = merge_config(dict(config), method_override)

        env = CloseRangeTrackingEnv(method_config)
        print(f"  Backend: {env._backend}")

        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(method_config.get("policy", {}).get("action_dim", 3))

        agent = PPOAgent(
            obs_dim=obs_dim, action_dim=action_dim, config=method_config, device="cpu"
        )

        metrics = evaluate_method(
            env,
            agent,
            method_config,
            method_name,
            num_episodes=episodes,
            seeds=seeds,
            scenarios=scenarios,
            save_trajectories=False,
            output_dir=output_dir,
        )
        all_method_metrics.append(metrics)
        env.close()

        print(
            f"  Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f} | "
            f"Success: {metrics['success_rate']:.2%} | "
            f"Score Win: {metrics['score_win_rate']:.2%} | "
            f"Crash: {metrics['crash_rate']:.2%} | "
            f"OOB: {metrics['out_of_bounds_rate']:.2%}"
        )
        if metrics.get("per_scenario"):
            for sc_name, sc_metrics in metrics["per_scenario"].items():
                print(
                    f"    [{sc_name}] Success: {sc_metrics['success_rate']:.2%} | "
                    f"Score Win: {sc_metrics['score_win_rate']:.2%} | "
                    f"Range: {sc_metrics['mean_final_range_m']:.1f} m"
                )

    elapsed = time.time() - start_time

    # Save JSON
    json_path = os.path.join(output_dir, "prediction_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_method_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMetrics JSON saved to: {json_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, "prediction_metrics.csv")
    scalar_keys = [
        "method",
        "scenario",
        "seed",
        "episodes",
        "instant_success_rate",
        "score_win_rate",
        "mean_return",
        "mean_final_range_m",
        "mean_final_ata_deg",
        "prediction_rmse_m",
        "prediction_fallback_rate",
        "timeout_rate",
        "crash_rate",
        "out_of_bounds_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for m in all_method_metrics:
            writer.writerow({k: m.get(k, "") for k in scalar_keys})
    print(f"Metrics CSV saved to: {csv_path}")

    # Save per-scenario CSV
    scenario_rows = []
    for m in all_method_metrics:
        method_name = m["method"]
        per_scenario = m.get("per_scenario", {})
        for sc_name, sc_metrics in per_scenario.items():
            row = {"method": method_name, "scenario": sc_name}
            row.update(
                {
                    k: sc_metrics.get(
                        k,
                        sc_metrics.get(
                            k.replace("instant_", "").replace(
                                "prediction_", "mean_prediction_"
                            ),
                            "",
                        ),
                    )
                    for k in scalar_keys[2:]
                }
            )
            scenario_rows.append(row)

    scenario_csv_path = os.path.join(output_dir, "scenario_metrics.csv")
    with open(scenario_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "scenario"] + scalar_keys[2:])
        writer.writeheader()
        writer.writerows(scenario_rows)
    print(f"Scenario CSV saved to: {scenario_csv_path}")

    # Statistical comparison
    comparison = compare_methods(all_method_metrics, baseline_name="no_prediction")

    # Generate summary.md
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Stage 6B: Simple-Backend Prediction Benchmark Summary\n\n")
        f.write(
            "> **Warning**: This benchmark evaluates mechanism and comparative trends.\n"
        )
        f.write(
            "> Smoke or small-run results must not be presented as final paper conclusions.\n"
        )
        f.write(
            "> Full runs with sufficient seeds and episodes are required for statistical claims.\n\n"
        )

        f.write("## Configuration\n\n")
        f.write(f"- **Date**: {datetime.now().isoformat()}\n")
        f.write(f"- **Episodes per seed**: {episodes}\n")
        f.write(f"- **Seeds**: {seeds}\n")
        f.write(f"- **Scenarios**: {scenarios}\n")
        f.write(f"- **Methods**: {list(methods_cfg.keys())}\n")
        f.write(f"- **Elapsed time**: {elapsed:.1f}s\n\n")

        f.write("## Aggregated Metrics\n\n")
        f.write(
            "| Method | Episodes | Success | Score Win | Return | Final Range (m) | Final ATA (deg) | Timeout | Crash | OOB |\n"
        )
        f.write(
            "|--------|----------|---------|-----------|--------|-----------------|-----------------|---------|-------|-----|\n"
        )
        for m in all_method_metrics:
            f.write(
                f"| {m['method']} | {m['episodes']} | "
                f"{m.get('instant_success_rate', m.get('success_rate', 0)):.2%} | "
                f"{m.get('score_win_rate', 0):.2%} | "
                f"{m['mean_return']:.2f} ± {m['std_return']:.2f} | "
                f"{m['mean_final_range_m']:.1f} | "
                f"{m['mean_final_ata_deg']:.1f} | "
                f"{m['timeout_rate']:.2%} | "
                f"{m['crash_rate']:.2%} | "
                f"{m['out_of_bounds_rate']:.2%} |\n"
            )

        f.write("\n## Per-Scenario Metrics\n\n")
        for m in all_method_metrics:
            method_name = m["method"]
            per_scenario = m.get("per_scenario", {})
            if per_scenario:
                f.write(f"### {method_name}\n\n")
                f.write(
                    "| Scenario | Success | Score Win | Return | Final Range (m) |\n"
                )
                f.write(
                    "|----------|---------|-----------|--------|-----------------|\n"
                )
                for sc_name, sc_metrics in per_scenario.items():
                    f.write(
                        f"| {sc_name} | "
                        f"{sc_metrics.get('success_rate', 0):.2%} | "
                        f"{sc_metrics.get('score_win_rate', 0):.2%} | "
                        f"{sc_metrics.get('mean_return', 0):.2f} | "
                        f"{sc_metrics.get('mean_final_range_m', 0):.1f} |\n"
                    )
                f.write("\n")

        f.write("## Pairwise Comparison (vs no_prediction)\n\n")
        f.write(
            "| Comparison | Baseline Return | Treatment Return | Delta | Relative Delta (%) |\n"
        )
        f.write(
            "|------------|-----------------|------------------|-------|--------------------|\n"
        )
        for key, comp in comparison.get("pairwise", {}).items():
            baseline_val = comp.get("baseline_value", np.nan)
            treatment_val = comp.get("treatment_value", np.nan)
            delta = comp.get("delta", np.nan)
            rel = comp.get("relative_delta_pct", np.nan)
            f.write(
                f"| {key} | "
                f"{baseline_val:.2f} | {treatment_val:.2f} | "
                f"{delta:+.2f} | "
                f"{rel:+.1f}% |\n"
            )

        f.write("\n---\n")
        f.write("Generated by `run_stage6b_simple_benchmark.py`\n")

    print(f"Summary markdown saved to: {summary_path}")

    # Save run metadata
    metadata = get_run_metadata(config)
    metadata["benchmark"] = {
        "episodes": episodes,
        "seeds": seeds,
        "scenarios": scenarios,
        "methods": list(methods_cfg.keys()),
        "elapsed_seconds": elapsed,
    }
    meta_path = save_run_metadata(output_dir, metadata)
    print(f"Run metadata saved to: {meta_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Stage 6B Simple-Backend Benchmark")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to benchmark config YAML"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run minimal smoke benchmark"
    )
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per seed")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="Random seeds"
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+", default=None, help="Fixed scenario names"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory override"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["simple", "jsbsim"],
        help="Override backend type",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing output directory",
    )
    args = parser.parse_args()

    try:
        run_benchmark(
            args.config,
            smoke=args.smoke,
            episodes=args.episodes,
            seeds=args.seeds,
            scenarios=args.scenarios,
            output_dir=args.output_dir,
            backend=args.backend,
            force=args.force,
        )
    except Exception as exc:
        print(f"\nERROR: Benchmark failed with exception: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
