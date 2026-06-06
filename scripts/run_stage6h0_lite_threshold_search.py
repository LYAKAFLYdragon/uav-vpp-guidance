#!/usr/bin/env python3
"""
Stage 6H.0-lite: Mode-Switch Threshold Optimization Preflight.

Samples threshold configs from a 6-D search space and evaluates them
against candidate geometries, regression baselines, and negative controls.

Search space:
    aspect_enter_threshold_deg: [10, 15, 20, 25]
    aspect_exit_threshold_deg:  [20, 30, 45, null]
    range_enter_m:              [1500, 2000, 2500, 3000]
    closing_speed_enter_mps:    [80, 120, 160]
    hold_policy:                [episode_latch, min_hold_2s, hysteresis_exit]
    fallback_mode:              [los_rate, hybrid]

Default sample-size = 60 (not full 1152 grid).

Usage:
    # Dry-run (no simulation)
    python scripts/run_stage6h0_lite_threshold_search.py --dry-run

    # Real run with regression baseline file
    python scripts/run_stage6h0_lite_threshold_search.py \
        --sample-size 60 \
        --sampling-method latin_hypercube \
        --seed 0 \
        --regression-baseline-file outputs/stage6h0_regression_baseline_search/regression_baseline_candidates.csv \
        --output-dir outputs/stage6h0_lite_threshold_search
"""

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
    load_experiment_config,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import merge_config


SEARCH_SPACE = {
    "aspect_enter_threshold_deg": [10.0, 15.0, 20.0, 25.0],
    "aspect_exit_threshold_deg": [20.0, 30.0, 45.0, None],
    "range_enter_m": [1500.0, 2000.0, 2500.0, 3000.0],
    "closing_speed_enter_mps": [80.0, 120.0, 160.0],
    "hold_policy": ["episode_latch", "min_hold_2s", "hysteresis_exit"],
    "fallback_mode": ["los_rate", "hybrid"],
}

CANDIDATE_GEOMETRIES = {
    "pt20": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": -500},
    "pt29": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 200, "aspect_angle_deg": 0, "altitude_diff_m": 0},
    "pt38": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": 500},
    "near_range_1800": {"initial_range_m": 1800, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": 0},
}

NEGATIVE_CONTROLS = {
    "neg_aspect_60": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 60, "altitude_diff_m": 0},
    "neg_aspect_90": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 90, "altitude_diff_m": 0},
    "neg_low_closing": {"initial_range_m": 2000, "ego_speed_mps": 150, "target_speed_mps": 140, "aspect_angle_deg": 0, "altitude_diff_m": 0},
}

ACCEPTANCE = {
    "candidate_min_success_rate": 0.95,
    "regression_max_degradation_pp": 5.0,
    "negative_control_max_false_activation_rate": 0.05,
}


def _make_env_agent(config, method_override, allow_random_policy=False):
    method_config = merge_config(copy.deepcopy(config), copy.deepcopy(method_override))
    env = CloseRangeTrackingEnv(method_config)
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(method_config.get("policy", {}).get("action_dim", 3))
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device="cpu")

    ckpt = method_override.get("checkpoint")
    ckpt_exists = ckpt and os.path.exists(ckpt)
    if ckpt_exists:
        agent.load(ckpt)
    elif allow_random_policy:
        pass
    else:
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    policy_type = "trained_ppo" if ckpt_exists else ("random_policy" if allow_random_policy else "missing_checkpoint")
    return env, agent, method_config, policy_type, ckpt


def _sample_configs(sample_size, method, seed):
    rng = np.random.default_rng(seed)
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    n_dims = len(keys)

    if method == "grid":
        from itertools import product
        all_configs = []
        for combo in product(*values):
            all_configs.append(dict(zip(keys, combo)))
        if len(all_configs) <= sample_size:
            return all_configs
        indices = rng.choice(len(all_configs), size=sample_size, replace=False)
        return [all_configs[i] for i in indices]

    if method == "random":
        configs = []
        for _ in range(sample_size):
            cfg = {}
            for k, v in zip(keys, values):
                cfg[k] = rng.choice(v)
            configs.append(cfg)
        return configs

    if method == "latin_hypercube":
        # Latin Hypercube: for each dim, split into sample_size bins,
        # pick one random value per bin, then shuffle across dims.
        configs = []
        dim_samples = []
        for dim_idx, dim_values in enumerate(values):
            n_vals = len(dim_values)
            bins = np.array_split(np.arange(sample_size), n_vals)
            bin_choices = []
            for b in bins:
                if len(b) > 0:
                    choices = [rng.choice(dim_values) for _ in b]
                    bin_choices.extend(choices)
            # Pad or truncate to sample_size
            if len(bin_choices) < sample_size:
                extra = [rng.choice(dim_values) for _ in range(sample_size - len(bin_choices))]
                bin_choices.extend(extra)
            bin_choices = bin_choices[:sample_size]
            rng.shuffle(bin_choices)
            dim_samples.append(bin_choices)

        for i in range(sample_size):
            cfg = {k: dim_samples[d][i] for d, k in enumerate(keys)}
            configs.append(cfg)
        return configs

    raise ValueError(f"Unknown sampling method: {method}")


def _load_regression_baselines(csv_path):
    baselines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_candidate", "").lower() in ("true", "1", "yes"):
                baselines.append({
                    "scenario_id": row["scenario_id"],
                    "aspect_angle_deg": float(row["aspect_angle_deg"]),
                    "initial_range_m": float(row["initial_range_m"]),
                    "ego_speed_mps": float(row["ego_speed_mps"]),
                    "target_speed_mps": float(row["target_speed_mps"]),
                    "altitude_diff_m": float(row["altitude_diff_m"]),
                    "baseline_variant": row["variant"],
                    "baseline_success_rate": float(row["success_rate"]),
                })
    return baselines


def _build_mode_switch_config(cfg):
    ms = {
        "enabled": True,
        "aspect_threshold_deg": cfg["aspect_enter_threshold_deg"],
        "range_threshold_m": cfg["range_enter_m"],
        "closing_speed_threshold_mps": cfg["closing_speed_enter_mps"],
    }
    if cfg["aspect_exit_threshold_deg"] is not None:
        ms["exit_aspect_threshold_deg"] = cfg["aspect_exit_threshold_deg"]
    if cfg["hold_policy"] == "min_hold_2s":
        ms["exit_policy"] = "min_hold"
        ms["min_hold_time_s"] = 2.0
    elif cfg["hold_policy"] == "hysteresis_exit":
        ms["exit_policy"] = "hysteresis"
        ms["exit_aspect_threshold_deg"] = cfg.get("aspect_exit_threshold_deg", 45.0)
    return ms


def run_threshold_search(
    output_dir: str,
    config_path: str = "config/experiment/stage6g5_wide_geometry_smoke.yaml",
    sample_size: int = 60,
    sampling_method: str = "latin_hypercube",
    seed: int = 0,
    candidate_geometries=None,
    regression_baseline_file: str = None,
    negative_controls=None,
    episodes_per_point: int = 3,
    eval_seeds=None,
    dry_run: bool = False,
    allow_random_policy: bool = False,
):
    if eval_seeds is None:
        eval_seeds = [0]
    if candidate_geometries is None:
        candidate_geometries = list(CANDIDATE_GEOMETRIES.keys())
    if negative_controls is None:
        negative_controls = list(NEGATIVE_CONTROLS.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Check regression baseline
    regression_baseline_missing = False
    regression_baselines = []
    if regression_baseline_file and os.path.exists(regression_baseline_file):
        regression_baselines = _load_regression_baselines(regression_baseline_file)
    elif not dry_run:
        regression_baseline_missing = True

    requested_config = {
        "experiment_name": "stage6h0_lite_threshold_search",
        "search_space": SEARCH_SPACE,
        "sample_size": sample_size,
        "sampling_method": sampling_method,
        "seed": seed,
        "candidate_geometries": candidate_geometries,
        "regression_baseline_file": regression_baseline_file,
        "regression_baseline_missing": regression_baseline_missing,
        "negative_controls": negative_controls,
        "episodes_per_point": episodes_per_point,
        "eval_seeds": eval_seeds,
        "dry_run": dry_run,
        "allow_random_policy": allow_random_policy,
        "acceptance_criteria": ACCEPTANCE,
    }
    with open(output_path / "threshold_search_plan.json", "w", encoding="utf-8") as f:
        json.dump(requested_config, f, indent=2, default=str)

    sampled_configs = _sample_configs(sample_size, sampling_method, seed)
    with open(output_path / "threshold_configs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SEARCH_SPACE.keys()) + ["config_id"])
        writer.writeheader()
        for idx, cfg in enumerate(sampled_configs):
            row = copy.deepcopy(cfg)
            row["config_id"] = f"cfg_{idx:04d}"
            writer.writerow(row)

    if dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would evaluate {len(sampled_configs)} sampled configs")
        print(f"Candidate geometries: {len(candidate_geometries)}")
        print(f"Regression baselines: {len(regression_baselines)}")
        print(f"Negative controls: {len(negative_controls)}")
        print("No simulation executed.")
        _write_empty_csv(output_path / "raw_episodes.csv",
                         ["config_id", "scenario", "episode_index", "is_success", "reason"])
        _write_empty_csv(output_path / "threshold_summary.csv",
                         ["config_id", "candidate_success_rate", "regression_degradation_pp",
                          "false_activation_rate", "accepted"])
        _write_summary_md(output_path / "threshold_search_summary.md", [], sampled_configs,
                          regression_baseline_missing=regression_baseline_missing)
        return requested_config

    if regression_baseline_missing:
        raise FileNotFoundError(
            "Regression baseline file missing. Run find_stage6h0_regression_baseline.py first, "
            "or use --dry-run."
        )

    print(f"\n=== Stage 6H.0-lite: Threshold Search ===")
    print(f"Sampled configs: {len(sampled_configs)}")
    print(f"Candidates: {len(candidate_geometries)} | Baselines: {len(regression_baselines)} | Negatives: {len(negative_controls)}")

    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    method_override = config.get("methods", {}).get("no_prediction", {})

    all_episodes = []
    config_results = []

    for cfg_idx, cfg in enumerate(sampled_configs):
        config_id = f"cfg_{cfg_idx:04d}"
        print(f"\n--- Config {config_id} ---")
        print(f"  enter_aspect={cfg['aspect_enter_threshold_deg']} exit_aspect={cfg['aspect_exit_threshold_deg']} "
              f"range={cfg['range_enter_m']} closing={cfg['closing_speed_enter_mps']} "
              f"hold={cfg['hold_policy']} fallback={cfg['fallback_mode']}")

        variant_config = copy.deepcopy(config)
        variant_config.setdefault("guidance", {})
        variant_config["guidance"]["direct_track_mode"] = False
        variant_config["guidance"]["mode"] = cfg["fallback_mode"]
        variant_config["guidance"]["mode_switch"] = _build_mode_switch_config(cfg)

        env, agent, method_config, policy_type, ckpt = _make_env_agent(
            variant_config, method_override, allow_random_policy=allow_random_policy
        )

        # Evaluate candidate geometries
        candidate_success = 0
        candidate_total = 0
        for scen_name in candidate_geometries:
            pt = CANDIDATE_GEOMETRIES[scen_name]
            scenario = build_geometry_scenario(
                pt["initial_range_m"], pt["ego_speed_mps"], pt["target_speed_mps"],
                pt["aspect_angle_deg"], pt["altitude_diff_m"], base_altitude_m=5000.0,
            )
            for ev_seed in eval_seeds:
                for ep_idx in range(episodes_per_point):
                    episode_seed = ev_seed * 100000 + cfg_idx * 1000 + hash(scen_name) % 100 + ep_idx
                    set_seed(episode_seed)
                    try:
                        ep_result, _ = evaluate_single_episode(
                            env, agent, method_config, scenario=scenario, seed=episode_seed,
                            save_trajectory=False, method_name="no_prediction",
                        )
                        ep_result["config_id"] = config_id
                        ep_result["scenario"] = scen_name
                        ep_result["scenario_type"] = "candidate"
                        ep_result["episode_index"] = ep_idx
                        all_episodes.append(ep_result)
                        candidate_total += 1
                        if ep_result.get("is_success"):
                            candidate_success += 1
                    except Exception as exc:
                        all_episodes.append({
                            "config_id": config_id, "scenario": scen_name, "scenario_type": "candidate",
                            "episode_index": ep_idx, "is_success": False, "reason": f"exception:{exc}",
                        })
                        candidate_total += 1

        # Evaluate regression baselines
        regression_degradation_pp = 0.0
        if regression_baselines:
            baseline_success = 0
            baseline_total = 0
            for bl in regression_baselines:
                scenario = build_geometry_scenario(
                    bl["initial_range_m"], bl["ego_speed_mps"], bl["target_speed_mps"],
                    bl["aspect_angle_deg"], bl["altitude_diff_m"], base_altitude_m=5000.0,
                )
                for ev_seed in eval_seeds:
                    for ep_idx in range(episodes_per_point):
                        episode_seed = ev_seed * 100000 + cfg_idx * 1000 + hash(bl["scenario_id"]) % 100 + ep_idx
                        set_seed(episode_seed)
                        try:
                            ep_result, _ = evaluate_single_episode(
                                env, agent, method_config, scenario=scenario, seed=episode_seed,
                                save_trajectory=False, method_name="no_prediction",
                            )
                            ep_result["config_id"] = config_id
                            ep_result["scenario"] = bl["scenario_id"]
                            ep_result["scenario_type"] = "regression"
                            all_episodes.append(ep_result)
                            baseline_total += 1
                            if ep_result.get("is_success"):
                                baseline_success += 1
                        except Exception as exc:
                            all_episodes.append({
                                "config_id": config_id, "scenario": bl["scenario_id"],
                                "scenario_type": "regression", "is_success": False,
                                "reason": f"exception:{exc}",
                            })
                            baseline_total += 1
            baseline_sr = baseline_success / max(1, baseline_total)
            # Degradation vs average baseline success rate
            avg_baseline_sr = np.mean([b["baseline_success_rate"] for b in regression_baselines])
            regression_degradation_pp = max(0.0, (avg_baseline_sr - baseline_sr) * 100)

        # Evaluate negative controls
        false_activation_count = 0
        negative_total = 0
        for scen_name in negative_controls:
            pt = NEGATIVE_CONTROLS[scen_name]
            scenario = build_geometry_scenario(
                pt["initial_range_m"], pt["ego_speed_mps"], pt["target_speed_mps"],
                pt["aspect_angle_deg"], pt["altitude_diff_m"], base_altitude_m=5000.0,
            )
            for ev_seed in eval_seeds:
                for ep_idx in range(episodes_per_point):
                    episode_seed = ev_seed * 100000 + cfg_idx * 1000 + hash(scen_name) % 100 + ep_idx
                    set_seed(episode_seed)
                    try:
                        ep_result, _ = evaluate_single_episode(
                            env, agent, method_config, scenario=scenario, seed=episode_seed,
                            save_trajectory=False, method_name="no_prediction",
                        )
                        ep_result["config_id"] = config_id
                        ep_result["scenario"] = scen_name
                        ep_result["scenario_type"] = "negative"
                        all_episodes.append(ep_result)
                        negative_total += 1
                        # False activation: mode_switch_effective=True on a negative control
                        if ep_result.get("mode_switch_effective", False):
                            false_activation_count += 1
                    except Exception as exc:
                        all_episodes.append({
                            "config_id": config_id, "scenario": scen_name,
                            "scenario_type": "negative", "is_success": False,
                            "reason": f"exception:{exc}",
                        })
                        negative_total += 1

        false_activation_rate = false_activation_count / max(1, negative_total)
        candidate_sr = candidate_success / max(1, candidate_total)

        accepted = (
            candidate_sr >= ACCEPTANCE["candidate_min_success_rate"]
            and regression_degradation_pp <= ACCEPTANCE["regression_max_degradation_pp"]
            and false_activation_rate <= ACCEPTANCE["negative_control_max_false_activation_rate"]
        )

        config_results.append({
            "config_id": config_id,
            **cfg,
            "candidate_success_rate": candidate_sr,
            "candidate_success_count": candidate_success,
            "candidate_total": candidate_total,
            "regression_degradation_pp": regression_degradation_pp,
            "false_activation_rate": false_activation_rate,
            "false_activation_count": false_activation_count,
            "negative_total": negative_total,
            "accepted": accepted,
        })

        status = "ACCEPTED" if accepted else "REJECTED"
        print(f"  -> {status} | candidate={candidate_sr:.2f} degradation={regression_degradation_pp:.1f}pp false_act={false_activation_rate:.2f}")

        env.close()

    # Save outputs
    _write_csv_from_dicts(output_path / "raw_episodes.csv", all_episodes,
                          ["config_id", "scenario", "scenario_type", "episode_index",
                           "is_success", "reason", "mode_switch_effective",
                           "effective_guidance_mode", "virtual_point_source"])

    _write_csv_from_dicts(output_path / "threshold_summary.csv", config_results,
                          ["config_id", "aspect_enter_threshold_deg", "aspect_exit_threshold_deg",
                           "range_enter_m", "closing_speed_enter_mps", "hold_policy", "fallback_mode",
                           "candidate_success_rate", "regression_degradation_pp",
                           "false_activation_rate", "accepted"])

    accepted = [r for r in config_results if r["accepted"]]
    rejected = [r for r in config_results if not r["accepted"]]
    _write_csv_from_dicts(output_path / "accepted_thresholds.csv", accepted,
                          ["config_id", "aspect_enter_threshold_deg", "aspect_exit_threshold_deg",
                           "range_enter_m", "closing_speed_enter_mps", "hold_policy", "fallback_mode",
                           "candidate_success_rate", "regression_degradation_pp", "false_activation_rate"])
    _write_csv_from_dicts(output_path / "rejected_thresholds.csv", rejected,
                          ["config_id", "aspect_enter_threshold_deg", "aspect_exit_threshold_deg",
                           "range_enter_m", "closing_speed_enter_mps", "hold_policy", "fallback_mode",
                           "candidate_success_rate", "regression_degradation_pp", "false_activation_rate"])

    _write_summary_md(output_path / "threshold_search_summary.md", config_results, sampled_configs,
                      regression_baseline_missing=regression_baseline_missing)
    return requested_config


def _write_empty_csv(path: Path, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _write_csv_from_dicts(path: Path, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_summary_md(path: Path, config_results: list, sampled_configs: list, regression_baseline_missing: bool = False):
    accepted = [r for r in config_results if r.get("accepted")]
    lines = ["# Stage 6H.0-lite: Threshold Search Summary", ""]
    lines.append(f"**Total configs evaluated**: {len(sampled_configs)}")
    lines.append(f"**Accepted**: {len(accepted)} | **Rejected**: {len(config_results) - len(accepted)}")
    lines.append("")

    if regression_baseline_missing:
        lines.append("⚠️ **Regression baseline file was missing.**")
        lines.append("Threshold search requires a validated regression baseline before real runs.")
        lines.append("")

    if not config_results:
        lines.append("*Status: dry-run only. No real episodes executed.*")
    elif not accepted:
        lines.append("## ⚠️ No threshold configs passed acceptance criteria.")
        lines.append("")
        lines.append("All sampled configs failed at least one criterion:")
        lines.append("- candidate_success_rate ≥ 0.95")
        lines.append("- regression_degradation ≤ 5 percentage points")
        lines.append("- false_activation_rate ≤ 0.05")
        lines.append("")
        lines.append("**Next step**: Adjust search space, increase sample size, or investigate why thresholds are unstable.")
    else:
        lines.append("## ✅ Accepted Threshold Configs")
        lines.append("")
        lines.append("| Config | Enter Aspect | Exit Aspect | Range | Closing Speed | Hold Policy | Fallback | Candidate SR | Degradation | False Act |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in accepted:
            lines.append(
                f"| {r['config_id']} | {r['aspect_enter_threshold_deg']}° | {r['aspect_exit_threshold_deg']}° | "
                f"{r['range_enter_m']}m | {r['closing_speed_enter_mps']}m/s | {r['hold_policy']} | {r['fallback_mode']} | "
                f"{r['candidate_success_rate']*100:.1f}% | {r['regression_degradation_pp']:.1f}pp | {r['false_activation_rate']*100:.1f}% |"
            )
        lines.append("")
        lines.append("These configs meet all acceptance criteria and can be used for further validation.")

    lines.append("")
    lines.append("> **Paper-safe note**: Results limited to sampled configs and tested geometries. "
        "No universal claims about optimal thresholds are made.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6H.0-lite Threshold Search")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--sampling-method", type=str, default="latin_hypercube",
                        choices=["random", "latin_hypercube", "grid"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-geometries", type=str, nargs="+",
                        default=list(CANDIDATE_GEOMETRIES.keys()))
    parser.add_argument("--regression-baseline-file", type=str, default=None)
    parser.add_argument("--negative-controls", type=str, nargs="+",
                        default=list(NEGATIVE_CONTROLS.keys()))
    parser.add_argument("--episodes-per-point", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-random-policy", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6h0_lite_threshold_search")
    parser.add_argument("--config", type=str, default="config/experiment/stage6g5_wide_geometry_smoke.yaml")
    args = parser.parse_args()

    run_threshold_search(
        output_dir=args.output_dir,
        config_path=args.config,
        sample_size=args.sample_size,
        sampling_method=args.sampling_method,
        seed=args.seed,
        candidate_geometries=args.candidate_geometries,
        regression_baseline_file=args.regression_baseline_file,
        negative_controls=args.negative_controls,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
    )


if __name__ == "__main__":
    main()
