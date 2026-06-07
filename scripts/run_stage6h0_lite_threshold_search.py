#!/usr/bin/env python3
"""Stage 6H.0-lite: Mode-Switch Threshold Optimization Preflight.

Samples threshold configs from a 6-D search space and evaluates them
against candidate geometries, regression baselines, and negative controls.

All scenarios originate from ScenarioRegistry exclusively.
No legacy aspect-angle builder paths remain.

Search space:
    aspect_enter_threshold_deg: [10, 15, 20, 25]
    aspect_exit_threshold_deg:  [20, 30, 45, null]
    range_enter_m:              [1500, 2000, 2500, 3000]
    closing_speed_enter_mps:    [80, 120, 160]
    hold_policy:                [episode_latch, min_hold_2s, hysteresis_exit]
    fallback_mode:              [los_rate, hybrid]

Usage:
    # Dry-run (no simulation)
    python scripts/run_stage6h0_lite_threshold_search.py --dry-run

    # Smoke run (small sample, fast)
    python scripts/run_stage6h0_lite_threshold_search.py \
        --sample-size 5 --sampling-method random --episodes-per-point 1 \
        --mode exploratory --output-dir outputs/stage6h0f3_smoke

    # Formal LHS60 (paper-safe)
    python scripts/run_stage6h0_lite_threshold_search.py \
        --sample-size 60 --sampling-method latin_hypercube --seed 0 \
        --mode formal \
        --regression-baseline-file outputs/stage6h0f2_formal_baseline/regression_baseline.csv \
        --output-dir outputs/stage6h0f3_formal_lhs60
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
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.geometry_scenarios import build_explicit_scenario
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
    load_experiment_config,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import merge_config
from uav_vpp_guidance.utils.geometry_validator import validate_scenario_geometry

# ---------------------------------------------------------------------------
# Contract: no legacy builder may be imported or used in this module.
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "aspect_enter_threshold_deg": [10.0, 15.0, 20.0, 25.0],
    "aspect_exit_threshold_deg": [20.0, 30.0, 45.0, None],
    "crossing_aspect_threshold_deg": [None, 0.0, 15.0, 30.0, 45.0],
    "range_enter_m": [1500.0, 2000.0, 2500.0, 3000.0],
    "closing_speed_enter_mps": [80.0, 120.0, 160.0],
    "hold_policy": ["episode_latch", "min_hold_2s", "hysteresis_exit"],
    "fallback_mode": ["los_rate", "hybrid"],
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
                    "scenario_type": row.get("scenario_type", ""),
                    "aspect_angle_deg": float(row.get("aspect_angle_deg", 0)),
                    "initial_range_m": float(row.get("initial_range_m", 0)),
                    "ego_speed_mps": float(row.get("ego_speed_mps", 0)),
                    "target_speed_mps": float(row.get("target_speed_mps", 0)),
                    "altitude_diff_m": float(row.get("altitude_diff_m", 0)),
                    "baseline_variant": row.get("variant", "no_prediction"),
                    "baseline_success_rate": float(row.get("success_rate", 0)),
                })
    return baselines


def _build_mode_switch_config(cfg):
    ms = {
        "enabled": True,
        "aspect_threshold_deg": cfg["aspect_enter_threshold_deg"],
        "crossing_aspect_threshold_deg": cfg.get("crossing_aspect_threshold_deg"),
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


def _scenario_from_registry(name):
    """Load a scenario dict from ScenarioRegistry.

    Returns the scenario and its validated geometry family.
    """
    scen = ScenarioRegistry.get(name)
    if scen is None:
        raise ValueError(f"Scenario '{name}' not found in ScenarioRegistry")
    report = validate_scenario_geometry(scen)
    return scen, report["classified_family"]


def run_threshold_search(
    output_dir: str,
    config_path: str = "config/experiment/stage6g5_wide_geometry_smoke.yaml",
    sample_size: int = 60,
    sampling_method: str = "latin_hypercube",
    seed: int = 0,
    candidate_set: str = "candidate_search",
    regression_baseline_file: str = None,
    negative_set: str = "negative_control",
    episodes_per_point: int = 3,
    eval_seeds=None,
    dry_run: bool = False,
    allow_random_policy: bool = False,
    mode: str = "exploratory",
):
    if eval_seeds is None:
        eval_seeds = [0]

    # Initialize registry and resolve scenario sets
    initialize_canonical_scenarios()
    candidate_names = ScenarioRegistry.list_names(candidate_set)
    negative_names = ScenarioRegistry.list_names(negative_set)

    if not candidate_names:
        raise ValueError(f"Candidate set '{candidate_set}' is empty or not found in ScenarioRegistry")
    if not negative_names:
        raise ValueError(f"Negative set '{negative_set}' is empty or not found in ScenarioRegistry")

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
        "mode": mode,
        "search_space": SEARCH_SPACE,
        "sample_size": sample_size,
        "sampling_method": sampling_method,
        "seed": seed,
        "candidate_set": candidate_set,
        "candidate_scenarios": candidate_names,
        "regression_baseline_file": regression_baseline_file,
        "regression_baseline_missing": regression_baseline_missing,
        "negative_set": negative_set,
        "negative_scenarios": negative_names,
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
        print(f"Mode: {mode}")
        print(f"Would evaluate {len(sampled_configs)} sampled configs")
        print(f"Candidate scenarios ({candidate_set}): {len(candidate_names)}")
        print(f"Regression baselines: {len(regression_baselines)}")
        print(f"Negative controls ({negative_set}): {len(negative_names)}")
        print("No simulation executed.")
        _write_empty_csv(output_path / "raw_episodes.csv",
                         ["config_id", "scenario_id", "geometry_family", "scenario_type",
                          "episode_index", "is_success", "reason",
                          "effective_guidance_mode", "mode_switch_effective"])
        _write_empty_csv(output_path / "threshold_summary.csv",
                         ["config_id", "candidate_success_rate", "regression_degradation_pp",
                          "false_activation_rate", "accepted"])
        _write_empty_csv(output_path / "accepted_thresholds.csv",
                         ["config_id", "aspect_enter_threshold_deg", "aspect_exit_threshold_deg",
                          "range_enter_m", "closing_speed_enter_mps", "hold_policy", "fallback_mode",
                          "candidate_success_rate", "regression_degradation_pp", "false_activation_rate"])
        _write_empty_csv(output_path / "rejected_thresholds.csv",
                         ["config_id", "aspect_enter_threshold_deg", "aspect_exit_threshold_deg",
                          "range_enter_m", "closing_speed_enter_mps", "hold_policy", "fallback_mode",
                          "candidate_success_rate", "regression_degradation_pp", "false_activation_rate"])
        _write_summary_md(output_path / "threshold_search_summary.md", [], sampled_configs,
                          regression_baseline_missing=regression_baseline_missing, mode=mode)
        return requested_config

    if regression_baseline_missing:
        if mode == "formal":
            raise RuntimeError(
                "Regression baseline file is required but missing. "
                "Formal threshold search cannot proceed without a validated "
                "non-tail-chase VPP baseline. Run regression baseline recovery first, "
                "or use --mode exploratory. "
                "Use --regression-baseline-file to provide the baseline file."
            )
        else:
            print("WARNING: Regression baseline file missing. Exploratory mode continues.")
            print("  Candidate and negative-control evaluations will proceed.")
            print("  Results are NOT paper-safe formal threshold results.")

    print("\n=== Stage 6H.0-lite: Threshold Search ===")
    print(f"Mode: {mode} | Sampled configs: {len(sampled_configs)}")
    print(f"Candidates: {len(candidate_names)} | Baselines: {len(regression_baselines)} | Negatives: {len(negative_names)}")

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

        # Evaluate candidate scenarios
        candidate_success = 0
        candidate_total = 0
        family_success = {}
        family_total = {}
        for scen_name in candidate_names:
            scenario, family = _scenario_from_registry(scen_name)
            family_success.setdefault(family, 0)
            family_total.setdefault(family, 0)
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
                        ep_result["scenario_id"] = scen_name
                        ep_result["geometry_family"] = family
                        ep_result["scenario_type"] = "candidate"
                        ep_result["episode_index"] = ep_idx
                        all_episodes.append(ep_result)
                        candidate_total += 1
                        family_total[family] += 1
                        if ep_result.get("is_success"):
                            candidate_success += 1
                            family_success[family] += 1
                    except Exception as exc:
                        all_episodes.append({
                            "config_id": config_id, "scenario_id": scen_name,
                            "geometry_family": family, "scenario_type": "candidate",
                            "episode_index": ep_idx, "is_success": False,
                            "reason": f"exception:{exc}",
                            "effective_guidance_mode": "unknown",
                            "mode_switch_effective": False,
                        })
                        candidate_total += 1
                        family_total[family] += 1

        # Evaluate regression baselines
        regression_degradation_pp = 0.0
        if regression_baselines:
            baseline_success = 0
            baseline_total = 0
            for bl in regression_baselines:
                if bl.get("scenario_type"):
                    scenario = build_explicit_scenario(
                        bl["scenario_type"],
                        bl["initial_range_m"], bl["ego_speed_mps"], bl["target_speed_mps"],
                        altitude_diff_m=bl["altitude_diff_m"], base_altitude_m=5000.0,
                    )
                else:
                    raise ValueError(
                        f"Baseline {bl['scenario_id']} missing scenario_type. "
                        "All baselines must use explicit scenario types."
                    )
                _, family = validate_scenario_geometry(scenario), validate_scenario_geometry(scenario)["classified_family"]
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
                            ep_result["scenario_id"] = bl["scenario_id"]
                            ep_result["geometry_family"] = family
                            ep_result["scenario_type"] = "regression"
                            all_episodes.append(ep_result)
                            baseline_total += 1
                            if ep_result.get("is_success"):
                                baseline_success += 1
                        except Exception as exc:
                            all_episodes.append({
                                "config_id": config_id, "scenario_id": bl["scenario_id"],
                                "geometry_family": family, "scenario_type": "regression",
                                "is_success": False, "reason": f"exception:{exc}",
                                "effective_guidance_mode": "unknown",
                                "mode_switch_effective": False,
                            })
                            baseline_total += 1
            baseline_sr = baseline_success / max(1, baseline_total)
            avg_baseline_sr = np.mean([b["baseline_success_rate"] for b in regression_baselines])
            regression_degradation_pp = max(0.0, (avg_baseline_sr - baseline_sr) * 100)

        # Evaluate negative controls
        false_activation_count = 0
        negative_total = 0
        for scen_name in negative_names:
            scenario, family = _scenario_from_registry(scen_name)
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
                        ep_result["scenario_id"] = scen_name
                        ep_result["geometry_family"] = family
                        ep_result["scenario_type"] = "negative"
                        all_episodes.append(ep_result)
                        negative_total += 1
                        if ep_result.get("mode_switch_effective", False):
                            false_activation_count += 1
                    except Exception as exc:
                        all_episodes.append({
                            "config_id": config_id, "scenario_id": scen_name,
                            "geometry_family": family, "scenario_type": "negative",
                            "is_success": False, "reason": f"exception:{exc}",
                            "effective_guidance_mode": "unknown",
                            "mode_switch_effective": False,
                        })
                        negative_total += 1

        false_activation_rate = false_activation_count / max(1, negative_total)
        candidate_sr = candidate_success / max(1, candidate_total)

        # Per-family breakdown
        family_breakdown = {}
        for fam in family_total:
            family_breakdown[fam] = {
                "success_rate": family_success.get(fam, 0) / max(1, family_total[fam]),
                "success_count": family_success.get(fam, 0),
                "total": family_total[fam],
            }

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
            "family_breakdown": family_breakdown,
        })

        status = "ACCEPTED" if accepted else "REJECTED"
        print(f"  -> {status} | candidate={candidate_sr:.2f} degradation={regression_degradation_pp:.1f}pp false_act={false_activation_rate:.2f}")
        for fam, info in sorted(family_breakdown.items()):
            print(f"      {fam}: {info['success_count']}/{info['total']} = {info['success_rate']:.1%}")

        env.close()

    # Save outputs
    _write_csv_from_dicts(output_path / "raw_episodes.csv", all_episodes,
                          ["config_id", "scenario_id", "geometry_family", "scenario_type",
                           "episode_index", "is_success", "reason",
                           "effective_guidance_mode", "mode_switch_effective"])

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
                      regression_baseline_missing=regression_baseline_missing, mode=mode)
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


def _write_summary_md(path: Path, config_results: list, sampled_configs: list, regression_baseline_missing: bool = False, mode: str = "exploratory"):
    accepted = [r for r in config_results if r.get("accepted")]
    lines = ["# Stage 6H.0-lite: Threshold Search Summary", ""]
    lines.append(f"**Mode**: {mode}")
    lines.append(f"**Total configs evaluated**: {len(sampled_configs)}")
    lines.append(f"**Accepted**: {len(accepted)} | **Rejected**: {len(config_results) - len(accepted)}")
    lines.append("")

    if mode == "exploratory":
        lines.append("> ⚠️ **Exploratory mode**. Results are NOT paper-safe formal threshold results.")
        lines.append("> Formal acceptance requires `--mode formal` with a validated regression baseline.")
        lines.append("")

    if regression_baseline_missing:
        lines.append("⚠️ **Regression baseline file was missing.**")
        if mode == "exploratory":
            lines.append("Exploratory run proceeded without regression checks.")
        else:
            lines.append("Formal run requires a validated regression baseline.")
        lines.append("")

    if not config_results:
        lines.append("*Status: dry-run only. No real episodes executed.*")
    elif not accepted:
        lines.append("## ⚠️ No threshold configs passed acceptance criteria.")
        lines.append("")
        lines.append("All sampled configs failed at least one criterion:")
        lines.append("- candidate_success_rate ≥ 0.95")
        if mode == "formal":
            lines.append("- regression_degradation ≤ 5 percentage points")
        lines.append("- false_activation_rate ≤ 0.05")
        lines.append("")
        lines.append("**Next step**: Adjust search space, increase sample size, or investigate why thresholds are unstable.")
    else:
        label = "✅ Accepted (Formal)" if mode == "formal" else "🧪 Accepted (Exploratory)"
        lines.append(f"## {label} Threshold Configs")
        lines.append("")
        lines.append("| Config | Enter Aspect | Exit Aspect | Range | Closing Speed | Hold Policy | Fallback | Candidate SR | Degradation | False Act |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in accepted:
            lines.append(
                f"| {r['config_id']} | {r['aspect_enter_threshold_deg']}° | {r['aspect_exit_threshold_deg']}° | "
                f"{r['range_enter_m']}m | {r['closing_speed_enter_mps']}m/s | {r['hold_policy']} | {r['fallback_mode']} | "
                f"{r['candidate_success_rate']*100:.1f}% | {r['regression_degradation_pp']:.1f}pp | {r['false_activation_rate']*100:.1f}% |"
            )
        # Per-family breakdown
        lines.append("")
        lines.append("### Per-Family Candidate Breakdown")
        lines.append("")
        lines.append("| Config | Family | Success Rate | Count |")
        lines.append("|---|---|---|---|")
        for r in accepted:
            for fam, info in sorted(r.get("family_breakdown", {}).items()):
                lines.append(
                    f"| {r['config_id']} | {fam} | {info['success_rate']*100:.1f}% | {info['success_count']}/{info['total']} |"
                )
        lines.append("")
        if mode == "formal":
            lines.append("These configs meet all formal acceptance criteria and can be locked for bilevel initialization.")
        else:
            lines.append("These configs meet candidate + negative-control criteria but lack formal regression validation.")

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
    parser.add_argument("--candidate-set", type=str, default="candidate_search",
                        help="ScenarioRegistry set tag for candidate scenarios")
    parser.add_argument("--regression-set", type=str, default="regression_baseline",
                        help="ScenarioRegistry set tag for regression baselines (used for catalog only)")
    parser.add_argument("--negative-set", type=str, default="negative_control",
                        help="ScenarioRegistry set tag for negative controls")
    parser.add_argument("--mode", type=str, default="exploratory",
                        choices=["exploratory", "formal"],
                        help="exploratory: baseline optional; formal: baseline required")
    parser.add_argument("--regression-baseline-file", type=str, default=None,
                        help="Path to regression baseline CSV (required in formal mode)")
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
        candidate_set=args.candidate_set,
        regression_baseline_file=args.regression_baseline_file,
        negative_set=args.negative_set,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
