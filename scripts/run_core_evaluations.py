#!/usr/bin/env python3
"""Run core experiment evaluations (P0-A, P0-B, P1-A, P1-B).

This script handles methods not in run_paper_benchmark.py's METHODS dict.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import initialize_canonical_scenarios
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.evaluation.statistical_comparison import paired_t_test, cohens_d


REGISTRY_PATH = Path("config/checkpoint_registry.yaml")


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def get_training_checkpoint(registry: dict, key: str, seed: int = 0) -> str:
    entry = registry["training"][key]
    ckpt = entry["checkpoint"]
    if "{seed}" in ckpt:
        ckpt = ckpt.format(seed=seed)
    return ckpt


def get_training_output_dir(registry: dict, key: str, seed: int = 0) -> str:
    entry = registry["training"][key]
    out = entry["output_dir"]
    if "{seed}" in out:
        out = out.format(seed=seed)
    return out


def evaluate_method(
    config_path: str,
    checkpoint_path: str,
    method_name: str,
    seeds: list,
    gains: dict = None,
):
    """Evaluate a single checkpoint on a set of scenarios and seeds."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve includes for train_prediction_vpp_ppo configs
    includes = config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if inc_full.exists():
            with open(inc_full, "r", encoding="utf-8") as fi:
                merged = {**merged, **yaml.safe_load(fi)}
    config = {**merged, **config}

    # Override gains if provided (for bilevel)
    if gains:
        if "guidance" not in config:
            config["guidance"] = {}
        if "gains" not in config["guidance"]:
            config["guidance"]["gains"] = {}
        config["guidance"]["gains"].update(copy.deepcopy(gains))

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])

    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    if Path(checkpoint_path).exists():
        agent.load(checkpoint_path)
        print(f"  Loaded: {checkpoint_path}")
    else:
        print(f"  WARNING: Missing checkpoint {checkpoint_path}")

    scenarios = list(config.get("scenarios", {}).values())
    if not scenarios:
        # fallback to canonical regression suite for bilevel configs
        from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry
        scenarios = ScenarioRegistry.get_regression_suite()[:4]

    episodes = []
    for scen in scenarios:
        for seed in seeds:
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=config,
                scenario=scen,
                seed=seed,
                save_trajectory=False,
                method_name=method_name,
            )
            result["method"] = method_name
            result["scenario"] = scen.get("name", "unknown")
            episodes.append(result)

    env.close()
    return pd.DataFrame(episodes)


def save_results(df: pd.DataFrame, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "raw_episodes.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    summary = {
        "method": name,
        "n_episodes": len(df),
        "success_rate": float(df["is_success"].astype(float).mean()),
        "mean_return": float(df["return"].astype(float).mean()),
        "std_return": float(df["return"].astype(float).std()),
        "crash_rate": float(df["is_crash"].astype(float).mean()),
        "oob_rate": float(df["is_out_of_bounds"].astype(float).mean()),
        "timeout_rate": float(df["is_timeout"].astype(float).mean()),
    }
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {csv_path}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Mean return: {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")


def compare_methods(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str):
    r1 = df1["return"].astype(float).values
    r2 = df2["return"].astype(float).values
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    t_result = paired_t_test(r1, r2)
    d_result = cohens_d(r1, r2)
    p_val = t_result["p_value"]
    d_val = d_result["d"]
    print(f"  Paired t-test ({label2} vs {label1}): p={p_val:.4f}, d={d_val:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_PATH,
        help="Path to checkpoint registry YAML",
    )
    args = parser.parse_args()

    seeds = args.seeds
    results_root = Path("docs/results")
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))

    initialize_canonical_scenarios()

    # --- P0-A: VPP Ablation ---
    print("\n>>> P0-A: VPP Ablation")
    df_vpp = evaluate_method(
        "config/experiment/train_no_prediction_vpp_ppo.yaml",
        get_training_checkpoint(registry, "p0a_vpp"),
        "vpp_no_pred",
        seeds,
    )
    save_results(df_vpp, results_root / "p0a_vpp_ablation", "vpp_no_pred")

    df_no_vpp = evaluate_method(
        "config/experiment/train_no_vpp_direct_command.yaml",
        get_training_checkpoint(registry, "p0a_no_vpp"),
        "no_vpp_zero_offset",
        seeds,
    )
    save_results(df_no_vpp, results_root / "p0a_no_vpp_ablation", "no_vpp_zero_offset")
    compare_methods(df_vpp, df_no_vpp, "VPP", "No-VPP")

    # --- P0-B: Bilevel Ablation ---
    print("\n>>> P0-B: Bilevel Ablation")
    # Load best gains from bilevel results
    bilevel_out = get_training_output_dir(registry, "p0b_bilevel")
    bilevel_results = json.load(
        open(Path(bilevel_out) / "bilevel_results.json")
    )
    best_gains = bilevel_results.get("best_gains", {})
    best_episode = bilevel_results.get("best_policy_episode", 10)
    ckpt = str(Path(bilevel_out) / f"checkpoints/policy_ep{best_episode}.pt")

    df_single = evaluate_method(
        "config/experiment/train_no_prediction_vpp_ppo.yaml",
        get_training_checkpoint(registry, "no_prediction_vpp_ppo"),
        "single_layer",
        seeds,
    )
    save_results(df_single, results_root / "p0b_single_layer", "single_layer")

    df_bilevel = evaluate_method(
        "config/experiment/train_no_prediction_vpp_ppo.yaml",
        ckpt,
        "bilevel",
        seeds,
        gains=best_gains,
    )
    save_results(df_bilevel, results_root / "p0b_bilevel_ablation", "bilevel")
    compare_methods(df_single, df_bilevel, "Single", "Bilevel")

    # --- P1-A: Maneuvering Target ---
    print("\n>>> P1-A: Maneuvering Target")
    methods_p1a = {
        "no_prediction": (
            "config/experiment/train_no_prediction_vpp_ppo_maneuver.yaml",
            get_training_checkpoint(registry, "p1a_no_pred"),
        ),
        "cv_prediction": (
            "config/experiment/train_vpp_ppo_cv_maneuver.yaml",
            get_training_checkpoint(registry, "p1a_cv"),
        ),
        "ca_prediction": (
            "config/experiment/train_vpp_ppo_ca_maneuver.yaml",
            get_training_checkpoint(registry, "p1a_ca"),
        ),
    }
    dfs_p1a = {}
    for name, (cfg, ckpt) in methods_p1a.items():
        print(f"  Evaluating {name}...")
        if not Path(ckpt).exists():
            print(f"    SKIP: missing {ckpt}")
            continue
        df = evaluate_method(cfg, ckpt, name, seeds)
        dfs_p1a[name] = df
        save_results(df, results_root / "p1a_maneuver_target" / name, name)

    if "no_prediction" in dfs_p1a:
        for name, df in dfs_p1a.items():
            if name != "no_prediction":
                compare_methods(dfs_p1a["no_prediction"], df, "No-Pred", name)

    # --- P1-B: Neural Predictors ---
    print("\n>>> P1-B: Neural Predictors on Maneuvering Target")
    methods_p1b = {
        "lstm_frozen": (
            "config/experiment/train_vpp_ppo_lstm_frozen_maneuver.yaml",
            get_training_checkpoint(registry, "p1b_lstm"),
        ),
        "gru_frozen": (
            "config/experiment/train_vpp_ppo_gru_frozen_maneuver.yaml",
            get_training_checkpoint(registry, "p1b_gru"),
        ),
    }
    dfs_p1b = {}
    for name, (cfg, ckpt) in methods_p1b.items():
        print(f"  Evaluating {name}...")
        if not Path(ckpt).exists():
            print(f"    SKIP: missing {ckpt}")
            continue
        df = evaluate_method(cfg, ckpt, name, seeds)
        dfs_p1b[name] = df
        save_results(df, results_root / "p1b_neural_maneuver" / name, name)

    if "no_prediction" in dfs_p1a:
        for name, df in dfs_p1b.items():
            compare_methods(dfs_p1a["no_prediction"], df, "No-Pred", name)

    print("\n========================================")
    print("All core evaluations complete!")
    print("========================================")


if __name__ == "__main__":
    main()
