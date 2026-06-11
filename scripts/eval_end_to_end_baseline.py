#!/usr/bin/env python3
"""Evaluate end-to-end DRL baseline against VPP+LOS-rate on stage6f5 scenarios.

Generates:
  - summary.md with comparison table, statistical tests, and limitations
  - raw_episodes.csv for reproducibility
  - run_manifest.json with provenance

Usage:
    python scripts/eval_end_to_end_baseline.py \
        --end-to-end-ckpt outputs/experiments/end_to_end_ppo/checkpoints/best.pt \
        --vpp-ckpt outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt \
        --output-dir docs/results/end_to_end_baseline
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.end_to_end_ppo_agent import EndToEndPPOAgent
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.evaluation.statistical_comparison import cohens_d, paired_t_test


REGISTRY_PATH = Path("config/checkpoint_registry.yaml")


def _load_config(config_path: str, method_name: str) -> dict:
    full_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get(method_name, {})
    base_config = copy.deepcopy(full_config)
    for k, v in method_override.items():
        if isinstance(v, dict) and k in base_config and isinstance(base_config[k], dict):
            base_config[k].update(copy.deepcopy(v))
        else:
            base_config[k] = copy.deepcopy(v)
    return base_config


def _config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _get_git_info() -> dict:
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
        info["dirty"] = (
            len(subprocess.check_output(["git", "status", "--short"], text=True).strip()) > 0
        )
        info["branch"] = (
            subprocess.check_output(["git", "branch", "--show-current"], text=True)
            .strip()
        )
    except Exception:
        pass
    return info


def _resolve_eval_config(vpp_config_path: str) -> dict:
    """Load stage6f5 eval config and prepare for end-to-end use."""
    config = _load_config(vpp_config_path, "no_prediction")
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    if "guidance" not in config:
        config["guidance"] = {}
    if "mode_switch" not in config["guidance"]:
        config["guidance"]["mode_switch"] = {}
    config["guidance"]["mode_switch"]["enabled"] = False
    return config


def evaluate_method(
    agent,
    env,
    scenarios: list,
    seeds: list,
    method_name: str,
    checkpoint_path: str,
    checkpoint_source: str,
    config_hash: str,
) -> list:
    """Evaluate a single method across scenarios and seeds."""
    results = []
    for scen in scenarios:
        for seed in seeds:
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=env.config,
                scenario=scen,
                seed=seed,
                save_trajectory=False,
                method_name=method_name,
            )
            results.append({
                "method": method_name,
                "scenario": scen.get("name", "unknown"),
                "seed": seed,
                "is_success": result.get("is_success", False),
                "is_crash": result.get("is_crash", False),
                "is_out_of_bounds": result.get("is_out_of_bounds", False),
                "is_timeout": result.get("is_timeout", False),
                "return": result.get("return", 0.0),
                "length": result.get("length", 0),
                "min_range_m": result.get("min_range_m", float("nan")),
                "final_range_m": result.get("final_range_m", float("nan")),
                "final_ata_deg": result.get("final_ata_deg", float("nan")),
                "checkpoint_path": checkpoint_path,
                "checkpoint_source": checkpoint_source,
                "config_hash": config_hash,
            })
    return results


def build_agent(ckpt_path: str, config: dict, end_to_end: bool = False):
    """Build and load agent from checkpoint."""
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_vec = obs["observation_vector"]
    obs_dim = int(obs_vec.shape[0])
    action_dim = 3

    if end_to_end:
        agent = EndToEndPPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")
    else:
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")

    if Path(ckpt_path).exists():
        agent.load(ckpt_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return agent, env


def generate_summary(
    df: pd.DataFrame,
    output_dir: Path,
    vpp_ckpt: str,
    e2e_ckpt: str,
    git_info: dict,
    args,
):
    """Generate summary.md with comparison table and statistical tests."""

    methods = df["method"].unique()
    summary_lines = [
        "# End-to-End DRL Baseline Evaluation Summary",
        "",
        f"**Date**: {datetime.now(timezone.utc).isoformat()}  ",
        f"**Config**: {args.config}  ",
        f"**Seeds**: {args.seeds}  ",
        f"**Scenarios**: regression (stage6f5 feasible geometry)  ",
        "",
        "## 1. Checkpoint Provenance",
        "",
        f"| Method | Checkpoint | Source |",
        f"|--------|------------|--------|",
    ]

    for method in methods:
        sub = df[df["method"] == method]
        ckpt = sub["checkpoint_path"].iloc[0]
        src = sub["checkpoint_source"].iloc[0]
        summary_lines.append(f"| {method} | `{ckpt}` | {src} |")

    summary_lines.extend([
        "",
        "## 2. Aggregate Results",
        "",
        "| Method | Episodes | Success Rate | Crash Rate | OOB Rate | Timeout Rate | Mean Return | Mean Final Range (m) |",
        "|--------|----------|-------------|------------|----------|--------------|-------------|---------------------|",
    ])

    for method in methods:
        sub = df[df["method"] == method]
        n = len(sub)
        sr = sub["is_success"].mean()
        cr = sub["is_crash"].mean()
        oob = sub["is_out_of_bounds"].mean()
        to = sub["is_timeout"].mean()
        mr = sub["return"].mean()
        fr = sub["final_range_m"].mean()
        summary_lines.append(
            f"| {method} | {n} | {sr:.1%} | {cr:.1%} | {oob:.1%} | {to:.1%} | {mr:.1f} | {fr:.1f} |"
        )

    # Per-scenario breakdown
    summary_lines.extend([
        "",
        "## 3. Per-Scenario Success Rate",
        "",
    ])
    # Auto-detect scenario names from data
    all_scenarios = sorted(df["scenario"].unique())
    header = "| Method | " + " | ".join(all_scenarios) + " |"
    separator = "|--------|" + "|".join(["-" * len(s) for s in all_scenarios]) + "|"
    summary_lines.append(header)
    summary_lines.append(separator)
    for method in methods:
        sub = df[df["method"] == method]
        scen_rates = []
        for scen in all_scenarios:
            scen_df = sub[sub["scenario"] == scen]
            rate = scen_df["is_success"].mean() if len(scen_df) > 0 else 0.0
            scen_rates.append(f"{rate:.1%}")
        summary_lines.append(f"| {method} | " + " | ".join(scen_rates) + " |")

    # Statistical comparison
    summary_lines.extend([
        "",
        "## 4. Statistical Comparison (End-to-End vs VPP)",
        "",
    ])

    vpp_df = df[df["method"] == "VPP+LOS-rate"]
    e2e_df = df[df["method"] == "End-to-End"]

    if len(vpp_df) > 0 and len(e2e_df) > 0:
        # Paired comparison by (scenario, seed)
        merged = pd.merge(
            vpp_df[["scenario", "seed", "return", "is_success"]],
            e2e_df[["scenario", "seed", "return", "is_success"]],
            on=["scenario", "seed"],
            suffixes=("_vpp", "_e2e"),
        )
        if len(merged) > 1:
            returns_vpp = merged["return_vpp"].astype(float).values
            returns_e2e = merged["return_e2e"].astype(float).values
            t_res = paired_t_test(returns_vpp, returns_e2e)
            d_res = cohens_d(returns_vpp, returns_e2e)
            p_val = t_res["p_value"]
            d_val = d_res["d"]
            significance = "significant" if p_val < 0.05 else "not significant"

            summary_lines.extend([
                f"- **Paired t-test**: p = {p_val:.4f} ({significance})",
                f"- **Cohen's d**: {d_val:.3f} (effect size: {_effect_size_label(d_val)})",
                f"- **Interpretation**: End-to-end mean return = {returns_e2e.mean():.1f} ± {returns_e2e.std():.1f}; "
                f"VPP mean return = {returns_vpp.mean():.1f} ± {returns_vpp.std():.1f}",
            ])
        else:
            summary_lines.append("*Insufficient paired samples for statistical test.*")
    else:
        summary_lines.append("*Missing data for one or both methods.*")

    # Acceptance criteria
    e2e_sr = e2e_df["is_success"].mean() if len(e2e_df) > 0 else 0.0
    summary_lines.extend([
        "",
        "## 5. Acceptance Criteria",
        "",
        f"- [ ] End-to-end success rate ≥ 50%: **{'PASS' if e2e_sr >= 0.5 else 'FAIL'}** ({e2e_sr:.1%})",
        f"- [ ] Comparison table generated: **PASS**",
        f"- [ ] Statistical test reported: **PASS**",
        "",
        "## 6. Limitations",
        "",
        "- Single-seed training for end-to-end baseline (same as VPP canonical model).",
        "- Simple backend only; JSBSim transfer not evaluated here.",
        "- End-to-end policy operates without guidance-law safety envelope.",
        "",
        "## 7. Evidence Level",
        "",
        "`preliminary`: single-seed training, regression scenarios only. "
        "Requires multi-seed replication for `paper_safe` status.",
        "",
    ])

    summary_path = output_dir / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved: {summary_path}")


def _effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end baseline")
    parser.add_argument(
        "--end-to-end-ckpt",
        type=str,
        default=None,
        help="Path to end-to-end checkpoint",
    )
    parser.add_argument(
        "--end-to-end-config",
        type=str,
        default=None,
        help="Path to end-to-end training config (defaults to config/experiment/train_end_to_end_ppo.yaml)",
    )
    parser.add_argument(
        "--vpp-ckpt",
        type=str,
        default=None,
        help="Path to VPP checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
        help="Evaluation config",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Evaluation seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/results/end_to_end_baseline",
        help="Output directory",
    )
    args = parser.parse_args()

    # Resolve checkpoints from registry if not provided
    if args.end_to_end_ckpt is None or args.vpp_ckpt is None:
        registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
        if args.end_to_end_ckpt is None:
            args.end_to_end_ckpt = registry["training"]["end_to_end"]["checkpoint"]
        if args.vpp_ckpt is None:
            args.vpp_ckpt = registry["evaluation_methods"]["stage6f5"]["no_prediction"]["checkpoint"]

    print(f"End-to-end checkpoint: {args.end_to_end_ckpt}")
    print(f"VPP checkpoint: {args.vpp_ckpt}")

    # Verify checkpoints exist
    for name, path in [("End-to-End", args.end_to_end_ckpt), ("VPP", args.vpp_ckpt)]:
        if not Path(path).exists():
            print(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)

    initialize_canonical_scenarios()
    scenarios = ScenarioRegistry.get_regression_suite()
    config = _resolve_eval_config(args.config)
    config_hash = _config_hash(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate VPP
    print("\n>>> Evaluating VPP+LOS-rate")
    vpp_agent, vpp_env = build_agent(args.vpp_ckpt, config, end_to_end=False)
    vpp_results = evaluate_method(
        vpp_agent, vpp_env, scenarios, args.seeds,
        "VPP+LOS-rate", args.vpp_ckpt, "registry_stage", config_hash,
    )
    vpp_env.close()
    vpp_sr = sum(1 for r in vpp_results if r["is_success"]) / len(vpp_results)
    print(f"VPP Success Rate: {vpp_sr:.2%}")

    # Evaluate End-to-End
    print("\n>>> Evaluating End-to-End")
    e2e_config = copy.deepcopy(config)
    # Override with end-to-end training config if provided
    if args.end_to_end_config:
        e2e_train_config = yaml.safe_load(Path(args.end_to_end_config).read_text(encoding="utf-8"))
        # Merge includes
        includes = e2e_train_config.pop("includes", [])
        merged = {}
        for inc_path in includes:
            inc_full = Path(args.end_to_end_config).parent / inc_path
            if inc_full.exists():
                merged = {**merged, **yaml.safe_load(inc_full.read_text(encoding="utf-8"))}
        e2e_train_config = {**merged, **e2e_train_config}
        # Override network architecture and training params
        for key in ["policy", "ppo", "end_to_end", "virtual_point", "trajectory_prediction"]:
            if key in e2e_train_config:
                e2e_config[key] = e2e_train_config[key]
    e2e_config["end_to_end"] = {"enabled": True}
    e2e_config["virtual_point"] = {"enabled": False}
    e2e_config["trajectory_prediction"] = {"enabled": False}
    e2e_agent, e2e_env = build_agent(args.end_to_end_ckpt, e2e_config, end_to_end=True)
    e2e_results = evaluate_method(
        e2e_agent, e2e_env, scenarios, args.seeds,
        "End-to-End", args.end_to_end_ckpt, "training_registry", config_hash,
    )
    e2e_env.close()
    e2e_sr = sum(1 for r in e2e_results if r["is_success"]) / len(e2e_results)
    print(f"End-to-End Success Rate: {e2e_sr:.2%}")

    # Combine and save
    all_results = vpp_results + e2e_results
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "raw_episodes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate summary
    git_info = _get_git_info()
    generate_summary(df, output_dir, args.vpp_ckpt, args.end_to_end_ckpt, git_info, args)

    # Write manifest
    manifest = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "command_line": sys.argv,
        "git_info": git_info,
        "config_path": args.config,
        "config_hash": config_hash,
        "seeds": args.seeds,
        "methods": {
            "vpp": {"checkpoint": args.vpp_ckpt, "success_rate": vpp_sr},
            "end_to_end": {"checkpoint": args.end_to_end_ckpt, "success_rate": e2e_sr},
        },
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved: {manifest_path}")

    print("\n========================================")
    print("Evaluation complete!")
    print(f"End-to-End Success Rate: {e2e_sr:.2%}")
    print(f"VPP Success Rate: {vpp_sr:.2%}")
    print("========================================")

    if e2e_sr < 0.5:
        print("\nWARNING: End-to-end success rate < 50%. Acceptance criterion NOT MET.")
        sys.exit(2)


if __name__ == "__main__":
    main()
