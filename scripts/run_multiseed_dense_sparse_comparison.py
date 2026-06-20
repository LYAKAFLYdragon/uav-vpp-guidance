#!/usr/bin/env python3
"""
Multi-seed full adversarial curriculum comparison:
    Dense VPP / Dense No-VPP / Sparse VPP / Sparse No-VPP

Each condition runs Stage-1 (curriculum) → Stage-2 (target + pursuer fine-tune)
→ Stage-3 (harder target + pursuer fine-tune), 100K steps per stage.

A shared adversarial target is trained per seed using the Dense-VPP Stage-1
pursuer as the frozen opponent, then all four conditions use that same target
for fair comparison.

Final evaluation reports OSR, crash, timeout, and per-scenario success rates
over 4 smoke-test scenarios × 50 episodes each.

Usage:
    source activate jsbenv
    python scripts/run_multiseed_dense_sparse_comparison.py \
        --seeds 0 1 2 \
        --output-root outputs/adversarial_curriculum_pilot/multi_seed_dense_sparse \
        --use-swanlab --swanlab-project uav-vpp-guidance --swanlab-exp ds_pilot_v1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.train_curriculum_ppo import train_ppo_curriculum
from scripts.train_adversarial_target import train_target
from scripts.train_adversarial_pursuer import train_pursuer
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.utils.swanlab_logger import SwanLabLogger

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

STAGE1_CONFIGS = {
    "dense_vpp": "config/experiment/dense_curriculum_pilot.yaml",
    "dense_no_vpp": "config/experiment/dense_curriculum_pilot_no_vpp.yaml",
    "sparse_vpp": "config/experiment/sparse_curriculum_pilot.yaml",
    "sparse_no_vpp": "config/experiment/sparse_curriculum_pilot_no_vpp.yaml",
}

STAGE2_PURSuer_CONFIGS = {
    "vpp": "config/adversarial/train_pursuer_v2_pilot_aggressive.yaml",
    "no_vpp": "config/adversarial/train_pursuer_v2_no_vpp_pilot.yaml",
}

STAGE3_PURSuer_CONFIGS = {
    "vpp": "config/adversarial/train_pursuer_v2_pilot_aggressive_stage3.yaml",
    "no_vpp": "config/adversarial/train_pursuer_v2_no_vpp_pilot.yaml",
}

TARGET_CONFIG = "config/adversarial/train_target_pilot.yaml"

EVAL_SCENARIOS = [
    "smoke_head_on",
    "smoke_crossing_left",
    "smoke_crossing_right",
    "smoke_tail_chase",
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load config and recursively apply includes."""
    base = load_yaml_config(config_path)
    includes = base.pop("includes", [])
    merged: Dict[str, Any] = {}
    cfg_dir = os.path.dirname(config_path)
    for inc in includes:
        inc_full = os.path.join(cfg_dir, inc)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_experiment_config(inc_full))
    return merge_config(merged, base)


def make_swanlab_logger(project: Optional[str], experiment: Optional[str], config: Dict[str, Any]) -> Optional[Any]:
    if not project or not experiment:
        return None
    return SwanLabLogger(
        project=project,
        experiment=experiment,
        config=config,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_stage1(
    condition: str,
    seed: int,
    output_dir: str,
    smoke: bool,
    use_swanlab: bool,
    swanlab_project: Optional[str],
    swanlab_exp_base: Optional[str],
) -> str:
    """Train Stage-1 pursuer vs maneuver-library bandit. Returns best checkpoint path."""
    config_path = STAGE1_CONFIGS[condition]
    config = load_experiment_config(config_path)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = f"{condition}_stage1_s{seed}"

    os.makedirs(output_dir, exist_ok=True)

    swanlab_logger = None
    if use_swanlab:
        exp_name = f"{swanlab_exp_base}_{condition}_s{seed}_s1" if swanlab_exp_base else None
        swanlab_logger = make_swanlab_logger(
            swanlab_project, exp_name,
            {"condition": condition, "seed": seed, "stage": 1, "config": config_path},
        )

    set_seed(seed)
    try:
        train_ppo_curriculum(
            config, output_dir, smoke=smoke, algorithm="ppo", swanlab_logger=swanlab_logger
        )
    finally:
        if swanlab_logger is not None:
            swanlab_logger.finish()

    ckpt = os.path.join(output_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(output_dir, "checkpoints", "last.pt")
    return ckpt


def run_target_stage(
    stage: int,
    prev_pursuer_ckpt: str,
    output_dir: str,
    seed: int,
    difficulty: str,
    smoke: bool,
    use_swanlab: bool,
    swanlab_project: Optional[str],
    swanlab_exp_base: Optional[str],
) -> str:
    """Train an RL target against a frozen pursuer. Returns target checkpoint."""
    config = load_experiment_config(TARGET_CONFIG)
    config["experiment"]["name"] = f"shared_target_stage{stage}_s{seed}"
    config["experiment"]["seed"] = seed
    config["pursuer_checkpoint"] = prev_pursuer_ckpt

    # Difficulty-specific target behavior
    config.setdefault("env", {}).setdefault("bandit", {})["difficulty"] = difficulty
    if difficulty == "hard":
        selector = config["env"]["bandit"].setdefault("selector", {})
        selector["reaction_time_s"] = 1.5
        selector["selector_noise"] = 0.25
        selector["param_perturb_scale"] = 0.25

    os.makedirs(output_dir, exist_ok=True)

    swanlab_logger = None
    if use_swanlab:
        exp_name = f"{swanlab_exp_base}_shared_target_s{seed}_s{stage}" if swanlab_exp_base else None
        swanlab_logger = make_swanlab_logger(
            swanlab_project, exp_name,
            {"seed": seed, "stage": stage, "difficulty": difficulty},
        )

    set_seed(seed)
    try:
        train_target(config, output_dir, smoke=smoke, swanlab_logger=swanlab_logger, log_prefix=f"s{stage}/target/")
    finally:
        if swanlab_logger is not None:
            swanlab_logger.finish()

    return os.path.join(output_dir, "checkpoints", "best.pt")


def run_pursuer_stage(
    condition: str,
    stage: int,
    prev_pursuer_ckpt: str,
    target_ckpt: str,
    output_dir: str,
    seed: int,
    smoke: bool,
    use_swanlab: bool,
    swanlab_project: Optional[str],
    swanlab_exp_base: Optional[str],
) -> str:
    """Fine-tune pursuer against a frozen target. Returns pursuer checkpoint."""
    vpp_key = "no_vpp" if "no_vpp" in condition else "vpp"
    config_path = (STAGE2_PURSuer_CONFIGS if stage == 2 else STAGE3_PURSuer_CONFIGS)[vpp_key]
    config = load_experiment_config(config_path)
    config["experiment"]["name"] = f"{condition}_stage{stage}_s{seed}"
    config["experiment"]["seed"] = seed
    config["pursuer_checkpoint"] = prev_pursuer_ckpt
    config["target_checkpoint"] = target_ckpt

    os.makedirs(output_dir, exist_ok=True)

    swanlab_logger = None
    if use_swanlab:
        exp_name = f"{swanlab_exp_base}_{condition}_s{seed}_s{stage}" if swanlab_exp_base else None
        swanlab_logger = make_swanlab_logger(
            swanlab_project, exp_name,
            {"condition": condition, "seed": seed, "stage": stage},
        )

    set_seed(seed)
    try:
        train_pursuer(config, output_dir, smoke=smoke, swanlab_logger=swanlab_logger, log_prefix=f"s{stage}/pursuer/")
    finally:
        if swanlab_logger is not None:
            swanlab_logger.finish()

    ckpt = os.path.join(output_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(output_dir, "checkpoints", "last.pt")
    return ckpt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _policy_config_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    import torch
    if not os.path.exists(checkpoint_path):
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        return ckpt.get("config", {}).get("policy")
    except Exception:
        return None


def evaluate_pursuer_per_scenario(
    pursuer_ckpt: str,
    target_ckpt: str,
    base_config: Dict[str, Any],
    scenarios: List[str],
    episodes_per_scenario: int = 50,
    seeds: List[int] = None,
) -> Dict[str, Any]:
    """Evaluate a pursuer on explicit scenarios. Returns per-scenario and overall stats."""
    if seeds is None:
        seeds = [0, 1, 2, 3]

    import torch
    from uav_vpp_guidance.agents.ppo_agent import PPOAgent
    from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
    from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
    from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry, initialize_canonical_scenarios

    initialize_canonical_scenarios()

    config = copy.deepcopy(base_config)
    env = AdversarialJSBSimEnv(config)

    # Load pursuer
    pursuer_cfg = dict(config)
    ckpt_policy = _policy_config_from_checkpoint(pursuer_ckpt)
    if ckpt_policy:
        pursuer_cfg["policy"] = ckpt_policy
    pursuer_agent = PPOAgent(
        obs_dim=16, action_dim=3, config=pursuer_cfg,
        device=config.get("ppo", {}).get("device", "cpu"),
    )
    pursuer_agent.load(pursuer_ckpt)
    pursuer_agent.network.eval()

    # Load target
    target_cfg = dict(config)
    target_cfg["ppo"] = config.get("target_ppo", config.get("ppo", {}))
    target_cfg["policy"] = config.get("target_policy", config.get("policy", {}))
    ckpt_target_policy = _policy_config_from_checkpoint(target_ckpt)
    if ckpt_target_policy:
        target_cfg["policy"] = ckpt_target_policy
    target_agent = AdversarialTargetAgent(config=target_cfg)
    target_agent.load(target_ckpt)
    target_agent.eval()

    per_scenario: Dict[str, List[Dict[str, Any]]] = {s: [] for s in scenarios}

    try:
        for seed in seeds:
            for scenario_name in scenarios:
                scenario = ScenarioRegistry.get(scenario_name)
                if scenario is None:
                    print(f"[EVAL] Warning: scenario {scenario_name} not found")
                    continue
                for ep in range(episodes_per_scenario // max(1, len(seeds))):
                    ep_seed = seed * 100000 + ep
                    p_obs, t_obs = env.reset(scenario=scenario, seed=ep_seed)
                    for _ in range(env.max_steps):
                        p_action = pursuer_agent.get_deterministic_action(p_obs["observation_vector"])
                        t_action = target_agent.get_deterministic_action(t_obs["observation_vector"])
                        p_obs, t_obs, _, _, terminated, truncated, info = env.step(p_action, t_action)
                        if terminated or truncated:
                            reason = info.get("termination", {}).get("reason", "unknown")
                            per_scenario[scenario_name].append({
                                "seed": seed,
                                "episode": ep,
                                "reason": reason,
                                "success": reason == "success",
                                "crash": reason == "crash",
                                "out_of_bounds": reason == "out_of_bounds",
                                "timeout": reason == "timeout",
                            })
                            break
    finally:
        env.close()

    scenario_stats = {}
    all_records: List[Dict[str, Any]] = []
    for name, records in per_scenario.items():
        all_records.extend(records)
        n = len(records)
        scenario_stats[name] = {
            "n": n,
            "success_rate": sum(1 for r in records if r["success"]) / max(1, n),
            "crash_rate": sum(1 for r in records if r["crash"]) / max(1, n),
            "out_of_bounds_rate": sum(1 for r in records if r["out_of_bounds"]) / max(1, n),
            "timeout_rate": sum(1 for r in records if r["timeout"]) / max(1, n),
        }

    n_total = len(all_records)
    overall = {
        "n": n_total,
        "success_rate": sum(1 for r in all_records if r["success"]) / max(1, n_total),
        "crash_rate": sum(1 for r in all_records if r["crash"]) / max(1, n_total),
        "out_of_bounds_rate": sum(1 for r in all_records if r["out_of_bounds"]) / max(1, n_total),
        "timeout_rate": sum(1 for r in all_records if r["timeout"]) / max(1, n_total),
    }

    return {
        "overall": overall,
        "per_scenario": scenario_stats,
        "records": per_scenario,
    }


# ---------------------------------------------------------------------------
# Aggregation / plotting
# ---------------------------------------------------------------------------

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-condition, per-seed results."""
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_condition.setdefault(r["condition"], []).append(r)

    summary = {}
    for condition, items in by_condition.items():
        osrs = [it["eval"]["overall"]["success_rate"] for it in items]
        summary[condition] = {
            "num_seeds": len(items),
            "mean_osr": float(np.mean(osrs)),
            "std_osr": float(np.std(osrs)),
            "min_osr": float(np.min(osrs)),
            "max_osr": float(np.max(osrs)),
            "seeds": {it["seed"]: it["eval"] for it in items},
        }

    # Hypothesis tests
    def _delta(cond_a: str, cond_b: str) -> Dict[str, Any]:
        diffs = []
        for seed in summary[cond_a]["seeds"]:
            if seed in summary[cond_b]["seeds"]:
                diffs.append(
                    summary[cond_a]["seeds"][seed]["overall"]["success_rate"]
                    - summary[cond_b]["seeds"][seed]["overall"]["success_rate"]
                )
        arr = np.array(diffs)
        return {
            "mean": float(np.mean(arr)) if len(arr) else None,
            "std": float(np.std(arr)) if len(arr) else None,
            "values": [float(x) for x in arr],
            "n": len(arr),
        }

    if "dense_vpp" in summary and "dense_no_vpp" in summary:
        summary["delta_vpp_dense"] = _delta("dense_vpp", "dense_no_vpp")
    if "sparse_vpp" in summary and "sparse_no_vpp" in summary:
        summary["delta_vpp_sparse"] = _delta("sparse_vpp", "sparse_no_vpp")
    if "sparse_no_vpp" in summary and "dense_no_vpp" in summary:
        summary["delta_sparse_no_vpp_vs_dense_no_vpp"] = _delta("sparse_no_vpp", "dense_no_vpp")

    summary["hypothesis_1"] = {
        "description": "Sparse reward amplifies VPP advantage over No-VPP",
        "test": "delta_vpp_sparse > delta_vpp_dense",
        "delta_vpp_sparse_mean": summary.get("delta_vpp_sparse", {}).get("mean"),
        "delta_vpp_dense_mean": summary.get("delta_vpp_dense", {}).get("mean"),
        "amplified": (
            summary.get("delta_vpp_sparse", {}).get("mean") is not None
            and summary.get("delta_vpp_dense", {}).get("mean") is not None
            and summary["delta_vpp_sparse"]["mean"] > summary["delta_vpp_dense"]["mean"]
        ),
    }
    summary["hypothesis_2"] = {
        "description": "Sparse reward helps No-VPP as well",
        "test": "sparse_no_vpp_osr > dense_no_vpp_osr",
        "sparse_no_vpp_mean_osr": summary.get("sparse_no_vpp", {}).get("mean_osr"),
        "dense_no_vpp_mean_osr": summary.get("dense_no_vpp", {}).get("mean_osr"),
        "helps": (
            summary.get("sparse_no_vpp", {}).get("mean_osr") is not None
            and summary.get("dense_no_vpp", {}).get("mean_osr") is not None
            and summary["sparse_no_vpp"]["mean_osr"] > summary["dense_no_vpp"]["mean_osr"]
        ),
    }

    return summary


def plot_comparison(summary: Dict[str, Any], output_path: str):
    """Bar plot of mean OSR per condition with error bars."""
    conditions = ["dense_vpp", "dense_no_vpp", "sparse_vpp", "sparse_no_vpp"]
    labels = ["Dense+VPP", "Dense+No-VPP", "Sparse+VPP", "Sparse+No-VPP"]
    means = [summary[c]["mean_osr"] * 100 if c in summary else 0.0 for c in conditions]
    stds = [summary[c]["std_osr"] * 100 if c in summary else 0.0 for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Overall Success Rate (OSR) %")
    ax.set_title("Dense vs Sparse Reward: VPP vs No-VPP (multi-seed)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 2,
                f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=10)

    # Delta annotations
    d_dense = summary.get("delta_vpp_dense", {})
    d_sparse = summary.get("delta_vpp_sparse", {})
    if d_dense.get("mean") is not None and d_sparse.get("mean") is not None:
        text = (
            f"VPP boost (Dense): {d_dense['mean']*100:+.1f}%\n"
            f"VPP boost (Sparse): {d_sparse['mean']*100:+.1f}%"
        )
        ax.text(0.95, 0.95, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-seed dense/sparse VPP vs No-VPP adversarial curriculum.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-root", type=str, default="outputs/adversarial_curriculum_pilot/multi_seed_dense_sparse")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with minimal steps")
    parser.add_argument("--eval-episodes-per-scenario", type=int, default=50)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="uav-vpp-guidance")
    parser.add_argument("--swanlab-exp", type=str, default="ds_pilot_v1")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Subset of conditions to run (default all 4)")
    args = parser.parse_args()

    conditions = args.conditions or list(STAGE1_CONFIGS.keys())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Top-level SwanLab logger for summary
    swanlab_summary = None
    if args.use_swanlab:
        swanlab_summary = make_swanlab_logger(
            args.swanlab_project, args.swanlab_exp,
            {"seeds": args.seeds, "conditions": conditions, "smoke": args.smoke},
        )

    manifest: Dict[str, Any] = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "seeds": args.seeds,
        "conditions": conditions,
    }
    per_run_results: List[Dict[str, Any]] = []

    try:
        for seed in args.seeds:
            seed_dir = output_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # ---- Shared target for this seed (trained from dense VPP Stage-1) ----
            shared_stage1_dir = seed_dir / "shared_dense_vpp_stage1"
            shared_stage2_target_dir = seed_dir / "shared_target_stage2"
            shared_stage3_target_dir = seed_dir / "shared_target_stage3"

            if not args.skip_training:
                print("\n" + "=" * 80)
                print(f"[SEED {seed}] Training shared Dense-VPP Stage-1 pursuer & target")
                print("=" * 80)
                shared_stage1_ckpt = run_stage1(
                    "dense_vpp", seed, str(shared_stage1_dir), args.smoke,
                    args.use_swanlab, args.swanlab_project, args.swanlab_exp,
                )

                shared_stage2_target_ckpt = run_target_stage(
                    2, shared_stage1_ckpt, str(shared_stage2_target_dir), seed,
                    "medium", args.smoke, args.use_swanlab, args.swanlab_project, args.swanlab_exp,
                )

                shared_stage3_target_ckpt = run_target_stage(
                    3, shared_stage1_ckpt, str(shared_stage3_target_dir), seed,
                    "hard", args.smoke, args.use_swanlab, args.swanlab_project, args.swanlab_exp,
                )
            else:
                shared_stage1_ckpt = str(shared_stage1_dir / "checkpoints" / "best.pt")
                shared_stage2_target_ckpt = str(shared_stage2_target_dir / "checkpoints" / "best.pt")
                shared_stage3_target_ckpt = str(shared_stage3_target_dir / "checkpoints" / "best.pt")

            manifest.setdefault("shared_targets", {}).setdefault(seed, {
                "stage1_pursuer": shared_stage1_ckpt,
                "stage2_target": shared_stage2_target_ckpt,
                "stage3_target": shared_stage3_target_ckpt,
            })

            # ---- Each condition ----
            for condition in conditions:
                print("\n" + "=" * 80)
                print(f"[SEED {seed} | {condition}] Full pipeline")
                print("=" * 80)

                cond_dir = seed_dir / condition
                stage1_dir = cond_dir / "stage1"
                stage2_dir = cond_dir / "stage2" / "pursuer"
                stage3_dir = cond_dir / "stage3" / "pursuer"

                if not args.skip_training:
                    stage1_ckpt = run_stage1(
                        condition, seed, str(stage1_dir), args.smoke,
                        args.use_swanlab, args.swanlab_project, args.swanlab_exp,
                    )
                    stage2_ckpt = run_pursuer_stage(
                        condition, 2, stage1_ckpt, shared_stage2_target_ckpt,
                        str(stage2_dir), seed, args.smoke,
                        args.use_swanlab, args.swanlab_project, args.swanlab_exp,
                    )
                    stage3_ckpt = run_pursuer_stage(
                        condition, 3, stage2_ckpt, shared_stage3_target_ckpt,
                        str(stage3_dir), seed, args.smoke,
                        args.use_swanlab, args.swanlab_project, args.swanlab_exp,
                    )
                else:
                    stage3_ckpt = str(stage3_dir / "checkpoints" / "best.pt")

                # ---- Evaluation ----
                print(f"\n[EVAL] {condition} seed {seed}")
                vpp_key = "no_vpp" if "no_vpp" in condition else "vpp"
                eval_config_path = STAGE3_PURSuer_CONFIGS[vpp_key]
                eval_config = load_experiment_config(eval_config_path)
                eval_config["target_checkpoint"] = shared_stage3_target_ckpt
                eval_config["pursuer_checkpoint"] = stage3_ckpt

                eval_res = evaluate_pursuer_per_scenario(
                    pursuer_ckpt=stage3_ckpt,
                    target_ckpt=shared_stage3_target_ckpt,
                    base_config=eval_config,
                    scenarios=EVAL_SCENARIOS,
                    episodes_per_scenario=args.eval_episodes_per_scenario,
                    seeds=args.eval_seeds,
                )

                run_result = {
                    "condition": condition,
                    "seed": seed,
                    "stage3_ckpt": stage3_ckpt,
                    "target_ckpt": shared_stage3_target_ckpt,
                    "eval": eval_res,
                }
                per_run_results.append(run_result)
                manifest.setdefault("runs", []).append(run_result)

                if swanlab_summary is not None:
                    swanlab_summary.log({
                        f"eval/{condition}_seed{seed}_osr": eval_res["overall"]["success_rate"],
                        f"eval/{condition}_seed{seed}_crash": eval_res["overall"]["crash_rate"],
                        f"eval/{condition}_seed{seed}_timeout": eval_res["overall"]["timeout_rate"],
                    }, step=seed)

        # ---- Aggregation ----
        summary = aggregate_results(per_run_results)
        manifest["summary"] = summary
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["status"] = "completed"

        manifest_path = output_root / "multi_seed_comparison.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n[SUMMARY] Saved manifest to {manifest_path}")

        plot_path = output_root / "vpp_vs_no_vpp_sparse_vs_dense.png"
        plot_comparison(summary, str(plot_path))

        if swanlab_summary is not None:
            for condition in conditions:
                if condition in summary:
                    swanlab_summary.log({
                        f"summary/{condition}_mean_osr": summary[condition]["mean_osr"],
                        f"summary/{condition}_std_osr": summary[condition]["std_osr"],
                    }, step=0)
            for key in ["delta_vpp_dense", "delta_vpp_sparse", "delta_sparse_no_vpp_vs_dense_no_vpp"]:
                if key in summary and summary[key]["mean"] is not None:
                    swanlab_summary.log({
                        f"summary/{key}_mean": summary[key]["mean"],
                        f"summary/{key}_std": summary[key]["std"],
                    }, step=0)
            swanlab_summary.log({
                "summary/hypothesis_1_amplified": summary["hypothesis_1"]["amplified"],
                "summary/hypothesis_2_helps": summary["hypothesis_2"]["helps"],
            }, step=0)
            print(json.dumps(summary, indent=2, ensure_ascii=False))

    finally:
        if swanlab_summary is not None:
            swanlab_summary.finish()


if __name__ == "__main__":
    main()
