#!/usr/bin/env python3
"""
Evaluate the curriculum + adversarial pipeline and generate paper figures.

Reads the manifest produced by `scripts/train_curriculum_adversarial.py` and:
  1. Evaluates each stage's pursuer against the stage's target.
  2. Aggregates per-stage and per-scenario statistics.
  3. Generates:
       - curriculum_progress.png/pdf
       - adversarial_before_after.png/pdf
       - trajectory_3d.png/pdf

Usage:
    python scripts/evaluate_curriculum_adversarial.py \
        --manifest outputs/adversarial_curriculum/manifest.json \
        --output-dir outputs/adversarial_curriculum
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.evaluate_adversarial import run_adversarial_episode, SCENARIO_BUILDERS
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load and merge experiment config."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: Dict[str, Any] = {}
    cfg_dir = os.path.dirname(config_path)
    for inc_path in includes:
        inc_full = os.path.join(cfg_dir, inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stage_eval_curve(stage_dir: str, stage: str) -> Tuple[List[int], List[float]]:
    """Load (steps, success_rates) for a stage."""
    if stage == "stage1":
        log_path = Path(stage_dir) / "stage1" / "logs" / "eval_log.csv"
    else:
        log_path = Path(stage_dir) / stage / "pursuer" / "logs" / "adversarial_training_log.csv"

    if not log_path.exists():
        return [], []

    steps, rates = [], []
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            if stage == "stage1":
                rates.append(float(row["success_rate"]))
            else:
                rates.append(float(row["capture_rate"]))
    return steps, rates


def _load_config_for_checkpoint(ckpt_path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Load the config snapshot saved next to a checkpoint."""
    snapshot = Path(ckpt_path).parent.parent / "config_snapshot.yaml"
    if snapshot.exists():
        return load_experiment_config(str(snapshot))
    return fallback


def _policy_config_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Extract the policy architecture stored in a PPO checkpoint."""
    if not Path(checkpoint_path).exists():
        return None
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        return ckpt.get("config", {}).get("policy")
    except Exception:
        return None


def evaluate_stage(
    stage: str,
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    scenarios: List[str],
    num_episodes: int,
    seeds: List[int],
) -> Dict[str, Any]:
    """Evaluate one stage's pursuer against its target."""
    pursuer_ckpt = manifest["stages"][stage]["pursuer_ckpt"]
    # Smoke runs may only save last.pt; fall back if best.pt is missing.
    if not Path(pursuer_ckpt).exists():
        pursuer_ckpt = str(Path(pursuer_ckpt).parent / "last.pt")

    if stage == "stage1":
        target_ckpt = None
    else:
        # Derive target checkpoint from pursuer checkpoint path:
        #   .../stage2/pursuer/checkpoints/best.pt -> .../stage2/target/checkpoints/best.pt
        target_ckpt = str(Path(pursuer_ckpt).parent.parent.parent / "target" / "checkpoints" / "best.pt")
        if not Path(target_ckpt).exists():
            target_ckpt = str(Path(target_ckpt).parent / "last.pt")

    # Load the exact config used during training so policy architecture matches.
    stage_config = _load_config_for_checkpoint(pursuer_ckpt, config)
    ckpt_policy = _policy_config_from_checkpoint(pursuer_ckpt)
    if ckpt_policy:
        stage_config["policy"] = ckpt_policy
    env = AdversarialJSBSimEnv(stage_config)

    ppo_cfg = stage_config.get("ppo", {})
    policy_cfg = stage_config.get("policy", {})
    pursuer_agent = PPOAgent(16, 3, stage_config)
    pursuer_agent.load(pursuer_ckpt)
    pursuer_agent.network.eval()

    # Always instantiate a target agent. For stage1 the action is ignored by
    # the environment (maneuver-library target), but run_adversarial_episode
    # still needs a non-None object.
    target_cfg = dict(stage_config)
    target_cfg["ppo"] = stage_config.get("target_ppo", ppo_cfg)
    target_cfg["policy"] = stage_config.get("target_policy", policy_cfg)
    target_agent = AdversarialTargetAgent(config=target_cfg)
    if target_ckpt is not None and Path(target_ckpt).exists():
        target_agent.load(target_ckpt)
    target_agent.eval()

    per_scenario: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for sc_name in scenarios:
            builder = SCENARIO_BUILDERS.get(sc_name)
            if builder is None:
                continue
            scenario = builder()
            episodes: List[Dict[str, Any]] = []
            for seed in seeds:
                for ep in range(num_episodes // max(1, len(seeds))):
                    ep_seed = seed * 10000 + ep
                    # For stage1 the target is non-RL (maneuver library); env handles it.
                    result = run_adversarial_episode(
                        env, pursuer_agent, target_agent, scenario, ep_seed
                    )
                    result["seed"] = seed
                    result["episode"] = ep
                    result["scenario"] = sc_name
                    episodes.append(result)
            per_scenario[sc_name] = episodes
    finally:
        env.close()

    all_eps = [e for eps in per_scenario.values() for e in eps]
    captures = sum(1 for e in all_eps if e["captured"])
    return {
        "stage": stage,
        "pursuer_ckpt": pursuer_ckpt,
        "target_ckpt": target_ckpt,
        "num_episodes": len(all_eps),
        "capture_rate": captures / max(1, len(all_eps)),
        "mean_final_range_m": float(np.mean([e["final_range_m"] for e in all_eps])),
        "mean_min_range_m": float(np.mean([e["min_range_m"] for e in all_eps])),
        "mean_length": float(np.mean([e["length"] for e in all_eps])),
        "per_scenario": {
            sc: {
                "capture_rate": sum(1 for e in eps if e["captured"]) / max(1, len(eps)),
                "mean_final_range_m": float(np.mean([e["final_range_m"] for e in eps])),
            }
            for sc, eps in per_scenario.items()
        },
    }


def collect_trajectory(
    pursuer_ckpt: str,
    target_ckpt: str,
    config: Dict[str, Any],
    scenario_name: str = "tail_chase_2000m",
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect 3D positions for one adversarial episode."""
    from scripts.evaluate_adversarial import SCENARIO_BUILDERS

    stage_config = _load_config_for_checkpoint(pursuer_ckpt, config)
    ckpt_policy = _policy_config_from_checkpoint(pursuer_ckpt)
    if ckpt_policy:
        stage_config["policy"] = ckpt_policy
    env = AdversarialJSBSimEnv(stage_config)
    pursuer_agent = PPOAgent(16, 3, stage_config)
    pursuer_agent.load(pursuer_ckpt)
    pursuer_agent.network.eval()

    target_cfg = dict(stage_config)
    target_cfg["ppo"] = stage_config.get("target_ppo", stage_config.get("ppo", {}))
    target_cfg["policy"] = stage_config.get("target_policy", stage_config.get("policy", {}))
    target_agent = AdversarialTargetAgent(config=target_cfg)
    target_agent.load(target_ckpt)
    target_agent.eval()

    scenario = SCENARIO_BUILDERS[scenario_name]()
    p_obs, t_obs = env.reset(scenario=scenario, seed=seed)

    p_pos, t_pos = [], []
    try:
        for _ in range(env.max_steps):
            p_action = pursuer_agent.get_deterministic_action(p_obs["observation_vector"])
            t_action = target_agent.get_deterministic_action(t_obs["observation_vector"])
            p_obs, t_obs, _, _, terminated, truncated, info = env.step(p_action, t_action)

            p_state = info.get("pursuer_state", {})
            t_state = info.get("target_state", {})
            p_pos.append(p_state.get("position_m", p_state.get("position_neu", np.zeros(3))))
            t_pos.append(t_state.get("position_m", t_state.get("position_neu", np.zeros(3))))

            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "pursuer": np.asarray(p_pos),
        "target": np.asarray(t_pos),
    }


def setup_plotting_style() -> bool:
    """Set matplotlib / seaborn style consistent with paper materials."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        sns.set_context("paper")
        try:
            plt.rcParams["font.family"] = "Times New Roman"
        except Exception:
            pass
        plt.rcParams["figure.dpi"] = 300
        return True
    except ImportError:
        return False


def plot_curriculum_progress(
    stage_dirs: Dict[str, str],
    output_dir: str,
) -> None:
    """Plot success/capture rate vs training steps across stages."""
    if not setup_plotting_style():
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"stage1": "#2196F3", "stage2": "#FF9800", "stage3": "#4CAF50"}

    for stage, stage_dir in stage_dirs.items():
        steps, rates = load_stage_eval_curve(stage_dir, stage)
        if steps:
            ax.plot(steps, rates, label=stage.upper(), color=colors.get(stage), linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success / Capture Rate")
    ax.set_title("Curriculum-Adversarial Training Progress")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curriculum_progress.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "curriculum_progress.pdf"))
    plt.close()


def plot_adversarial_before_after(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Bar chart: capture rate before (stage1) and after (stage3) adversarial training."""
    if not setup_plotting_style():
        return

    import matplotlib.pyplot as plt

    labels = [r["stage"] for r in results]
    rates = [r["capture_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, rates, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax.set_ylabel("Capture Rate")
    ax.set_title("Adversarial Curriculum: Stage-wise Capture Rate")
    ax.set_ylim(0, 1.1)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_capture_rates.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "stage_capture_rates.pdf"))
    plt.close()


def plot_trajectory_3d(
    trajectory: Dict[str, np.ndarray],
    output_dir: str,
) -> None:
    """Plot 3D trajectory of pursuer and target."""
    if not setup_plotting_style():
        return

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    p = trajectory["pursuer"]
    t = trajectory["target"]
    if len(p) == 0 or len(t) == 0:
        plt.close()
        return

    ax.plot(p[:, 0], p[:, 1], p[:, 2], label="Pursuer", color="#2196F3", linewidth=1.5)
    ax.plot(t[:, 0], t[:, 1], t[:, 2], label="Target", color="#F44336", linewidth=1.5)
    ax.scatter(p[0, 0], p[0, 1], p[0, 2], color="#2196F3", marker="o", s=50)
    ax.scatter(t[0, 0], t[0, 1], t[0, 2], color="#F44336", marker="o", s=50)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Typical Adversarial Episode Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_3d.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "trajectory_3d.pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate curriculum + adversarial pipeline."
    )
    parser.add_argument(
        "--manifest", type=str, default="outputs/adversarial_curriculum/manifest.json",
        help="Path to train_curriculum_adversarial.py manifest.",
    )
    parser.add_argument(
        "--config", type=str, default="config/adversarial/train_pursuer_v2.yaml",
        help="Base adversarial config for evaluation.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/adversarial_curriculum",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=30,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+", default=[
            "tail_chase_2000m", "head_on_4000m", "crossing_90deg_2000m",
        ],
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    config = load_experiment_config(args.config)

    output_dir = args.output_dir
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Evaluate each stage
    stage_results: List[Dict[str, Any]] = []
    for stage in ["stage1", "stage2", "stage3"]:
        if stage not in manifest.get("stages", {}):
            continue
        result = evaluate_stage(
            stage, manifest, config, args.scenarios, args.num_episodes, args.seeds
        )
        stage_results.append(result)
        print(
            f"[{stage}] capture_rate={result['capture_rate']:.1%} "
            f"mean_final_range={result['mean_final_range_m']:.0f}m"
        )

    # Save JSON
    eval_results = {
        "stage_results": stage_results,
        "metadata": {
            "manifest": args.manifest,
            "config": args.config,
            "num_episodes": args.num_episodes,
            "seeds": args.seeds,
            "scenarios": args.scenarios,
        },
    }
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    # Figures
    stage_dirs = {stage: str(Path(manifest["stages"][stage]["pursuer_ckpt"]).parents[2])
                  for stage in manifest.get("stages", {})}
    plot_curriculum_progress(stage_dirs, fig_dir)
    plot_adversarial_before_after(stage_results, fig_dir)

    # 3D trajectory from the last available stage
    last_stage = stage_results[-1]
    if last_stage.get("target_ckpt"):
        traj = collect_trajectory(
            last_stage["pursuer_ckpt"], last_stage["target_ckpt"], config
        )
        plot_trajectory_3d(traj, fig_dir)

    print(f"\nResults saved to {output_dir}/eval_results.json")
    print(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
