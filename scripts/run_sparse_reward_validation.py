"""
Sparse reward + trajectory-level relabelling validation experiment.

Runs a head-to-head comparison of:
  - Dense baseline
  - Sparse without relabelling
  - Sparse with linear terminal relabelling
  - Sparse with Gaussian event relabelling

Outputs are written to outputs/sparse_reward_comparison/.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_vpp_guidance.training.train_no_prediction_vpp_ppo import (
    load_experiment_config,
    train_ppo,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import merge_config


def make_base_config():
    """Load the current best dense config and force CPU training."""
    cfg = load_experiment_config("config/experiment/maneuver_target_vpp_pilot.yaml")
    cfg.setdefault("ppo", {})["device"] = "cpu"
    return cfg


def build_variant_config(base_config: dict, variant: str) -> dict:
    """Create a config dict for one experimental condition."""
    cfg = merge_config({}, base_config)

    if variant == "dense":
        return cfg

    # Sparse variants share the same event / outcome magnitudes.
    # Lock reward is intentionally set to 0.0 so that the conditions are
    # genuinely sparse and the ablation isolates the relabelling effect.
    cfg.setdefault("reward", {})["use_sparse"] = True
    cfg["sparse_reward"] = {
        "enabled": True,
        "terminal_success": 10.0,
        "terminal_crash": -10.0,
        "terminal_failure": -5.0,
        "event_range_m": 1200.0,
        "event_ata_deg": 45.0,
        "event_reward": 1.0,
        "lock_reward": 0.0,
        "gaussian_window": 50,
        "gaussian_sigma_ratio": 0.5,
        "relabelling": {"enabled": True, "terminal_kernel": "linear", "event_kernel": "gaussian"},
    }

    if variant == "sparse_no_relabel":
        cfg["sparse_reward"]["relabelling"]["enabled"] = False
        # In the no-relabel condition the event reward is only useful when
        # relabelled; without relabelling the agent receives terminal rewards only.
        cfg["sparse_reward"]["event_reward"] = 0.0
    elif variant == "sparse_linear":
        cfg["sparse_reward"]["relabelling"]["terminal_kernel"] = "linear"
        cfg["sparse_reward"]["relabelling"]["event_kernel"] = "none"
    elif variant == "sparse_gaussian":
        cfg["sparse_reward"]["relabelling"]["terminal_kernel"] = "none"
        cfg["sparse_reward"]["relabelling"]["event_kernel"] = "gaussian"
    elif variant == "sparse_full":
        pass  # default: linear terminal + gaussian events
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return cfg


def apply_hard_overrides(cfg: dict) -> dict:
    """Make the task harder: longer episodes and a more aggressive bandit."""
    cfg.setdefault("env", {})["max_high_level_steps"] = 400
    bandit = cfg.setdefault("env", {}).setdefault("bandit", {})
    bandit["difficulty"] = "hard"
    selector = bandit.setdefault("selector", {})
    selector["reaction_time_s"] = 1.5
    selector["selector_noise"] = 0.25
    selector["param_perturb_scale"] = 0.25
    # Ensure all aggressive maneuvers are available.
    selector["allowed_maneuvers"] = [
        "straight_level", "coordinated_turn", "barrel_roll", "dive",
        "high_yoyo", "low_yoyo", "scissors", "split_s", "immelmann",
        "break_turn", "defensive_spiral", "extension",
    ]
    return cfg


def run_training(variant: str, cfg: dict, output_dir: str, smoke: bool = False,
                 total_timesteps: int = 10000) -> str:
    """Train a single condition and return the output directory."""
    cfg["experiment"]["name"] = variant
    cfg["experiment"]["seed"] = 0

    ppo_cfg = cfg.setdefault("ppo", {})
    if smoke:
        ppo_cfg["total_timesteps"] = 1024
    else:
        ppo_cfg["total_timesteps"] = int(total_timesteps)
    ppo_cfg["rollout_steps"] = 1024
    ppo_cfg["minibatch_size"] = 256
    ppo_cfg["update_epochs"] = 10

    eval_cfg = cfg.setdefault("evaluation", {})
    eval_cfg["eval_interval"] = 1000
    eval_cfg["eval_episodes"] = 10
    eval_cfg["seeds"] = [0, 1, 2]

    chk_cfg = cfg.setdefault("checkpoint", {})
    chk_cfg["save_interval"] = 10000
    chk_cfg["save_best"] = True
    chk_cfg["save_last"] = True

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training variant: {variant}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    train_ppo(cfg, output_dir, smoke=smoke)
    return output_dir


def evaluate_checkpoint(
    cfg: dict,
    checkpoint_path: str,
    scenarios: Dict[str, dict],
    episodes_per_scenario: int = 50,
) -> Dict:
    """Evaluate a checkpoint per scenario and return aggregate metrics."""
    env = CloseRangeTrackingEnv(cfg)
    env.set_domain_rand_scale(0.0)

    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(cfg.get("policy", {}).get("action_dim", 3))

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=cfg, device="cpu")
    agent.load(checkpoint_path)

    per_scenario = {}
    overall = defaultdict(list)

    for scenario_name, scenario in scenarios.items():
        metrics = {
            "returns": [],
            "sparse_returns": [],
            "lengths": [],
            "success": [],
            "crash": [],
            "timeout": [],
            "out_of_bounds": [],
            "final_range": [],
            "final_ata": [],
            "min_range": [],
        }

        for ep in range(episodes_per_scenario):
            seed = ep + 1000 * (list(scenarios.keys()).index(scenario_name) + 1)
            obs = env.reset(scenario=scenario, seed=seed)
            ep_return = 0.0
            ep_length = 0
            min_range = float("inf")
            final_range = 0.0
            final_ata = 0.0
            reason = "timeout"

            while True:
                obs_vec = obs["observation_vector"]
                action = agent.get_deterministic_action(obs_vec)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                ep_length += 1
                # Canonical sparse terminal return for fair cross-condition comparison.
                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    if reason == "success":
                        sparse_ep_return = 10.0
                    elif reason == "crash":
                        sparse_ep_return = -10.0
                    elif reason in ("timeout", "out_of_bounds"):
                        sparse_ep_return = -5.0
                    else:
                        sparse_ep_return = 0.0
                range_m = info.get("range_m", float("nan"))
                ata_deg = info.get("ata_deg", float("nan"))
                if np.isfinite(range_m):
                    min_range = min(min_range, float(range_m))
                    final_range = float(range_m)
                if np.isfinite(ata_deg):
                    final_ata = float(ata_deg)

                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break

            metrics["returns"].append(ep_return)
            metrics["sparse_returns"].append(sparse_ep_return)
            metrics["lengths"].append(ep_length)
            metrics["success"].append(1.0 if reason == "success" else 0.0)
            metrics["crash"].append(1.0 if reason == "crash" else 0.0)
            metrics["timeout"].append(1.0 if reason == "timeout" else 0.0)
            metrics["out_of_bounds"].append(1.0 if reason == "out_of_bounds" else 0.0)
            metrics["final_range"].append(final_range)
            metrics["final_ata"].append(final_ata)
            metrics["min_range"].append(min_range)

        per_scenario[scenario_name] = {
            "success_rate": float(np.mean(metrics["success"])),
            "crash_rate": float(np.mean(metrics["crash"])),
            "timeout_rate": float(np.mean(metrics["timeout"])),
            "out_of_bounds_rate": float(np.mean(metrics["out_of_bounds"])),
            "mean_return": float(np.mean(metrics["sparse_returns"])),
            "std_return": float(np.std(metrics["sparse_returns"])),
            "env_mean_return": float(np.mean(metrics["returns"])),
            "env_std_return": float(np.std(metrics["returns"])),
            "mean_length": float(np.mean(metrics["lengths"])),
            "mean_final_range_m": float(np.nanmean(metrics["final_range"])),
            "mean_final_ata_deg": float(np.nanmean(np.abs(metrics["final_ata"]))),
            "mean_min_range_m": float(np.nanmean(metrics["min_range"])),
        }

        for key in ["success", "crash", "timeout", "out_of_bounds", "sparse_returns"]:
            overall[key].extend(metrics[key])

    env.close()

    overall_metrics = {
        "success_rate": float(np.mean(overall["success"])),
        "crash_rate": float(np.mean(overall["crash"])),
        "timeout_rate": float(np.mean(overall["timeout"])),
        "out_of_bounds_rate": float(np.mean(overall["out_of_bounds"])),
        "mean_return": float(np.mean(overall["sparse_returns"])),
        "std_return": float(np.std(overall["sparse_returns"])),
    }

    return {"overall": overall_metrics, "per_scenario": per_scenario}


def parse_training_curves(output_dir: str) -> pd.DataFrame:
    """Read episode training log and return a smoothed dataframe."""
    path = os.path.join(output_dir, "logs", "episode_train_log.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["smoothed_return"] = df["episode_return"].rolling(window=20, min_periods=1).mean()
    return df


def compute_convergence_episode(df: pd.DataFrame, threshold: float = 0.5) -> int:
    """First episode where the 20-episode rolling success rate exceeds threshold."""
    if df.empty or "success" not in df.columns:
        return -1
    smoothed = df["success"].rolling(window=20, min_periods=1).mean()
    idx = np.where(smoothed >= threshold)[0]
    if len(idx) == 0:
        return -1
    return int(df.iloc[idx[0]]["episode"])


def plot_learning_curves(results: Dict[str, dict], output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "dense": "black",
        "sparse_no_relabel": "red",
        "sparse_linear": "blue",
        "sparse_gaussian": "green",
        "sparse_full": "orange",
    }

    for variant, data in results.items():
        df = data.get("train_df")
        if df is None or df.empty:
            continue
        label = variant.replace("_", " ").title()
        ax.plot(df["episode"], df["smoothed_return"], label=label, color=colors.get(variant, None), alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (20-ep smoothed)")
    ax.set_title("Learning Curves: Dense vs Sparse Reward Variants")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_scenario_breakdown(results: Dict[str, dict], output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(results.keys())
    scenarios = list(next(iter(results.values()))["eval"]["per_scenario"].keys())

    x = np.arange(len(scenarios))
    width = 0.8 / len(variants)
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, variant in enumerate(variants):
        rates = [results[variant]["eval"]["per_scenario"][s]["success_rate"] for s in scenarios]
        ax.bar(x + i * width, rates, width, label=variant.replace("_", " ").title(), color=colors[i])

    ax.set_ylabel("Success Rate")
    ax.set_title("Per-Scenario Success Rate")
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(scenarios)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_comparison_table(results: Dict[str, dict], output_path: str):
    lines = [
        "# Sparse Reward Validation: Comparison Table",
        "",
        "| Variant | OSR | Crash Rate | Timeout Rate | Mean Return | Convergence Episode |",
        "|---------|-----|------------|--------------|-------------|---------------------|",
    ]
    for variant, data in results.items():
        ovr = data["eval"]["overall"]
        conv = data.get("convergence_episode", -1)
        conv_str = str(conv) if conv > 0 else "N/A"
        lines.append(
            f"| {variant} | {ovr['success_rate']:.2%} | {ovr['crash_rate']:.2%} | "
            f"{ovr['timeout_rate']:.2%} | {ovr['mean_return']:.2f} ± {ovr['std_return']:.2f} | {conv_str} |"
        )

    lines.extend(["", "## Per-Scenario Success Rates", ""])
    scenarios = list(next(iter(results.values()))["eval"]["per_scenario"].keys())
    header = "| Variant | " + " | ".join(scenarios) + " |"
    lines.append(header)
    lines.append("|" + "-" * (len(header) - 2) + "|")
    for variant, data in results.items():
        rates = [f"{data['eval']['per_scenario'][s]['success_rate']:.2%}" for s in scenarios]
        lines.append(f"| {variant} | " + " | ".join(rates) + " |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_report(results: Dict[str, dict], output_path: str):
    if "dense" not in results:
        print("Skipping report: dense baseline not available.")
        return
    dense = results["dense"]["eval"]["overall"]
    best_sparse = (
        results.get("sparse_gaussian")
        or results.get("sparse_full")
        or results.get("sparse_linear")
        or results.get("sparse_no_relabel")
    )
    no_relabel = results.get("sparse_no_relabel")

    lines = [
        "# Dense vs Sparse Reward Validation Report",
        "",
        "## Executive Summary",
        "",
    ]

    if best_sparse is None:
        lines.append("- **No sparse condition available yet.** Only dense baseline has been evaluated.")
    elif best_sparse["eval"]["overall"]["success_rate"] >= 0.9 * dense["success_rate"]:
        lines.append(
            "- **Sparse + trajectory relabelling reaches the dense baseline.** "
            f"Best sparse OSR = {best_sparse['eval']['overall']['success_rate']:.2%} "
            f"vs Dense OSR = {dense['success_rate']:.2%}."
        )
    else:
        lines.append(
            "- **Sparse reward does not yet match dense performance.** "
            f"Best sparse OSR = {best_sparse['eval']['overall']['success_rate']:.2%} "
            f"vs Dense OSR = {dense['success_rate']:.2%}."
        )

    if no_relabel and no_relabel["eval"]["overall"]["success_rate"] < 0.2:
        lines.append(
            "- **Relabelling is necessary:** Sparse-NoRelabel essentially fails "
            f"(OSR = {no_relabel['eval']['overall']['success_rate']:.2%}), "
            "confirming the R2SP hypothesis."
        )
    elif no_relabel:
        lines.append(
            "- **Unexpectedly, sparse without relabelling shows non-trivial learning.** "
            "Further investigation is needed."
        )

    lines.extend(["", "## Detailed Findings", ""])

    # Disadvantage analysis
    scenarios = list(next(iter(results.values()))["eval"]["per_scenario"].keys())
    if "disadvantage" in scenarios:
        dense_dis = results["dense"]["eval"]["per_scenario"]["disadvantage"]["success_rate"]
        if best_sparse is not None:
            # Use the same sparse condition identified above for consistency.
            sparse_dis = best_sparse["eval"]["per_scenario"]["disadvantage"]["success_rate"]
            lines.append(
                f"- **Disadvantage scenario:** Dense = {dense_dis:.2%}, "
                f"Best sparse = {sparse_dis:.2%}. "
                + (
                    "Sparse performs comparably or better, suggesting reduced geometric bias."
                    if sparse_dis >= dense_dis * 0.9 else
                    "Sparse lags here; geometric shaping may still provide useful guidance for this hard scenario."
                )
            )
        else:
            lines.append(f"- **Disadvantage scenario (dense only):** Dense = {dense_dis:.2%}.")

    lines.extend(["", "## Next Steps", "", "1. Run multi-seed stability experiments."])
    if best_sparse is None or best_sparse["eval"]["overall"]["success_rate"] < dense["success_rate"]:
        lines.append("2. Tune sparse-reward hyperparameters (event magnitude, Gaussian window/bandwidth).")
    lines.append("3. Evaluate CBF compatibility under sparse rewards if not already tested.")
    lines.append("4. Analyze trajectories to check for human-preferred geometry bias.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (1K steps).")
    parser.add_argument("--variants", "--variant", nargs="+", dest="variants",
                        default=["dense", "sparse_no_relabel", "sparse_linear", "sparse_gaussian"],
                        help="Which conditions to run.")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes per scenario.")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and re-evaluate existing checkpoints.")
    parser.add_argument("--output-dir", default="outputs/sparse_reward_comparison_jsbsim",
                        help="Root directory for this experiment.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training.")
    parser.add_argument("--cbf", action="store_true", help="Enable CBF safety filter.")
    parser.add_argument("--hard-mode", action="store_true",
                        help="Use harder bandit and longer episodes to stress-test relabelling.")
    parser.add_argument("--total-timesteps", type=int, default=10000,
                        help="Training steps per variant.")
    args = parser.parse_args()

    root_dir = args.output_dir
    os.makedirs(root_dir, exist_ok=True)

    base_config = make_base_config()
    scenarios = base_config.get("scenarios", {})

    results_path = os.path.join(root_dir, "results.json")
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            # Remove non-serializable train_df if present from a previous partial run.
            for variant in results:
                results[variant].pop("train_df", None)
        except Exception as exc:
            print(f"Warning: could not load existing results: {exc}")
            results = {}

    def configure_variant_cfg(variant):
        cfg = build_variant_config(base_config, variant)
        cfg.setdefault("experiment", {})["seed"] = args.seed
        if args.cbf:
            cfg.setdefault("cbf", {})["enabled"] = True
        if args.hard_mode:
            cfg = apply_hard_overrides(cfg)
        return cfg

    # Phase 1: training
    if not args.skip_training:
        for variant in args.variants:
            cfg = configure_variant_cfg(variant)
            output_dir = os.path.join(root_dir, variant)
            t0 = time.time()
            run_training(variant, cfg, output_dir, smoke=args.smoke,
                        total_timesteps=args.total_timesteps)
            print(f"Training {variant} took {time.time() - t0:.1f}s")

    # Phase 2: evaluation
    for variant in args.variants:
        cfg = configure_variant_cfg(variant)
        output_dir = os.path.join(root_dir, variant)
        checkpoint = os.path.join(output_dir, "checkpoints", "best.pt")
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join(output_dir, "checkpoints", "last.pt")
        if not os.path.exists(checkpoint):
            print(f"Warning: no checkpoint found for {variant}")
            continue

        print(f"\nEvaluating {variant} ...")
        eval_metrics = evaluate_checkpoint(cfg, checkpoint, scenarios, episodes_per_scenario=args.episodes)
        df = parse_training_curves(output_dir)
        convergence = compute_convergence_episode(df)

        results[variant] = {
            "eval": eval_metrics,
            "train_df": df,
            "convergence_episode": convergence,
            "output_dir": output_dir,
        }

        # Incrementally persist so partial runs are not lost.
        serializable = {}
        for v, data in results.items():
            serializable[v] = {
                "eval": data["eval"],
                "convergence_episode": data["convergence_episode"],
                "output_dir": data["output_dir"],
            }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"Saved intermediate results to {results_path}")

    # Phase 3: artifacts (only generate when at least one variant is available)
    if results:
        plot_learning_curves(results, os.path.join(root_dir, "learning_curves.png"))
        plot_per_scenario_breakdown(results, os.path.join(root_dir, "per_scenario_breakdown.png"))
        write_comparison_table(results, os.path.join(root_dir, "comparison_table.md"))
        if "dense" in results:
            write_report(results, os.path.join(root_dir, "dense_vs_sparse_report.md"))
        else:
            print("Skipping report: dense baseline not present in results yet.")

    print(f"\nValidation complete. Results written to {root_dir}/")


if __name__ == "__main__":
    main()
