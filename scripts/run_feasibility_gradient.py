#!/usr/bin/env python3
"""Feasibility-gradient validation across a controlled difficulty sweep.

Trains a separate policy for each scenario in a 5-step difficulty ladder
(from an easy speed-advantage geometry down to the standard `disadvantage`
geometry), then evaluates every checkpoint on all gradient scenarios.  The
result is a success-rate curve that shows at which difficulty level isolated
solo training fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


SCENARIOS = [
    "disadvantage_easy_speed_advantage",
    "disadvantage_bridge_1",
    "disadvantage_bridge_2",
    "disadvantage_bridge_3",
    "disadvantage",
]

SCENARIO_LABELS = [
    "easy (λ=1.56)",
    "bridge_1 (λ=1.37)",
    "bridge_2 (λ=1.20)",
    "bridge_3 (λ=1.05)",
    "standard (λ=0.91)",
]

# Use the 60k curriculum config as base so the guidance protections
# (altitude hold + roll-angle limit), position-advantage reward, and
# world-frame VPP settings are active.
BASE_CONFIG = Path("config/experiment/disadvantage_curriculum.yaml")
FEASIBILITY_CONFIG = Path("config/experiment/feasibility_gradient_scenarios.yaml")
OUTPUT_DIR = Path("outputs/feasibility_gradient")
TIMESTEPS = 10_000
EVAL_EPISODES = 20
EVAL_SEEDS = [0, 1, 2]


def build_single_scenario_config(scenario_name: str) -> Path:
    """Write a temp config that trains on exactly one scenario."""
    cfg_dir = OUTPUT_DIR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"train_{scenario_name}.yaml"

    base_abs = (Path.cwd() / BASE_CONFIG).as_posix()
    bridge_abs = (Path.cwd() / FEASIBILITY_CONFIG).as_posix()

    cfg = f"""includes:
  - {base_abs}
  - {bridge_abs}

curriculum:
  stage_gate_sr: 0.50
  stages:
    - progress_end: 1.0
      scenario_names:
        - {scenario_name}
"""
    cfg_path.write_text(cfg, encoding="utf-8")
    return cfg_path


def train_scenario(scenario_name: str) -> Path:
    """Run a short solo training run and return the checkpoint directory."""
    cfg_path = build_single_scenario_config(scenario_name)
    run_dir = OUTPUT_DIR / f"train_{scenario_name}"

    cmd = [
        sys.executable,
        "scripts/train_curriculum_ppo.py",
        "--config",
        str(cfg_path),
        "--output-dir",
        str(run_dir),
        "--backend",
        "jsbsim",
        "--device",
        "cuda",
        "--total-timesteps",
        str(TIMESTEPS),
        "--eval-scenario",
        scenario_name,
    ]
    print(f"\n[train] {scenario_name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # The best checkpoint selected by evaluation success rate.
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    if not best_ckpt.exists():
        # Fallback to the final checkpoint if no best.pt was written.
        best_ckpt = run_dir / "checkpoints" / "final.pt"
    return best_ckpt


def load_experiment_config(config_path: Path) -> dict:
    base_config = load_yaml_config(str(config_path))
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, base_config)


def summarize(records: list) -> dict:
    n = len(records)
    successes = sum(1 for r in records if r.get("is_success"))
    crashes = sum(1 for r in records if r.get("is_crash"))
    oobs = sum(1 for r in records if r.get("is_out_of_bounds"))
    timeouts = sum(1 for r in records if r.get("reason") == "timeout")
    returns = [r.get("return", np.nan) for r in records]
    min_ranges = [r.get("min_range_m", np.nan) for r in records]
    final_ranges = [r.get("final_range_m", np.nan) for r in records]
    return {
        "n": n,
        "success_rate": successes / n if n else 0.0,
        "crash_rate": crashes / n if n else 0.0,
        "out_of_bounds_rate": oobs / n if n else 0.0,
        "timeout_rate": timeouts / n if n else 0.0,
        "mean_return": float(np.nanmean(returns)) if returns else 0.0,
        "std_return": float(np.nanstd(returns)) if returns else 0.0,
        "mean_min_range_m": float(np.nanmean(min_ranges)) if min_ranges else 0.0,
        "mean_final_range_m": float(np.nanmean(final_ranges)) if final_ranges else 0.0,
    }


def evaluate_checkpoint(checkpoint: Path, scenario_name: str, eval_scenarios: dict):
    """Evaluate a trained policy on every scenario in eval_scenarios."""
    cfg_path = build_single_scenario_config(scenario_name)
    config = load_experiment_config(cfg_path)

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cuda")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(checkpoint)

    per_scenario = {}
    all_records = []
    for scen_name, scenario in eval_scenarios.items():
        records = []
        for seed in EVAL_SEEDS:
            for ep in range(EVAL_EPISODES):
                result, _ = evaluate_single_episode(
                    env=env,
                    agent=agent,
                    config=config,
                    scenario=scenario,
                    seed=seed * 10000 + ep,
                    save_trajectory=False,
                    method_name=f"train_{scenario_name}",
                )
                records.append(result)
        per_scenario[scen_name] = summarize(records)
        all_records.extend(records)

    env.close()
    overall = summarize(all_records)
    return per_scenario, overall


def plot_results(diagonal_sr: list, matrix: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SCENARIOS))

    # Diagonal: trained on scenario i, evaluated on scenario i.
    ax.plot(x, [sr * 100 for sr in diagonal_sr], marker="o", linewidth=2, label="in-distribution (trained & eval same scenario)")

    # Off-diagonal curves for the two easiest and two hardest reference scenarios.
    for ref in [SCENARIOS[0], SCENARIOS[-1]]:
        values = [matrix[train][ref]["success_rate"] * 100 for train in SCENARIOS]
        ax.plot(x, values, marker="s", linestyle="--", alpha=0.7, label=f"eval on {ref}")

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Feasibility gradient: isolated 10k-step training per scenario")
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(results: dict, output_path: Path):
    lines = [
        "# Feasibility Gradient Validation Report",
        "",
        "This report validates whether short (10k-step) isolated PPO training can",
        "solve a controlled ladder of lead-turn difficulty.  The ladder ranges from",
        "a very easy speed-advantage geometry down to the standard `disadvantage`",
        "geometry.",
        "",
        "## Setup",
        "",
        f"- Training timesteps per scenario: **{TIMESTEPS:,}**",
        f"- Evaluation: **{EVAL_EPISODES}** episodes × **{len(EVAL_SEEDS)}** seeds per scenario",
        f"- Backend: JSBSim F-16",
        f"- Device: NVIDIA 4090 (CUDA)",
        "",
        "## Scenarios",
        "",
    ]
    for label, name in zip(SCENARIO_LABELS, SCENARIOS):
        lines.append(f"- `{name}`: {label}")
    lines.extend(["", "## Results", ""])

    lines.append("| Trained on \\ Eval on | " + " | ".join(SCENARIO_LABELS) + " |")
    lines.append("|" + "---|" * (len(SCENARIOS) + 1))
    for train_name, label in zip(SCENARIOS, SCENARIO_LABELS):
        row = [label]
        for eval_name in SCENARIOS:
            sr = results["matrix"][train_name][eval_name]["success_rate"] * 100
            row.append(f"{sr:.0f}%")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "### Diagonal (train = eval) success rates", ""])
    for label, sr in zip(SCENARIO_LABELS, results["diagonal_success_rates"]):
        lines.append(f"- {label}: {sr*100:.0f}%")

    lines.extend(["", "## Interpretation", ""])
    diagonal = results["diagonal_success_rates"]
    if diagonal[-1] >= 0.5:
        lines.append("The standard `disadvantage` scenario is solvable with 10k isolated steps.")
    elif diagonal[-2] >= 0.5:
        lines.append(
            "Isolated training succeeds up to `disadvantage_bridge_3` but fails on "
            "the standard `disadvantage` geometry; a mixed curriculum is required for the hardest case."
        )
    else:
        lines.append(
            "Isolated training is not sufficient even for moderate bridge scenarios; "
            "a mixed curriculum appears necessary to bootstrap lead-turn behavior."
        )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "scenarios": SCENARIOS,
        "scenario_labels": SCENARIO_LABELS,
        "timesteps": TIMESTEPS,
        "eval_episodes": EVAL_EPISODES,
        "eval_seeds": EVAL_SEEDS,
        "checkpoints": {},
        "matrix": {},
        "diagonal_success_rates": [],
    }

    # Train one policy per scenario.
    for scenario_name in SCENARIOS:
        checkpoint = train_scenario(scenario_name)
        results["checkpoints"][scenario_name] = str(checkpoint)
        print(f"[checkpoint] {scenario_name} -> {checkpoint}")

    # Load the full scenario dictionary once for evaluation.
    full_cfg_path = build_single_scenario_config(SCENARIOS[0])
    full_config = load_experiment_config(full_cfg_path)
    eval_scenarios = full_config.get("scenarios", {})

    # Evaluate every trained policy on every scenario.
    diagonal = []
    for scenario_name in SCENARIOS:
        checkpoint = Path(results["checkpoints"][scenario_name])
        per_scenario, overall = evaluate_checkpoint(
            checkpoint=checkpoint,
            scenario_name=scenario_name,
            eval_scenarios=eval_scenarios,
        )
        results["matrix"][scenario_name] = per_scenario
        diagonal.append(per_scenario[scenario_name]["success_rate"])

        print(f"\n[eval] trained on {scenario_name}: overall SR={overall['success_rate']*100:.1f}%")
        for ev_name, ev_res in per_scenario.items():
            print(
                f"  {ev_name:<35} SR={ev_res['success_rate']*100:>5.1f}% "
                f"crash={ev_res['crash_rate']*100:>5.1f}% timeout={ev_res['timeout_rate']*100:>5.1f}%"
            )

    results["diagonal_success_rates"] = diagonal

    # Save JSON results.
    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved results: {results_path}")

    # Plot.
    plot_path = OUTPUT_DIR / "success_rate_curve.png"
    plot_results(diagonal, results["matrix"], plot_path)
    print(f"Saved plot: {plot_path}")

    # Report.
    report_path = Path("docs/feasibility_gradient_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, report_path)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
