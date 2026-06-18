#!/usr/bin/env python3
"""
Generate CBF safety analysis figures.

Reads evaluation JSONs produced by ``evaluate_cbf_safety.py`` and optionally
re-runs a single episode to obtain per-step trajectories for time-series and
safe-set plots.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "paper_materials" / "scripts"))

from interpretability_common import build_env_and_agent, save_figure, setup_paper_style


def load_summary(results_dir: str) -> Dict[str, Any]:
    path = Path(results_dir) / "eval_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rollout_episode(env, agent, scenario: Dict[str, Any], seed: int, use_cbf: bool):
    """Run one episode and record per-step telemetry."""
    if use_cbf:
        env._use_cbf = True
        if env._cbf_filter is not None:
            env._cbf_filter.reset()
    else:
        env._use_cbf = False

    obs = env.reset(seed=seed, scenario=scenario)
    done = False

    traj = {
        "t": [],
        "range_m": [],
        "h": [],
        "h_dot": [],
        "action_norm": [],
        "cbf_active": [],
        "cbf_solve_time_ms": [],
        "cbf_filtered": [],
        "own_x": [],
        "own_y": [],
        "target_x": [],
        "target_y": [],
    }

    d_min = 500.0
    if use_cbf and env._cbf_filter is not None:
        d_min = env._cbf_filter.d_min

    dt = env.env_config.get("high_level_dt", 0.2)
    step = 0
    while not done:
        if agent is None:
            action = np.zeros(3, dtype=np.float64)
        else:
            action = agent.select_action(
                obs["observation_vector"], deterministic=True, store=False
            )[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rel = info.get("relative_state") or obs.get("relative_state", {})
        range_m = float(rel.get("range_m", np.nan))
        own_pos = info.get("own_state", {}).get("position_m", [np.nan, np.nan, np.nan])
        target_pos = info.get("target_state", {}).get("position_m", [np.nan, np.nan, np.nan])

        traj["t"].append(step * dt)
        traj["range_m"].append(range_m)
        traj["h"].append(range_m - d_min)
        traj["h_dot"].append(float(rel.get("range_rate_mps", np.nan)))
        traj["action_norm"].append(float(np.linalg.norm(action)))
        cbf_info = info.get("cbf")
        traj["cbf_active"].append(bool(cbf_info.get("active", False)) if cbf_info else False)
        traj["cbf_filtered"].append(bool(info.get("cbf_filtered", False)))
        traj["cbf_solve_time_ms"].append(
            float(cbf_info.get("solve_time_ms", 0.0)) if cbf_info else 0.0
        )
        traj["own_x"].append(float(own_pos[0]))
        traj["own_y"].append(float(own_pos[1]))
        traj["target_x"].append(float(target_pos[0]))
        traj["target_y"].append(float(target_pos[1]))
        step += 1
        if step > 512:
            break

    return {k: np.asarray(v) for k, v in traj.items()}


def plot_statistical_comparison(
    baseline_dir: str,
    cbf_dir: str,
    output_dir: str,
):
    """Bar charts and box plots comparing baseline vs CBF."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    setup_paper_style()

    base = load_summary(baseline_dir)
    cbf = load_summary(cbf_dir)
    scenarios = base.get("scenarios", ["favorable", "neutral", "challenging", "disadvantage"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Success / crash / oob / timeout rates.
    ax = axes[0, 0]
    metrics = ["success_rate", "crash_rate", "oob_rate", "timeout_rate"]
    x = np.arange(len(scenarios))
    width = 0.35
    for j, (label, data) in enumerate([("Baseline", base), ("CBF", cbf)]):
        vals = np.array([data["per_scenario"][s][m] * 100 for m in metrics for s in scenarios]).reshape(
            len(metrics), len(scenarios)
        )
        # Stack bars per scenario: show success and crash side-by-side?
        # Simpler: grouped bar for success rate only.
        pass
    # Actually, let's plot success/crash grouped.
    base_success = [base["per_scenario"][s]["success_rate"] * 100 for s in scenarios]
    cbf_success = [cbf["per_scenario"][s]["success_rate"] * 100 for s in scenarios]
    base_crash = [base["per_scenario"][s]["crash_rate"] * 100 for s in scenarios]
    cbf_crash = [cbf["per_scenario"][s]["crash_rate"] * 100 for s in scenarios]

    ax.bar(x - width / 2, base_success, width, label="Baseline success", color="C0")
    ax.bar(x + width / 2, cbf_success, width, label="CBF success", color="C1")
    ax.bar(x - width / 2, base_crash, width, bottom=base_success, label="Baseline crash", color="C2")
    ax.bar(x + width / 2, cbf_crash, width, bottom=cbf_success, label="CBF crash", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Termination rates per scenario")
    ax.legend(loc="best", fontsize=7)
    ax.set_ylim(0, 110)

    # (b) Minimum separation distance distribution.
    ax = axes[0, 1]
    base_mins = [ep["min_range_m"] for ep in base["raw_episodes"]]
    cbf_mins = [ep["min_range_m"] for ep in cbf["raw_episodes"]]
    bp = ax.boxplot(
        [base_mins, cbf_mins],
        labels=["Baseline", "CBF"],
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], ["C0", "C1"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    d_min = cbf.get("cbf_config", {}).get("params", {}).get("d_min", 500.0)
    ax.axhline(d_min, color="red", linestyle="--", label=f"$d_{{min}}={d_min}$m")
    ax.set_ylabel("Minimum range (m)")
    ax.set_title("Minimum separation distance")
    ax.legend(loc="best")

    # (c) CBF activation rate.
    ax = axes[1, 0]
    if cbf.get("cbf_enabled"):
        rates = [
            cbf["per_scenario"][s].get("cbf_intervention_rate", 0.0) * 100
            for s in scenarios
        ]
        sns.barplot(x=scenarios, y=rates, ax=ax, palette="muted")
        ax.set_ylabel("Intervention rate (%)")
        ax.set_title("CBF QP activation rate")
        ax.set_ylim(0, max(rates + [10]) * 1.1)
    else:
        ax.text(0.5, 0.5, "CBF disabled", ha="center", va="center", transform=ax.transAxes)

    # (d) CBF solve time distribution.
    ax = axes[1, 1]
    solve_times = [
        t
        for ep in cbf["raw_episodes"]
        for t in ep.get("cbf_solve_times_ms", [])
    ]
    if solve_times:
        ax.hist(solve_times, bins=30, color="C1", edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(solve_times), color="red", linestyle="--", label="Mean")
        ax.set_xlabel("Solve time (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"CBF solve time (mean={np.mean(solve_times):.3f} ms)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No QP activations", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    save_figure(fig, Path(output_dir), "statistical_comparison")
    plt.close(fig)


def plot_time_series(
    checkpoint_path: str,
    scenario: Dict[str, Any],
    scenario_name: str,
    output_dir: str,
):
    """Plot distance, h, action norm, and CBF activation for one episode."""
    import matplotlib.pyplot as plt

    setup_paper_style()

    env_base, agent, _ = build_env_and_agent(checkpoint_path)
    env_cbf, _, _ = build_env_and_agent(checkpoint_path)
    env_cbf.config["cbf"] = {"enabled": True, "params": {"solver": "scipy"}}
    # The env was already built without CBF; easiest is to instantiate a new one.
    env_cbf.close()
    cfg = env_base.config.copy()
    cfg["cbf"] = {"enabled": True, "params": {"solver": "scipy"}}
    env_cbf = type(env_base)(cfg)

    try:
        traj_base = rollout_episode(env_base, agent, scenario, seed=42, use_cbf=False)
        traj_cbf = rollout_episode(env_cbf, agent, scenario, seed=42, use_cbf=True)
    finally:
        env_base.close()
        env_cbf.close()

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    ax = axes[0]
    ax.plot(traj_base["t"], traj_base["range_m"], label="No CBF", color="C0")
    ax.plot(traj_cbf["t"], traj_cbf["range_m"], label="CBF", color="C1")
    d_min = env_cbf._cbf_filter.d_min if env_cbf._cbf_filter is not None else 500.0
    ax.axhline(d_min, color="red", linestyle="--", label=f"$d_{{min}}={d_min}$m")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"Scenario: {scenario_name}")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(traj_base["t"], traj_base["h"], label="No CBF", color="C0")
    ax.plot(traj_cbf["t"], traj_cbf["h"], label="CBF", color="C1")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_ylabel("$h(x)$ (m)")
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(traj_base["t"], traj_base["action_norm"], label="No CBF", color="C0")
    ax.plot(traj_cbf["t"], traj_cbf["action_norm"], label="CBF", color="C1")
    ax.set_ylabel("$\\|a\\|$")
    ax.legend(loc="best")

    ax = axes[3]
    active_steps = traj_cbf["t"][traj_cbf["cbf_active"]]
    ax.scatter(active_steps, np.ones_like(active_steps), marker="|", s=200, color="red", label="CBF active")
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CBF activation")
    ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, Path(output_dir), "time_series_comparison")
    plt.close(fig)


def plot_safe_set(
    checkpoint_path: str,
    scenario: Dict[str, Any],
    scenario_name: str,
    output_dir: str,
):
    """2D contour of h(x) in the horizontal plane with overlaid trajectory."""
    import matplotlib.pyplot as plt

    setup_paper_style()

    env_cbf, agent, _ = build_env_and_agent(checkpoint_path)
    env_cbf.config["cbf"] = {"enabled": True, "params": {"solver": "scipy"}}
    env_cbf.close()
    cfg = env_cbf.config.copy()
    cfg["cbf"] = {"enabled": True, "params": {"solver": "scipy"}}
    env_cbf = type(env_cbf)(cfg)

    try:
        traj = rollout_episode(env_cbf, agent, scenario, seed=42, use_cbf=True)
    finally:
        env_cbf.close()

    d_min = env_cbf._cbf_filter.d_min if env_cbf._cbf_filter is not None else 500.0
    target_pos = np.array([traj["target_x"][0], traj["target_y"][0]])

    # Grid around target.
    margin = 2500.0
    x = np.linspace(target_pos[0] - margin, target_pos[0] + margin, 200)
    y = np.linspace(target_pos[1] - margin, target_pos[1] + margin, 200)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - target_pos[0]) ** 2 + (Y - target_pos[1]) ** 2)
    H = R - d_min

    fig, ax = plt.subplots(figsize=(7, 6))
    levels = [-500, -250, 0, 250, 500, 1000, 1500, 2000]
    cs = ax.contourf(X, Y, H, levels=levels, cmap="RdYlGn", alpha=0.6)
    ax.contour(X, Y, H, levels=[0], colors="red", linewidths=2, linestyles="--")
    ax.plot(traj["own_x"], traj["own_y"], color="black", linewidth=1.5, label="CBF trajectory")
    ax.plot(traj["target_x"][0], traj["target_y"][0], "b*", markersize=12, label="Target start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_title(f"Safe set and trajectory ({scenario_name}), $d_{{min}}={d_min}$m")
    ax.legend(loc="best")
    fig.colorbar(cs, ax=ax, label="$h(x)$ (m)")
    fig.tight_layout()
    save_figure(fig, Path(output_dir), "safe_set")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CBF safety analysis figures")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--cbf-dir", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="disadvantage")
    parser.add_argument("--output-dir", type=str, default="outputs/cbf_safety/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = args.scenario
    from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

    cfg_path = PROJECT_ROOT / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml"
    base = load_yaml_config(cfg_path)
    includes = base.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = cfg_path.parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(inc_full))
    scenario = merge_config(merged, base).get("scenarios", {}).get(scenario_name, {})

    print("Generating statistical comparison...")
    plot_statistical_comparison(args.baseline_dir, args.cbf_dir, output_dir)

    print("Generating time-series comparison...")
    plot_time_series(args.checkpoint, scenario, scenario_name, output_dir)

    print("Generating safe-set plot...")
    plot_safe_set(args.checkpoint, scenario, scenario_name, output_dir)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
