"""Generate final 4-condition CBF comparison figures for the paper.

Conditions:
- Baseline
- CBF-PM   (point-mass Jacobian, d_min=500)
- CBF-FD   (JSBSim finite-difference Jacobian, d_min=500)
- CBF-FD-E (JSBSim FD + flight-envelope clip, d_min=500)

Outputs:
- paper_materials/figures/cbf_randomized/per_scenario_min_range_4cond.{png,pdf}
- paper_materials/figures/cbf_randomized/time_series_4cond.{png,pdf}
- paper_materials/figures/cbf_randomized/qp_time_comparison.{png,pdf}
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.compression": 9,
})


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def condition_stats(episodes: List[Dict[str, Any]], is_cbf: bool) -> Dict[str, np.ndarray]:
    success = np.array([e["captured"] for e in episodes], dtype=float)
    crash = np.array([e["crash"] for e in episodes], dtype=float)
    oob = np.array([e["out_of_bounds"] for e in episodes], dtype=float)
    min_range = np.array([e["min_range_m"] for e in episodes if np.isfinite(e.get("min_range_m", np.nan))], dtype=float)
    result = {"success": success, "crash": crash, "oob": oob, "min_range": min_range}
    if is_cbf:
        cbf_active = np.array([e["cbf_active_steps"] / max(e["length"], 1) for e in episodes], dtype=float)
        qp_time = np.array([np.mean(e.get("cbf_solve_times_ms", [0.0])) for e in episodes], dtype=float)
        result["cbf_active"] = cbf_active
        result["qp_time"] = qp_time
    return result


def plot_per_scenario_min_range(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
) -> None:
    scenarios: List[str] = []
    for _, _, data, _ in conditions:
        scenarios.extend(data.get("per_scenario", {}).keys())
    scenarios = sorted(set(scenarios))
    n_scen = len(scenarios)
    n_cond = len(conditions)

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    width = 0.2
    x = np.arange(n_scen)
    palette = sns.color_palette("muted", n_cond)

    for i, (label, _, data, is_cbf) in enumerate(conditions):
        means, stds = [], []
        for sc in scenarios:
            eps = [e for e in data.get("raw_episodes", []) if e.get("scenario") == sc]
            st = condition_stats(eps, is_cbf)
            means.append(float(np.mean(st["min_range"])))
            stds.append(float(np.std(st["min_range"], ddof=1)))
        offset = width * (i - (n_cond - 1) / 2)
        ax.bar(x + offset, means, width, yerr=stds, label=label, color=palette[i], capsize=2, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
    ax.set_ylabel("Minimum range (m)")
    ax.axhline(500, color="red", linestyle="--", linewidth=0.8, label="$d_{\\min}=500$")
    ax.legend(loc="upper right", frameon=True, fontsize=7)
    sns.despine(ax=ax)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"per_scenario_min_range_4cond.{ext}")
    plt.close(fig)


def plot_time_series(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
) -> None:
    scenarios = ["head_on_4000m", "offset_45deg_2000m", "tail_chase_2000m"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    palette = sns.color_palette("muted", len(conditions))

    for ax, sc in zip(axes, scenarios):
        for i, (label, _, data, is_cbf) in enumerate(conditions):
            eps = [e for e in data.get("raw_episodes", []) if e.get("scenario") == sc]
            if not eps:
                continue
            ep = eps[0]
            traj = ep["trajectory"]
            steps = np.array([r["step"] for r in traj])
            ranges = np.array([r["range_m"] for r in traj])
            ax.plot(steps, ranges, label=label if sc == scenarios[0] else "", color=palette[i], linewidth=1.2)
            if is_cbf:
                active = np.array([bool(r.get("cbf_active", False)) for r in traj])
                if active.any():
                    ax.fill_between(steps, 0, ranges.max() * 1.1, where=active, color=palette[i], alpha=0.12)
        ax.set_title(sc.replace("_", " "), fontsize=9)
        ax.set_xlabel("Decision step", fontsize=8)
        ax.axhline(500, color="red", linestyle="--", linewidth=0.7)
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Range (m)")
    axes[0].legend(loc="upper right", frameon=True, fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"time_series_4cond.{ext}")
    plt.close(fig)


def plot_qp_time(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
) -> None:
    """Bar plot of overall mean QP solve time for the three CBF methods."""
    cbf_conditions = [(label, data) for label, _, data, is_cbf in conditions if is_cbf]
    labels = [label for label, _ in cbf_conditions]
    means = [data["overall"].get("mean_cbf_solve_time_ms", 0.0) for _, data in cbf_conditions]
    maxes = [data["overall"].get("max_cbf_solve_time_ms", 0.0) for _, data in cbf_conditions]

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    x = np.arange(len(labels))
    palette = sns.color_palette("muted", len(labels))
    bars = ax.bar(x, means, color=palette, edgecolor="black", linewidth=0.5)
    # Add max as error whisker
    ax.errorbar(x, means, yerr=[np.zeros(len(maxes)), np.array(maxes) - np.array(means)], fmt="none", ecolor="black", capsize=3, linewidth=0.8)

    # 5 Hz real-time budget line (200 ms).
    ax.axhline(200.0, color="red", linestyle="--", linewidth=0.8, label="5 Hz budget (200 ms)")

    for bar, m in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f"{m:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("QP solve time (ms)")
    ax.set_ylim(0, max(maxes) * 1.2)
    ax.legend(loc="upper right", frameon=True, fontsize=7)
    sns.despine(ax=ax)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"qp_time_comparison.{ext}")
    plt.close(fig)


def main() -> None:
    root = Path("outputs/cbf_adversarial")
    conditions: List[Tuple[str, str, Dict[str, Any], bool]] = [
        ("Baseline", "baseline", load_json(str(root / "randomized" / "baseline" / "eval_results.json")), False),
        ("CBF-PM", "cbf_pm", load_json(str(root / "randomized" / "cbf500" / "eval_results.json")), True),
        ("CBF-FD", "cbf_fd", load_json(str(root / "randomized_jsbsim_fd" / "cbf500" / "eval_results.json")), True),
        ("CBF-FD-E", "cbf_fd_e", load_json(str(root / "randomized_jsbsim_fd_envelope" / "cbf500" / "eval_results.json")), True),
    ]

    output_dir = Path("paper_materials/figures/cbf_randomized")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_per_scenario_min_range(conditions, output_dir)
    plot_time_series(conditions, output_dir)
    plot_qp_time(conditions, output_dir)
    print("Final CBF paper figures saved to", output_dir)


if __name__ == "__main__":
    main()
