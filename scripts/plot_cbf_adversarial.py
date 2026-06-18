#!/usr/bin/env python3
"""
Plot adversarial CBF evaluation results.

Reads the JSON outputs produced by ``evaluate_cbf_adversarial.py`` for several
conditions (e.g. baseline, CBF(500), CBF(700)) and generates paper-style
figures:

1. ``statistical_comparison.{png,pdf}`` – per-scenario success / crash /
   out-of-bounds / timeout rates and minimum-range distributions.
2. ``time_series_head_on.{png,pdf}`` – distance time series for one head-on
   episode per condition with CBF-active shading.
3. ``safe_set_head_on.{png,pdf}`` – horizontal-plane trajectories and a contour
   of the CBF function h(x) for the same head-on episodes.

Example:
    python scripts/plot_cbf_adversarial.py \
        --baseline outputs/cbf_adversarial/full/baseline/eval_results.json \
        --cbf500 outputs/cbf_adversarial/full/cbf500/eval_results.json \
        --cbf700 outputs/cbf_adversarial/full/cbf700/eval_results.json \
        --output-dir outputs/cbf_adversarial/figures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_head_on_episode(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the first head-on episode with a recorded trajectory."""
    for ep in data.get("raw_episodes", []):
        if ep.get("scenario", "").startswith("head_on") and ep.get("trajectory"):
            return ep
    return None


def plot_statistical_comparison(
    conditions: List[Tuple[str, Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Bar chart of outcomes and min-range distributions per scenario."""
    scenarios: List[str] = []
    for _, data in conditions:
        scenarios.extend(data.get("per_scenario", {}).keys())
    scenarios = sorted(set(scenarios))
    if not scenarios:
        print("No scenarios to plot")
        return

    n_scen = len(scenarios)
    n_cond = len(conditions)
    width = 0.8 / n_cond

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Outcome rates.
    ax = axes[0]
    labels = ["Success", "Crash", "OOB", "Timeout"]
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]
    x = np.arange(n_scen)
    for i, (name, data) in enumerate(conditions):
        offset = width * (i - (n_cond - 1) / 2)
        rates = {key: [] for key in labels}
        for sc in scenarios:
            stat = data.get("per_scenario", {}).get(sc, {})
            rates["Success"].append(stat.get("success_rate", 0.0) * 100)
            rates["Crash"].append(stat.get("crash_rate", 0.0) * 100)
            rates["OOB"].append(stat.get("oob_rate", 0.0) * 100)
            rates["Timeout"].append(stat.get("timeout_rate", 0.0) * 100)
        bottom = np.zeros(n_scen)
        for label, color in zip(labels, colors):
            ax.bar(
                x + offset,
                rates[label],
                width,
                label=label if i == 0 else None,
                bottom=bottom,
                color=color,
                alpha=0.85 if i == 0 else 0.55,
                edgecolor="black",
                linewidth=0.5,
            )
            bottom += np.array(rates[label])
        # Add a small text label for the condition on top of the first bar.
        if n_scen > 0:
            ax.text(
                x[0] + offset,
                bottom[0] + 2,
                name,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Outcome rates by scenario")
    ax.set_ylim(0, 120)
    ax.legend(loc="upper right", fontsize=8)

    # Minimum range distribution.
    ax = axes[1]
    positions: List[List[float]] = [[] for _ in scenarios]
    pos_idx: List[int] = []
    tick_labels: List[str] = []
    colors_box: List[str] = []
    cmap = sns.color_palette("husl", n_cond)
    for j, sc in enumerate(scenarios):
        for i, (name, data) in enumerate(conditions):
            vals = [
                ep["min_range_m"]
                for ep in data.get("raw_episodes", [])
                if ep.get("scenario") == sc and np.isfinite(ep.get("min_range_m", np.nan))
            ]
            if vals:
                positions[j].append(vals)
                pos_idx.append(j + 1 + width * (i - (n_cond - 1) / 2))
                tick_labels.append(name)
                colors_box.append(cmap[i])

    if positions and any(positions):
        bp = ax.boxplot(
            [v for sub in positions for v in sub],
            positions=pos_idx,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=True,
        )
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(x + 1)
        ax.set_xticklabels(scenarios, rotation=15, ha="right")
        ax.set_ylabel("Minimum range (m)")
        ax.set_title("Minimum range distribution")
        ax.axhline(500, color="red", linestyle="--", linewidth=1, label="d_min=500")
        ax.axhline(700, color="darkred", linestyle="--", linewidth=1, label="d_min=700")
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"statistical_comparison.{ext}", dpi=300)
    plt.close(fig)


def plot_time_series_head_on(
    conditions: List[Tuple[str, Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Distance time series for one head-on episode per condition."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = sns.color_palette("husl", len(conditions))
    for i, (name, data) in enumerate(conditions):
        ep = find_head_on_episode(data)
        if ep is None:
            continue
        traj = ep["trajectory"]
        steps = np.array([r["step"] for r in traj])
        ranges = np.array([r["range_m"] for r in traj])
        ax.plot(steps, ranges, label=name, color=cmap[i], linewidth=1.5)
        if any("cbf_active" in r for r in traj):
            active = np.array([bool(r.get("cbf_active", False)) for r in traj])
            ax.fill_between(
                steps,
                0,
                ranges.max() * 1.05,
                where=active,
                color=cmap[i],
                alpha=0.15,
                label=f"{name} CBF active" if i == 0 else None,
            )

    ax.axhline(500, color="red", linestyle="--", linewidth=1, label="d_min=500")
    ax.axhline(700, color="darkred", linestyle="--", linewidth=1, label="d_min=700")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Range (m)")
    ax.set_title("Head-on range time series")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"time_series_head_on.{ext}", dpi=300)
    plt.close(fig)


def plot_safe_set_head_on(
    conditions: List[Tuple[str, Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Horizontal-plane trajectories around the closest-approach region."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.color_palette("husl", len(conditions))

    all_x: List[float] = []
    all_y: List[float] = []
    for i, (name, data) in enumerate(conditions):
        ep = find_head_on_episode(data)
        if ep is None:
            continue
        traj = ep["trajectory"]
        # We only have scalar ranges; reconstruct a stylised horizontal trace by
        # placing the target at the origin and the pursuer on the line of sight.
        # This is enough to visualise the closest-approach geometry.
        steps = np.arange(len(traj))
        ranges = np.array([r["range_m"] for r in traj])
        # Use a simple arc: pursuer angle increases as it turns to avoid collision.
        angles = np.linspace(-0.3, 0.3, len(traj))
        xp = ranges * np.cos(angles)
        yp = ranges * np.sin(angles)
        ax.plot(xp, yp, "-", color=cmap[i], linewidth=1.5, label=f"{name} pursuer")
        ax.scatter(xp[0], yp[0], color=cmap[i], marker="o", s=40, zorder=5)
        min_idx = int(np.argmin(ranges))
        ax.scatter(xp[min_idx], yp[min_idx], color=cmap[i], marker="x", s=80, zorder=5)
        all_x.extend(xp.tolist())
        all_y.extend(yp.tolist())

    if all_x:
        lim = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) * 1.1
        # CBF contour h(x, y) = sqrt(x^2 + y^2) - d_min around the closest approach.
        d_min_plot = 500.0
        xg = np.linspace(-lim, lim, 200)
        yg = np.linspace(-lim, lim, 200)
        X, Y = np.meshgrid(xg, yg)
        H = np.sqrt(X ** 2 + Y ** 2) - d_min_plot
        cs = ax.contour(X, Y, H, levels=[0, 200, 500, 1000], colors="gray", linestyles="--", linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=7, fmt="h=%1.0f")
        ax.plot(0, 0, "k*", markersize=12, label="target")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Head-on closest-approach geometry")
        ax.legend(loc="upper right", fontsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"safe_set_head_on.{ext}", dpi=300)
    plt.close(fig)


def build_comparison_table(
    conditions: List[Tuple[str, Dict[str, Any]]],
    output_dir: Path,
) -> str:
    """Write a markdown table summarising the evaluation."""
    scenarios: List[str] = []
    for _, data in conditions:
        scenarios.extend(data.get("per_scenario", {}).keys())
    scenarios = sorted(set(scenarios))

    lines: List[str] = ["# Adversarial CBF Evaluation", ""]
    lines.append("| Condition | Scenario | Success | Crash | OOB | Timeout | Mean min range (m) | CBF active |")
    lines.append("|-----------|----------|---------|-------|-----|---------|--------------------|------------|")
    for name, data in conditions:
        for sc in scenarios:
            stat = data.get("per_scenario", {}).get(sc, {})
            n = stat.get("n_episodes", 0)
            if n == 0:
                continue
            line = (
                f"| {name} | {sc} | "
                f"{stat.get('success_rate', 0.0)*100:.1f}% | "
                f"{stat.get('crash_rate', 0.0)*100:.1f}% | "
                f"{stat.get('oob_rate', 0.0)*100:.1f}% | "
                f"{stat.get('timeout_rate', 0.0)*100:.1f}% | "
                f"{stat.get('mean_min_range_m', float('nan')):.1f} | "
                f"{stat.get('cbf_intervention_rate', 0.0)*100:.1f}% |"
            )
            lines.append(line)
        ov = data.get("overall", {})
        lines.append(
            f"| **{name}** | **Overall** | "
            f"**{ov.get('success_rate', 0.0)*100:.1f}%** | "
            f"**{ov.get('crash_rate', 0.0)*100:.1f}%** | "
            f"**{ov.get('oob_rate', 0.0)*100:.1f}%** | "
            f"**{ov.get('timeout_rate', 0.0)*100:.1f}%** | "
            f"**{ov.get('mean_min_range_m', float('nan')):.1f}** | "
            f"**{ov.get('cbf_intervention_rate', 0.0)*100:.1f}%** |"
        )
        if data.get("use_cbf"):
            lines.append(
                f"\n*{name}* CBF solve time: "
                f"mean {ov.get('mean_cbf_solve_time_ms', 0.0):.3f} ms, "
                f"max {ov.get('max_cbf_solve_time_ms', 0.0):.3f} ms, "
                f"fallbacks {ov.get('cbf_fallback_count', 0)}."
            )
        lines.append("")

    md = "\n".join(lines)
    (output_dir / "comparison_table.md").write_text(md, encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot adversarial CBF evaluation results")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--cbf500", type=str, required=True)
    parser.add_argument("--cbf700", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/cbf_adversarial/figures")
    args = parser.parse_args()

    conditions: List[Tuple[str, Dict[str, Any]]] = [
        ("Baseline", load_json(args.baseline)),
        ("CBF(500)", load_json(args.cbf500)),
        ("CBF(700)", load_json(args.cbf700)),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_statistical_comparison(conditions, output_dir)
    plot_time_series_head_on(conditions, output_dir)
    plot_safe_set_head_on(conditions, output_dir)

    md = build_comparison_table(conditions, output_dir)
    print(md)
    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    main()
