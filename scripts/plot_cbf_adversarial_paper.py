#!/usr/bin/env python3
"""
Generate paper-grade figures and LaTeX tables for adversarial CBF evaluation.

Reads the JSON outputs produced by ``evaluate_cbf_adversarial.py`` for three
conditions (Baseline, CBF(500), CBF(700)) and produces:

- ``outputs/cbf_adversarial/figures_paper/table_latex.tex``
- ``outputs/cbf_adversarial/figures_paper/per_scenario_table.tex``
- ``outputs/cbf_adversarial/figures_paper/latex_macros.tex``
- ``outputs/cbf_adversarial/figures_paper/per_scenario_min_range.{png,pdf}``
- ``outputs/cbf_adversarial/figures_paper/time_series_multi_scenario.{png,pdf}``
- ``outputs/cbf_adversarial/figures_paper/cbf_activation_pattern.{png,pdf}``
- ``outputs/cbf_adversarial/figures_paper/safe_set_multi_trajectory.{png,pdf}``
- ``outputs/cbf_adversarial/stats_test.json``

Example:
    python scripts/plot_cbf_adversarial_paper.py \
        --baseline outputs/cbf_adversarial/full_v2/baseline/eval_results.json \
        --cbf500 outputs/cbf_adversarial/full_v2/cbf500/eval_results.json \
        --cbf700 outputs/cbf_adversarial/full_v2/cbf700/eval_results.json \
        --output-dir outputs/cbf_adversarial/figures_paper \
        --stats-dir outputs/cbf_adversarial
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Paper-grade typography.
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


def group_by_seed(episodes: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for ep in episodes:
        seed = int(ep.get("seed", 0))
        groups.setdefault(seed, []).append(ep)
    return groups


def _per_episode_metrics(
    episodes: List[Dict[str, Any]], is_cbf: bool
) -> Dict[str, np.ndarray]:
    """Return arrays of per-episode metric values."""
    success = np.array([e["captured"] for e in episodes], dtype=float)
    crash = np.array([e["crash"] for e in episodes], dtype=float)
    oob = np.array([e["out_of_bounds"] for e in episodes], dtype=float)
    timeout = np.array([e["survived"] for e in episodes], dtype=float)
    min_range = np.array(
        [e["min_range_m"] for e in episodes if np.isfinite(e.get("min_range_m", np.nan))],
        dtype=float,
    )
    result = {
        "success": success,
        "crash": crash,
        "oob": oob,
        "timeout": timeout,
        "min_range": min_range,
    }
    if is_cbf:
        cbf_active = np.array(
            [e["cbf_active_steps"] / max(e["length"], 1) for e in episodes], dtype=float
        )
        qp_time = np.array(
            [
                np.mean(e.get("cbf_solve_times_ms", [0.0]))
                for e in episodes
            ],
            dtype=float,
        )
        result["cbf_active"] = cbf_active
        result["qp_time"] = qp_time
    return result


def condition_stats(
    episodes: List[Dict[str, Any]], is_cbf: bool, level: str = "episode"
) -> Dict[str, np.ndarray]:
    """
    Compute metric arrays for a condition.

    - ``level='episode'``: one value per episode (n = len(episodes)).
    - ``level='seed'``: one value per seed (mean across episodes in each seed).
    """
    if level == "episode":
        return _per_episode_metrics(episodes, is_cbf)
    groups = group_by_seed(episodes)
    seeds = sorted(groups.keys())
    per_ep = {s: _per_episode_metrics(groups[s], is_cbf) for s in seeds}
    result: Dict[str, np.ndarray] = {"seeds": np.array(seeds)}
    keys = ["success", "crash", "oob", "timeout", "min_range"]
    if is_cbf:
        keys += ["cbf_active", "qp_time"]
    for key in keys:
        result[key] = np.array([np.mean(per_ep[s][key]) for s in seeds], dtype=float)
    return result


def mean_std_str(arr: np.ndarray, decimals: int = 1) -> str:
    return f"${np.mean(arr):.{decimals}f} \\pm {np.std(arr, ddof=1):.{decimals}f}$"


def write_main_table(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
    std_level: str,
) -> None:
    rows: List[str] = []
    n_total = sum(len(data.get("raw_episodes", [])) for _, _, data, _ in conditions[:1])
    for label, _, data, is_cbf in conditions:
        eps = data.get("raw_episodes", [])
        st = condition_stats(eps, is_cbf, level=std_level)
        row = (
            f"{label} & "
            f"{mean_std_str(st['success']*100)} & "
            f"{mean_std_str(st['crash']*100)} & "
            f"{mean_std_str(st['oob']*100)} & "
            f"{mean_std_str(st['min_range'])} & "
        )
        if is_cbf:
            row += (
                f"{mean_std_str(st['cbf_active']*100)} & "
                f"{mean_std_str(st['qp_time'], decimals=3)} \\\\"
            )
        else:
            row += "— & — \\\\"
        rows.append(row)

    std_desc = "episodes" if std_level == "episode" else "seeds"
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Adversarial CBF evaluation results (mean$\pm$std across {std_desc}, $n={n_total}$ episodes per condition).}}",
        r"\label{tab:cbf_adversarial}",
        r"\begin{tabular}{lccccccl}",
        r"\toprule",
        r"Method & Success (\%) & Crash (\%) & OOB (\%) & Min range (m) & CBF active (\%) & QP time (ms) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (output_dir / "table_latex.tex").write_text("\n".join(tex), encoding="utf-8")


def write_per_scenario_table(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
    std_level: str,
) -> None:
    scenarios: List[str] = []
    for _, _, data, _ in conditions:
        scenarios.extend(data.get("per_scenario", {}).keys())
    scenarios = sorted(set(scenarios))
    std_desc = "episodes" if std_level == "episode" else "seeds"

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Per-scenario minimum range (m) for adversarial CBF evaluation (mean$\pm$std across {std_desc}).}}",
        r"\label{tab:cbf_adversarial_per_scenario}",
        r"\begin{tabular}{l" + "c" * len(conditions) + "}",
        r"\toprule",
        "Scenario & " + " & ".join(label for label, _, _, _ in conditions) + r" \\",
        r"\midrule",
    ]
    for sc in scenarios:
        cells = [sc.replace("_", r"\_")]
        for label, _, data, is_cbf in conditions:
            eps = [e for e in data.get("raw_episodes", []) if e.get("scenario") == sc]
            if not eps:
                cells.append("—")
                continue
            st = condition_stats(eps, is_cbf, level=std_level)
            cells.append(mean_std_str(st["min_range"]))
        tex.append(" & ".join(cells) + r" \\")
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (output_dir / "per_scenario_table.tex").write_text("\n".join(tex), encoding="utf-8")


def write_latex_macros(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
    std_level: str,
) -> None:
    lines = [r"% Adversarial CBF LaTeX macros"]
    for label, key, data, is_cbf in conditions:
        eps = data.get("raw_episodes", [])
        st = condition_stats(eps, is_cbf, level=std_level)
        prefix = {
            "Baseline": "cbfBaseline",
            "CBF(500)": "cbfFive",
            "CBF(700)": "cbfSeven",
            "CBF(500)-FD": "cbfFd",
        }.get(label, label.replace("(", "").replace(")", "").replace(" ", "").replace("-", ""))
        lines.append(rf"\newcommand{{\{prefix}Success}}{{{np.mean(st['success'])*100:.1f}}}")
        lines.append(rf"\newcommand{{\{prefix}Crash}}{{{np.mean(st['crash'])*100:.1f}}}")
        lines.append(rf"\newcommand{{\{prefix}Oob}}{{{np.mean(st['oob'])*100:.1f}}}")
        lines.append(rf"\newcommand{{\{prefix}MinRange}}{{{np.mean(st['min_range']):.1f}}}")
        if is_cbf:
            lines.append(rf"\newcommand{{\{prefix}Active}}{{{np.mean(st['cbf_active'])*100:.1f}}}")
            lines.append(rf"\newcommand{{\{prefix}QpTime}}{{{np.mean(st['qp_time']):.3f}}}")
    (output_dir / "latex_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_statistical_tests(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    stats_dir: Path,
    stats_name: str,
) -> None:
    baseline_eps = conditions[0][2].get("raw_episodes", [])
    scenarios = sorted({e.get("scenario", "unknown") for e in baseline_eps})

    results: Dict[str, Any] = {"n_per_condition": len(baseline_eps)}
    for comp_label, _, data, is_cbf in conditions[1:]:
        if not is_cbf:
            continue
        cbf_eps = data.get("raw_episodes", [])
        comp: Dict[str, Any] = {}
        for scope in ["overall"] + scenarios:
            if scope == "overall":
                base_vals = np.array([e["min_range_m"] for e in baseline_eps if np.isfinite(e["min_range_m"])])
                cbf_vals = np.array([e["min_range_m"] for e in cbf_eps if np.isfinite(e["min_range_m"])])
            else:
                base_vals = np.array([e["min_range_m"] for e in baseline_eps if e.get("scenario") == scope and np.isfinite(e["min_range_m"])])
                cbf_vals = np.array([e["min_range_m"] for e in cbf_eps if e.get("scenario") == scope and np.isfinite(e["min_range_m"])])
            u, p = stats.mannwhitneyu(base_vals, cbf_vals, alternative="two-sided")
            n1, n2 = len(base_vals), len(cbf_vals)
            cles = u / (n1 * n2)  # Probability that a random CBF episode has larger min range.
            # Rank-biserial correlation (positive -> CBF tends larger).
            u_min = min(u, n1 * n2 - u)
            r_biserial = 1.0 - (2.0 * u_min) / (n1 * n2)
            comp[scope] = {
                "baseline_n": n1,
                "cbf_n": n2,
                "baseline_mean": float(np.mean(base_vals)),
                "cbf_mean": float(np.mean(cbf_vals)),
                "U_statistic": float(u),
                "p_value": float(p),
                "cles": float(cles),
                "rank_biserial_r": float(r_biserial),
            }
        results[comp_label] = comp

    (stats_dir / stats_name).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def plot_per_scenario_min_range(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
    std_level: str,
) -> None:
    scenarios: List[str] = []
    for _, _, data, _ in conditions:
        scenarios.extend(data.get("per_scenario", {}).keys())
    scenarios = sorted(set(scenarios))
    n_scen = len(scenarios)
    n_cond = len(conditions)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    width = 0.25
    x = np.arange(n_scen)
    palette = sns.color_palette("muted", n_cond)

    for i, (label, _, data, is_cbf) in enumerate(conditions):
        means, stds = [], []
        for sc in scenarios:
            eps = [e for e in data.get("raw_episodes", []) if e.get("scenario") == sc]
            st = condition_stats(eps, is_cbf, level=std_level)
            means.append(float(np.mean(st["min_range"])))
            stds.append(float(np.std(st["min_range"], ddof=1)))
        offset = width * (i - (n_cond - 1) / 2)
        ax.bar(x + offset, means, width, yerr=stds, label=label, color=palette[i], capsize=2, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
    ax.set_ylabel("Minimum range (m)")
    ax.axhline(500, color="red", linestyle="--", linewidth=0.8, label="$d_{\\min}=500$")
    ax.axhline(700, color="darkred", linestyle="--", linewidth=0.8, label="$d_{\\min}=700$")
    ax.legend(loc="upper right", frameon=True, fontsize=7)
    sns.despine(ax=ax)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"per_scenario_min_range.{ext}")
    plt.close(fig)


def plot_time_series_multi_scenario(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
) -> None:
    """Time series for one representative episode per scenario (head-on, offset, tail_chase)."""
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
        ax.axhline(700, color="darkred", linestyle="--", linewidth=0.7)
        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Range (m)")
    axes[0].legend(loc="upper right", frameon=True, fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"time_series_multi_scenario.{ext}")
    plt.close(fig)


def plot_cbf_activation_pattern(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
) -> None:
    """Scatter: episode index vs CBF-active fraction, colored by scenario."""
    cbf_conditions = [(label, data) for label, _, data, is_cbf in conditions if is_cbf]
    if not cbf_conditions:
        return
    fig, axes = plt.subplots(1, len(cbf_conditions), figsize=(7.0, 2.2), sharey=True)
    if len(cbf_conditions) == 1:
        axes = [axes]
    scenario_list = sorted({e.get("scenario", "unknown") for e in cbf_conditions[0][1].get("raw_episodes", [])})
    palette = sns.color_palette("muted", len(scenario_list))
    sc_map = {s: palette[i] for i, s in enumerate(scenario_list)}

    for ax, (label, data) in zip(axes, cbf_conditions):
        eps = data.get("raw_episodes", [])
        for s in scenario_list:
            s_eps = [e for e in eps if e.get("scenario") == s]
            x = np.arange(len(s_eps))
            y = [e["cbf_active_steps"] / max(e["length"], 1) for e in s_eps]
            ax.scatter(x, np.array(y) * 100, label=s.replace("_", " "), color=sc_map[s], s=20, alpha=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Episode index", fontsize=8)
        ax.set_ylim(0, 60)
        sns.despine(ax=ax)
    axes[0].set_ylabel("CBF active (\%)")
    axes[-1].legend(loc="upper right", frameon=True, fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"cbf_activation_pattern.{ext}")
    plt.close(fig)


def plot_safe_set_multi_trajectory(
    conditions: List[Tuple[str, str, Dict[str, Any], bool]],
    output_dir: Path,
) -> None:
    """Stylised horizontal-plane closest-approach geometry with CBF contours."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    palette = sns.color_palette("muted", len(conditions))

    all_x: List[float] = []
    all_y: List[float] = []
    for i, (label, _, data, is_cbf) in enumerate(conditions):
        eps = [e for e in data.get("raw_episodes", []) if e.get("scenario", "").startswith("head_on")]
        if not eps:
            continue
        ep = eps[0]
        traj = ep["trajectory"]
        ranges = np.array([r["range_m"] for r in traj])
        # Stylised arc: pursuer turns to avoid collision.
        angles = np.linspace(-0.4, 0.4, len(traj))
        xp = ranges * np.cos(angles)
        yp = ranges * np.sin(angles)
        active = np.array([bool(r.get("cbf_active", False)) for r in traj])
        ax.plot(xp, yp, "-", color=palette[i], linewidth=1.2, label=label)
        # Thicker segment where CBF is active.
        if active.any():
            ax.plot(xp[active], yp[active], "-", color=palette[i], linewidth=2.5)
        min_idx = int(np.argmin(ranges))
        ax.scatter(xp[min_idx], yp[min_idx], color=palette[i], marker="x", s=60, zorder=5)
        all_x.extend(xp.tolist())
        all_y.extend(yp.tolist())

    if all_x:
        lim = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) * 1.1
        xg = np.linspace(-lim, lim, 200)
        yg = np.linspace(-lim, lim, 200)
        X, Y = np.meshgrid(xg, yg)
        H = np.sqrt(X ** 2 + Y ** 2) - 500.0
        cs = ax.contour(X, Y, H, levels=[0, 200, 500, 1000], colors="gray", linestyles="--", linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=7, fmt="$h=%1.0f$")
        ax.plot(0, 0, "k*", markersize=10, label="target")
        # Arrow showing CBF-induced deviation near closest approach.
        ax.annotate(
            "",
            xy=(700, -200),
            xytext=(300, 50),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=8,
        )
        ax.text(720, -220, "CBF deviation", fontsize=7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.legend(loc="upper right", frameon=True, fontsize=7)
        sns.despine(ax=ax)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"safe_set_multi_trajectory.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-grade adversarial CBF figures")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--cbf500", type=str, required=True)
    parser.add_argument("--cbf700", type=str, required=True)
    parser.add_argument("--cbf500-fd", type=str, default=None, help="Optional JSBSim finite-difference CBF(500) results.")
    parser.add_argument("--output-dir", type=str, default="outputs/cbf_adversarial/figures_paper")
    parser.add_argument("--stats-dir", type=str, default="outputs/cbf_adversarial")
    parser.add_argument("--stats-name", type=str, default="stats_test.json")
    parser.add_argument(
        "--std-level",
        type=str,
        choices=["episode", "seed"],
        default="episode",
        help="Level at which to compute standard deviations (episode or seed).",
    )
    args = parser.parse_args()

    conditions: List[Tuple[str, str, Dict[str, Any], bool]] = [
        ("Baseline", "baseline", load_json(args.baseline), False),
        ("CBF(500)", "cbf500", load_json(args.cbf500), True),
        ("CBF(700)", "cbf700", load_json(args.cbf700), True),
    ]
    if args.cbf500_fd:
        conditions.append(("CBF(500)-FD", "cbf500_fd", load_json(args.cbf500_fd), True))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    write_main_table(conditions, output_dir, args.std_level)
    write_per_scenario_table(conditions, output_dir, args.std_level)
    write_latex_macros(conditions, output_dir, args.std_level)
    run_statistical_tests(conditions, stats_dir, args.stats_name)

    plot_per_scenario_min_range(conditions, output_dir, args.std_level)
    plot_time_series_multi_scenario(conditions, output_dir)
    plot_cbf_activation_pattern(conditions, output_dir)
    plot_safe_set_multi_trajectory(conditions, output_dir)

    print("Paper-grade outputs saved to", output_dir)
    print("Statistical tests saved to", stats_dir / args.stats_name)


if __name__ == "__main__":
    main()
