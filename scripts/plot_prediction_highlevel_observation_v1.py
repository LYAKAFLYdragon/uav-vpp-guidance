#!/usr/bin/env python
"""Generate paper-oriented plots from an existing H0/H1/H2 analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "h0_current_geometry": "H0 当前观测",
    "h1_frozen_predicted_geometry": "H1 预测观测",
    "h2_retrained_predicted_geometry": "H2 再适配",
}


def _save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    written = []
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    return written


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value


def _configure_fonts():
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _plot_information_flow(output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    stages = [
        ("冻结预测器", "预测相对位置", "#d9edf7"),
        ("预测几何", "距离 / 距离率\nATA / AA", "#e8f5e9"),
        ("高层决策", "PPO 选择低层策略", "#fff3cd"),
        ("低层执行", "固定策略 → VPP", "#fce4ec"),
        ("控制链", "制导 → PID → JSBSim", "#ede7f6"),
        ("闭环结果", "终端与几何指标", "#e0f2f1"),
    ]
    x_positions = np.linspace(0.35, 10.85, len(stages))
    for index, (top, bottom, color) in enumerate(stages):
        box = FancyBboxPatch(
            (x_positions[index], 0.7), 1.75, 1.25,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=0.9, edgecolor="#444444", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x_positions[index] + 0.875, 1.55, top, ha="center", va="center", fontsize=10)
        ax.text(x_positions[index] + 0.875, 1.08, bottom, ha="center", va="center", fontsize=8.5)
        if index < len(stages) - 1:
            ax.annotate(
                "", xy=(x_positions[index + 1] - 0.08, 1.325),
                xytext=(x_positions[index] + 1.83, 1.325),
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
            )
    ax.text(6.5, 0.25, "H1：不更新高层 PPO；H2：仅重新训练高层 PPO；低层策略和执行链始终冻结", ha="center", fontsize=8.5)
    result = _save_figure(fig, output_dir, "prediction_decision_information_flow")
    plt.close(fig)
    return result


def _plot_result_comparison(analysis_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    per_seed = _read_csv(analysis_path.parent / "metrics_per_seed.csv")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    if per_seed:
        for row in per_seed:
            value = _float(row, "success_rate")
            if np.isfinite(value):
                key = (row.get("method", ""), row.get("opponent", ""), row.get("training_seed", ""))
                grouped.setdefault(key, []).append(value)
    if not grouped:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        for row in payload.get("summaries", []):
            value = row.get("success_rate")
            if value is not None and np.isfinite(float(value)):
                key = (row.get("method", ""), row.get("opponent", ""), "all")
                grouped.setdefault(key, []).append(float(value))
    if not grouped:
        return []

    methods = [method for method in METHOD_LABELS if any(key[0] == method for key in grouped)]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    styles = {"expert": ("o", "-"), "end_to_end": ("s", "--")}
    for opponent in ("expert", "end_to_end"):
        for seed in sorted({key[2] for key in grouped if key[1] == opponent}):
            values = [float(np.mean(grouped.get((method, opponent, seed), [np.nan]))) for method in methods]
            if not any(np.isfinite(values)):
                continue
            marker, linestyle = styles[opponent]
            ax.plot(
                range(len(methods)), values, marker=marker, linestyle=linestyle,
                linewidth=1.0, markersize=5, label=f"{opponent}, seed {seed}",
            )
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods])
    ax.set_ylabel("胜率")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    result = _save_figure(fig, output_dir, "H0_H1_H2_result_comparison")
    plt.close(fig)
    return result


def _plot_attitude_heatmap(analysis_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = _read_csv(analysis_path.parent / "metrics_by_attitude.csv")
    if not rows:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        rows = payload.get("summaries", [])
    scenarios = ["head_on", "crossing_left", "crossing_right", "neutral", "disadvantage"]
    methods = [method for method in METHOD_LABELS if any(row.get("method") == method for row in rows)]
    if not methods:
        return []
    matrix = np.full((len(methods), len(scenarios)), np.nan)
    for i, method in enumerate(methods):
        for j, scenario in enumerate(scenarios):
            values = [
                _float(row, "success_rate")
                for row in rows
                if row.get("method") == method and row.get("scenario") == scenario
            ]
            values = [value for value in values if np.isfinite(value)]
            if values:
                matrix[i, j] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(scenarios)), ["迎头", "左交叉", "右交叉", "中性", "劣势"])
    ax.set_yticks(range(len(methods)), [METHOD_LABELS[method] for method in methods])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("初始态势")
    ax.set_ylabel("配置")
    fig.colorbar(image, ax=ax, label="胜率")
    result = _save_figure(fig, output_dir, "attitude_result_heatmap")
    plt.close(fig)
    return result


def plot(analysis_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_fonts()

    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    rows = payload.get("summaries", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(_plot_information_flow(output_dir))
    if not rows:
        return written

    methods = sorted({row["method"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    matrix = []
    labels = []
    for method in methods:
        matrix.append([next((row["success_rate"] for row in rows if row["method"] == method and row["scenario"] == scenario), float("nan")) for scenario in scenarios])
        labels.append(method)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(scenarios)), scenarios, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("H0/H1/H2 success-rate heatmap")
    fig.colorbar(image, ax=ax, label="success rate")
    fig.tight_layout()
    path = output_dir / "success_rate_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped = {}
    for row in rows:
        value = row.get("mean_prediction_error_m")
        if value is not None and np.isfinite(float(value)):
            grouped.setdefault(row["method"], []).append(float(value))
    box_methods = [method for method in methods if grouped.get(method)]
    if box_methods:
        ax.boxplot([grouped[method] for method in box_methods], labels=box_methods, showmeans=True)
    ax.set_ylabel("mean prediction error (m)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = output_dir / "prediction_error_boxplot.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)
    written.extend(_plot_result_comparison(analysis_path, output_dir))
    written.extend(_plot_attitude_heatmap(analysis_path, output_dir))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    written = plot(args.analysis.resolve(), args.output_dir.resolve())
    print("\n".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
