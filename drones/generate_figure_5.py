import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from generate_figures_1_3 import (
    COMPARISON_DIRS,
    CONTROLLER_LABELS,
    CONTROLLER_ORDER,
    OUT_DIR,
    SOURCE_DATA_DIR,
    TASK_LABELS,
    TASK_ORDER,
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_ORDER,
    add_panel_label,
    apply_publication_style,
    source_base_fields,
    export_figure,
    load_json,
    detect_variant,
    normalize_termination_reason,
)


# Figure contract
# Core conclusion:
#   Episode-level geometry health depends on the joint reading of forward-bias
#   shift, damage exchange, and termination semantics rather than on any single
#   scalar alone.
# Evidence chain:
#   Panels a-d: each controller/encounter cell maps pre-merge Forward Bias to
#   episode damage margin, showing how broad, narrow, and mixed variants occupy
#   different geometry-health regimes and how similar bias ranges can still end
#   in different termination semantics.
# Archetype:
#   Quantitative grid with episode-level clouds and variant centroids.
# Journal/export contract:
#   Double-column manuscript figure with editable PDF/SVG text and source data.


ROOT = Path(__file__).resolve().parents[1]

X_LIMITS = (-720, 560)
Y_LIMITS = (-48, 72)
X_LABEL = r"Pre-merge Forward Bias $\bar{d}_{\mathrm{fwd}}^{\mathrm{pre}}$ (m)"
Y_LABEL = r"Episode damage margin $m_{\mathrm{dmg}}$"

TERMINATION_LABELS = {
    "favorable_timeout": "Favorable timeout",
    "counterpart_crash": "Counterpart crash/OOB",
    "unfavorable_timeout": "Unfavorable timeout",
    "ego_crash": "Ego crash/OOB",
}

TERMINATION_MARKERS = {
    "favorable_timeout": "o",
    "counterpart_crash": "s",
    "unfavorable_timeout": "^",
    "ego_crash": "X",
}

LABEL_OFFSETS = {
    ("expert", "head_on", "baseline"): (8, 6),
    ("expert", "head_on", "broad"): (8, -2),
    ("expert", "head_on", "narrow"): (8, 6),
    ("expert", "head_on", "mixed"): (8, 10),
    ("expert", "crossing_feasible", "baseline"): (8, 6),
    ("expert", "crossing_feasible", "broad"): (8, -8),
    ("expert", "crossing_feasible", "narrow"): (8, 4),
    ("expert", "crossing_feasible", "mixed"): (8, 10),
    ("end_to_end", "head_on", "baseline"): (8, 6),
    ("end_to_end", "head_on", "broad"): (8, 4),
    ("end_to_end", "head_on", "narrow"): (8, 4),
    ("end_to_end", "head_on", "mixed"): (8, 8),
    ("end_to_end", "crossing_feasible", "baseline"): (8, 6),
    ("end_to_end", "crossing_feasible", "broad"): (8, 0),
    ("end_to_end", "crossing_feasible", "narrow"): (8, 8),
    ("end_to_end", "crossing_feasible", "mixed"): (8, 12),
}


def stable_jitter(key: str, amplitude: float) -> float:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 2.0 * amplitude


def load_episode_dataframe() -> pd.DataFrame:
    rows = []
    for controller in CONTROLLER_ORDER:
        for source_variant in ["broad", "narrow", "mixed"]:
            path = COMPARISON_DIRS[(controller, source_variant)] / "aggregate" / "episode_records.json"
            obj = load_json(path)
            for ep in obj["episodes"]:
                variant = detect_variant(ep["method_label"])
                if source_variant != "broad" and variant == "baseline":
                    continue
                rows.append(
                    {
                        "controller": ep["opponent_stage"],
                        "task": ep["task"],
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        "forward_bias_m": ep["pre_merge_vp_forward_bias_m"],
                        "damage_margin": ep["damage_margin"],
                        "termination": normalize_termination_reason(ep["combat_reason"]),
                        "seed": ep["seed"],
                    }
                )

    df = pd.DataFrame(rows)
    df = (
        df.sort_values(["controller", "task", "variant", "seed"])
        .drop_duplicates(
            subset=["controller", "task", "variant", "seed", "forward_bias_m", "damage_margin", "termination"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    df["plot_x"] = df.apply(
        lambda row: row["forward_bias_m"]
        + stable_jitter(
            f"{row['controller']}::{row['task']}::{row['variant']}::{row['termination']}::{row['seed']}::x",
            9.0,
        ),
        axis=1,
    )
    df["plot_y"] = df.apply(
        lambda row: row["damage_margin"]
        + stable_jitter(
            f"{row['controller']}::{row['task']}::{row['variant']}::{row['termination']}::{row['seed']}::y",
            1.0,
        ),
        axis=1,
    )
    return df


def build_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.25))
    panel_order = [
        ("expert", "head_on"),
        ("expert", "crossing_feasible"),
        ("end_to_end", "head_on"),
        ("end_to_end", "crossing_feasible"),
    ]

    for idx, ((controller, task), ax) in enumerate(zip(panel_order, axes.flat)):
        subset = df[(df["controller"] == controller) & (df["task"] == task)].copy()

        ax.axhspan(0, Y_LIMITS[1], color="#F8FBF9", alpha=0.48, zorder=0)
        ax.axvspan(X_LIMITS[0], 0, color="#F4F7FB", alpha=0.65, zorder=0)
        ax.axhline(0, color="#6B7280", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
        ax.axvline(0, color="#6B7280", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)

        for variant in VARIANT_ORDER:
            variant_rows = subset[subset["variant"] == variant]
            if variant_rows.empty:
                continue
            color = VARIANT_COLORS[variant]
            for termination, marker in TERMINATION_MARKERS.items():
                points = variant_rows[variant_rows["termination"] == termination]
                if points.empty:
                    continue
                ax.scatter(
                    points["plot_x"],
                    points["plot_y"],
                    s=58,
                    marker=marker,
                    color=color,
                    alpha=0.72,
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=2,
                )

            mean_x = variant_rows["forward_bias_m"].mean()
            mean_y = variant_rows["damage_margin"].mean()
            ax.scatter(
                mean_x,
                mean_y,
                s=188,
                marker="D",
                color=color,
                edgecolor="black",
                linewidth=0.85,
                zorder=4,
            )
            dx, dy = LABEL_OFFSETS[(controller, task, variant)]
            ax.annotate(
                VARIANT_LABELS[variant].lower(),
                xy=(mean_x, mean_y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.2,
                color="#222222",
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                zorder=5,
            )

        ax.set_xlim(*X_LIMITS)
        ax.set_ylim(*Y_LIMITS)
        ax.grid(axis="both", color="#E7E7E7", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{CONTROLLER_LABELS[controller]}\n{TASK_LABELS[task]}",
            fontsize=8.8,
            pad=6,
        )
        if idx % 2 == 0:
            ax.set_ylabel(Y_LABEL, fontsize=8.0)
        if idx >= 2:
            ax.set_xlabel(X_LABEL, fontsize=8.0)
        add_panel_label(ax, chr(ord("a") + idx))

    variant_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=VARIANT_COLORS[variant],
            markeredgecolor="black",
            markersize=6.8,
            label=VARIANT_LABELS[variant],
        )
        for variant in VARIANT_ORDER
    ]
    termination_handles = [
        Line2D(
            [0],
            [0],
            marker=TERMINATION_MARKERS[key],
            color="#4B5563",
            markerfacecolor="#4B5563",
            markeredgecolor="white",
            markeredgewidth=0.45,
            linestyle="None",
            markersize=6.3,
            label=TERMINATION_LABELS[key],
        )
        for key in ["favorable_timeout", "counterpart_crash", "unfavorable_timeout", "ego_crash"]
    ]

    fig.subplots_adjust(top=0.74, bottom=0.10, left=0.10, right=0.98, wspace=0.20, hspace=0.30)
    fig.suptitle(
        "Episode-level geometry-health map",
        fontsize=10.0,
        y=0.975,
    )
    fig.text(
        0.5,
        0.946,
        "Large diamonds show variant means. Smaller points show individual episodes with slight deterministic jitter for visibility.",
        ha="center",
        va="center",
        fontsize=6.9,
        color="#475467",
    )
    fig.text(0.17, 0.885, "Color: interface variant", fontsize=6.9, color="#344054", ha="left", va="center")
    fig.text(
        0.62,
        0.885,
        "Marker: termination semantics",
        fontsize=6.9,
        color="#344054",
        ha="left",
        va="center",
    )
    fig.legend(
        handles=variant_handles,
        loc="upper center",
        bbox_to_anchor=(0.29, 0.872),
        ncol=4,
        fontsize=6.9,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.legend(
        handles=termination_handles,
        loc="upper center",
        bbox_to_anchor=(0.80, 0.872),
        ncol=2,
        fontsize=6.9,
        columnspacing=1.1,
        handletextpad=0.5,
    )

    export_figure(fig, OUT_DIR / "figure5_geometry_health_map")


def save_source_data(df: pd.DataFrame) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_rows = []
    for row in df.itertuples(index=False):
        export_row = source_base_fields(row.controller, row.task, row.variant)
        export_row.update(
            {
                "seed": int(row.seed),
                "forward_bias_pre_merge_m": float(row.forward_bias_m),
                "damage_margin": float(row.damage_margin),
                "termination_semantics": TERMINATION_LABELS[row.termination],
                "termination_semantics_code": row.termination,
            }
        )
        export_rows.append(export_row)

    pd.DataFrame(export_rows).to_csv(
        SOURCE_DATA_DIR / "figure5_geometry_health_map.csv",
        index=False,
    )


def main() -> None:
    apply_publication_style()
    df = load_episode_dataframe()
    save_source_data(df)
    build_figure(df)
    print(f"Figure 5 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
