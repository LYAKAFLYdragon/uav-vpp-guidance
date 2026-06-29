import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from generate_figures_1_3 import (
    COMPARISON_DIRS,
    OUT_DIR,
    SOURCE_DATA_DIR,
    TERMINATION_LABELS,
    VARIANT_COLORS,
    normalize_termination_reason,
    add_panel_label,
    apply_publication_style,
    export_figure,
    load_json,
    source_base_fields,
)


# Figure contract
# Core conclusion:
#   A paired crossing-encounter case under the expert controller shows that the
#   broad and mixed variants can share a favorable outcome label while
#   exhibiting very different trajectory structure, damage exchange, and
#   forward-bias evolution.
# Evidence chain:
#   Panel a: Broad 3-D trajectory ends with counterpart crash/OOB despite
#            a negative damage margin.
#   Panel b: Mixed 3-D trajectory from the same seed sustains a longer,
#            bounded interaction and ends with favorable timeout plus positive
#            damage margin.
#   Panels c-d: d_fwd(t) exposes the different geometry evolution behind those
#            superficially similar outcomes.
# Archetype:
#   Asymmetric mixed-modality figure with plan-view hero panels and subordinate
#   time-series panels.
# Journal/export contract:
#   Double-column manuscript figure with editable PDF/SVG text and a matching
#   source-data CSV.


ROOT = Path(__file__).resolve().parents[1]

CONTROLLER = "expert"
TASK = "crossing_feasible"
TASK_LABEL = "Crossing encounter"
CONTROLLER_LABEL = "Expert controller"
VARIANT_ORDER = ["broad", "mixed"]
METHOD_FILTERS = {
    "broad": ("combat finetune best", "combat_finetune_best"),
    "mixed": ("mixed crossing-restored", "mixed_crossing_restored"),
}

COUNTERPART_COLOR = "#98A2B3"
RAW_LINE_ALPHA = 0.22
PREMERGE_SHADE = "#F2F4F7"
FIRST_PASS_COLOR = "#6B7280"
ROLLING_WINDOW_STEPS = 5
TRAJECTORY_VIEW_ELEV = 20
TRAJECTORY_VIEW_AZIM = -58
Z_ASPECT_SCALE = 0.56
GROUND_FOOTPRINT_ALPHA = 0.060
GROUND_FOOTPRINT_EDGE_ALPHA = 0.18
GROUND_TRACE_ALPHA = 0.16
HEIGHT_GUIDE_ALPHA = 0.26
HEIGHT_GUIDE_LINEWIDTH = 0.8


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotonic-chain convex hull for a subtle ground footprint polygon."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def add_ground_footprint(
    ax: plt.Axes,
    xs: pd.Series,
    ys: pd.Series,
    z_floor: float,
    color: str,
) -> None:
    hull = _convex_hull(list(zip(xs.tolist(), ys.tolist())))
    if len(hull) < 3:
        return

    verts = [[(x, y, z_floor) for x, y in hull]]
    face_rgba = matplotlib.colors.to_rgba(color, GROUND_FOOTPRINT_ALPHA)
    edge_rgba = matplotlib.colors.to_rgba(color, GROUND_FOOTPRINT_EDGE_ALPHA)
    poly = Poly3DCollection(
        verts,
        facecolors=[face_rgba],
        edgecolors=[edge_rgba],
        linewidths=0.6,
        zorder=0,
    )
    ax.add_collection3d(poly)


def add_height_guides(
    ax: plt.Axes,
    event_rows: list[pd.Series],
    z_floor: float,
    color: str,
) -> None:
    guide_color = matplotlib.colors.to_rgba(color, HEIGHT_GUIDE_ALPHA)
    floor_color = matplotlib.colors.to_rgba(color, 0.22)
    for row in event_rows:
        x = row["ego_x_m"]
        y = row["ego_y_m"]
        z = row["ego_z_m"]
        ax.plot(
            [x, x],
            [y, y],
            [z_floor, z],
            color=guide_color,
            linewidth=HEIGHT_GUIDE_LINEWIDTH,
            linestyle=(0, (2, 2)),
            zorder=1,
        )
        ax.scatter(
            [x],
            [y],
            [z_floor],
            s=18,
            color=[floor_color],
            edgecolor="none",
            zorder=1,
        )


def load_variant_episodes(variant: str) -> list[dict]:
    path = COMPARISON_DIRS[(CONTROLLER, variant)] / "aggregate" / "episode_records.json"
    obj = load_json(path)
    filters = METHOD_FILTERS[variant]
    return [
        ep
        for ep in obj["episodes"]
        if ep["task"] == TASK
        and ep["opponent_stage"] == CONTROLLER
        and any(token in ep["method_label"].lower() for token in filters)
    ]


def choose_representative_seed(broad_eps: list[dict], mixed_eps: list[dict]) -> int:
    broad_by_seed = {ep["seed"]: ep for ep in broad_eps}
    mixed_by_seed = {ep["seed"]: ep for ep in mixed_eps}
    shared_seeds = sorted(set(broad_by_seed) & set(mixed_by_seed))

    candidates = []
    for seed in shared_seeds:
        broad = broad_by_seed[seed]
        mixed = mixed_by_seed[seed]
        if broad["combat_reason"] != "target_crash_or_out_of_bounds":
            continue
        if mixed["combat_reason"] != "timeout_hp_advantage":
            continue
        candidates.append(
            (
                broad["damage_margin"],
                -mixed["damage_margin"],
                -mixed["total_time_s"],
                seed,
            )
        )

    if not candidates:
        raise RuntimeError("No paired representative seed matched the broad-vs-mixed selection rule.")

    candidates.sort()
    return candidates[0][-1]


def select_episode(episodes: list[dict], seed: int) -> dict:
    return next(ep for ep in episodes if ep["seed"] == seed)


def build_case_dataframe(episodes_by_variant: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for variant, episode in episodes_by_variant.items():
        trajectory = episode["trajectory"]
        x0 = trajectory[0]["ego_pos_x"]
        y0 = trajectory[0]["ego_pos_y"]
        first_pass_step = int(episode["first_pass_step"] or 0)
        termination_code = normalize_termination_reason(episode["combat_reason"])
        for step in trajectory:
            rows.append(
                {
                    "variant": variant,
                    "seed": episode["seed"],
                    "time_s": step["time_s"],
                    "step": step["step"],
                    "ego_x_m": step["ego_pos_x"] - x0,
                    "ego_y_m": step["ego_pos_y"] - y0,
                    "ego_z_m": step["ego_pos_z"],
                    "target_x_m": step["target_pos_x"] - x0,
                    "target_y_m": step["target_pos_y"] - y0,
                    "target_z_m": step["target_pos_z"],
                    "d_fwd_m": step["vp_forward_bias_m"],
                    "d_fwd_m_smooth": None,
                    "range_m": step["range_m"],
                    "pre_merge": step["pre_merge"],
                    "post_merge": step["post_merge"],
                    "first_pass_step": first_pass_step,
                    "termination_code": termination_code,
                    "combat_reason": episode["combat_reason"],
                    "damage_margin": episode["damage_margin"],
                    "total_time_s": episode["total_time_s"],
                }
            )

    df = pd.DataFrame(rows)
    df["d_fwd_m_smooth"] = (
        df.groupby("variant")["d_fwd_m"]
        .transform(lambda s: s.rolling(window=ROLLING_WINDOW_STEPS, min_periods=1, center=True).mean())
    )
    return df


def pretty_reason(reason: str) -> str:
    return TERMINATION_LABELS[normalize_termination_reason(reason)]


def build_figure(case_df: pd.DataFrame, episodes_by_variant: dict[str, dict], seed: int) -> None:
    fig = plt.figure(figsize=(7.2, 6.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.92], hspace=0.24, wspace=0.18)

    plan_axes = [
        fig.add_subplot(grid[0, 0], projection="3d"),
        fig.add_subplot(grid[0, 1], projection="3d"),
    ]
    trace_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    x_values = pd.concat([case_df["ego_x_m"], case_df["target_x_m"]])
    y_values = pd.concat([case_df["ego_y_m"], case_df["target_y_m"]])
    z_values = pd.concat([case_df["ego_z_m"], case_df["target_z_m"]])
    x_pad = 0.07 * (x_values.max() - x_values.min())
    y_pad = 0.07 * (y_values.max() - y_values.min())
    z_pad = 0.08 * (z_values.max() - z_values.min())
    x_limits = (x_values.min() - x_pad, x_values.max() + x_pad)
    y_limits = (y_values.min() - y_pad, y_values.max() + y_pad)
    z_limits = (z_values.min() - z_pad, z_values.max() + z_pad)
    z_floor = z_limits[0]

    d_pad = 0.08 * (case_df["d_fwd_m"].max() - case_df["d_fwd_m"].min())
    d_limits = (case_df["d_fwd_m"].min() - d_pad, case_df["d_fwd_m"].max() + d_pad)

    for col, variant in enumerate(VARIANT_ORDER):
        episode = episodes_by_variant[variant]
        subset = case_df[case_df["variant"] == variant].copy()
        color = VARIANT_COLORS[variant]
        first_pass_step = int(episode["first_pass_step"])
        first_pass = subset[subset["step"] == first_pass_step].iloc[0]
        start_row = subset.iloc[0]
        end_row = subset.iloc[-1]

        plan_ax = plan_axes[col]
        add_ground_footprint(plan_ax, subset["ego_x_m"], subset["ego_y_m"], z_floor, color)
        plan_ax.plot(
            subset["target_x_m"],
            subset["target_y_m"],
            zs=z_floor,
            color=COUNTERPART_COLOR,
            linewidth=1.0,
            alpha=0.12,
            zorder=0,
        )
        plan_ax.plot(
            subset["ego_x_m"],
            subset["ego_y_m"],
            zs=z_floor,
            color=color,
            linewidth=1.2,
            alpha=GROUND_TRACE_ALPHA,
            zorder=0,
        )
        add_height_guides(plan_ax, [start_row, first_pass, end_row], z_floor, color)
        plan_ax.plot(
            subset["target_x_m"],
            subset["target_y_m"],
            subset["target_z_m"],
            color=COUNTERPART_COLOR,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=1,
        )
        plan_ax.plot(
            subset["ego_x_m"],
            subset["ego_y_m"],
            subset["ego_z_m"],
            color=color,
            linewidth=2.3,
            solid_capstyle="round",
            zorder=2,
        )
        plan_ax.scatter(
            [subset["ego_x_m"].iloc[0], subset["target_x_m"].iloc[0]],
            [subset["ego_y_m"].iloc[0], subset["target_y_m"].iloc[0]],
            [subset["ego_z_m"].iloc[0], subset["target_z_m"].iloc[0]],
            s=28,
            color=[color, COUNTERPART_COLOR],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        plan_ax.scatter(
            [subset["ego_x_m"].iloc[-1], subset["target_x_m"].iloc[-1]],
            [subset["ego_y_m"].iloc[-1], subset["target_y_m"].iloc[-1]],
            [subset["ego_z_m"].iloc[-1], subset["target_z_m"].iloc[-1]],
            s=46,
            marker="X",
            color=[color, COUNTERPART_COLOR],
            linewidth=0.8,
            zorder=4,
        )
        plan_ax.scatter(
            first_pass["ego_x_m"],
            first_pass["ego_y_m"],
            first_pass["ego_z_m"],
            s=42,
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        plan_ax.set_title(
            f"{variant.capitalize()} variant",
            fontsize=8.8,
            pad=8,
        )
        plan_ax.text2D(
            0.03,
            0.97,
            (
                f"{pretty_reason(episode['combat_reason'])}\n"
                f"Damage margin {episode['damage_margin']:+.1f}, duration {episode['total_time_s']:.1f} s"
            ),
            transform=plan_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#344054",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D0D5DD"},
        )
        plan_ax.set_xlim(*x_limits)
        plan_ax.set_ylim(*y_limits)
        plan_ax.set_zlim(*z_limits)
        plan_ax.set_box_aspect(
            (
                x_limits[1] - x_limits[0],
                y_limits[1] - y_limits[0],
                (z_limits[1] - z_limits[0]) * Z_ASPECT_SCALE,
            )
        )
        plan_ax.view_init(elev=TRAJECTORY_VIEW_ELEV, azim=TRAJECTORY_VIEW_AZIM)
        plan_ax.grid(color="#E7EBF0", linewidth=0.45, alpha=0.7)
        plan_ax.xaxis.pane.set_facecolor((0.985, 0.990, 0.998, 1.0))
        plan_ax.yaxis.pane.set_facecolor((0.985, 0.990, 0.998, 1.0))
        plan_ax.zaxis.pane.set_facecolor((0.958, 0.972, 0.988, 1.0))
        plan_ax.xaxis.pane.set_edgecolor("#E9EDF3")
        plan_ax.yaxis.pane.set_edgecolor("#E9EDF3")
        plan_ax.zaxis.pane.set_edgecolor("#E3E8EF")
        plan_ax.xaxis.set_major_locator(MaxNLocator(4))
        plan_ax.yaxis.set_major_locator(MaxNLocator(4))
        plan_ax.zaxis.set_major_locator(MaxNLocator(4))
        plan_ax.tick_params(axis="both", which="major", labelsize=6.8, pad=0)
        plan_ax.tick_params(axis="z", which="major", labelsize=6.8, pad=1)
        plan_ax.set_xlabel("x (m)", labelpad=1)
        if col == 0:
            plan_ax.set_ylabel("y (m)", labelpad=2)
        else:
            plan_ax.set_ylabel("")
        plan_ax.set_zlabel("z (m)", labelpad=2)
        plan_ax.text2D(
            -0.14,
            1.02,
            chr(ord("a") + col),
            transform=plan_ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        trace_ax = trace_axes[col]
        trace_ax.axvspan(0, first_pass["time_s"], color=PREMERGE_SHADE, zorder=0)
        trace_ax.axhline(0, color="#4B5563", linestyle="--", linewidth=0.9, zorder=1)
        trace_ax.axvline(first_pass["time_s"], color=FIRST_PASS_COLOR, linestyle=":", linewidth=1.0, zorder=1)
        trace_ax.plot(
            subset["time_s"],
            subset["d_fwd_m"],
            color=color,
            alpha=RAW_LINE_ALPHA,
            linewidth=1.0,
            zorder=2,
        )
        trace_ax.plot(
            subset["time_s"],
            subset["d_fwd_m_smooth"],
            color=color,
            linewidth=2.0,
            zorder=3,
        )
        trace_ax.text(
            0.03,
            0.94,
            "Shaded: pre-merge",
            transform=trace_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#475467",
        )
        trace_ax.annotate(
            "First pass",
            xy=(first_pass["time_s"], d_limits[1] * 0.80),
            xytext=(10, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7,
            color=FIRST_PASS_COLOR,
        )
        trace_ax.set_xlim(0, subset["time_s"].max() * 1.02)
        trace_ax.set_ylim(*d_limits)
        trace_ax.grid(color="#ECECEC", linewidth=0.65)
        trace_ax.set_axisbelow(True)
        trace_ax.set_xlabel("Time (s)")
        if col == 0:
            trace_ax.set_ylabel(r"$d_{\mathrm{fwd}}(t)$ (m)")
        add_panel_label(trace_ax, chr(ord("c") + col))

    fig.subplots_adjust(top=0.865, bottom=0.095)
    fig.suptitle(
        f"Representative paired case: {CONTROLLER_LABEL} / {TASK_LABEL} (same seed)",
        fontsize=10.0,
        y=0.973,
    )
    fig.text(
        0.5,
        0.943,
        "Colored line: ego trajectory. Gray line: counterpart trajectory. Circle: start. Diamond: first pass. Cross: end. Faint floor projections and dotted guide stems aid depth reading.",
        ha="center",
        va="center",
        fontsize=6.9,
        color="#475467",
    )
    fig.text(
        0.5,
        0.01,
        f"Representative seed: {seed}. Thick lines in panels c-d show a centered 1.0 s moving average of raw $d_{{\\mathrm{{fwd}}}}(t)$.",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#475467",
    )

    export_figure(fig, OUT_DIR / "figure4_representative_case")


def save_source_data(case_df: pd.DataFrame) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_rows = []
    for row in case_df.itertuples(index=False):
        export_row = source_base_fields(CONTROLLER, TASK, row.variant)
        export_row.update(
            {
                "seed": int(row.seed),
                "time_s": float(row.time_s),
                "step_index": int(row.step),
                "ego_x_m": float(row.ego_x_m),
                "ego_y_m": float(row.ego_y_m),
                "ego_z_m": float(row.ego_z_m),
                "counterpart_x_m": float(row.target_x_m),
                "counterpart_y_m": float(row.target_y_m),
                "counterpart_z_m": float(row.target_z_m),
                "forward_bias_trace_m": float(row.d_fwd_m),
                "forward_bias_trace_smooth_m": float(row.d_fwd_m_smooth),
                "range_m": float(row.range_m),
                "is_pre_merge": bool(row.pre_merge),
                "is_post_merge": bool(row.post_merge),
                "first_pass_step": int(row.first_pass_step),
                "termination_semantics": TERMINATION_LABELS[row.termination_code],
                "termination_semantics_code": row.termination_code,
                "damage_margin": float(row.damage_margin),
                "episode_duration_s": float(row.total_time_s),
            }
        )
        export_rows.append(export_row)

    pd.DataFrame(export_rows).to_csv(
        SOURCE_DATA_DIR / "figure4_representative_case.csv",
        index=False,
    )


def main() -> None:
    apply_publication_style()
    broad_eps = load_variant_episodes("broad")
    mixed_eps = load_variant_episodes("mixed")
    seed = choose_representative_seed(broad_eps, mixed_eps)
    episodes_by_variant = {
        "broad": select_episode(broad_eps, seed),
        "mixed": select_episode(mixed_eps, seed),
    }
    case_df = build_case_dataframe(episodes_by_variant)
    save_source_data(case_df)
    build_figure(case_df, episodes_by_variant, seed)
    print(f"Figure 4 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
