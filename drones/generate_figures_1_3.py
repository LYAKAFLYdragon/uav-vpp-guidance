import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd


# Figure contract
# Core conclusion:
#   Fig. 1: The geometry-basis interface inserts an auditable semantic layer
#   without changing the policy backbone or 3-D action head.
#   Fig. 2: Outcome rate alone is insufficient because identical or similar win
#   rates can arise from very different termination semantics.
#   Fig. 3: Forward Bias reveals large geometry shifts and helps interpret broad,
#   narrow, and mixed interface trade-offs.
# Figure archetype:
#   Fig. 1 -> schematic-led composite
#   Fig. 2 -> quantitative grid
#   Fig. 3 -> quantitative grid
# Target journal/output:
#   Drones manuscript; editable SVG and PDF plus PNG preview
# Backend:
#   Python only


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "drones" / "figures"
SOURCE_DATA_DIR = OUT_DIR / "source_data"
COMPARISON_ROOT = ROOT / "outputs" / "jsbsim_hrl_comparison"


COMPARISON_DIRS = {
    ("expert", "broad"): COMPARISON_ROOT
    / "tactical_basis_headon_mvp_combat_finetune_best_expert_10seed_20260629",
    ("expert", "narrow"): COMPARISON_ROOT
    / "tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_expert_10seed_20260629",
    ("expert", "mixed"): COMPARISON_ROOT
    / "tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_expert_10seed_20260629",
    ("end_to_end", "broad"): COMPARISON_ROOT
    / "tactical_basis_headon_mvp_combat_finetune_best_end_to_end_10seed_20260629",
    ("end_to_end", "narrow"): COMPARISON_ROOT
    / "tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_end_to_end_10seed_20260629",
    ("end_to_end", "mixed"): COMPARISON_ROOT
    / "tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_end_to_end_10seed_20260629",
}

VARIANT_ORDER = ["baseline", "broad", "narrow", "mixed"]
TASK_ORDER = ["head_on", "crossing_feasible"]
CONTROLLER_ORDER = ["expert", "end_to_end"]

VARIANT_LABELS = {
    "baseline": "Baseline",
    "broad": "Broad",
    "narrow": "Narrow",
    "mixed": "Mixed",
}

TASK_LABELS = {
    "head_on": "Frontal encounter",
    "crossing_feasible": "Crossing encounter",
}

CONTROLLER_LABELS = {
    "expert": "Expert controller",
    "end_to_end": "End-to-end learned controller",
}

VARIANT_COLORS = {
    "baseline": "#4D4D4D",
    "broad": "#B64342",
    "narrow": "#7A8E3A",
    "mixed": "#2F7F7B",
}

TERMINATION_COLORS = {
    "favorable_timeout": "#2F9E8F",
    "counterpart_crash": "#8FA8C9",
    "unfavorable_timeout": "#D7D7D7",
    "ego_crash": "#D96C5F",
}

TERMINATION_LABELS = {
    "favorable_timeout": "Favorable timeout",
    "counterpart_crash": "Counterpart crash/OOB",
    "unfavorable_timeout": "Unfavorable timeout",
    "ego_crash": "Ego crash/OOB",
}

TERMINATION_ORDER = [
    "favorable_timeout",
    "counterpart_crash",
    "unfavorable_timeout",
    "ego_crash",
]


def source_base_fields(controller_code: str, encounter_code: str, variant_code: str) -> dict[str, str]:
    return {
        "controller": CONTROLLER_LABELS[controller_code],
        "controller_code": controller_code,
        "encounter": TASK_LABELS[encounter_code],
        "encounter_code": encounter_code,
        "variant": VARIANT_LABELS[variant_code],
        "variant_code": variant_code,
    }


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def export_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def detect_variant(method_name: str) -> str:
    text = method_name.lower()
    if "mixed crossing-restored" in text or "mixed_crossing_restored" in text:
        return "mixed"
    if "narrow extents" in text or "narrow_extents" in text:
        return "narrow"
    if "combat finetune best" in text or "combat_finetune_best" in text:
        return "broad"
    if "jsbsim compare" in text or (
        "reset075_no_mode_switch_longscale00" in text and "tactical_basis" not in text
    ):
        return "baseline"
    raise ValueError(f"Unrecognized method string: {method_name}")


def normalize_termination_reason(reason: str) -> str:
    mapping = {
        "timeout_hp_advantage": "favorable_timeout",
        "timeout_hp_disadvantage": "unfavorable_timeout",
        "ego_crash_or_out_of_bounds": "ego_crash",
        "target_crash_or_out_of_bounds": "counterpart_crash",
    }
    if reason not in mapping:
        raise ValueError(f"Unexpected termination reason: {reason}")
    return mapping[reason]


def load_termination_dataframe() -> pd.DataFrame:
    rows = []
    for controller in CONTROLLER_ORDER:
        for variant_source in ["broad", "narrow", "mixed"]:
            path = COMPARISON_DIRS[(controller, variant_source)] / "aggregate" / "episode_records.json"
            obj = load_json(path)
            episodes = obj["episodes"]
            target_variants = ["baseline", variant_source] if variant_source == "broad" else [variant_source]
            for variant in target_variants:
                for task in TASK_ORDER:
                    selected = [
                        ep
                        for ep in episodes
                        if ep["task"] == task
                        and ep["opponent_stage"] == controller
                        and detect_variant(ep["method_label"]) == variant
                    ]
                    if not selected:
                        continue
                    total = len(selected)
                    favorable = sum(1 for ep in selected if ep["combat_outcome"] == "win")
                    counts = {key: 0 for key in TERMINATION_ORDER}
                    for ep in selected:
                        counts[normalize_termination_reason(ep["combat_reason"])] += 1
                    for category in TERMINATION_ORDER:
                        rows.append(
                            {
                                "controller": controller,
                                "task": task,
                                "variant": variant,
                                "category": category,
                                "count": counts[category],
                                "episodes": total,
                                "favorable_rate": favorable / total,
                            }
                        )

    df = pd.DataFrame(rows)
    df = (
        df.sort_values(["controller", "task", "variant", "category"])
        .drop_duplicates(subset=["controller", "task", "variant", "category"], keep="first")
        .reset_index(drop=True)
    )
    return df


def load_forward_bias_dataframe() -> pd.DataFrame:
    rows = []
    for controller in CONTROLLER_ORDER:
        source_map = {
            "baseline": "broad",
            "broad": "broad",
            "narrow": "narrow",
            "mixed": "mixed",
        }
        for variant in VARIANT_ORDER:
            source_variant = source_map[variant]
            path = COMPARISON_DIRS[(controller, source_variant)] / "aggregate" / "combat_geometry_diagnostics.json"
            obj = load_json(path)
            for row in obj["rows"]:
                detected = detect_variant(row["method"])
                if detected != variant:
                    continue
                if row["opponent_stage"] != controller:
                    continue
                rows.append(
                    {
                        "controller": controller,
                        "task": row["task"],
                        "variant": variant,
                        "forward_bias_m": row["pre_merge_vp_forward_bias_m"],
                    }
                )
    df = pd.DataFrame(rows)
    df = (
        df.sort_values(["controller", "task", "variant"])
        .drop_duplicates(subset=["controller", "task", "variant"], keep="first")
        .reset_index(drop=True)
    )
    return df


def build_figure_1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    neutral_fill = "#EEF2F7"
    neutral_edge = "#7A8AA1"
    basis_fill = "#F6E8EC"
    basis_edge = "#A65A73"
    diag_fill = "#E7F3F2"
    diag_edge = "#2F7F7B"
    accent = "#2F7F7B"

    def box(x, y, w, h, text, fc, ec, fontsize=8, weight="normal", align="center"):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.1,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha=align,
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color="#222222",
        )

    # Column headers
    ax.text(0.15, 0.83, "Unchanged\npolicy backbone", ha="center", va="bottom", fontsize=8.8, fontweight="bold", linespacing=0.95)
    ax.text(0.50, 0.83, "New geometry-basis\nsemantics layer", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=basis_edge, linespacing=0.95)
    ax.text(0.84, 0.83, "Existing guidance\nstack", ha="center", va="bottom", fontsize=8.8, fontweight="bold", linespacing=0.95)

    # Left column
    box(0.04, 0.63, 0.22, 0.10, "Observations", neutral_fill, neutral_edge, weight="bold")
    box(0.04, 0.49, 0.22, 0.10, "Policy network", neutral_fill, neutral_edge)
    box(0.04, 0.35, 0.22, 0.10, "3-D action head\n$[u_x, u_y, u_z]$", neutral_fill, neutral_edge)

    # Middle column
    outer = patches.FancyBboxPatch(
        (0.33, 0.29),
        0.34,
        0.48,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=basis_edge,
        facecolor=basis_fill,
    )
    ax.add_patch(outer)
    box(0.37, 0.63, 0.26, 0.08, "$a_{\\mathrm{ll}}$: lead--lag", "#FFFFFF", basis_edge, fontsize=8)
    box(0.37, 0.53, 0.26, 0.08, "$a_{\\mathrm{io}}$: inside--outside", "#FFFFFF", basis_edge, fontsize=8)
    box(0.37, 0.43, 0.26, 0.08, "$a_{\\mathrm{cd}}$: climb--descent", "#FFFFFF", basis_edge, fontsize=8)
    box(0.37, 0.32, 0.26, 0.08, "$\\Delta \\mathbf{p}^{\\mathrm{gb}} = a_{\\mathrm{ll}}\\mathbf{b}_{\\mathrm{ll}} + a_{\\mathrm{io}}\\mathbf{b}_{\\mathrm{io}} + a_{\\mathrm{cd}}\\mathbf{b}_{\\mathrm{cd}}$", "#FFFDFD", basis_edge, fontsize=8)
    ax.text(0.50, 0.27, "Task-conditioned extents and basis frames", ha="center", va="top", fontsize=7.5, color=basis_edge)

    # Right column
    box(0.74, 0.63, 0.22, 0.10, "Anchor selection", neutral_fill, neutral_edge, weight="bold")
    box(0.74, 0.49, 0.22, 0.10, "Virtual point", neutral_fill, neutral_edge)
    box(0.74, 0.35, 0.22, 0.10, "LOS guidance\n+ flight dynamics", neutral_fill, neutral_edge)

    # Bottom diagnostics
    diag = patches.FancyBboxPatch(
        (0.05, 0.06),
        0.90,
        0.15,
        boxstyle="round,pad=0.014,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=diag_edge,
        facecolor=diag_fill,
    )
    ax.add_patch(diag)
    ax.text(0.50, 0.18, "Auditable telemetry and diagnostics", ha="center", va="center", fontsize=9, fontweight="bold", color=diag_edge)
    ax.text(
        0.50,
        0.11,
        "coefficients   |   extents   |   world offset   |   Forward Bias $d_{\\mathrm{fwd}}$   |   termination semantics",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#2B4A4A",
    )

    arrow_kw = dict(arrowstyle="-|>", color=accent, linewidth=1.6, mutation_scale=12)
    ax.annotate("", xy=(0.33, 0.54), xytext=(0.26, 0.54), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.74, 0.54), xytext=(0.67, 0.54), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.50, 0.21), xytext=(0.50, 0.29), arrowprops=arrow_kw)

    ax.text(0.15, 0.30, "Policy capacity unchanged", ha="center", va="top", fontsize=7.5, color=neutral_edge)
    ax.text(0.84, 0.30, "Downstream guidance unchanged", ha="center", va="top", fontsize=7.5, color=neutral_edge)

    export_figure(fig, OUT_DIR / "figure1_geometry_basis_interface")


def build_figure_2(termination_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), sharex=True, sharey=True)
    axes = axes.flatten()

    handles = []
    labels = []
    variant_positions = range(len(VARIANT_ORDER))

    for idx, (controller, task) in enumerate(
        [(c, t) for c in CONTROLLER_ORDER for t in TASK_ORDER]
    ):
        ax = axes[idx]
        subset = termination_df[
            (termination_df["controller"] == controller) & (termination_df["task"] == task)
        ]
        bottom = [0] * len(VARIANT_ORDER)
        for category in TERMINATION_ORDER:
            counts = []
            for variant in VARIANT_ORDER:
                row = subset[
                    (subset["variant"] == variant) & (subset["category"] == category)
                ].iloc[0]
                counts.append(int(row["count"]))
            bars = ax.bar(
                variant_positions,
                counts,
                bottom=bottom,
                width=0.72,
                color=TERMINATION_COLORS[category],
                edgecolor="#444444",
                linewidth=0.6,
                label=TERMINATION_LABELS[category],
            )
            if idx == 0:
                handles.append(bars[0])
                labels.append(TERMINATION_LABELS[category])
            bottom = [b + c for b, c in zip(bottom, counts)]

        for x_pos, variant in enumerate(VARIANT_ORDER):
            rate = subset[subset["variant"] == variant]["favorable_rate"].iloc[0]
            ax.text(
                x_pos,
                10.25,
                f"$r_{{\\mathrm{{fav}}}}$={rate:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=VARIANT_COLORS[variant],
            )

        ax.set_title(
            f"{CONTROLLER_LABELS[controller]}\n{TASK_LABELS[task]}",
            fontsize=8.8,
            pad=8,
        )
        ax.set_xticks(list(variant_positions))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=7.5)
        ax.set_ylim(0, 10.8)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        ax.set_axisbelow(True)
        if idx % 2 == 0:
            ax.set_ylabel("Episodes (count of 10)", fontsize=8)
        add_panel_label(ax, chr(ord("a") + idx))

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=6.9,
        columnspacing=1.3,
        handlelength=1.5,
    )
    fig.subplots_adjust(top=0.82, wspace=0.20, hspace=0.34)
    export_figure(fig, OUT_DIR / "figure2_termination_semantics")


def build_figure_3(forward_bias_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), sharex=True, sharey=True)
    axes = axes.flatten()
    x = list(range(len(VARIANT_ORDER)))
    y_min, y_max = -750, 550

    for idx, (controller, task) in enumerate(
        [(c, t) for c in CONTROLLER_ORDER for t in TASK_ORDER]
    ):
        ax = axes[idx]
        subset = forward_bias_df[
            (forward_bias_df["controller"] == controller) & (forward_bias_df["task"] == task)
        ].copy()
        subset["variant"] = pd.Categorical(subset["variant"], categories=VARIANT_ORDER, ordered=True)
        subset = subset.sort_values("variant")
        values = subset["forward_bias_m"].tolist()

        ax.axhspan(0, y_max, color="#F6EEE8", alpha=0.45, zorder=0)
        ax.axhspan(y_min, 0, color="#EEF3F8", alpha=0.60, zorder=0)
        ax.axhline(0, color="#666666", linestyle="--", linewidth=0.9, zorder=1)
        ax.plot(x, values, color="#A0A0A0", linewidth=1.2, zorder=2)

        for xpos, variant, value in zip(x, VARIANT_ORDER, values):
            ax.vlines(xpos, 0, value, color=VARIANT_COLORS[variant], linewidth=1.1, alpha=0.85, zorder=2)
            ax.scatter(
                xpos,
                value,
                s=46,
                color=VARIANT_COLORS[variant],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            offset = 22 if value >= 0 else -28
            va = "bottom" if value >= 0 else "top"
            ax.text(
                xpos,
                value + offset,
                f"{value:.0f}",
                ha="center",
                va=va,
                fontsize=7.0,
                color=VARIANT_COLORS[variant],
            )

        if idx in (0, 2):
            ax.text(-0.40, 365, "Forward placement", fontsize=7.2, color="#8A5A40")
            ax.text(-0.40, -660, "Lagging placement", fontsize=7.2, color="#58728D")

        ax.set_title(
            f"{CONTROLLER_LABELS[controller]}\n{TASK_LABELS[task]}",
            fontsize=8.8,
            pad=8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=7.5)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([-600, -300, 0, 300, 500])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        ax.set_axisbelow(True)
        if idx % 2 == 0:
            ax.set_ylabel("Pre-merge Forward Bias (m)", fontsize=8)
        add_panel_label(ax, chr(ord("a") + idx))

    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.34)
    export_figure(fig, OUT_DIR / "figure3_forward_bias")


def save_source_data(termination_df: pd.DataFrame, forward_bias_df: pd.DataFrame) -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    termination_export = []
    for row in termination_df.itertuples(index=False):
        export_row = source_base_fields(row.controller, row.task, row.variant)
        export_row.update(
            {
                "termination_semantics": TERMINATION_LABELS[row.category],
                "termination_semantics_code": row.category,
                "termination_count": int(row.count),
                "episode_count": int(row.episodes),
                "favorable_outcome_rate": float(row.favorable_rate),
            }
        )
        termination_export.append(export_row)

    forward_bias_export = []
    for row in forward_bias_df.itertuples(index=False):
        export_row = source_base_fields(row.controller, row.task, row.variant)
        export_row.update(
            {
                "forward_bias_pre_merge_m": float(row.forward_bias_m),
            }
        )
        forward_bias_export.append(export_row)

    pd.DataFrame(termination_export).to_csv(
        SOURCE_DATA_DIR / "figure2_termination_semantics.csv",
        index=False,
    )
    pd.DataFrame(forward_bias_export).to_csv(
        SOURCE_DATA_DIR / "figure3_forward_bias.csv",
        index=False,
    )


def main() -> None:
    apply_publication_style()
    termination_df = load_termination_dataframe()
    forward_bias_df = load_forward_bias_dataframe()
    save_source_data(termination_df, forward_bias_df)
    build_figure_1()
    build_figure_2(termination_df)
    build_figure_3(forward_bias_df)
    print(f"Figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
