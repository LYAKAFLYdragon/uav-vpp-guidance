#!/usr/bin/env python3
"""Build paper-facing read-only materials from a verified Taxonomy60 analysis.

The command is deliberately downstream of the frozen formal run.  It verifies
the analysis-package hashes before producing opponent capability cards, a
minimal M0--M4 ablation report, and descriptive figures.  It never invokes the
simulator or loads a policy checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
PRIMARY_OPPONENTS = ("expert", "end_to_end")
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
METHOD_LABELS = {
    "fixed_head_on": "M0 fixed head-on",
    "fixed_crossing": "M1 fixed crossing",
    "static_oracle_gate": "M2 static gate",
    "canonical_ppo": "M3 canonical PPO",
    "ppo_unbootstrapped_crossing": "M4 PPO, no bootstrap",
}
REFERENCE_METHODS = METHODS[:3] + ("ppo_unbootstrapped_crossing",)
STATE_ORDER = ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry")
REQUIRED_ANALYSIS_FILES = (
    "analysis_manifest.json",
    "episode_index.csv",
    "method_state_opponent_summary.csv",
    "paired_ppo_deltas.csv",
    "mode_guard_transparency.csv",
    "dynamic_taxonomy_phase_episode_index.csv",
    "phase_geometry_mechanism_summary.csv",
)


class PaperMaterialsError(RuntimeError):
    """Raised when an evidence package cannot support a paper-facing export."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int:
    return int(float(value))


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _wilson(wins: int, resolved: int) -> tuple[float | None, float | None]:
    if resolved <= 0:
        return None, None
    z = 1.959963984540054
    proportion = wins / resolved
    denominator = 1.0 + z * z / resolved
    centre = (proportion + z * z / (2.0 * resolved)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / resolved + z * z / (4.0 * resolved * resolved)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise PaperMaterialsError("refusing to write an empty table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _verify_analysis_package(analysis_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ANALYSIS_FILES if not (analysis_dir / name).is_file()]
    if missing:
        raise PaperMaterialsError("analysis package is missing {}".format(missing))
    manifest = _load_json(analysis_dir / "analysis_manifest.json")
    if (
        manifest.get("source_id") != SOURCE_ID
        or manifest.get("formal_verdict") != "pass"
        or manifest.get("selected_study_route") != "paper_closure"
        or int(manifest.get("record_count", 0)) != 900
        or int(manifest.get("raw_phase_record_count", 0)) != 900
    ):
        raise PaperMaterialsError("analysis manifest does not describe the valid 900-record paper-closure source")
    for filename, expected in manifest.get("output_sha256", {}).items():
        path = analysis_dir / filename
        if not path.is_file() or _sha256(path) != expected:
            raise PaperMaterialsError("analysis output hash mismatch: {}".format(filename))
    return manifest


def _group(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    return groups


def _read_opponent_registry(formal_root: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    config_path = formal_root / "{}-expert".format(SOURCE_ID) / "resolved_config.yaml"
    if not config_path.is_file():
        raise PaperMaterialsError("formal expert resolved config is missing")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    registry = config.get("opponent_registry")
    env = config.get("env")
    if not isinstance(registry, Mapping) or not isinstance(env, Mapping):
        raise PaperMaterialsError("resolved config lacks opponent registry or environment contract")
    return dict(registry), dict(env)


def _common_reference_metrics(
    episode_rows: Sequence[Mapping[str, Any]], phase_rows: Sequence[Mapping[str, Any]], opponent: str, method: str
) -> dict[str, Any]:
    records = [row for row in episode_rows if row["opponent"] == opponent and row["method"] == method]
    phases = [row for row in phase_rows if row["opponent"] == opponent and row["method"] == method]
    if len(records) != 60 or len(phases) != 60:
        raise PaperMaterialsError("common reference matrix is incomplete for {}/{}".format(opponent, method))
    wins = sum(row["outcome"] == "win" for row in records)
    losses = sum(row["outcome"] == "loss" for row in records)
    draws = sum(row["outcome"] == "draw" for row in records)
    resolved = wins + losses
    lower, upper = _wilson(wins, resolved)
    return {
        "episodes": len(records),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "resolved_win_rate": wins / resolved if resolved else None,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "mean_hp_advantage": _mean(_float(row.get("hp_advantage")) for row in records),
        "mean_ego_hp": _mean(_float(row.get("ego_hp")) for row in records),
        "mean_target_hp": _mean(_float(row.get("target_hp")) for row in records),
        "mean_range_m": _mean(_float(row.get("mean_range_m")) for row in records),
        "mean_abs_ata_deg": _mean(_float(row.get("mean_abs_ata_deg")) for row in records),
        "mean_own_speed_mps": _mean(_float(row.get("mean_speed_mps")) for row in records),
        "mean_own_altitude_m": _mean(_float(row.get("mean_altitude_m")) for row in records),
        "mean_ego_attack_zone_fraction": _mean(_float(row.get("ego_attack_zone_fraction")) for row in phases),
        "mean_target_attack_zone_fraction": _mean(_float(row.get("target_attack_zone_fraction")) for row in phases),
        "episodes_with_post_merge": sum(_int(row["post_merge_steps"]) > 0 for row in phases),
        "episodes_with_re_entry": sum(_int(row["re_entry_steps"]) > 0 for row in phases),
        "mean_clipping_fraction": _mean(_float(row.get("clipping_fraction")) for row in phases),
    }


def build_opponent_cards(
    episode_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
    env: Mapping[str, Any],
) -> dict[str, Any]:
    cards: dict[str, Any] = {}
    for opponent in OPPONENTS:
        definition = registry.get(opponent)
        if not isinstance(definition, Mapping):
            raise PaperMaterialsError("resolved registry lacks {}".format(opponent))
        config = definition.get("config") if isinstance(definition.get("config"), Mapping) else {}
        expected_names = config.get("expected_feature_names") if isinstance(config, Mapping) else None
        cards[opponent] = {
            "opponent": opponent,
            "evidence_role": "primary" if opponent in PRIMARY_OPPONENTS else "supporting_provenance_limited",
            "implementation": {
                "type": definition.get("type"),
                "class": definition.get("class"),
                "checkpoint": definition.get("checkpoint"),
                "checkpoint_sha256": definition.get("checkpoint_sha256"),
                "delegate_class": config.get("delegate_class") if isinstance(config, Mapping) else None,
                "expected_observation_feature_count": len(expected_names) if isinstance(expected_names, list) else None,
                "expected_observation_feature_names": expected_names if isinstance(expected_names, list) else None,
                "training_provenance_boundary": (
                    "rule_based_no_checkpoint" if definition.get("type") == "rule_based"
                    else "checkpoint_is_frozen_but_training_recipe_is_not_reconstructed_by_this_formal_source"
                ),
            },
            "common_evaluation_interface": {
                "backend": "jsbsim_strict",
                "aircraft_model": env.get("aircraft_model"),
                "high_level_dt_s": env.get("high_level_dt"),
                "low_level_dt_s": env.get("low_level_dt"),
                "action_repeat": env.get("action_repeat"),
                "episode_horizon_high_level_steps": env.get("max_high_level_steps"),
            },
            "common_reference_pressure": {
                method: _common_reference_metrics(episode_rows, phase_rows, opponent, method)
                for method in ("fixed_head_on", "fixed_crossing", "static_oracle_gate")
            },
            "interpretation_boundary": (
                "This card describes pressure against the frozen common-reference controllers on the Taxonomy60 envelope. "
                "It is not an Elo score, absolute strength ranking, or a reconstruction of unrecorded training data."
            ),
        }
    return cards


def _fmt(value: Any, digits: int = 3) -> str:
    parsed = _float(value)
    return "--" if parsed is None else "{:.{}f}".format(parsed, digits)


def _card_markdown(cards: Mapping[str, Any]) -> str:
    lines = [
        "# Taxonomy60 Opponent Capability Cards v3",
        "",
        "**Source:** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`  ",
        "**Scope:** frozen common-reference controllers on the held-out Taxonomy60 envelope; not an absolute opponent ranking.",
    ]
    for opponent in OPPONENTS:
        card = cards[opponent]
        implementation = card["implementation"]
        interface = card["common_evaluation_interface"]
        lines.extend(
            [
                "",
                "## {}".format(opponent),
                "",
                "- Evidence role: `{}`.".format(card["evidence_role"]),
                "- Implementation: `{}`; `{}`.".format(implementation.get("type"), implementation.get("class")),
                "- Checkpoint SHA-256: `{}`.".format(implementation.get("checkpoint_sha256") or "not applicable"),
                "- Interface: strict JSBSim F-16, high-level {:.1f} s, low-level {:.7f} s, action repeat {}, horizon {}.".format(
                    float(interface["high_level_dt_s"]), float(interface["low_level_dt_s"]),
                    interface["action_repeat"], interface["episode_horizon_high_level_steps"],
                ),
                "- Training provenance boundary: {}".format(implementation["training_provenance_boundary"]),
                "",
                "| Common reference | W/L/D | Resolved win rate [Wilson 95% CI] | HP advantage | Target attack-zone fraction | Post-merge / re-entry episodes |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, metrics in card["common_reference_pressure"].items():
            lines.append(
                "| {} | {} / {} / {} | {} [{}, {}] | {} | {} | {} / {} |".format(
                    METHOD_LABELS[method], metrics["wins"], metrics["losses"], metrics["draws"],
                    _fmt(metrics["resolved_win_rate"]), _fmt(metrics["wilson_95_lower"]), _fmt(metrics["wilson_95_upper"]),
                    _fmt(metrics["mean_hp_advantage"], 2), _fmt(metrics["mean_target_attack_zone_fraction"]),
                    metrics["episodes_with_post_merge"], metrics["episodes_with_re_entry"],
                )
            )
        lines.append("\n{}".format(card["interpretation_boundary"]))
    return "\n".join(lines) + "\n"


def _ablation_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    mode_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Taxonomy60 M0--M4 Ablation Report",
        "",
        "The five methods share the frozen JSBSim, predictor, VPP, guidance, PID, terminal and scenario contracts. "
        "The report is descriptive: opponents are never pooled and paired deltas are not significance tests.",
        "",
        "## Overall Primary-Split Outcomes",
        "",
        "| Opponent | Method | W/L/D | Resolved win rate [Wilson 95% CI] | Guard-confounded records |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for opponent in PRIMARY_OPPONENTS:
        for method in METHODS:
            row = next(
                item for item in summary_rows
                if item["opponent"] == opponent and item["method"] == method and item["initial_class"] == "all"
            )
            lines.append(
                "| {} | {} | {} / {} / {} | {} [{}, {}] | {} |".format(
                    opponent, METHOD_LABELS[method], row["wins"], row["losses"], row["draws"],
                    _fmt(row["win_rate_resolved"]), _fmt(row["wilson_95_lower"]), _fmt(row["wilson_95_upper"]),
                    row["guard_confounded_records"],
                )
            )
    lines.extend(["", "## Paired Canonical-PPO Deltas", "", "| Opponent | Reference | Delta | Better / worse | Guard-confounded pairs |", "| --- | --- | ---: | ---: | ---: |"])
    for opponent in PRIMARY_OPPONENTS:
        for reference in REFERENCE_METHODS:
            comparison = "canonical_ppo_minus_{}".format(reference)
            row = next(item for item in paired_rows if item["opponent"] == opponent and item["initial_class"] == "all" and item["comparison"] == comparison)
            lines.append(
                "| {} | {} | {} | {} / {} | {} |".format(
                    opponent, METHOD_LABELS[reference], _fmt(row["paired_win_delta"]),
                    row["canonical_ppo_better"], row["canonical_ppo_worse"], row["guard_confounded_pair_count"],
                )
            )
    guard_counts = Counter((row["opponent"], row["method"]) for row in mode_rows if row["guard_confounded"] == "True")
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- M0 and M1 isolate the two frozen specialists; M2 is a static initial-label mapping, not a dynamic oracle.",
            "- M3 and M4 retain the deterministic recovery override. Guard-confounded M3/M4 episodes are deployed mixed-interface evidence, not pure PPO-routing causal evidence.",
            "- Guard counts are {}. A zero or small paired delta does not establish equivalence, and no single-split pattern authorizes retraining.".format(
                "; ".join("{} {}={}".format(opponent, method, count) for (opponent, method), count in sorted(guard_counts.items()))
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    if not figure.get_constrained_layout():
        figure.tight_layout()
    paths = [output_dir / "{}.png".format(stem), output_dir / "{}.pdf".format(stem)]
    figure.savefig(paths[0], dpi=350, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_heatmap(summary_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    _set_plot_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.0), sharey=True, constrained_layout=True)
    for axis, opponent in zip(axes, OPPONENTS):
        matrix: list[list[float]] = []
        annotations: list[list[str]] = []
        for method in METHODS:
            values: list[float] = []
            labels: list[str] = []
            for state in STATE_ORDER:
                row = next(item for item in summary_rows if item["opponent"] == opponent and item["method"] == method and item["initial_class"] == state)
                rate = _float(row["win_rate_resolved"])
                values.append(rate if rate is not None else math.nan)
                labels.append("{}/{}".format(row["wins"], row["n_resolved"]) if rate is not None else "--")
            matrix.append(values)
            annotations.append(labels)
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        axis.set_title(opponent.replace("_", " "))
        axis.set_xticks(range(len(STATE_ORDER)), [state.replace("_", "\n") for state in STATE_ORDER])
        axis.set_yticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
        for row_index, row in enumerate(annotations):
            for column_index, text in enumerate(row):
                axis.text(column_index, row_index, text, ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=list(axes), shrink=0.82, pad=0.018, label="Resolved win rate")
    figure.suptitle("Taxonomy60: state x opponent x frozen method (W/N resolved)", y=1.03)
    return _save_figure(figure, output_dir, "taxonomy60_state_opponent_method_heatmap")


def plot_paired_delta_forest(paired_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    _set_plot_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 7.8), sharex=True, sharey=True)
    comparisons = [(method, METHOD_LABELS[method]) for method in REFERENCE_METHODS]
    labels = ["{} vs {}".format(state, label.replace("M", "M", 1)) for state in STATE_ORDER for _, label in comparisons]
    maximum_abs_delta = max(
        abs(_float(row["paired_win_delta"]) or 0.0) for row in paired_rows
    )
    horizontal_limit = max(0.10, maximum_abs_delta + 0.05)
    for axis, opponent in zip(axes, OPPONENTS):
        plotted: list[tuple[int, float, bool]] = []
        index = 0
        for state in STATE_ORDER:
            for reference, _ in comparisons:
                row = next(item for item in paired_rows if item["opponent"] == opponent and item["initial_class"] == state and item["comparison"] == "canonical_ppo_minus_{}".format(reference))
                delta = _float(row["paired_win_delta"])
                if delta is not None:
                    plotted.append((index, delta, _int(row["guard_confounded_pair_count"]) == 0))
                index += 1
        axis.axvline(0.0, color="black", linewidth=0.8, zorder=1)
        for y, delta, unconfounded in plotted:
            axis.plot(delta, y, "D" if unconfounded else "o", color="#176D65" if unconfounded else "#9A6A37", markersize=4.5, zorder=2)
        axis.set_title(opponent.replace("_", " "))
        axis.set_xlim(-horizontal_limit, horizontal_limit)
        axis.set_xlabel("Paired win delta: M3 - reference")
        axis.grid(axis="x", color="#d7d7d7", linewidth=0.6)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    figure.suptitle("Descriptive paired deltas: diamond = no guard-confounded pair; circle = mixed-interface pair", y=1.02)
    return _save_figure(figure, output_dir, "taxonomy60_paired_delta_forest")


def plot_representative_trajectories(phase_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    """Use the lexicographically first expert/M3 episode per state, not outcome selection."""
    _set_plot_style()
    selected: list[Mapping[str, Any]] = []
    for state in STATE_ORDER:
        candidates = sorted(
            (row for row in phase_rows if row["opponent"] == "expert" and row["method"] == "canonical_ppo" and row["initial_class"] == state),
            key=lambda row: str(row["scenario_id"]),
        )
        if not candidates:
            raise PaperMaterialsError("representative selection lacks expert/M3 {}".format(state))
        selected.append(candidates[0])
    figure, axes = plt.subplots(1, 5, figsize=(15.0, 3.4), sharex=False, sharey=False)
    selection_rows: list[dict[str, Any]] = []
    for axis, row in zip(axes, selected):
        raw = _load_json(Path(row["source_raw_path"]))
        trajectory = raw.get("trajectory", [])
        ego_x = [float(frame["ego_pos_x"]) / 1000.0 for frame in trajectory]
        ego_y = [float(frame["ego_pos_y"]) / 1000.0 for frame in trajectory]
        target_x = [float(frame["target_pos_x"]) / 1000.0 for frame in trajectory]
        target_y = [float(frame["target_pos_y"]) / 1000.0 for frame in trajectory]
        axis.plot(ego_x, ego_y, color="#176D65", linewidth=1.5, label="Ego")
        axis.plot(target_x, target_y, color="#B6503C", linewidth=1.3, label="Target")
        axis.scatter([ego_x[0], target_x[0]], [ego_y[0], target_y[0]], color=["#176D65", "#B6503C"], s=16, zorder=3)
        axis.scatter([ego_x[-1], target_x[-1]], [ego_y[-1], target_y[-1]], color=["#176D65", "#B6503C"], marker="x", s=24, zorder=3)
        axis.set_title("{}\n{}".format(row["initial_class"], row["outcome"]), fontweight="bold")
        axis.set_xlabel("East (km)")
        axis.grid(color="#dedede", linewidth=0.5)
        axis.set_aspect("equal", adjustable="box")
        selection_rows.append({
            "selection_rule": "lexicographically_first_expert_canonical_ppo_per_initial_state",
            "initial_class": row["initial_class"], "scenario_id": row["scenario_id"],
            "outcome": row["outcome"], "terminal_reason": row["terminal_reason"],
            "source_raw_path": row["source_raw_path"], "source_raw_sha256": row["source_raw_sha256"],
        })
    axes[0].set_ylabel("North (km)")
    axes[0].legend(loc="best", frameon=False, fontsize=7)
    figure.suptitle("Pre-specified representative expert-opponent trajectories: canonical PPO", y=1.04)
    return _save_figure(figure, output_dir, "taxonomy60_representative_trajectories"), selection_rows


def build_materials(analysis_dir: Path, formal_root: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir).resolve()
    formal_root = Path(formal_root).resolve()
    output_dir = Path(output_dir).resolve()
    analysis_manifest = _verify_analysis_package(analysis_dir)
    if output_dir.exists():
        if not overwrite:
            raise PaperMaterialsError("output directory already exists; use --overwrite only for disposable exports")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    episode_rows = _read_csv(analysis_dir / "episode_index.csv")
    summary_rows = _read_csv(analysis_dir / "method_state_opponent_summary.csv")
    paired_rows = _read_csv(analysis_dir / "paired_ppo_deltas.csv")
    mode_rows = _read_csv(analysis_dir / "mode_guard_transparency.csv")
    phase_rows = _read_csv(analysis_dir / "dynamic_taxonomy_phase_episode_index.csv")
    if len(episode_rows) != 900 or len(phase_rows) != 900:
        raise PaperMaterialsError("analysis index is not a complete 900-record matrix")
    registry, env = _read_opponent_registry(formal_root)
    cards = build_opponent_cards(episode_rows, phase_rows, registry, env)
    (output_dir / "opponent_capability_cards_v3.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "opponent_capability_cards_v3.md").write_text(_card_markdown(cards), encoding="utf-8")
    (output_dir / "m0_m4_ablation_report.md").write_text(_ablation_markdown(summary_rows, paired_rows, mode_rows), encoding="utf-8")
    _write_csv(output_dir / "m0_m4_ablation_cells.csv", summary_rows)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    figure_paths = []
    figure_paths.extend(plot_heatmap(summary_rows, figures_dir))
    figure_paths.extend(plot_paired_delta_forest(paired_rows, figures_dir))
    trajectory_paths, selection_rows = plot_representative_trajectories(phase_rows, figures_dir)
    figure_paths.extend(trajectory_paths)
    _write_csv(output_dir / "representative_trajectory_selection.csv", selection_rows)
    report = [
        "# Taxonomy60 Paper-Closure Materials",
        "",
        "- Source: `{}`.".format(SOURCE_ID),
        "- Input analysis route: `paper_closure`, verified 900-record universe.",
        "- Outputs: three opponent cards, an M0--M4 report, complete cell table, two descriptive aggregate figures, and pre-specified representative trajectories.",
        "- Boundary: the third opponent is supporting/provenance-limited; all plots retain separate panels and no opponent pooling.",
        "- Boundary: no trajectory was simulated or altered; representative traces follow the lexicographic selection rule recorded in `representative_trajectory_selection.csv`.",
    ]
    (output_dir / "paper_materials_handoff.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_hashes = {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in sorted(output_dir.rglob("*")) if path.is_file()
    }
    manifest = {
        "source_id": SOURCE_ID,
        "kind": "read_only_paper_materials",
        "analysis_dir": str(analysis_dir),
        "analysis_manifest_sha256": _sha256(analysis_dir / "analysis_manifest.json"),
        "analysis_raw_phase_joint_sha256": analysis_manifest.get("raw_phase_joint_sha256"),
        "formal_root": str(formal_root),
        "record_count": len(episode_rows),
        "opponent_roles": {opponent: cards[opponent]["evidence_role"] for opponent in OPPONENTS},
        "figure_files": [path.relative_to(output_dir).as_posix() for path in figure_paths],
        "prohibited_actions": analysis_manifest.get("prohibited_actions", []),
        "output_sha256": output_hashes,
    }
    (output_dir / "paper_materials_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_materials(args.analysis_dir, args.formal_root, args.output_dir, overwrite=args.overwrite)
    except PaperMaterialsError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
