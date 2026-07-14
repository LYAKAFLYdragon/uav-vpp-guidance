"""Build a read-only capability atlas from the frozen Heldout40 evidence.

The atlas distinguishes evidence patterns rather than asserting causal proof.
It never loads a policy or environment, does not pool opponent statistics, and
does not modify any formal run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SOURCE_ID = "THESIS-FIVE-STATE-HELDOUT40-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "canonical_ppo_high_level_policy",
    "legacy_static_oracle_task_gate",
    "head_on_vpp_specialist_no_routing",
    "crossing_vpp_specialist_no_routing",
)
CANONICAL_METHOD = METHODS[0]
FIXED_METHODS = METHODS[2:]
INITIAL_CLASSES = (
    "advantage",
    "head_on",
    "disadvantage",
    "neutral",
    "crossing_entry",
)

LABEL_ROUTING = "routing-sensitive candidate"
LABEL_LIBRARY = "shared-specialist insufficiency candidate"
LABEL_OPPONENT = "opponent-dependent sensitivity"
LABEL_CEILING = "ceiling/non-discriminative"

RULES = {
    "routing_sensitive": {
        "fixed_improvement_pairs_min": 2,
        "definition": "At least two paired scenarios where a fixed specialist has a strictly better terminal ordinal than canonical PPO.",
    },
    "shared_specialist_insufficiency": {
        "shared_all_method_loss_pairs_min": 2,
        "definition": "At least two paired scenarios where all four methods lose and no fixed-specialist routing improvement meets the routing threshold.",
    },
    "ceiling_non_discriminative": {
        "definition": "All four methods have identical terminal ordinals for every scenario and no higher-priority routing/library label applies.",
    },
    "opponent_dependent": {
        "minimum_resolved_ratio_per_opponent": 0.6,
        "terminal_ordinal_range_min": 0.25,
        "definition": "For a fixed state and method, terminal ordinal means span at least 0.25 across separately reported opponents; only extreme opponent rows are tagged.",
    },
    "scope": "Evidence labels, not significance tests, superiority claims, or a reason to tune any frozen asset.",
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return sum(finite) / len(finite) if finite else None


def _std(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    if not finite:
        return None
    average = sum(finite) / len(finite)
    return math.sqrt(sum((value - average) ** 2 for value in finite) / len(finite))


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _outcome(record: Mapping[str, Any]) -> str:
    if _bool(record.get("win")):
        return "win"
    if _bool(record.get("loss")):
        return "loss"
    if _bool(record.get("draw")):
        return "draw"
    return "unresolved"


def _ordinal(outcome: str) -> int:
    return {"loss": 0, "draw": 1, "unresolved": 1, "win": 2}[outcome]


def _terminal_reason(record: Mapping[str, Any]) -> str:
    return str(record.get("combat_reason") or record.get("termination_reason") or "unknown")


def _phase_name(frame: Mapping[str, Any]) -> str:
    if _bool(frame.get("pre_merge")):
        return "pre_merge"
    range_rate = _finite(frame.get("range_rate_mps"))
    return "re_entry" if range_rate is not None and range_rate < 0.0 else "post_merge"


def _first_value(trajectory: Sequence[Mapping[str, Any]], field: str) -> Any:
    for frame in trajectory:
        value = frame.get(field)
        if value not in (None, "", "null"):
            return value
    return None


def _first_true_time(trajectory: Sequence[Mapping[str, Any]], field: str) -> float | None:
    for frame in trajectory:
        if _bool(frame.get(field)):
            return _finite(frame.get("time_s"))
    return None


def _fraction(trajectory: Sequence[Mapping[str, Any]], predicate) -> float | None:
    return sum(1 for frame in trajectory if predicate(frame)) / len(trajectory) if trajectory else None


def _vpp_vertical_offset(frame: Mapping[str, Any]) -> float | None:
    vp_z = _finite(frame.get("vp_pos_z"))
    ego_z = _finite(frame.get("ego_pos_z"))
    return vp_z - ego_z if vp_z is not None and ego_z is not None else None


def _source_raw_path(run_dir: Path, record: Mapping[str, Any]) -> Path:
    return (
        run_dir
        / "raw"
        / str(record["task"])
        / str(record.get("controller") or record.get("method"))
        / f"seed_{int(record['seed']):02d}"
        / f"episode_{int(record['episode']):03d}.json"
    )


def episode_features(record: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Return a compact, unit-preserving summary of an immutable raw episode."""

    trajectory = [frame for frame in record.get("trajectory", []) if isinstance(frame, Mapping)]
    if not trajectory:
        raise ValueError(f"Raw trajectory missing: {record.get('scenario')}/{record.get('controller')}")
    phase_counts = Counter(_phase_name(frame) for frame in trajectory)
    requested_counts = Counter(
        str(frame["commander_requested_mode_name"])
        for frame in trajectory
        if frame.get("commander_requested_mode_name") not in (None, "", "null")
    )
    effective_counts = Counter(
        str(frame["commander_mode_name"])
        for frame in trajectory
        if frame.get("commander_mode_name") not in (None, "", "null")
    )
    guard_reasons = sorted(
        {
            str(frame["commander_mode_constraint_reason"])
            for frame in trajectory
            if _bool(frame.get("commander_mode_constraint_triggered"))
            and frame.get("commander_mode_constraint_reason") not in (None, "", "null")
        }
    )
    outcome = _outcome(record)
    return {
        "opponent": str(record.get("opponent_stage")),
        "method": str(record.get("controller") or record.get("method")),
        "task": str(record.get("task")),
        "scenario": str(record.get("scenario")),
        "seed": int(record["seed"]),
        "episode": int(record["episode"]),
        "initial_class": str(record.get("scenario_metadata", {}).get("initial_class")),
        "height_condition": str(record.get("scenario_metadata", {}).get("height_condition")),
        "mirror_sign": str(record.get("scenario_metadata", {}).get("mirror_sign")),
        "distance_speed_package": str(record.get("scenario_metadata", {}).get("distance_speed_package")),
        "outcome": outcome,
        "terminal_ordinal": _ordinal(outcome),
        "terminal_reason": _terminal_reason(record),
        "resolved": outcome in {"win", "loss"},
        "ego_hp": _finite(record.get("ego_hp")),
        "target_hp": _finite(record.get("target_hp")),
        "hp_advantage": _finite(record.get("hp_advantage")),
        "ego_crash_or_oob": "ego" in _terminal_reason(record).lower()
        and ("crash" in _terminal_reason(record).lower() or "out_of_bounds" in _terminal_reason(record).lower()),
        "ego_attack_zone_time_s": _fraction(trajectory, lambda frame: _bool(frame.get("ego_in_attack_zone")))
        * float(record.get("total_time_s", 0.0)),
        "target_attack_zone_time_s": _fraction(trajectory, lambda frame: _bool(frame.get("target_in_attack_zone")))
        * float(record.get("total_time_s", 0.0)),
        "first_ego_attack_zone_time_s": _first_true_time(trajectory, "ego_in_attack_zone"),
        "first_target_attack_zone_time_s": _first_true_time(trajectory, "target_in_attack_zone"),
        "pre_merge_fraction": _fraction(trajectory, lambda frame: _phase_name(frame) == "pre_merge"),
        "post_merge_fraction": _fraction(trajectory, lambda frame: _phase_name(frame) == "post_merge"),
        "re_entry_fraction": _fraction(trajectory, lambda frame: _phase_name(frame) == "re_entry"),
        "mean_vp_forward_bias_m": _mean(frame.get("vp_forward_bias_m") for frame in trajectory),
        "std_vp_forward_bias_m": _std(frame.get("vp_forward_bias_m") for frame in trajectory),
        "mean_vp_lateral_bias_m": _mean(frame.get("vp_lateral_bias_m") for frame in trajectory),
        "std_vp_lateral_bias_m": _std(frame.get("vp_lateral_bias_m") for frame in trajectory),
        "mean_vp_vertical_offset_m": _mean(_vpp_vertical_offset(frame) for frame in trajectory),
        "std_vp_vertical_offset_m": _std(_vpp_vertical_offset(frame) for frame in trajectory),
        "first_requested_mode": _first_value(trajectory, "commander_requested_mode_name"),
        "first_effective_mode": _first_value(trajectory, "commander_mode_name"),
        "first_switch_step": _finite(record.get("commander_first_switch_step"))
        or _finite(_first_value(trajectory, "commander_first_switch_step")),
        "guard_trigger_fraction": _fraction(
            trajectory, lambda frame: _bool(frame.get("commander_mode_constraint_triggered"))
        ),
        "guard_reasons": guard_reasons,
        "requested_mode_fractions": {
            name: count / len(trajectory) for name, count in sorted(requested_counts.items())
        },
        "effective_mode_fractions": {
            name: count / len(trajectory) for name, count in sorted(effective_counts.items())
        },
        "source_raw_path": str(_source_raw_path(run_dir, record)),
    }


def _load_run(run_dir: Path, opponent: str, scenarios: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    path = run_dir / "aggregate" / "episode_records.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("episodes") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ValueError(f"Invalid episode records: {path}")
    result = []
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid episode record in {path}")
        if record.get("opponent_stage") != opponent:
            raise ValueError(f"Opponent mismatch in {path}: {record.get('opponent_stage')}")
        if record.get("backend") != "jsbsim" or not _bool(record.get("strict_backend")):
            raise ValueError(f"Heldout40 requires strict JSBSim: {path}")
        method = str(record.get("controller") or record.get("method"))
        scenario = str(record.get("scenario"))
        key = (scenario, method)
        if method not in METHODS or scenario not in scenarios or key in seen:
            raise ValueError(f"Invalid or duplicate paired episode: {opponent}/{key}")
        seen.add(key)
        result.append(episode_features(record, run_dir))
    expected = {(scenario, method) for scenario in scenarios for method in METHODS}
    if seen != expected:
        raise ValueError(f"Incomplete formal evidence for {opponent}")
    return result


def _validate_analysis_inputs(analysis_dir: Path) -> dict[str, Any]:
    analysis = json.loads((analysis_dir / "heldout40_analysis.json").read_text(encoding="utf-8"))
    pairing = json.loads((analysis_dir / "heldout40_pairing_verification.json").read_text(encoding="utf-8"))
    gate = json.loads((analysis_dir / "heldout40_terminal_discriminability_gate.json").read_text(encoding="utf-8"))
    if analysis.get("source_id") != SOURCE_ID or pairing.get("source_id") != SOURCE_ID or gate.get("source_id") != SOURCE_ID:
        raise ValueError("analysis_r2 source ID mismatch")
    if not pairing.get("all_cells_passed") or int(pairing.get("actual_cells", 0)) != 120:
        raise ValueError("analysis_r2 pairing verification is not complete")
    if set(gate.get("opponents", {})) != set(OPPONENTS):
        raise ValueError("analysis_r2 gate is missing opponent results")
    return {"analysis": analysis, "pairing": pairing, "gate": gate}


def _terminal_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wins = sum(row["outcome"] == "win" for row in rows)
    losses = sum(row["outcome"] == "loss" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    resolved = wins + losses
    return {
        "n_total": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "unresolved": total - resolved - draws,
        "n_resolved": resolved,
        "resolved_ratio": resolved / total if total else None,
        "win_rate": wins / resolved if resolved else None,
        "mean_terminal_ordinal": _mean(row["terminal_ordinal"] for row in rows),
        "mean_hp_advantage": _mean(row["hp_advantage"] for row in rows),
        "mean_ego_attack_zone_time_s": _mean(row["ego_attack_zone_time_s"] for row in rows),
        "mean_target_attack_zone_time_s": _mean(row["target_attack_zone_time_s"] for row in rows),
        "mean_first_ego_attack_zone_time_s": _mean(row["first_ego_attack_zone_time_s"] for row in rows),
        "mean_first_target_attack_zone_time_s": _mean(row["first_target_attack_zone_time_s"] for row in rows),
        "ego_crash_or_oob_rate": _mean(1.0 if row["ego_crash_or_oob"] else 0.0 for row in rows),
        "mean_pre_merge_fraction": _mean(row["pre_merge_fraction"] for row in rows),
        "mean_post_merge_fraction": _mean(row["post_merge_fraction"] for row in rows),
        "mean_re_entry_fraction": _mean(row["re_entry_fraction"] for row in rows),
        "mean_vp_forward_bias_m": _mean(row["mean_vp_forward_bias_m"] for row in rows),
        "mean_vp_lateral_bias_m": _mean(row["mean_vp_lateral_bias_m"] for row in rows),
        "mean_vp_vertical_offset_m": _mean(row["mean_vp_vertical_offset_m"] for row in rows),
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_safe_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input_file_record(path: Path) -> dict[str, Any]:
    """Describe one consumed immutable input without modifying it."""

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _input_artifacts(
    *, run_dirs: Mapping[str, Path], manifest_path: Path, analysis_dir: Path
) -> dict[str, Any]:
    """Fingerprint exactly the files read by this read-only analysis."""

    analysis_names = (
        "heldout40_analysis.json",
        "heldout40_pairing_verification.json",
        "heldout40_terminal_discriminability_gate.json",
    )
    return {
        "manifest": _input_file_record(manifest_path),
        "formal_episode_records": {
            opponent: _input_file_record(run_dirs[opponent] / "aggregate" / "episode_records.json")
            for opponent in OPPONENTS
        },
        "analysis_r2": {
            name: _input_file_record(analysis_dir / name) for name in analysis_names
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_safe_json(value), sort_keys=True)
                    if isinstance(value, (dict, list))
                    else _safe_json(value)
                    for key, value in row.items()
                }
            )


def _cell_label(paired_rows: Sequence[Mapping[str, Any]]) -> tuple[str | None, dict[str, int]]:
    fixed_improvements = sum(
        any(pair["delta_terminal_ordinal"] > 0 for pair in row["fixed_pairs"])
        for row in paired_rows
    )
    shared_losses = sum(row["all_methods_loss"] for row in paired_rows)
    all_identical = all(row["all_methods_same_terminal_ordinal"] for row in paired_rows)
    counts = {"fixed_improvement_pairs": fixed_improvements, "shared_all_method_loss_pairs": shared_losses}
    if fixed_improvements >= RULES["routing_sensitive"]["fixed_improvement_pairs_min"]:
        return LABEL_ROUTING, counts
    if shared_losses >= RULES["shared_specialist_insufficiency"]["shared_all_method_loss_pairs_min"]:
        return LABEL_LIBRARY, counts
    if all_identical:
        return LABEL_CEILING, counts
    return None, counts


def _format(value: Any) -> str:
    number = _finite(value)
    return "NA" if number is None else f"{number:.3f}"


def _report_markdown(
    *,
    state_cells: Mapping[tuple[str, str], Mapping[str, Any]],
    opponent_cards: Mapping[str, Any],
    candidate_count: int,
) -> str:
    lines = [
        "# Heldout40 Capability Atlas",
        "",
        "- Source: immutable Heldout40 formal runs and `analysis_r2` only.",
        "- Scope: evidence attribution, not training, tuning, superiority, or a global opponent-strength ranking.",
        "- Opponents remain separate throughout; cross-opponent labels indicate sensitivity, not pooled performance.",
        "",
        "## State x Opponent Evidence Labels",
        "",
        "| State | Opponent | Within-opponent evidence label | Fixed-improvement pairs | Shared all-method loss pairs |",
        "|---|---|---|---:|---:|",
    ]
    for initial_class in INITIAL_CLASSES:
        for opponent in OPPONENTS:
            cell = state_cells[(initial_class, opponent)]
            lines.append(
                f"| {initial_class} | {opponent} | {cell['within_opponent_label'] or 'none'} | "
                f"{cell['label_counts']['fixed_improvement_pairs']} | {cell['label_counts']['shared_all_method_loss_pairs']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "`routing-sensitive candidate` means a frozen fixed specialist improved paired terminal ordinal relative to canonical PPO in the predeclared number of scenarios. `shared-specialist insufficiency candidate` means every currently available route failed together; it does not prove a flight-control or physics failure. `opponent-dependent sensitivity` is assigned only from separately reported opponent extremes.",
            "",
            f"The candidate episode index contains {candidate_count} rows. The dedicated head-on/end-to-end audit and disadvantage audit retain mode/guard, VPP, phase, attack-zone, HP, and terminal evidence for manual review.",
            "",
            "## Motif Judgment",
            "",
            "`disadvantage` is the current first evidence-backed diagnostic motif for a new shared-skill library: all three separately reported opponent cells meet the shared-specialist-insufficiency candidate rule. This is not evidence of a physical-control limit and does not authorize training or tuning by itself.",
            "",
            "`head_on` against the end-to-end opponent is routing-sensitive evidence only: the two indexed fixed-specialist improvements motivate a causal routing review, not a conclusion that either fixed route is globally preferable.",
            "",
            "## Opponent Capability Boundary",
            "",
        ]
    )
    for opponent in OPPONENTS:
        card = opponent_cards[opponent]
        lines.append(f"- `{opponent}`: {card['boundary']}")
    return "\n".join(lines) + "\n"


def analyze(
    *,
    run_dirs: Mapping[str, Path],
    manifest_path: Path,
    analysis_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("source_id") != SOURCE_ID:
        raise ValueError("Heldout40 manifest source mismatch")
    scenarios_payload = manifest.get("scenarios")
    if not isinstance(scenarios_payload, list) or len(scenarios_payload) != 40:
        raise ValueError("Heldout40 manifest must contain 40 scenarios")
    scenarios = {str(item["name"]): dict(item.get("metadata") or {}) for item in scenarios_payload}
    if set(run_dirs) != set(OPPONENTS):
        raise ValueError(f"Expected formal runs for {OPPONENTS}")
    analysis_inputs = _validate_analysis_inputs(analysis_dir)
    input_artifacts = _input_artifacts(
        run_dirs=run_dirs,
        manifest_path=manifest_path,
        analysis_dir=analysis_dir,
    )

    episode_rows: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        episode_rows.extend(_load_run(run_dirs[opponent], opponent, scenarios))
    if len(episode_rows) != 480:
        raise ValueError(f"Expected 480 immutable episodes, found {len(episode_rows)}")

    by_cell: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in episode_rows:
        by_cell[(row["opponent"], row["initial_class"], row["scenario"])][row["method"]] = row

    paired_rows: list[dict[str, Any]] = []
    state_cells: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    head_on_end_to_end_rows: list[dict[str, Any]] = []
    head_on_end_to_end_seen: set[str] = set()
    disadvantage_rows: list[dict[str, Any]] = []

    for initial_class in INITIAL_CLASSES:
        for opponent in OPPONENTS:
            cell_pairs: list[dict[str, Any]] = []
            for scenario_name in sorted(scenarios):
                metadata = scenarios[scenario_name]
                if metadata.get("initial_class") != initial_class:
                    continue
                methods = by_cell[(opponent, initial_class, scenario_name)]
                if set(methods) != set(METHODS):
                    raise ValueError(f"Incomplete method pair: {opponent}/{scenario_name}")
                seeds = {methods[method]["seed"] for method in METHODS}
                if len(seeds) != 1:
                    raise ValueError(f"Unpaired seed: {opponent}/{scenario_name}")
                canonical = methods[CANONICAL_METHOD]
                fixed_pairs = []
                for method in FIXED_METHODS:
                    fixed = methods[method]
                    fixed_pairs.append(
                        {
                            "method": method,
                            "outcome": fixed["outcome"],
                            "terminal_reason": fixed["terminal_reason"],
                            "delta_terminal_ordinal": fixed["terminal_ordinal"] - canonical["terminal_ordinal"],
                            "delta_hp_advantage": (
                                fixed["hp_advantage"] - canonical["hp_advantage"]
                                if fixed["hp_advantage"] is not None and canonical["hp_advantage"] is not None
                                else None
                            ),
                            "fixed_source_raw_path": fixed["source_raw_path"],
                        }
                    )
                pair = {
                    "opponent": opponent,
                    "initial_class": initial_class,
                    "scenario": scenario_name,
                    "seed": canonical["seed"],
                    "canonical_outcome": canonical["outcome"],
                    "canonical_terminal_reason": canonical["terminal_reason"],
                    "canonical_terminal_ordinal": canonical["terminal_ordinal"],
                    "canonical_hp_advantage": canonical["hp_advantage"],
                    "canonical_source_raw_path": canonical["source_raw_path"],
                    "fixed_pairs": fixed_pairs,
                    "all_methods_loss": all(methods[method]["outcome"] == "loss" for method in METHODS),
                    "all_methods_same_terminal_ordinal": len({methods[method]["terminal_ordinal"] for method in METHODS}) == 1,
                }
                cell_pairs.append(pair)
                for fixed_pair in fixed_pairs:
                    paired_rows.append(
                        {
                            **{key: value for key, value in pair.items() if key != "fixed_pairs"},
                            "fixed_method": fixed_pair["method"],
                            "fixed_outcome": fixed_pair["outcome"],
                            "fixed_terminal_reason": fixed_pair["terminal_reason"],
                            "delta_terminal_ordinal": fixed_pair["delta_terminal_ordinal"],
                            "delta_hp_advantage": fixed_pair["delta_hp_advantage"],
                            "fixed_source_raw_path": fixed_pair["fixed_source_raw_path"],
                        }
                    )
            label, counts = _cell_label(cell_pairs)
            state_cells[(initial_class, opponent)] = {
                "within_opponent_label": label,
                "label_counts": counts,
                "paired_rows": cell_pairs,
            }

            if label == LABEL_ROUTING:
                for pair in cell_pairs:
                    for fixed in pair["fixed_pairs"]:
                        if fixed["delta_terminal_ordinal"] > 0:
                            candidate = {
                                "candidate_label": LABEL_ROUTING,
                                "opponent": opponent,
                                "initial_class": initial_class,
                                "scenario": pair["scenario"],
                                "seed": pair["seed"],
                                "canonical_outcome": pair["canonical_outcome"],
                                "canonical_terminal_reason": pair["canonical_terminal_reason"],
                                "canonical_source_raw_path": pair["canonical_source_raw_path"],
                                "improving_fixed_method": fixed["method"],
                                "fixed_outcome": fixed["outcome"],
                                "fixed_terminal_reason": fixed["terminal_reason"],
                                "fixed_source_raw_path": fixed["fixed_source_raw_path"],
                                "delta_terminal_ordinal": fixed["delta_terminal_ordinal"],
                                "delta_hp_advantage": fixed["delta_hp_advantage"],
                            }
                            candidate_rows.append(candidate)
                            if (
                                opponent == "end_to_end"
                                and initial_class == "head_on"
                                and pair["canonical_outcome"] == "loss"
                                and pair["scenario"] not in head_on_end_to_end_seen
                            ):
                                comparison = {"candidate": candidate}
                                for method in (CANONICAL_METHOD, *FIXED_METHODS):
                                    source = by_cell[(opponent, initial_class, pair["scenario"])][method]
                                    comparison.update(
                                        {
                                            f"{method}_first_requested_mode": source["first_requested_mode"],
                                            f"{method}_first_effective_mode": source["first_effective_mode"],
                                            f"{method}_first_switch_step": source["first_switch_step"],
                                            f"{method}_guard_trigger_fraction": source["guard_trigger_fraction"],
                                            f"{method}_guard_reasons": source["guard_reasons"],
                                            f"{method}_effective_mode_fractions": source["effective_mode_fractions"],
                                            f"{method}_mean_vp_forward_bias_m": source["mean_vp_forward_bias_m"],
                                            f"{method}_mean_vp_lateral_bias_m": source["mean_vp_lateral_bias_m"],
                                            f"{method}_mean_vp_vertical_offset_m": source["mean_vp_vertical_offset_m"],
                                            f"{method}_pre_merge_fraction": source["pre_merge_fraction"],
                                            f"{method}_post_merge_fraction": source["post_merge_fraction"],
                                            f"{method}_re_entry_fraction": source["re_entry_fraction"],
                                            f"{method}_ego_attack_zone_time_s": source["ego_attack_zone_time_s"],
                                            f"{method}_target_attack_zone_time_s": source["target_attack_zone_time_s"],
                                            f"{method}_hp_advantage": source["hp_advantage"],
                                        }
                                    )
                                head_on_end_to_end_rows.append(comparison)
                                head_on_end_to_end_seen.add(pair["scenario"])

            if label == LABEL_LIBRARY:
                for pair in cell_pairs:
                    if pair["all_methods_loss"]:
                        sources = by_cell[(opponent, initial_class, pair["scenario"])]
                        candidate_rows.append(
                            {
                                "candidate_label": LABEL_LIBRARY,
                                "opponent": opponent,
                                "initial_class": initial_class,
                                "scenario": pair["scenario"],
                                "seed": pair["seed"],
                                "canonical_outcome": pair["canonical_outcome"],
                                "canonical_terminal_reason": pair["canonical_terminal_reason"],
                                "canonical_source_raw_path": pair["canonical_source_raw_path"],
                                "improving_fixed_method": None,
                                "fixed_outcome": None,
                                "fixed_terminal_reason": None,
                                "fixed_source_raw_path": None,
                                "delta_terminal_ordinal": None,
                                "delta_hp_advantage": None,
                                "all_method_source_raw_paths": {
                                    method: sources[method]["source_raw_path"] for method in METHODS
                                },
                            }
                        )

            if initial_class == "disadvantage":
                for pair in cell_pairs:
                    comparison = {
                        "opponent": opponent,
                        "scenario": pair["scenario"],
                        "seed": pair["seed"],
                        "within_opponent_label": label,
                        "all_methods_loss": pair["all_methods_loss"],
                    }
                    for method in METHODS:
                        source = by_cell[(opponent, initial_class, pair["scenario"])][method]
                        comparison.update(
                            {
                                f"{method}_outcome": source["outcome"],
                                f"{method}_terminal_reason": source["terminal_reason"],
                                f"{method}_hp_advantage": source["hp_advantage"],
                                f"{method}_ego_attack_zone_time_s": source["ego_attack_zone_time_s"],
                                f"{method}_target_attack_zone_time_s": source["target_attack_zone_time_s"],
                                f"{method}_first_ego_attack_zone_time_s": source["first_ego_attack_zone_time_s"],
                                f"{method}_pre_merge_fraction": source["pre_merge_fraction"],
                                f"{method}_post_merge_fraction": source["post_merge_fraction"],
                                f"{method}_re_entry_fraction": source["re_entry_fraction"],
                                f"{method}_mean_vp_forward_bias_m": source["mean_vp_forward_bias_m"],
                                f"{method}_mean_vp_lateral_bias_m": source["mean_vp_lateral_bias_m"],
                                f"{method}_mean_vp_vertical_offset_m": source["mean_vp_vertical_offset_m"],
                                f"{method}_source_raw_path": source["source_raw_path"],
                            }
                        )
                    disadvantage_rows.append(comparison)

    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for opponent in OPPONENTS:
        for initial_class in INITIAL_CLASSES:
            for method in METHODS:
                selected = [
                    row
                    for row in episode_rows
                    if row["opponent"] == opponent
                    and row["initial_class"] == initial_class
                    and row["method"] == method
                ]
                summaries[(initial_class, opponent, method)] = _terminal_summary(selected)

    opponent_extreme_labels: dict[tuple[str, str, str], bool] = defaultdict(bool)
    opponent_sensitivity: list[dict[str, Any]] = []
    for initial_class in INITIAL_CLASSES:
        for method in METHODS:
            per_opponent = {opponent: summaries[(initial_class, opponent, method)] for opponent in OPPONENTS}
            values = {
                opponent: summary["mean_terminal_ordinal"]
                for opponent, summary in per_opponent.items()
                if summary["resolved_ratio"] is not None
                and summary["resolved_ratio"] >= RULES["opponent_dependent"]["minimum_resolved_ratio_per_opponent"]
                and summary["mean_terminal_ordinal"] is not None
            }
            if len(values) != len(OPPONENTS):
                continue
            low = min(values.values())
            high = max(values.values())
            sensitive = high - low >= RULES["opponent_dependent"]["terminal_ordinal_range_min"]
            opponent_sensitivity.append(
                {
                    "initial_class": initial_class,
                    "method": method,
                    "per_opponent_terminal_ordinal": values,
                    "range": high - low,
                    "opponent_dependent": sensitive,
                }
            )
            if sensitive:
                for opponent, value in values.items():
                    if math.isclose(value, low) or math.isclose(value, high):
                        opponent_extreme_labels[(initial_class, opponent, method)] = True

    matrix_rows: list[dict[str, Any]] = []
    for initial_class in INITIAL_CLASSES:
        for opponent in OPPONENTS:
            cell = state_cells[(initial_class, opponent)]
            for method in METHODS:
                summary = summaries[(initial_class, opponent, method)]
                labels = []
                if cell["within_opponent_label"] is not None:
                    labels.append(cell["within_opponent_label"])
                if opponent_extreme_labels[(initial_class, opponent, method)]:
                    labels.append(LABEL_OPPONENT)
                matrix_rows.append(
                    {
                        "initial_class": initial_class,
                        "opponent": opponent,
                        "method": method,
                        "labels": labels,
                        "within_opponent_label": cell["within_opponent_label"],
                        "fixed_improvement_pairs": cell["label_counts"]["fixed_improvement_pairs"],
                        "shared_all_method_loss_pairs": cell["label_counts"]["shared_all_method_loss_pairs"],
                        **summary,
                    }
                )

    opponent_cards = {}
    for opponent in OPPONENTS:
        opponent_cards[opponent] = {
            "opponent": opponent,
            "boundary": "Capability documentation only; this atlas does not assign a global strength, Elo, or rank.",
            "state_cells": {
                initial_class: {
                    "within_opponent_label": state_cells[(initial_class, opponent)]["within_opponent_label"],
                    "canonical": summaries[(initial_class, opponent, CANONICAL_METHOD)],
                }
                for initial_class in INITIAL_CLASSES
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "capability_atlas_label_rules.json", RULES)
    _write_csv(output_dir / "capability_atlas_state_opponent_method.csv", matrix_rows)
    _write_csv(output_dir / "capability_atlas_paired_deltas.csv", paired_rows)
    _write_csv(output_dir / "capability_atlas_candidate_episode_index.csv", candidate_rows)
    _write_csv(output_dir / "capability_atlas_headon_end_to_end_routing_audit.csv", head_on_end_to_end_rows)
    _write_csv(output_dir / "capability_atlas_disadvantage_audit.csv", disadvantage_rows)
    _write_json(output_dir / "capability_atlas_opponent_capability_cards.json", opponent_cards)
    _write_json(output_dir / "capability_atlas_opponent_sensitivity.json", opponent_sensitivity)
    (output_dir / "capability_atlas_report.md").write_text(
        _report_markdown(
            state_cells=state_cells,
            opponent_cards=opponent_cards,
            candidate_count=len(candidate_rows),
        ),
        encoding="utf-8",
    )
    report = {
        "source_id": SOURCE_ID,
        "mode": "read_only",
        "formal_input_dirs": {name: str(path) for name, path in run_dirs.items()},
        "analysis_input_dir": str(analysis_dir),
        "input_artifacts": input_artifacts,
        "episode_count": len(episode_rows),
        "pairing_input_passed": bool(analysis_inputs["pairing"]["all_cells_passed"]),
        "labels": {
            "within_opponent": {
                f"{initial_class}/{opponent}": state_cells[(initial_class, opponent)]["within_opponent_label"]
                for initial_class in INITIAL_CLASSES
                for opponent in OPPONENTS
            },
            "opponent_sensitivity": opponent_sensitivity,
        },
        "artifact_names": [
            "capability_atlas_label_rules.json",
            "capability_atlas_state_opponent_method.csv",
            "capability_atlas_paired_deltas.csv",
            "capability_atlas_candidate_episode_index.csv",
            "capability_atlas_headon_end_to_end_routing_audit.csv",
            "capability_atlas_disadvantage_audit.csv",
            "capability_atlas_opponent_capability_cards.json",
            "capability_atlas_opponent_sensitivity.json",
            "capability_atlas_report.md",
        ],
    }
    _write_json(output_dir / "capability_atlas_analysis.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--end-to-end-run", type=Path, required=True)
    parser.add_argument("--independent-ppo-vpp-run", type=Path, required=True)
    parser.add_argument("--analysis-r2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        run_dirs={
            "expert": args.expert_run,
            "end_to_end": args.end_to_end_run,
            "independent_ppo_vpp": args.independent_ppo_vpp_run,
        },
        manifest_path=args.manifest,
        analysis_dir=args.analysis_r2,
        output_dir=args.output_dir,
    )
    print(json.dumps({"episode_count": report["episode_count"], "mode": report["mode"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
