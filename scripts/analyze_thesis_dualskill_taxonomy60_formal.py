#!/usr/bin/env python3
"""Export read-only paper evidence from the frozen Taxonomy60 formal source.

The exporter intentionally reads the compact per-run summaries rather than
replaying, rerunning, or modifying any JSBSim episode.  It refuses any source
whose independent verifier did not select the pre-registered ``paper_closure``
route.  Guard-confounded PPO episodes remain in deployed-interface summaries,
but are flagged as ineligible for a pure routing-causality interpretation.
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
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (  # noqa: E402
    classify_relative_geometry,
)


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
PPO_METHODS = {"canonical_ppo", "ppo_unbootstrapped_crossing"}
REFERENCE_METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "ppo_unbootstrapped_crossing",
)
MANIFEST_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"


class FormalAnalysisError(RuntimeError):
    """Raised for an invalid or incomplete read-only evidence source."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def wilson_95_interval(wins: int, resolved: int) -> tuple[float | None, float | None]:
    """Return a two-sided 95% Wilson interval; draws are excluded upstream."""
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


def _scenario_metadata(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_yaml(manifest_path)
    if manifest.get("source_id") != SOURCE_ID:
        raise FormalAnalysisError("manifest Source ID mismatch")
    scenarios = manifest.get("scenarios", [])
    if len(scenarios) != 60:
        raise FormalAnalysisError("Taxonomy60 manifest must contain exactly 60 scenarios")
    result: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        name = scenario.get("name")
        metadata = scenario.get("metadata", {})
        if not isinstance(name, str) or not isinstance(metadata, Mapping):
            raise FormalAnalysisError("scenario lacks a usable name or metadata")
        required = ("initial_class", "height_condition", "mirror_sign", "task_registry_key", "scenario_seed")
        if any(key not in metadata for key in required):
            raise FormalAnalysisError("scenario {} lacks taxonomy metadata".format(name))
        result[name] = dict(metadata)
    if len(result) != 60:
        raise FormalAnalysisError("Taxonomy60 manifest scenario names are not unique")
    return result


def _guard_keys(route_decision: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for value in route_decision.get("guard_confounded_episode_ids", []):
        parts = str(value).split("|")
        if len(parts) != 3:
            raise FormalAnalysisError("malformed guard-confounded episode identity")
        keys.add((parts[0], parts[1], parts[2]))
    return keys


def _verify_analysis_entrypoint(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    verification_path = root / "formal_verification.json"
    decision_path = root / "taxonomy60_route_decision.json"
    execution_path = root / "formal_execution_manifest.json"
    if not all(path.is_file() for path in (verification_path, decision_path, execution_path)):
        raise FormalAnalysisError("formal root lacks verifier or execution receipts")
    verification = _load_json(verification_path)
    decision = _load_json(decision_path)
    execution = _load_json(execution_path)
    if verification.get("source_id") != SOURCE_ID or verification.get("verdict") != "pass":
        raise FormalAnalysisError("formal verification is not a passing Taxonomy60 receipt")
    if int(verification.get("records", 0)) != 900:
        raise FormalAnalysisError("formal verification does not cover 900 records")
    if decision.get("source_id") != SOURCE_ID or decision.get("formal_validity_verdict") != "pass":
        raise FormalAnalysisError("route decision does not describe a valid formal source")
    if decision.get("selected_study_route") != "paper_closure":
        raise FormalAnalysisError("read-only exporter is only defined for the paper-closure route")
    if execution.get("source_id") != SOURCE_ID or execution.get("execution_status") != "completed":
        raise FormalAnalysisError("formal execution receipt is incomplete")
    return verification, decision, execution


def load_summary_records(root: Path, metadata_by_scenario: Mapping[str, Mapping[str, Any]], guard_keys: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """Load and validate the 900 compact summary rows without touching episode data."""
    records: list[dict[str, Any]] = []
    expected = {(scenario, opponent, method) for scenario in metadata_by_scenario for opponent in OPPONENTS for method in METHODS}
    actual: set[tuple[str, str, str]] = set()
    for opponent in OPPONENTS:
        summary_path = root / "{}-{}".format(SOURCE_ID, opponent) / "summary.csv"
        if not summary_path.is_file():
            raise FormalAnalysisError("missing summary CSV for {}".format(opponent))
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 300:
            raise FormalAnalysisError("{} summary has {} rather than 300 records".format(opponent, len(rows)))
        for row in rows:
            scenario = row.get("scenario", "")
            method = row.get("method", "")
            key = (scenario, opponent, method)
            if scenario not in metadata_by_scenario or method not in METHODS or row.get("opponent_stage") != opponent:
                raise FormalAnalysisError("summary row lies outside the frozen record universe")
            if key in actual:
                raise FormalAnalysisError("duplicate summary record identity {}".format(key))
            if row.get("backend") != "jsbsim" or not _as_bool(row.get("strict_backend")):
                raise FormalAnalysisError("summary row violates the strict JSBSim contract")
            if _as_bool(row.get("backend_fallback_occurred")):
                raise FormalAnalysisError("summary row recorded backend fallback")
            if row.get("run_status") != "formal":
                raise FormalAnalysisError("summary row is not marked formal")
            expected_seed = int(metadata_by_scenario[scenario]["scenario_seed"])
            if int(row.get("seed", -1)) != expected_seed:
                raise FormalAnalysisError("summary row seed is not the frozen scenario seed")
            outcome = row.get("combat_outcome") or row.get("outcome")
            if outcome not in {"win", "loss", "draw"}:
                raise FormalAnalysisError("summary row has an unsupported outcome")
            normalized = {
                "source_id": SOURCE_ID,
                "scenario_id": scenario,
                "opponent": opponent,
                "method": method,
                "seed": expected_seed,
                "episode": int(row.get("episode", 0)),
                "task": row.get("task"),
                "initial_class": metadata_by_scenario[scenario]["initial_class"],
                "height_condition": metadata_by_scenario[scenario]["height_condition"],
                "mirror_sign": metadata_by_scenario[scenario]["mirror_sign"],
                "mirror_pair_id": metadata_by_scenario[scenario].get("mirror_pair_id"),
                "outcome": outcome,
                "terminal_reason": row.get("termination_reason"),
                "win": _as_bool(row.get("win")),
                "loss": _as_bool(row.get("loss")),
                "draw": _as_bool(row.get("draw")),
                "steps": _as_float(row.get("steps")),
                "total_time_s": _as_float(row.get("total_time_s")),
                "ego_hp": _as_float(row.get("ego_hp")),
                "target_hp": _as_float(row.get("target_hp")),
                "hp_advantage": _as_float(row.get("hp_advantage")),
                "mean_range_m": _as_float(row.get("mean_range_m")),
                "mean_abs_ata_deg": _as_float(row.get("mean_abs_ata_deg")),
                "mean_speed_mps": _as_float(row.get("mean_speed_mps")),
                "mean_altitude_m": _as_float(row.get("mean_altitude_m")),
                "final_speed_mps": _as_float(row.get("final_speed_mps")),
                "final_altitude_m": _as_float(row.get("final_altitude_m")),
                "envelope_fraction": _as_float(row.get("envelope_fraction")),
                "saturation_fraction": _as_float(row.get("saturation_fraction")),
                "prediction_valid_rate": _as_float(row.get("prediction_valid_rate")),
                "prediction_fallback_rate": _as_float(row.get("prediction_fallback_rate")),
                "mode_head_on_fraction": _as_float(row.get("commander_mode_head_on_fraction")),
                "mode_crossing_fraction": _as_float(row.get("commander_mode_crossing_fraction")),
                "mode_recovery_fraction": _as_float(row.get("commander_mode_recovery_fraction")),
                "mean_switch_count": _as_float(row.get("commander_mean_switch_count")),
                "first_switch_step": _as_float(row.get("commander_first_switch_step")),
                "guard_confounded": key in guard_keys,
                "routing_attribution_eligible": method not in PPO_METHODS or key not in guard_keys,
            }
            if normalized["win"] != (outcome == "win") or normalized["loss"] != (outcome == "loss"):
                raise FormalAnalysisError("summary outcome flags do not agree")
            records.append(normalized)
            actual.add(key)
    if actual != expected or len(records) != 900:
        raise FormalAnalysisError("summary records are not the exact 900-record formal matrix")
    return records


def _phase_name(frame: Mapping[str, Any]) -> str:
    """Use the existing frozen descriptive phase definition from Taxonomy30."""
    if _as_bool(frame.get("pre_merge")):
        return "pre_merge"
    range_rate = _as_float(frame.get("range_rate_mps"))
    return "re_entry" if range_rate is not None and range_rate < 0.0 else "post_merge"


def _dynamic_taxonomy(frame: Mapping[str, Any]) -> str:
    """Preserve the established `(AA, ATA)` classifier argument order."""
    return classify_relative_geometry(frame.get("aa_deg"), frame.get("ata_deg"))


def _mean_frame_values(frames: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return _mean(_as_float(frame.get(key)) for frame in frames)


def _mean_action_axis(frames: Sequence[Mapping[str, Any]], axis: int) -> float | None:
    values: list[float | None] = []
    for frame in frames:
        action = frame.get("vpp_action")
        values.append(_as_float(action[axis]) if isinstance(action, list) and len(action) == 3 else None)
    return _mean(values)


def _fraction_true(frames: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return sum(_as_bool(frame.get(key)) for frame in frames) / len(frames) if frames else None


def _first_true_step(frames: Sequence[Mapping[str, Any]], key: str) -> int | None:
    for index, frame in enumerate(frames, start=1):
        if _as_bool(frame.get(key)):
            return index
    return None


def _raw_phase_summary(
    raw: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    raw_path: Path,
    raw_sha256: str,
) -> dict[str, Any]:
    trajectory = raw.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory or not all(isinstance(frame, Mapping) for frame in trajectory):
        raise FormalAnalysisError("raw episode has no usable trajectory: {}".format(raw_path))
    frames = [dict(frame) for frame in trajectory]
    phase_counts = Counter(_phase_name(frame) for frame in frames)
    dynamic_counts = Counter(_dynamic_taxonomy(frame) for frame in frames)
    dynamic_phase_counts = Counter(
        "{}:{}".format(_dynamic_taxonomy(frame), _phase_name(frame)) for frame in frames
    )
    clipping_fraction = _mean(
        1.0
        if isinstance(frame.get("clipping"), Mapping) and _as_bool(frame["clipping"].get("any_saturated"))
        else 0.0
        for frame in frames
    )
    first = frames[0]
    last = frames[-1]
    return {
        "source_id": SOURCE_ID,
        "source_raw_path": str(raw_path),
        "source_raw_sha256": raw_sha256,
        "scenario_id": record["scenario_id"],
        "opponent": record["opponent"],
        "method": record["method"],
        "seed": record["seed"],
        "episode": record["episode"],
        "initial_class": record["initial_class"],
        "height_condition": record["height_condition"],
        "mirror_sign": record["mirror_sign"],
        "outcome": record["outcome"],
        "terminal_reason": record["terminal_reason"],
        "trajectory_steps": len(frames),
        "initial_dynamic_taxonomy": _dynamic_taxonomy(first),
        "terminal_dynamic_taxonomy": _dynamic_taxonomy(last),
        "pre_merge_steps": phase_counts["pre_merge"],
        "post_merge_steps": phase_counts["post_merge"],
        "re_entry_steps": phase_counts["re_entry"],
        "pre_merge_fraction": phase_counts["pre_merge"] / len(frames),
        "post_merge_fraction": phase_counts["post_merge"] / len(frames),
        "re_entry_fraction": phase_counts["re_entry"] / len(frames),
        "first_pass_complete_seen": any(_as_bool(frame.get("first_pass_complete")) for frame in frames),
        "first_pass_complete_step": _first_true_step(frames, "first_pass_complete"),
        "dynamic_taxonomy_step_counts": json.dumps(dict(sorted(dynamic_counts.items())), sort_keys=True),
        "dynamic_taxonomy_phase_step_counts": json.dumps(dict(sorted(dynamic_phase_counts.items())), sort_keys=True),
        "ego_attack_zone_fraction": _fraction_true(frames, "ego_in_attack_zone"),
        "target_attack_zone_fraction": _fraction_true(frames, "target_in_attack_zone"),
        "first_ego_attack_zone_step": _first_true_step(frames, "ego_in_attack_zone"),
        "first_target_attack_zone_step": _first_true_step(frames, "target_in_attack_zone"),
        "mean_range_m": _mean_frame_values(frames, "range_m"),
        "mean_range_rate_mps": _mean_frame_values(frames, "range_rate_mps"),
        "mean_ata_deg": _mean_frame_values(frames, "ata_deg"),
        "mean_aa_deg": _mean_frame_values(frames, "aa_deg"),
        "mean_speed_mps": _mean_frame_values(frames, "speed_mps"),
        "mean_altitude_m": _mean_frame_values(frames, "altitude_m"),
        "mean_vpp_action_x": _mean_action_axis(frames, 0),
        "mean_vpp_action_y": _mean_action_axis(frames, 1),
        "mean_vpp_action_z": _mean_action_axis(frames, 2),
        "mean_aoa_deg": _mean_frame_values(frames, "aoa_deg"),
        "mean_nz_command": _mean_frame_values(frames, "nz_cmd"),
        "mean_nz_actual": _mean_frame_values(frames, "nz_actual"),
        "clipping_fraction": clipping_fraction,
    }


def load_raw_phase_records(
    root: Path, summary_records: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Scan each frozen raw episode once and produce the dynamic phase contract."""
    summary_by_key = {
        (str(record["scenario_id"]), str(record["opponent"]), str(record["method"])): record
        for record in summary_records
    }
    raw_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    digest = hashlib.sha256()
    for opponent in OPPONENTS:
        raw_root = root / "{}-{}".format(SOURCE_ID, opponent) / "raw"
        raw_paths = sorted(raw_root.rglob("episode_*.json")) if raw_root.is_dir() else []
        if len(raw_paths) != 300:
            raise FormalAnalysisError("{} raw directory has {} rather than 300 episodes".format(opponent, len(raw_paths)))
        for raw_path in raw_paths:
            payload = raw_path.read_bytes()
            raw_sha256 = hashlib.sha256(payload).hexdigest()
            digest.update(raw_path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(raw_sha256.encode("ascii"))
            raw = json.loads(payload)
            if not isinstance(raw, Mapping):
                raise FormalAnalysisError("raw episode is not a JSON object")
            scenario = str(raw.get("scenario", ""))
            method = str(raw.get("controller") or raw.get("method") or "")
            key = (scenario, opponent, method)
            summary = summary_by_key.get(key)
            if summary is None or key in seen:
                raise FormalAnalysisError("raw episode does not pair one-to-one with a summary record")
            if (
                str(raw.get("opponent_stage")) != opponent
                or str(raw.get("task")) != str(summary["task"])
                or int(raw.get("seed", -1)) != int(summary["seed"])
                or str(raw.get("outcome")) != str(summary["outcome"])
                or str(raw.get("terminal_reason")) != str(summary["terminal_reason"])
            ):
                raise FormalAnalysisError("raw/summary provenance mismatch for {}".format(key))
            if str(raw.get("backend")) != "jsbsim" or not _as_bool(raw.get("strict_backend")):
                raise FormalAnalysisError("raw phase record violates the strict JSBSim contract")
            raw_records.append(_raw_phase_summary(raw, record=summary, raw_path=raw_path, raw_sha256=raw_sha256))
            seen.add(key)
    if seen != set(summary_by_key) or len(raw_records) != 900:
        raise FormalAnalysisError("raw phase records are not the exact 900-record formal matrix")
    return raw_records, digest.hexdigest()


def _group_records(
    records: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in fields)].append(record)
    return groups


def result_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize results without pooling opponents; draws stay outside Wilson denominators."""
    summary_rows: list[dict[str, Any]] = []
    for include_all in (False, True):
        source = records if not include_all else [dict(record, initial_class="all") for record in records]
        for (initial_class, opponent, method), group in sorted(_group_records(source, ("initial_class", "opponent", "method")).items()):
            wins = sum(bool(record["win"]) for record in group)
            losses = sum(bool(record["loss"]) for record in group)
            draws = sum(bool(record["draw"]) for record in group)
            resolved = wins + losses
            lower, upper = wilson_95_interval(wins, resolved)
            summary_rows.append(
                {
                    "initial_class": initial_class,
                    "opponent": opponent,
                    "opponent_role": "primary" if opponent in PRIMARY_OPPONENTS else "supporting_provenance_limited",
                    "method": method,
                    "n_total": len(group),
                    "n_resolved": resolved,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate_resolved": wins / resolved if resolved else None,
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                    "mean_range_m": _mean(record["mean_range_m"] for record in group),
                    "mean_abs_ata_deg": _mean(record["mean_abs_ata_deg"] for record in group),
                    "mean_ego_hp": _mean(record["ego_hp"] for record in group),
                    "mean_target_hp": _mean(record["target_hp"] for record in group),
                    "mean_hp_advantage": _mean(record["hp_advantage"] for record in group),
                    "mean_speed_mps": _mean(record["mean_speed_mps"] for record in group),
                    "mean_altitude_m": _mean(record["mean_altitude_m"] for record in group),
                    "mean_saturation_fraction": _mean(record["saturation_fraction"] for record in group),
                    "guard_confounded_records": sum(bool(record["guard_confounded"]) for record in group),
                }
            )
    return summary_rows


def dynamic_taxonomy_phase_rows(raw_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the frozen dynamic taxonomy x descriptive phase occupancy."""
    group_totals: Counter[tuple[str, str, str]] = Counter()
    occupancy: Counter[tuple[str, str, str, str, str]] = Counter()
    episode_presence: Counter[tuple[str, str, str, str, str]] = Counter()
    for record in raw_records:
        group = (str(record["initial_class"]), str(record["opponent"]), str(record["method"]))
        group_totals[group] += int(record["trajectory_steps"])
        dynamic_phase = json.loads(str(record["dynamic_taxonomy_phase_step_counts"]))
        for compound, count in dynamic_phase.items():
            dynamic_taxonomy, phase = compound.split(":", 1)
            key = (*group, dynamic_taxonomy, phase)
            occupancy[key] += int(count)
            if int(count) > 0:
                episode_presence[key] += 1
    rows: list[dict[str, Any]] = []
    for key, step_count in sorted(occupancy.items()):
        initial_class, opponent, method, dynamic_taxonomy, phase = key
        total_steps = group_totals[(initial_class, opponent, method)]
        rows.append(
            {
                "initial_class": initial_class,
                "opponent": opponent,
                "opponent_role": "primary" if opponent in PRIMARY_OPPONENTS else "supporting_provenance_limited",
                "method": method,
                "dynamic_taxonomy": dynamic_taxonomy,
                "phase": phase,
                "step_count": step_count,
                "group_total_steps": total_steps,
                "step_fraction_within_state_opponent_method": step_count / total_steps if total_steps else None,
                "episode_count_with_occupancy": episode_presence[key],
                "phase_semantics": "pre_merge; post_merge; or post_merge_closing_re_entry",
            }
        )
    return rows


def phase_geometry_mechanism_rows(raw_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize phase, geometry, VPP and command/response telemetry by formal cell."""
    rows: list[dict[str, Any]] = []
    for (initial_class, opponent, method), group in sorted(
        _group_records(raw_records, ("initial_class", "opponent", "method")).items()
    ):
        rows.append(
            {
                "initial_class": initial_class,
                "opponent": opponent,
                "opponent_role": "primary" if opponent in PRIMARY_OPPONENTS else "supporting_provenance_limited",
                "method": method,
                "episode_count": len(group),
                "episodes_with_first_pass": sum(bool(record["first_pass_complete_seen"]) for record in group),
                "episodes_with_post_merge": sum(int(record["post_merge_steps"]) > 0 for record in group),
                "episodes_with_re_entry": sum(int(record["re_entry_steps"]) > 0 for record in group),
                "mean_pre_merge_fraction": _mean(record["pre_merge_fraction"] for record in group),
                "mean_post_merge_fraction": _mean(record["post_merge_fraction"] for record in group),
                "mean_re_entry_fraction": _mean(record["re_entry_fraction"] for record in group),
                "mean_ego_attack_zone_fraction": _mean(record["ego_attack_zone_fraction"] for record in group),
                "mean_target_attack_zone_fraction": _mean(record["target_attack_zone_fraction"] for record in group),
                "mean_range_m": _mean(record["mean_range_m"] for record in group),
                "mean_range_rate_mps": _mean(record["mean_range_rate_mps"] for record in group),
                "mean_ata_deg": _mean(record["mean_ata_deg"] for record in group),
                "mean_aa_deg": _mean(record["mean_aa_deg"] for record in group),
                "mean_speed_mps": _mean(record["mean_speed_mps"] for record in group),
                "mean_altitude_m": _mean(record["mean_altitude_m"] for record in group),
                "mean_vpp_action_x": _mean(record["mean_vpp_action_x"] for record in group),
                "mean_vpp_action_y": _mean(record["mean_vpp_action_y"] for record in group),
                "mean_vpp_action_z": _mean(record["mean_vpp_action_z"] for record in group),
                "mean_aoa_deg": _mean(record["mean_aoa_deg"] for record in group),
                "mean_nz_command": _mean(record["mean_nz_command"] for record in group),
                "mean_nz_actual": _mean(record["mean_nz_actual"] for record in group),
                "mean_clipping_fraction": _mean(record["clipping_fraction"] for record in group),
                "interpretation_boundary": "descriptive phase/geometry telemetry; not a safety certificate or causal routing estimate",
            }
        )
    return rows


def paired_deltas(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return paired canonical-PPO deltas by state and opponent, never pooled."""
    by_cell: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        by_cell[(str(record["scenario_id"]), str(record["opponent"]))][str(record["method"])] = record
    rows: list[dict[str, Any]] = []
    for state in ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry", "all"):
        for opponent in OPPONENTS:
            cells = []
            for (scenario, cell_opponent), methods in by_cell.items():
                if cell_opponent != opponent or set(methods) != set(METHODS):
                    continue
                canonical = methods["canonical_ppo"]
                if state != "all" and canonical["initial_class"] != state:
                    continue
                cells.append(methods)
            for reference in REFERENCE_METHODS:
                eligible = [
                    methods
                    for methods in cells
                    if methods["canonical_ppo"]["outcome"] in {"win", "loss"}
                    and methods[reference]["outcome"] in {"win", "loss"}
                ]
                unconfounded = [
                    methods
                    for methods in eligible
                    if methods["canonical_ppo"]["routing_attribution_eligible"]
                    and methods[reference]["routing_attribution_eligible"]
                ]
                ppo_better = sum(
                    methods["canonical_ppo"]["win"] and methods[reference]["loss"] for methods in eligible
                )
                ppo_worse = sum(
                    methods["canonical_ppo"]["loss"] and methods[reference]["win"] for methods in eligible
                )
                rows.append(
                    {
                        "initial_class": state,
                        "opponent": opponent,
                        "opponent_role": "primary" if opponent in PRIMARY_OPPONENTS else "supporting_provenance_limited",
                        "comparison": "canonical_ppo_minus_{}".format(reference),
                        "n_cells": len(cells),
                        "n_pair_resolved": len(eligible),
                        "canonical_ppo_better": ppo_better,
                        "canonical_ppo_worse": ppo_worse,
                        "same_outcome": len(eligible) - ppo_better - ppo_worse,
                        "paired_win_delta": (ppo_better - ppo_worse) / len(eligible) if eligible else None,
                        "n_unconfounded_pair_resolved": len(unconfounded),
                        "guard_confounded_pair_count": len(eligible) - len(unconfounded),
                        "interpretation": "descriptive_paired_delta_not_significance_test",
                    }
                )
    return rows


def terminal_safety_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for include_all in (False, True):
        source = records if not include_all else [dict(record, initial_class="all") for record in records]
        for (initial_class, opponent, method), group in sorted(_group_records(source, ("initial_class", "opponent", "method")).items()):
            reasons = Counter(str(record["terminal_reason"]) for record in group)
            for reason, count in sorted(reasons.items()):
                rows.append(
                    {
                        "initial_class": initial_class,
                        "opponent": opponent,
                        "opponent_role": "primary" if opponent in PRIMARY_OPPONENTS else "supporting_provenance_limited",
                        "method": method,
                        "terminal_reason": reason,
                        "episode_count": count,
                        "fraction_of_group": count / len(group),
                        "safety_semantics": "timeout_outcome_not_kill_rate_or_safety_certification"
                        if reason.startswith("timeout")
                        else "inspect_terminal_actor_before_any_safety_interpretation",
                    }
                )
    return rows


def mode_guard_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["method"] not in PPO_METHODS:
            continue
        rows.append(
            {
                "scenario_id": record["scenario_id"],
                "initial_class": record["initial_class"],
                "height_condition": record["height_condition"],
                "mirror_sign": record["mirror_sign"],
                "opponent": record["opponent"],
                "opponent_role": "primary" if record["opponent"] in PRIMARY_OPPONENTS else "supporting_provenance_limited",
                "method": record["method"],
                "outcome": record["outcome"],
                "terminal_reason": record["terminal_reason"],
                "effective_head_on_fraction": record["mode_head_on_fraction"],
                "effective_crossing_fraction": record["mode_crossing_fraction"],
                "effective_recovery_fraction": record["mode_recovery_fraction"],
                "mean_switch_count": record["mean_switch_count"],
                "first_switch_step": record["first_switch_step"],
                "guard_confounded": record["guard_confounded"],
                "pure_routing_attribution_eligible": record["routing_attribution_eligible"],
                "note": "guard-confounded rows remain deployed-interface evidence only",
            }
        )
    return rows


def candidate_rows(records: Sequence[Mapping[str, Any]], route_decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = {(record["scenario_id"], record["opponent"], record["method"]): record for record in records}
    rows: list[dict[str, Any]] = []
    for cell in route_decision.get("cell_classifications", []):
        if "routing_sensitive_candidate" not in cell.get("labels", []):
            continue
        scenario = cell["scenario_id"]
        opponent = cell["opponent"]
        for method in METHODS:
            record = by_id[(scenario, opponent, method)]
            rows.append(
                {
                    "scenario_id": scenario,
                    "initial_class": record["initial_class"],
                    "height_condition": record["height_condition"],
                    "mirror_sign": record["mirror_sign"],
                    "opponent": opponent,
                    "opponent_role": "primary" if opponent in PRIMARY_OPPONENTS else "supporting_provenance_limited",
                    "method": method,
                    "outcome": record["outcome"],
                    "terminal_reason": record["terminal_reason"],
                    "guard_confounded": record["guard_confounded"],
                    "pure_routing_attribution_eligible": record["routing_attribution_eligible"],
                }
            )
    return rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fieldnames: Sequence[str] | None = None,
) -> None:
    """Write a table, preserving a header-only CSV for a valid zero-candidate result."""
    if not rows and not empty_fieldnames:
        raise FormalAnalysisError("refusing to write an empty analysis table without a schema")
    fieldnames = list(rows[0].keys()) if rows else list(empty_fieldnames or ())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    verification: Mapping[str, Any],
    decision: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    guard_count: int,
    raw_phase_digest: str,
) -> None:
    primary = [record for record in records if record["opponent"] in PRIMARY_OPPONENTS]
    outcomes = Counter((record["opponent"], record["outcome"]) for record in primary)
    lines = [
        "# Taxonomy60 Formal Evidence Closure",
        "",
        "**Source ID:** `{}`  ".format(SOURCE_ID),
        "**Route:** `{}`  ".format(decision["selected_study_route"]),
        "**Scope:** read-only export from the verified 900-record universe; no simulator, training, tuning, or result-driven rerun was run.",
        "",
        "## Independent Entry Check",
        "",
        "- Formal verifier verdict: `{}` with `{}` records.".format(verification["verdict"], verification["records"]),
        "- The pre-registered decision selects `paper_closure`; therefore training, tuning, scenario replacement, and result-driven reruns remain prohibited.",
        "- Primary evidence is reported separately for `expert` and `end_to_end`; `independent_ppo_vpp` remains supporting, provenance-limited evidence.",
        "- {} PPO records are guard-confounded. They remain deployed mixed-interface evidence but are excluded from pure routing-causality interpretation.".format(guard_count),
        "- The phase table scans all 900 raw trajectories under the existing `(AA, ATA)` taxonomy and pre-merge/post-merge/re-entry definitions; raw joint digest: `{}`.".format(raw_phase_digest),
        "",
        "## Primary Outcome Counts",
        "",
        "| Opponent | Wins | Losses | Draws |",
        "| --- | ---: | ---: | ---: |",
    ]
    for opponent in PRIMARY_OPPONENTS:
        lines.append(
            "| {} | {} | {} | {} |".format(
                opponent,
                outcomes[(opponent, "win")],
                outcomes[(opponent, "loss")],
                outcomes[(opponent, "draw")],
            )
        )
    lines.extend(
        [
            "",
            "These are all-method totals, not policy-level superiority claims. Policy comparisons are retained only in the paired, split-specific tables.",
            "",
            "## Paper-Closure Handoff",
            "",
            "1. Use `method_state_opponent_summary.csv` for the main/supplementary outcome table; Wilson intervals exclude draws from their denominator.",
            "2. Use `paired_ppo_deltas.csv` descriptively, never as a pooled significance test or generalization claim.",
            "3. Use `mode_guard_transparency.csv` and `routing_sensitive_candidates.csv` to state the recovery override boundary honestly.",
            "4. Use `dynamic_taxonomy_phase_episode_index.csv` and `phase_geometry_mechanism_summary.csv` only as descriptive trajectory evidence; phase labels are not extra training tasks.",
            "5. Use `terminal_safety_semantics.csv` to avoid describing timeout HP outcomes as kill rate, aerodynamic superiority, or safety certification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_analysis(root: Path, analysis_dir: Path, *, manifest_path: Path = MANIFEST_PATH, overwrite: bool = False) -> dict[str, Any]:
    """Export the complete read-only paper-closure package and return its manifest."""
    root = Path(root).resolve()
    analysis_dir = Path(analysis_dir).resolve()
    verification, decision, execution = _verify_analysis_entrypoint(root)
    metadata = _scenario_metadata(Path(manifest_path).resolve())
    guard = _guard_keys(decision)
    records = load_summary_records(root, metadata, guard)
    raw_phase_records, raw_phase_digest = load_raw_phase_records(root, records)
    if analysis_dir.exists():
        if not overwrite:
            raise FormalAnalysisError("analysis directory already exists; use --overwrite only for disposable exports")
        shutil.rmtree(analysis_dir)
    analysis_dir.mkdir(parents=True)
    tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "episode_index.csv": records,
        "method_state_opponent_summary.csv": result_summary(records),
        "paired_ppo_deltas.csv": paired_deltas(records),
        "terminal_safety_semantics.csv": terminal_safety_rows(records),
        "mode_guard_transparency.csv": mode_guard_rows(records),
        "routing_sensitive_candidates.csv": candidate_rows(records, decision),
        "dynamic_taxonomy_phase_episode_index.csv": raw_phase_records,
        "dynamic_taxonomy_phase_summary.csv": dynamic_taxonomy_phase_rows(raw_phase_records),
        "phase_geometry_mechanism_summary.csv": phase_geometry_mechanism_rows(raw_phase_records),
    }
    for filename, rows in tables.items():
        _write_csv(
            analysis_dir / filename,
            rows,
            empty_fieldnames=(
                (
                    "scenario_id", "initial_class", "height_condition", "mirror_sign", "opponent",
                    "opponent_role", "method", "outcome", "terminal_reason", "guard_confounded",
                    "pure_routing_attribution_eligible",
                )
                if filename == "routing_sensitive_candidates.csv"
                else None
            ),
        )
    _write_markdown(
        analysis_dir / "paper_closure_handoff.md",
        verification=verification,
        decision=decision,
        records=records,
        guard_count=len(guard),
        raw_phase_digest=raw_phase_digest,
    )
    inputs = {
        "formal_verification.json": _sha256(root / "formal_verification.json"),
        "taxonomy60_route_decision.json": _sha256(root / "taxonomy60_route_decision.json"),
        "formal_execution_manifest.json": _sha256(root / "formal_execution_manifest.json"),
        "taxonomy60_manifest.yaml": _sha256(Path(manifest_path).resolve()),
    }
    for opponent in OPPONENTS:
        inputs["{}_summary.csv".format(opponent)] = _sha256(root / "{}-{}".format(SOURCE_ID, opponent) / "summary.csv")
    output_hashes = {path.name: _sha256(path) for path in sorted(analysis_dir.iterdir()) if path.is_file()}
    artifact = {
        "source_id": SOURCE_ID,
        "analysis_kind": "read_only_paper_closure_export",
        "formal_verdict": verification["verdict"],
        "selected_study_route": decision["selected_study_route"],
        "record_count": len(records),
        "raw_phase_record_count": len(raw_phase_records),
        "raw_phase_joint_sha256": raw_phase_digest,
        "guard_confounded_record_count": len(guard),
        "execution_manifest_payload_sha256": execution.get("manifest_payload_sha256"),
        "input_sha256": inputs,
        "output_sha256": output_hashes,
        "primary_opponents": list(PRIMARY_OPPONENTS),
        "supporting_opponents": ["independent_ppo_vpp"],
        "prohibited_actions": decision.get("prohibited_actions", []),
    }
    (analysis_dir / "analysis_manifest.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="verified Taxonomy60 formal root")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="new analysis directory")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing disposable analysis directory")
    args = parser.parse_args()
    try:
        artifact = export_analysis(args.output_root, args.analysis_dir, manifest_path=args.manifest, overwrite=args.overwrite)
    except FormalAnalysisError as error:
        parser.error(str(error))
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
