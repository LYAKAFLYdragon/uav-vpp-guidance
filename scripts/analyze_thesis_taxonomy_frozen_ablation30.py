#!/usr/bin/env python3
"""Read-only Step 4 audit for the frozen thesis taxonomy ablation.

The analyzer deliberately consumes only raw evaluation artifacts.  It never
loads a policy, creates an environment, changes a configuration, or writes to
either formal run directory.  Its diagnostic labels are engineering triage
labels, not significance tests or retrospective policy baselines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (  # noqa: E402
    classify_relative_geometry,
)


ANALYSIS_VERSION = "thesis_taxonomy_ablation30_step4_v2"
SOURCE_ID = "THESIS-TAXONOMY-ABLATION30-V1"
METHODS = (
    "canonical_ppo_high_level_policy",
    "legacy_static_oracle_task_gate",
    "head_on_vpp_specialist_no_routing",
    "crossing_vpp_specialist_no_routing",
)
PPO_METHOD = METHODS[0]
ORACLE_METHOD = METHODS[1]
FIXED_METHODS = METHODS[2:]
INITIAL_CLASSES = (
    "advantage",
    "head_on",
    "disadvantage",
    "neutral",
    "crossing_entry",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(map(str, value))
                    if isinstance(value, (list, tuple, set))
                    else _json_safe(value)
                    for key, value in row.items()
                }
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_from_frame(frame: Mapping[str, Any], prefix: str) -> tuple[float, float, float] | None:
    values = [_finite(frame.get(f"{prefix}_pos_{axis}")) for axis in ("x", "y", "z")]
    if any(value is None for value in values):
        sequence = frame.get(f"{prefix}_pos_m")
        if isinstance(sequence, Sequence) and len(sequence) == 3:
            values = [_finite(item) for item in sequence]
    if any(value is None for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _subtract(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))  # type: ignore[return-value]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _angle_deg(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return None
    cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _velocity_between(
    first: Mapping[str, Any], second: Mapping[str, Any], prefix: str
) -> tuple[float, float, float] | None:
    first_pos = _vector_from_frame(first, prefix)
    second_pos = _vector_from_frame(second, prefix)
    if first_pos is None or second_pos is None:
        return None
    first_time = _finite(first.get("time_s"))
    second_time = _finite(second.get("time_s"))
    dt = (second_time - first_time) if first_time is not None and second_time is not None else None
    if dt is None or dt <= 1e-9:
        return None
    return tuple((next_value - current) / dt for current, next_value in zip(first_pos, second_pos))  # type: ignore[return-value]


def _initial_geometry(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute initial angles from positions over the first two logged steps."""

    if len(trajectory) < 2:
        return {
            "source": "unavailable",
            "geometry_state": "unknown",
            "own_to_target_los_angle_deg": None,
            "target_velocity_to_own_los_angle_deg": None,
            "los_elevation_deg": None,
            "specific_energy_difference_m2_s2": None,
        }
    first, second = trajectory[0], trajectory[1]
    own_position = _vector_from_frame(first, "ego")
    target_position = _vector_from_frame(first, "target")
    own_velocity = _velocity_between(first, second, "ego")
    target_velocity = _velocity_between(first, second, "target")
    if None in (own_position, target_position, own_velocity, target_velocity):
        return {
            "source": "unavailable",
            "geometry_state": "unknown",
            "own_to_target_los_angle_deg": None,
            "target_velocity_to_own_los_angle_deg": None,
            "los_elevation_deg": None,
            "specific_energy_difference_m2_s2": None,
        }
    own_to_target = _subtract(target_position, own_position)
    target_to_own = _subtract(own_position, target_position)
    own_angle = _angle_deg(own_velocity, own_to_target)
    target_angle = _angle_deg(target_velocity, target_to_own)
    horizontal_range = math.hypot(own_to_target[0], own_to_target[1])
    los_elevation = math.degrees(math.atan2(own_to_target[2], horizontal_range))
    own_energy = 9.80665 * own_position[2] + 0.5 * _norm(own_velocity) ** 2
    target_energy = 9.80665 * target_position[2] + 0.5 * _norm(target_velocity) ** 2
    return {
        "source": "first_two_trajectory_positions",
        "geometry_state": classify_relative_geometry(own_angle, target_angle),
        "own_to_target_los_angle_deg": own_angle,
        "target_velocity_to_own_los_angle_deg": target_angle,
        "los_elevation_deg": los_elevation,
        "specific_energy_difference_m2_s2": own_energy - target_energy,
    }


def _geometry_state_from_frame(frame: Mapping[str, Any]) -> str:
    # In this telemetry contract AA is the own-velocity-to-target-LOS angle,
    # while ATA is the target-velocity-to-own-LOS angle used by taxonomy v1.
    return classify_relative_geometry(frame.get("aa_deg"), frame.get("ata_deg"))


def _phase_name(frame: Mapping[str, Any]) -> str:
    if _bool(frame.get("pre_merge")):
        return "pre_merge"
    if _bool(frame.get("post_merge")):
        range_rate = _finite(frame.get("range_rate_mps"))
        return "re_entry" if range_rate is not None and range_rate < 0.0 else "post_merge"
    return "unclassified"


def _terminal_category(raw_reason: Any) -> str:
    reason = str(raw_reason or "unknown").lower()
    if "ego" in reason and ("crash" in reason or "out_of_bounds" in reason or "oob" in reason):
        return "ego_crash_or_oob"
    if "target" in reason and ("crash" in reason or "out_of_bounds" in reason or "oob" in reason):
        return "target_crash_or_oob"
    if "ego" in reason and "killed" in reason:
        return "ego_killed"
    if "target" in reason and "killed" in reason:
        return "target_killed"
    if "timeout" in reason and "advantage" in reason:
        return "timeout_hp_advantage"
    if "timeout" in reason and "disadvantage" in reason:
        return "timeout_hp_disadvantage"
    if "timeout" in reason and ("draw" in reason or "tie" in reason):
        return "timeout_draw"
    if "timeout" in reason:
        return "raw_timeout"
    return "other_or_unknown"


def _outcome(record: Mapping[str, Any]) -> str:
    if record["win"]:
        return "win"
    if record["loss"]:
        return "loss"
    if record["draw"]:
        return "draw"
    return "unresolved"


def _mean_finite(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return sum(finite) / len(finite) if finite else None


def _mode_counts(trajectory: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for frame in trajectory:
        value = frame.get(field)
        if value not in (None, "", "null"):
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _unique_values(trajectory: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    return sorted(
        {
            str(frame[field])
            for frame in trajectory
            if frame.get(field) not in (None, "", "null")
        }
    )


def _first_value(trajectory: Sequence[Mapping[str, Any]], field: str) -> Any:
    for frame in trajectory:
        value = frame.get(field)
        if value not in (None, "", "null"):
            return value
    return None


def _min_finite(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return min(finite) if finite else None


def _max_finite(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return max(finite) if finite else None


def _first_true_time_s(trajectory: Sequence[Mapping[str, Any]], field: str) -> float | None:
    for frame in trajectory:
        if _bool(frame.get(field)):
            return _finite(frame.get("time_s"))
    return None


def _true_fraction(trajectory: Sequence[Mapping[str, Any]], field: str) -> float | None:
    if not trajectory or not any(field in frame for frame in trajectory):
        return None
    return sum(_bool(frame.get(field)) for frame in trajectory) / len(trajectory)


def _mode_step_fractions(trajectory: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    counts = _mode_counts(trajectory, field)
    total = sum(counts.values())
    return {mode: count / total for mode, count in counts.items()} if total else {}


def _first_switch_step(trajectory: Sequence[Mapping[str, Any]]) -> int | None:
    for frame in trajectory:
        value = _finite(frame.get("commander_first_switch_step"))
        if value is not None:
            return int(value)
    for frame in trajectory:
        if _bool(frame.get("commander_mode_switched")):
            value = _finite(frame.get("step"))
            return int(value) if value is not None else None
    return None


def _target_speed_mps(trajectory: Sequence[Mapping[str, Any]]) -> list[float]:
    speeds: list[float] = []
    for first, second in zip(trajectory, trajectory[1:]):
        velocity = _velocity_between(first, second, "target")
        if velocity is not None:
            speeds.append(_norm(velocity))
    return speeds


def _vertical_vpp_offset_m(frame: Mapping[str, Any]) -> float | None:
    vp_z = _finite(frame.get("vp_pos_z"))
    ego_z = _finite(frame.get("ego_pos_z"))
    if vp_z is None or ego_z is None:
        return None
    return vp_z - ego_z


def _trajectory_telemetry(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compress one raw trajectory into paper-facing, unit-preserving features."""

    initial_range = _finite(trajectory[0].get("range_m"))
    final_range = _finite(trajectory[-1].get("range_m"))
    terminal_geometry = _initial_geometry(trajectory[-2:]) if len(trajectory) >= 2 else {}
    effective_fractions = _mode_step_fractions(trajectory, "commander_mode_name")
    requested_fractions = _mode_step_fractions(trajectory, "commander_requested_mode_name")
    has_mode_telemetry = bool(effective_fractions or requested_fractions)
    return {
        "initial_range_m": initial_range,
        "final_range_m": final_range,
        "minimum_range_m": _min_finite(frame.get("range_m") for frame in trajectory),
        "mean_range_m": _mean_finite(frame.get("range_m") for frame in trajectory),
        "mean_range_rate_mps": _mean_finite(frame.get("range_rate_mps") for frame in trajectory),
        "minimum_range_rate_mps": _min_finite(frame.get("range_rate_mps") for frame in trajectory),
        "maximum_range_rate_mps": _max_finite(frame.get("range_rate_mps") for frame in trajectory),
        "ego_attack_zone_entry_time_s": _first_true_time_s(trajectory, "ego_in_attack_zone"),
        "target_attack_zone_entry_time_s": _first_true_time_s(trajectory, "target_in_attack_zone"),
        "ego_attack_zone_step_fraction": _true_fraction(trajectory, "ego_in_attack_zone"),
        "target_attack_zone_step_fraction": _true_fraction(trajectory, "target_in_attack_zone"),
        "mean_ego_attack_score": _mean_finite(frame.get("ego_attack_score") for frame in trajectory),
        "mean_target_attack_score": _mean_finite(frame.get("target_attack_score") for frame in trajectory),
        "mean_vp_vertical_offset_m": _mean_finite(
            _vertical_vpp_offset_m(frame) for frame in trajectory
        ),
        "minimum_vp_vertical_offset_m": _min_finite(
            _vertical_vpp_offset_m(frame) for frame in trajectory
        ),
        "maximum_vp_vertical_offset_m": _max_finite(
            _vertical_vpp_offset_m(frame) for frame in trajectory
        ),
        "initial_altitude_m": _finite(trajectory[0].get("altitude_m")),
        "terminal_altitude_m": _finite(trajectory[-1].get("altitude_m")),
        "mean_altitude_m": _mean_finite(frame.get("altitude_m") for frame in trajectory),
        "mean_speed_mps": _mean_finite(frame.get("speed_mps") for frame in trajectory),
        "mean_target_altitude_m": _mean_finite(frame.get("target_pos_z") for frame in trajectory),
        "mean_target_speed_mps": _mean_finite(_target_speed_mps(trajectory)),
        "terminal_specific_energy_difference_m2_s2": terminal_geometry.get(
            "specific_energy_difference_m2_s2"
        ),
        "first_switch_step": _first_switch_step(trajectory),
        "has_mode_telemetry": has_mode_telemetry,
        "effective_mode_step_fractions": effective_fractions,
        "requested_mode_step_fractions": requested_fractions,
        "guard_override_step_fraction": (
            _true_fraction(trajectory, "commander_mode_constraint_triggered")
            if has_mode_telemetry
            else None
        ),
        "guard_override_reasons": (
            _unique_values(trajectory, "commander_mode_constraint_reason")
            if has_mode_telemetry
            else []
        ),
        "mean_nz_cmd_g": _mean_finite(frame.get("nz_cmd") for frame in trajectory),
        "mean_nz_g": _mean_finite(frame.get("nz_g") for frame in trajectory),
        "mean_abs_nz_tracking_error_g": _mean_finite(
            abs(command - actual)
            for frame in trajectory
            if (command := _finite(frame.get("nz_cmd"))) is not None
            and (actual := _finite(frame.get("nz_g"))) is not None
        ),
        "max_abs_nz_tracking_error_g": _max_finite(
            abs(command - actual)
            for frame in trajectory
            if (command := _finite(frame.get("nz_cmd"))) is not None
            and (actual := _finite(frame.get("nz_g"))) is not None
        ),
        "nz_saturation_step_fraction": _true_fraction(trajectory, "nz_saturated"),
        "roll_rate_saturation_step_fraction": _true_fraction(
            trajectory, "roll_rate_saturated"
        ),
        "throttle_saturation_step_fraction": _true_fraction(
            trajectory, "throttle_saturated"
        ),
    }


def _episode_record(
    raw_path: Path,
    split: str,
    manifest_scenarios: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    payload = raw_path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    raw = json.loads(payload)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Raw episode is not an object: {raw_path}")
    trajectory = raw.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory or not all(
        isinstance(frame, Mapping) for frame in trajectory
    ):
        raise ValueError(f"Raw episode has no usable trajectory: {raw_path}")
    controller = str(raw.get("controller") or raw.get("method") or "")
    if controller not in METHODS:
        raise ValueError(f"Unexpected method {controller!r} in {raw_path}")
    scenario = str(raw.get("scenario") or "")
    expected_metadata = manifest_scenarios.get(scenario)
    if expected_metadata is None:
        raise ValueError(f"Unexpected scenario {scenario!r} in {raw_path}")
    metadata = raw.get("scenario_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Raw episode has no scenario metadata: {raw_path}")
    for key in (
        "initial_class",
        "height_condition",
        "mirror_sign",
        "mirror_pair_id",
        "task_registry_key",
        "scenario_seed",
    ):
        if metadata.get(key) != expected_metadata.get(key):
            raise ValueError(
                f"Metadata mismatch for {scenario}/{controller}: {key}="
                f"{metadata.get(key)!r}, expected {expected_metadata.get(key)!r}"
            )
    if str(raw.get("task")) != str(expected_metadata.get("task_registry_key")):
        raise ValueError(f"Task mismatch for {scenario}/{controller}")
    if str(raw.get("backend", "")).lower() != "jsbsim" or not _bool(raw.get("strict_backend")):
        raise ValueError(f"Non-strict JSBSim evidence in {raw_path}")

    initial = _initial_geometry(trajectory)
    expected_class = str(expected_metadata["initial_class"])
    expected_geometry = "transition" if expected_class == "crossing_entry" else expected_class
    observed_geometry = initial["geometry_state"]
    initial_matches = observed_geometry == expected_geometry
    phase_counts: Counter[str] = Counter(_phase_name(frame) for frame in trajectory)
    taxonomy_counts: Counter[str] = Counter(_geometry_state_from_frame(frame) for frame in trajectory)
    frame_count = len(trajectory)
    switches = [
        value
        for frame in trajectory
        if (value := _finite(frame.get("commander_switch_count"))) is not None
    ]
    terminal_reason = raw.get("termination_reason") or raw.get("combat_reason")
    telemetry = _trajectory_telemetry(trajectory)
    record = {
        "opponent_stage": split,
        "source_raw_path": str(raw_path),
        "source_raw_sha256": source_hash,
        "scenario": scenario,
        "method": controller,
        "task": str(raw.get("task")),
        "seed": raw.get("seed"),
        "episode": raw.get("episode"),
        "initial_class": expected_class,
        "height_condition": str(expected_metadata["height_condition"]),
        "mirror_sign": str(expected_metadata["mirror_sign"]),
        "mirror_pair_id": str(expected_metadata["mirror_pair_id"]),
        "win": _bool(raw.get("win")),
        "loss": _bool(raw.get("loss")),
        "draw": _bool(raw.get("draw")),
        "outcome": "",  # Set immediately below to keep the serialized contract explicit.
        "termination_reason": str(terminal_reason or "unknown"),
        "terminal_category": _terminal_category(terminal_reason),
        "combat_reason": str(raw.get("combat_reason") or "unknown"),
        "survived": _bool(raw.get("survived")),
        "ego_hp": _finite(raw.get("ego_hp")),
        "target_hp": _finite(raw.get("target_hp")),
        "hp_advantage": _finite(raw.get("hp_advantage")),
        "trajectory_steps": frame_count,
        "initial_geometry_source": initial["source"],
        "initial_geometry_observed": observed_geometry,
        "initial_geometry_expected": expected_geometry,
        "initial_geometry_matches_manifest": initial_matches,
        "initial_own_to_target_los_angle_deg": initial["own_to_target_los_angle_deg"],
        "initial_target_velocity_to_own_los_angle_deg": initial[
            "target_velocity_to_own_los_angle_deg"
        ],
        "initial_los_elevation_deg": initial["los_elevation_deg"],
        "initial_specific_energy_difference_m2_s2": initial[
            "specific_energy_difference_m2_s2"
        ],
        "terminal_geometry_observed": _geometry_state_from_frame(trajectory[-1]),
        "geometry_transition_count": sum(
            current != previous
            for previous, current in zip(
                [_geometry_state_from_frame(frame) for frame in trajectory],
                [_geometry_state_from_frame(frame) for frame in trajectory][1:],
            )
        ),
        "pre_merge_step_fraction": phase_counts["pre_merge"] / frame_count,
        "post_merge_step_fraction": phase_counts["post_merge"] / frame_count,
        "re_entry_step_fraction": phase_counts["re_entry"] / frame_count,
        "unclassified_phase_step_fraction": phase_counts["unclassified"] / frame_count,
        "geometry_state_step_counts": dict(sorted(taxonomy_counts.items())),
        "mean_vp_forward_bias_m": _mean_finite(
            frame.get("vp_forward_bias_m") for frame in trajectory
        ),
        "mean_vp_lateral_bias_m": _mean_finite(
            frame.get("vp_lateral_bias_m") for frame in trajectory
        ),
        "first_requested_mode": _first_value(trajectory, "commander_requested_mode_name"),
        "first_effective_mode": _first_value(trajectory, "commander_mode_name"),
        "requested_mode_step_counts": _mode_counts(trajectory, "commander_requested_mode_name"),
        "effective_mode_step_counts": _mode_counts(trajectory, "commander_mode_name"),
        "effective_mode_constraint_reasons": _unique_values(
            trajectory, "commander_mode_constraint_reason"
        ),
        "max_commander_switch_count": int(max(switches)) if switches else None,
        "git_commit": raw.get("git_commit"),
        "config_sha256": raw.get("config_sha256"),
        "opponent_config": raw.get("opponent_config"),
        **telemetry,
    }
    record["outcome"] = _outcome(record)
    return record, source_hash


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Invalid manifest: {path}")
    if manifest.get("source_id") != SOURCE_ID:
        raise ValueError(f"Unexpected manifest source ID: {manifest.get('source_id')!r}")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 30:
        raise ValueError("Taxonomy manifest must define exactly 30 scenarios")
    by_name = {str(item["name"]): item["metadata"] for item in scenarios}
    if len(by_name) != 30:
        raise ValueError("Taxonomy manifest contains duplicate scenario names")
    return {"payload": manifest, "by_name": by_name}


def _load_split(
    split: str, run_dir: Path, manifest_scenarios: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_root = run_dir / "raw"
    if not raw_root.is_dir():
        raise ValueError(f"Missing raw evidence directory: {raw_root}")
    raw_paths = sorted(raw_root.rglob("*.json"))
    expected_count = len(manifest_scenarios) * len(METHODS)
    if len(raw_paths) != expected_count:
        raise ValueError(
            f"{split} has {len(raw_paths)} raw episodes; expected {expected_count}"
        )
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for raw_path in raw_paths:
        # The raw JSON is parsed exactly once. The classifier below receives
        # geometry only, never the scenario name used for manifest lookup.
        record, raw_hash = _episode_record(raw_path, split, manifest_scenarios)
        records.append(record)
        digest.update(raw_path.relative_to(raw_root).as_posix().encode("utf-8"))
        digest.update(raw_hash.encode("ascii"))
    group = defaultdict(list)
    for record in records:
        group[record["scenario"]].append(record)
    incomplete = {
        scenario: sorted(record["method"] for record in candidates)
        for scenario, candidates in group.items()
        if {record["method"] for record in candidates} != set(METHODS) or len(candidates) != 4
    }
    if set(group) != set(manifest_scenarios) or incomplete:
        raise ValueError(f"{split} scenario-method pairing is incomplete: {incomplete}")
    duplicate_keys = [
        key
        for key, count in Counter(
            (record["scenario"], record["method"], record["seed"], record["episode"])
            for record in records
        ).items()
        if count != 1
    ]
    if duplicate_keys:
        raise ValueError(f"{split} has duplicate scenario-method episodes: {duplicate_keys}")
    for scenario, candidates in group.items():
        seeds = {record["seed"] for record in candidates}
        if len(seeds) != 1:
            raise ValueError(f"{split}/{scenario} has non-shared method seeds: {seeds}")
    source_files = [
        run_dir / "run_manifest.json",
        run_dir / "resolved_config.yaml",
        run_dir / "aggregate" / "method_task_summary.json",
        run_dir / "aggregate" / "combat_geometry_diagnostics.json",
    ]
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise ValueError(f"{split} is missing formal source artifacts: {missing}")
    return records, {
        "run_dir": str(run_dir),
        "raw_file_count": len(raw_paths),
        "raw_payload_manifest_sha256": digest.hexdigest(),
        "source_files": {str(path.relative_to(run_dir)): _file_sha256(path) for path in source_files},
        "git_commits": sorted({str(record["git_commit"]) for record in records}),
        "config_hashes": sorted({str(record["config_sha256"]) for record in records}),
    }


def _wilson_interval(wins: int, resolved: int) -> tuple[float | None, float | None]:
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


def _result_summary(records: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in group_fields)].append(record)
    rows: list[dict[str, Any]] = []
    for key, candidates in sorted(groups.items()):
        wins = sum(_bool(item["win"]) for item in candidates)
        losses = sum(_bool(item["loss"]) for item in candidates)
        draws = sum(_bool(item["draw"]) for item in candidates)
        resolved = wins + losses
        lower, upper = _wilson_interval(wins, resolved)
        row = dict(zip(group_fields, key))
        row.update(
            {
                "n_total": len(candidates),
                "n_resolved": resolved,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "unresolved": len(candidates) - resolved - draws,
                "win_rate_resolved": wins / resolved if resolved else None,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "ego_crash_or_oob": sum(
                    item["terminal_category"] == "ego_crash_or_oob" for item in candidates
                ),
                "target_crash_or_oob": sum(
                    item["terminal_category"] == "target_crash_or_oob" for item in candidates
                ),
                "mean_pre_merge_step_fraction": _mean_finite(
                    item["pre_merge_step_fraction"] for item in candidates
                ),
                "mean_post_merge_step_fraction": _mean_finite(
                    item["post_merge_step_fraction"] for item in candidates
                ),
                "mean_re_entry_step_fraction": _mean_finite(
                    item["re_entry_step_fraction"] for item in candidates
                ),
            }
        )
        rows.append(row)
    return rows


def _pair_diagnostics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["opponent_stage"]), str(record["scenario"]))][
            str(record["method"])
        ] = record
    diagnostics: list[dict[str, Any]] = []
    for (split, scenario), methods in sorted(grouped.items()):
        if set(methods) != set(METHODS):
            raise ValueError(f"Incomplete paired diagnostic cell: {split}/{scenario}")
        ppo = methods[PPO_METHOD]
        oracle = methods[ORACLE_METHOD]
        fixed_head_on = methods[FIXED_METHODS[0]]
        fixed_crossing = methods[FIXED_METHODS[1]]
        fixed_any_win = _bool(fixed_head_on["win"]) or _bool(fixed_crossing["win"])
        fixed_both_loss = _bool(fixed_head_on["loss"]) and _bool(fixed_crossing["loss"])
        labels: list[str] = []
        if _bool(ppo["loss"]) and fixed_any_win:
            labels.append("routing_opportunity")
        if _bool(ppo["loss"]) and _bool(oracle["win"]):
            labels.append("learned_vs_oracle_gap")
        if _bool(oracle["loss"]) and fixed_any_win:
            labels.append("oracle_mapping_gap")
        if fixed_both_loss:
            labels.append("library_gap_candidate")
        if _bool(ppo["win"]) and fixed_both_loss:
            labels.append("composition_gain")
        if all(_bool(method["loss"]) for method in methods.values()):
            labels.append("all_method_failure")
        diagnostics.append(
            {
                "opponent_stage": split,
                "scenario": scenario,
                "initial_class": ppo["initial_class"],
                "height_condition": ppo["height_condition"],
                "mirror_sign": ppo["mirror_sign"],
                "mirror_pair_id": ppo["mirror_pair_id"],
                "ppo_outcome": ppo["outcome"],
                "oracle_outcome": oracle["outcome"],
                "fixed_head_on_outcome": fixed_head_on["outcome"],
                "fixed_crossing_outcome": fixed_crossing["outcome"],
                "ppo_terminal_category": ppo["terminal_category"],
                "oracle_terminal_category": oracle["terminal_category"],
                "fixed_head_on_terminal_category": fixed_head_on["terminal_category"],
                "fixed_crossing_terminal_category": fixed_crossing["terminal_category"],
                "labels": labels,
                "opponent_dependent": False,
                "library_gap_raw_geometry_audit": {
                    method: {
                        "initial_geometry": methods[method]["initial_geometry_observed"],
                        "terminal_geometry": methods[method]["terminal_geometry_observed"],
                        "terminal_category": methods[method]["terminal_category"],
                        "post_merge_fraction": methods[method]["post_merge_step_fraction"],
                        "re_entry_fraction": methods[method]["re_entry_step_fraction"],
                    }
                    for method in METHODS
                }
                if "library_gap_candidate" in labels
                else None,
            }
        )
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for diagnostic in diagnostics:
        by_scenario[str(diagnostic["scenario"])].append(diagnostic)
    for scenario, cells in by_scenario.items():
        if len(cells) != 2:
            raise ValueError(f"Scenario {scenario} is not represented in both opponent splits")
        label_sets = [{label for label in cell["labels"]} for cell in cells]
        differs = label_sets[0] != label_sets[1]
        for cell in cells:
            cell["opponent_dependent"] = differs
            if differs:
                cell["labels"] = [*cell["labels"], "opponent_dependent"]
    return diagnostics


def _mirror_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[(str(record["opponent_stage"]), str(record["mirror_pair_id"]), str(record["method"]))].append(
            record
        )
    rows: list[dict[str, Any]] = []
    for (split, pair_id, method), candidates in sorted(by_pair.items()):
        signs = {str(record["mirror_sign"]): record for record in candidates}
        if set(signs) != {"negative", "positive"}:
            raise ValueError(f"Incomplete mirror pair {split}/{pair_id}/{method}")
        negative = signs["negative"]
        positive = signs["positive"]
        rows.append(
            {
                "opponent_stage": split,
                "initial_class": negative["initial_class"],
                "height_condition": negative["height_condition"],
                "mirror_pair_id": pair_id,
                "method": method,
                "negative_outcome": negative["outcome"],
                "positive_outcome": positive["outcome"],
                "outcome_disagreement": negative["outcome"] != positive["outcome"],
                "negative_terminal_category": negative["terminal_category"],
                "positive_terminal_category": positive["terminal_category"],
            }
        )
    by_class_method: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class_method[(row["opponent_stage"], row["initial_class"], row["method"])].append(row)
    for candidates in by_class_method.values():
        count = sum(_bool(row["outcome_disagreement"]) for row in candidates)
        flag = count >= 2
        for row in candidates:
            row["class_height_pairs"] = len(candidates)
            row["class_mirror_outcome_disagreement_count"] = count
            row["class_mirror_sensitivity_flag"] = flag
    return rows


def _threshold_rows(diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for diagnostic in diagnostics:
        grouped[(str(diagnostic["opponent_stage"]), str(diagnostic["initial_class"]))].append(diagnostic)
    rows: list[dict[str, Any]] = []
    for (split, initial_class), candidates in sorted(grouped.items()):
        counts = Counter(label for candidate in candidates for label in candidate["labels"])
        label_signs = {
            label: sorted(
                {
                    str(candidate["mirror_sign"])
                    for candidate in candidates
                    if label in candidate["labels"]
                }
            )
            for label in (
                "routing_opportunity",
                "library_gap_candidate",
                "composition_gain",
            )
        }
        library_heights = sorted(
            {
                str(candidate["height_condition"])
                for candidate in candidates
                if "library_gap_candidate" in candidate["labels"]
            }
        )
        composition_without_ego_failure = sum(
            "composition_gain" in candidate["labels"]
            and candidate["ppo_terminal_category"] != "ego_crash_or_oob"
            for candidate in candidates
        )
        rows.append(
            {
                "opponent_stage": split,
                "initial_class": initial_class,
                "n_scenarios": len(candidates),
                "routing_opportunity_count": counts["routing_opportunity"],
                "routing_opportunity_mirror_signs": label_signs["routing_opportunity"],
                "systematic_routing_opportunity": counts["routing_opportunity"] >= 3
                and len(label_signs["routing_opportunity"]) == 2,
                "library_gap_candidate_count": counts["library_gap_candidate"],
                "library_gap_mirror_signs": label_signs["library_gap_candidate"],
                "library_gap_height_conditions": library_heights,
                "systematic_library_gap_candidate": counts["library_gap_candidate"] >= 4
                and len(label_signs["library_gap_candidate"]) == 2
                and len(library_heights) >= 2,
                "composition_gain_count": counts["composition_gain"],
                "composition_gain_without_ego_crash_or_oob_count": composition_without_ego_failure,
                "composition_evidence": composition_without_ego_failure >= 3,
                "all_method_failure_count": counts["all_method_failure"],
                "opponent_dependent_count": counts["opponent_dependent"],
            }
        )
    return rows


def _terminal_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], int] = Counter()
    for record in records:
        groups[
            (
                str(record["opponent_stage"]),
                str(record["initial_class"]),
                str(record["method"]),
                str(record["terminal_category"]),
                str(record["termination_reason"]),
            )
        ] += 1
    rows = [
        {
            "opponent_stage": split,
            "initial_class": initial_class,
            "method": method,
            "terminal_category": category,
            "termination_reason": reason,
            "episodes": count,
        }
        for (split, initial_class, method, category, reason), count in sorted(groups.items())
    ]
    return {
        "semantic_contract": {
            "ego_crash_or_oob": "Ego terminal failure; a negative outcome signal, not standalone controller-instability proof.",
            "target_crash_or_oob": "Opponent terminal event; a legal win when applicable, not proof of aerodynamic superiority.",
            "timeout_hp_advantage": "Resolved by terminal HP advantage, not a kill event.",
            "raw_timeout": "Unresolved or non-HP timeout; excluded from resolved win-rate denominator.",
        },
        "rows": rows,
    }


def _mean_record_field(records: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return _mean_finite(record.get(field) for record in records)


def _mean_mode_fraction(records: Sequence[Mapping[str, Any]], field: str, mode: str) -> float | None:
    values = [
        record[field].get(mode, 0.0)
        for record in records
        if isinstance(record.get(field), Mapping) and record[field]
    ]
    return _mean_finite(values)


def _telemetry_summary_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the missing checklist metrics without pooling opponent splits."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                str(record["opponent_stage"]),
                str(record["initial_class"]),
                str(record["method"]),
            )
        ].append(record)
    rows: list[dict[str, Any]] = []
    scalar_fields = (
        "initial_range_m",
        "final_range_m",
        "minimum_range_m",
        "mean_range_m",
        "mean_range_rate_mps",
        "minimum_range_rate_mps",
        "maximum_range_rate_mps",
        "mean_ego_attack_score",
        "mean_target_attack_score",
        "mean_vp_forward_bias_m",
        "mean_vp_lateral_bias_m",
        "mean_vp_vertical_offset_m",
        "minimum_vp_vertical_offset_m",
        "maximum_vp_vertical_offset_m",
        "initial_altitude_m",
        "terminal_altitude_m",
        "mean_altitude_m",
        "mean_speed_mps",
        "mean_target_altitude_m",
        "mean_target_speed_mps",
        "initial_los_elevation_deg",
        "initial_specific_energy_difference_m2_s2",
        "terminal_specific_energy_difference_m2_s2",
        "first_switch_step",
        "max_commander_switch_count",
        "guard_override_step_fraction",
        "mean_nz_cmd_g",
        "mean_nz_g",
        "mean_abs_nz_tracking_error_g",
        "max_abs_nz_tracking_error_g",
        "nz_saturation_step_fraction",
        "roll_rate_saturation_step_fraction",
        "throttle_saturation_step_fraction",
    )
    for (split, initial_class, method), candidates in sorted(groups.items()):
        row: dict[str, Any] = {
            "opponent_stage": split,
            "initial_class": initial_class,
            "method": method,
            "n_episodes": len(candidates),
            "ego_attack_zone_episode_entry_rate": sum(
                record["ego_attack_zone_entry_time_s"] is not None for record in candidates
            )
            / len(candidates),
            "target_attack_zone_episode_entry_rate": sum(
                record["target_attack_zone_entry_time_s"] is not None for record in candidates
            )
            / len(candidates),
            "mean_first_ego_attack_zone_entry_time_s": _mean_record_field(
                candidates, "ego_attack_zone_entry_time_s"
            ),
            "mean_first_target_attack_zone_entry_time_s": _mean_record_field(
                candidates, "target_attack_zone_entry_time_s"
            ),
            "mean_ego_attack_zone_step_fraction": _mean_record_field(
                candidates, "ego_attack_zone_step_fraction"
            ),
            "mean_target_attack_zone_step_fraction": _mean_record_field(
                candidates, "target_attack_zone_step_fraction"
            ),
            "mean_hp_advantage": _mean_record_field(candidates, "hp_advantage"),
            "mode_telemetry_episode_count": sum(
                _bool(record["has_mode_telemetry"]) for record in candidates
            ),
            "guard_override_reasons": sorted(
                {
                    reason
                    for record in candidates
                    for reason in record["guard_override_reasons"]
                }
            ),
        }
        row.update({f"mean_{field}": _mean_record_field(candidates, field) for field in scalar_fields})
        for mode in (
            "head_on_specialist",
            "crossing_specialist",
            "post_merge_recovery_specialist",
        ):
            row[f"mean_effective_{mode}_fraction"] = _mean_mode_fraction(
                candidates, "effective_mode_step_fractions", mode
            )
            row[f"mean_requested_{mode}_fraction"] = _mean_mode_fraction(
                candidates, "requested_mode_step_fractions", mode
            )
        rows.append(row)
    return rows


def _opponent_reference_pressure_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe opponent pressure using frozen non-PPO reference methods."""

    reference_methods = (ORACLE_METHOD, *FIXED_METHODS)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["method"] in reference_methods:
            groups[(str(record["opponent_stage"]), str(record["method"]))].append(record)
    rows: list[dict[str, Any]] = []
    for (split, method), candidates in sorted(groups.items()):
        wins = sum(_bool(record["win"]) for record in candidates)
        losses = sum(_bool(record["loss"]) for record in candidates)
        resolved = wins + losses
        rows.append(
            {
                "opponent_stage": split,
                "ego_reference_method": method,
                "n_total": len(candidates),
                "n_resolved": resolved,
                "ego_wins": wins,
                "ego_losses": losses,
                "ego_loss_rate_resolved": losses / resolved if resolved else None,
                "mean_ego_hp_advantage": _mean_record_field(candidates, "hp_advantage"),
                "mean_target_attack_zone_step_fraction": _mean_record_field(
                    candidates, "target_attack_zone_step_fraction"
                ),
                "target_attack_zone_episode_entry_rate": sum(
                    record["target_attack_zone_entry_time_s"] is not None for record in candidates
                )
                / len(candidates),
                "mean_first_target_attack_zone_entry_time_s": _mean_record_field(
                    candidates, "target_attack_zone_entry_time_s"
                ),
                "ego_crash_or_oob": sum(
                    record["terminal_category"] == "ego_crash_or_oob" for record in candidates
                ),
                "target_crash_or_oob": sum(
                    record["terminal_category"] == "target_crash_or_oob" for record in candidates
                ),
            }
        )
    return rows


def _opponent_capability_card(
    records: Sequence[Mapping[str, Any]],
    source_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build two interaction-conditioned capability cards, not an Elo ranking."""

    pressure_rows = _opponent_reference_pressure_rows(records)
    cards: list[dict[str, Any]] = []
    for split in ("expert", "end_to_end"):
        candidates = [record for record in records if record["opponent_stage"] == split]
        configurations = sorted(
            {
                json.dumps(_json_safe(record.get("opponent_config")), sort_keys=True)
                for record in candidates
                if record.get("opponent_config") is not None
            }
        )
        cards.append(
            {
                "opponent_stage": split,
                "identity_and_provenance": {
                    "raw_opponent_configs": [json.loads(item) for item in configurations],
                    "formal_source_id": source_id,
                    "interaction_episode_count": len(candidates),
                    "formal_splits_share_the_same_30_initial_ego_states": True,
                },
                "interface_and_training_boundary": {
                    "opponent_internal_observation_action_and_decision_period": "not logged by this evaluation telemetry",
                    "expert_training": "not applicable: raw config identifies a rule-based ExpertOpponent",
                    "end_to_end_training_budget_seed_and_selection": "not recoverable from this frozen raw telemetry; checkpoint path is recorded when present",
                },
                "interaction_conditioned_flight_behavior": {
                    "mean_target_speed_mps": _mean_record_field(candidates, "mean_target_speed_mps"),
                    "mean_target_altitude_m": _mean_record_field(candidates, "mean_target_altitude_m"),
                    "mean_target_attack_zone_step_fraction": _mean_record_field(
                        candidates, "target_attack_zone_step_fraction"
                    ),
                    "target_attack_zone_episode_entry_rate": sum(
                        record["target_attack_zone_entry_time_s"] is not None
                        for record in candidates
                    )
                    / len(candidates),
                    "mean_first_target_attack_zone_entry_time_s": _mean_record_field(
                        candidates, "target_attack_zone_entry_time_s"
                    ),
                    "mean_post_merge_step_fraction": _mean_record_field(
                        candidates, "post_merge_step_fraction"
                    ),
                    "mean_re_entry_step_fraction": _mean_record_field(
                        candidates, "re_entry_step_fraction"
                    ),
                },
                "reference_pressure_rows": [
                    row for row in pressure_rows if row["opponent_stage"] == split
                ],
                "interpretation_boundary": [
                    "Metrics are interaction-conditioned summaries across frozen ego references and the preregistered 30 scenes.",
                    "The card supports split-specific pressure descriptions, not an absolute opponent-strength ranking or Elo claim.",
                    "Opponent internal modes and training details not present in raw telemetry are explicitly unavailable rather than inferred.",
                ],
            }
        )
    return {"source_id": source_id, "cards": cards}, pressure_rows


def _build_opponent_capability_markdown(card: Mapping[str, Any]) -> str:
    lines = [
        "# 对手能力卡：THESIS-TAXONOMY-ABLATION30-V1",
        "",
        "## 范围",
        "",
        "本卡基于同一组 30 个冻结初始状态和三个非 PPO 我方参考方法（static Oracle、always-head-on、always-crossing）构建，只读且依赖交互条件。它不是 Elo 排名，也不声称对手具有绝对强弱次序。",
    ]
    for item in card["cards"]:
        identity = item["identity_and_provenance"]
        behavior = item["interaction_conditioned_flight_behavior"]
        lines.extend(
            [
                "",
                f"## {item['opponent_stage']}",
                "",
                "**身份与溯源**",
                "",
                f"- 冻结交互回合：{identity['interaction_episode_count']}。",
                f"- Raw config：`{json.dumps(identity['raw_opponent_configs'], ensure_ascii=True)}`。",
                "- 对手内部 observation/action 接口及训练元数据只在 raw provenance 可见时报告；未记录的信息明确标为不可得。",
                "",
                "**交互条件下的行为画像**",
                "",
                _markdown_table(
                    [behavior],
                    (
                        ("mean_target_speed_mps", "目标平均速度 (m/s)"),
                        ("mean_target_altitude_m", "目标平均高度 (m)"),
                        ("mean_target_attack_zone_step_fraction", "目标攻击区步占比"),
                        ("target_attack_zone_episode_entry_rate", "目标攻击区进入率"),
                        ("mean_first_target_attack_zone_entry_time_s", "首次目标攻击区时间 (s)"),
                        ("mean_post_merge_step_fraction", "Post-merge 占比"),
                        ("mean_re_entry_step_fraction", "Re-entry 占比"),
                    ),
                ),
                "",
                "**冻结参考方法下的压力**",
                "",
                _markdown_table(
                    item["reference_pressure_rows"],
                    (
                        ("ego_reference_method", "我方参考方法"),
                        ("n_resolved", "判定 N"),
                        ("ego_loss_rate_resolved", "我方负率"),
                        ("mean_ego_hp_advantage", "我方平均 HP 优势"),
                        ("mean_target_attack_zone_step_fraction", "目标攻击区占比"),
                        ("ego_crash_or_oob", "我方 crash/OOB"),
                    ),
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 声明边界",
            "",
            "- 不得依据本卡把任一对手称为普遍更强或更弱。",
            "- 所有比较受限于当前 JSBSim 资产、AoA60 终端协议、30 场景包线和冻结我方参考方法。",
            "- 后续若要使用 Elo 或 Bradley-Terry，必须增加固定对手池并独立冻结 rating protocol。",
        ]
    )
    return "\n".join(lines) + "\n"


def _expert_disadvantage_feasibility_audit(
    records: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit the five expert/disadvantage candidates without attributing causality."""

    candidate_cells = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic["opponent_stage"] == "expert"
        and diagnostic["initial_class"] == "disadvantage"
        and "library_gap_candidate" in diagnostic["labels"]
        and "all_method_failure" in diagnostic["labels"]
    ]
    by_split_scenario: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_split_scenario[(str(record["opponent_stage"]), str(record["scenario"]))].append(record)
    detail_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for candidate in sorted(candidate_cells, key=lambda item: str(item["scenario"])):
        scenario = str(candidate["scenario"])
        expert_records = by_split_scenario[("expert", scenario)]
        counterpart = {
            record["method"]: record
            for record in by_split_scenario[("end_to_end", scenario)]
        }
        counterpart_fixed_any_win = any(
            _bool(counterpart[method]["win"]) for method in FIXED_METHODS
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "height_condition": candidate["height_condition"],
                "mirror_sign": candidate["mirror_sign"],
                "expert_all_method_failure": True,
                "expert_terminal_categories": sorted(
                    {record["terminal_category"] for record in expert_records}
                ),
                "expert_ego_crash_or_oob_method_count": sum(
                    record["terminal_category"] == "ego_crash_or_oob"
                    for record in expert_records
                ),
                "expert_any_control_saturation": any(
                    any(
                        (_finite(record.get(field)) or 0.0) > 0.0
                        for field in (
                            "nz_saturation_step_fraction",
                            "roll_rate_saturation_step_fraction",
                            "throttle_saturation_step_fraction",
                        )
                    )
                    for record in expert_records
                ),
                "end_to_end_same_scenario_fixed_any_win": counterpart_fixed_any_win,
                "end_to_end_outcomes": {
                    method: counterpart[method]["outcome"] for method in METHODS
                },
            }
        )
        for record in sorted(expert_records, key=lambda item: str(item["method"])):
            detail_rows.append(
                {
                    "scenario": scenario,
                    "height_condition": record["height_condition"],
                    "mirror_sign": record["mirror_sign"],
                    "method": record["method"],
                    "outcome": record["outcome"],
                    "termination_reason": record["termination_reason"],
                    "terminal_category": record["terminal_category"],
                    "hp_advantage": record["hp_advantage"],
                    "initial_range_m": record["initial_range_m"],
                    "minimum_range_m": record["minimum_range_m"],
                    "final_range_m": record["final_range_m"],
                    "mean_range_rate_mps": record["mean_range_rate_mps"],
                    "ego_attack_zone_entry_time_s": record["ego_attack_zone_entry_time_s"],
                    "target_attack_zone_entry_time_s": record["target_attack_zone_entry_time_s"],
                    "ego_attack_zone_step_fraction": record["ego_attack_zone_step_fraction"],
                    "target_attack_zone_step_fraction": record["target_attack_zone_step_fraction"],
                    "mean_vp_forward_bias_m": record["mean_vp_forward_bias_m"],
                    "mean_vp_lateral_bias_m": record["mean_vp_lateral_bias_m"],
                    "mean_vp_vertical_offset_m": record["mean_vp_vertical_offset_m"],
                    "initial_altitude_m": record["initial_altitude_m"],
                    "terminal_altitude_m": record["terminal_altitude_m"],
                    "mean_abs_nz_tracking_error_g": record["mean_abs_nz_tracking_error_g"],
                    "max_abs_nz_tracking_error_g": record["max_abs_nz_tracking_error_g"],
                    "nz_saturation_step_fraction": record["nz_saturation_step_fraction"],
                    "roll_rate_saturation_step_fraction": record[
                        "roll_rate_saturation_step_fraction"
                    ],
                    "throttle_saturation_step_fraction": record[
                        "throttle_saturation_step_fraction"
                    ],
                    "effective_mode_step_fractions": record["effective_mode_step_fractions"],
                    "guard_override_step_fraction": record["guard_override_step_fraction"],
                    "end_to_end_same_scenario_outcome": counterpart[record["method"]]["outcome"],
                    "end_to_end_same_scenario_fixed_any_win": counterpart_fixed_any_win,
                    "source_raw_path": record["source_raw_path"],
                }
            )
    counterpart_fixed_successes = sum(
        _bool(row["end_to_end_same_scenario_fixed_any_win"]) for row in scenario_rows
    )
    any_saturation = sum(_bool(row["expert_any_control_saturation"]) for row in scenario_rows)
    audit = {
        "scope": "expert/disadvantage all-method-failure library-gap candidates only",
        "candidate_scenario_count": len(scenario_rows),
        "candidate_scenarios": scenario_rows,
        "counterpart_end_to_end_fixed_any_win_count": counterpart_fixed_successes,
        "candidate_scenarios_with_any_logged_control_saturation": any_saturation,
        "decision": {
            "shared_skill_library_training_approved": False,
            "reason": "Frozen telemetry shows an expert-only candidate set with mixed terminal paths; it cannot yet exclude opponent conditioning or common execution-chain effects.",
            "next_gate": "Complete opponent capability interpretation and a dedicated low-level feasibility protocol before any specialist training.",
        },
    }
    return audit, scenario_rows, detail_rows


def _build_feasibility_markdown(audit: Mapping[str, Any]) -> str:
    rows = audit["candidate_scenarios"]
    lines = [
        "# Expert/Disadvantage 低层 Feasibility 审计",
        "",
        "## 范围",
        "",
        "本只读审计仅检查同时带有 `library_gap_candidate` 与 `all_method_failure` 标签的 `expert/disadvantage` 单元。它不训练、不调参，也不从单个终端事件推断控制器的因果缺陷。",
        "",
        "## 候选场景",
        "",
        _markdown_table(
            rows,
            (
                ("scenario", "场景"),
                ("height_condition", "高度"),
                ("mirror_sign", "镜像"),
                ("expert_terminal_categories", "expert 终端类别"),
                ("expert_ego_crash_or_oob_method_count", "我方 crash/OOB 方法数"),
                ("expert_any_control_saturation", "任一记录饱和"),
                ("end_to_end_same_scenario_fixed_any_win", "E2E 固定技能任一胜"),
            ),
        ),
        "",
        "## 排除逻辑",
        "",
        f"- 候选数：{audit['candidate_scenario_count']}。",
        f"- 同一物理场景在 end-to-end 下至少有一个固定 specialist 胜的数量：{audit['counterpart_end_to_end_fixed_any_win_count']}/{audit['candidate_scenario_count']}。",
        f"- 存在记录到的 command/response saturation 的候选数：{audit['candidate_scenarios_with_any_logged_control_saturation']}/{audit['candidate_scenario_count']}。",
        "- 终端类别混合且对应 end-to-end 胜负不同，因此不能把失败唯一归因为缺少 specialist，也不能唯一归因为共享 guidance/PID 链。",
        "",
        "## 决策",
        "",
        "- 本审计**不批准**启动共享技能库训练。",
        "- 五个场景应保留为负证据；在定义任何低层 feasibility 实验前，先依据详细 CSV 检查 VPP 前/侧/垂向几何、距离演化、攻击区时序以及 command-vs-response 误差。",
        "- 饱和比例为零不构成飞行安全认证，也不排除所有共享执行链限制。",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append("-" if not math.isfinite(value) else f"{value:.3f}")
            elif isinstance(value, list):
                values.append(", ".join(str(item) for item in value) if value else "-")
            elif value is None:
                values.append("-")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _build_report(
    pairing: Mapping[str, Any],
    initial_verification: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    mirror_rows: Sequence[Mapping[str, Any]],
) -> str:
    class_summary = [
        row
        for row in summary_rows
        if set(row) >= {"opponent_stage", "initial_class", "method"}
        and row.get("initial_class") is not None
    ]
    systematic_library = [
        row
        for row in threshold_rows
        if row["systematic_library_gap_candidate"]
    ]
    systematic_routing = [row for row in threshold_rows if row["systematic_routing_opportunity"]]
    composition = [row for row in threshold_rows if row["composition_evidence"]]
    mirror_flag_keys = {
        (row["opponent_stage"], row["initial_class"], row["method"])
        for row in mirror_rows
        if row["class_mirror_sensitivity_flag"]
    }
    library_cells = [
        row for row in diagnostics if "library_gap_candidate" in row["labels"]
    ]
    lines = [
        "# Thesis Taxonomy Frozen Ablation30: Step 4 Result Audit",
        "",
        "## Frozen Scope",
        "",
        "- Source ID: `THESIS-TAXONOMY-ABLATION30-V1`; analysis version: "
        f"`{ANALYSIS_VERSION}`.",
        "- Evidence only: 30 preregistered scenarios × 4 frozen methods × 2 opponent splits = 240 raw JSBSim episodes.",
        "- This is a read-only diagnosis. No policy, reward, VPP, guidance, PID, seed, or scenario was changed or rerun.",
        "- The invalid zero-episode `c049369` initialization directory is not an input to this analysis.",
        "",
        "## Integrity And Taxonomy",
        "",
        f"- Pairing: {pairing['total_records']}/{pairing['expected_total_records']} episodes; all scenario-method cells are complete.",
        f"- Initial geometry recomputation: {initial_verification['match_count']}/{initial_verification['record_count']} records agree with the preregistered class (crossing-entry expects the explicit `transition` state).",
        "- Initial quadrant is recomputed from the first two telemetry positions and velocities; scenario names are not classifier inputs.",
        "- Step-wise phase labels are `pre_merge`, `post_merge`, and post-merge closing `re_entry`; they are descriptive telemetry summaries, not new task labels.",
        "",
        "## Split-Wise Results",
        "",
        _markdown_table(
            class_summary,
            (
                ("opponent_stage", "Opponent"),
                ("initial_class", "Initial class"),
                ("method", "Method"),
                ("n_total", "N total"),
                ("n_resolved", "N resolved"),
                ("wins", "W"),
                ("losses", "L"),
                ("win_rate_resolved", "Resolved win rate"),
                ("ego_crash_or_oob", "Ego crash/OOB"),
            ),
        ),
        "",
        "Results are deliberately not pooled across opponent splits. Each initial class has six explicit scenarios, so these descriptive counts are not significance claims.",
        "",
        "## Diagnostic Labels",
        "",
        "- `routing_opportunity`: PPO loss while at least one frozen fixed specialist wins.",
        "- `library_gap_candidate`: both fixed specialists lose. It is a feasibility candidate only, not proof that a new skill is required.",
        "- `composition_gain`: PPO win while both fixed specialists lose; this remains limited evidence unless it occurs in at least 3/6 scenarios without added ego crash/OOB.",
        "",
        _markdown_table(
            threshold_rows,
            (
                ("opponent_stage", "Opponent"),
                ("initial_class", "Initial class"),
                ("routing_opportunity_count", "Routing"),
                ("systematic_routing_opportunity", "Systematic routing"),
                ("library_gap_candidate_count", "Library candidates"),
                ("systematic_library_gap_candidate", "Systematic library"),
                ("composition_gain_count", "Composition"),
                ("composition_evidence", "Composition evidence"),
                ("opponent_dependent_count", "Opponent-dependent"),
            ),
        ),
        "",
        "## Mirror And Terminal Audit",
        "",
        f"- Mirror sensitivity flags: {len(mirror_flag_keys)} method/class/split combinations (a flag requires disagreement in at least 2 of 3 height pairs).",
        f"- Library-gap raw-geometry audits retained: {len(library_cells)} scenario × opponent cells, with terminal categories and phase geometry in `diagnostic_label_by_episode.csv`.",
        "- Terminal categories distinguish ego and target crash/OOB, kills, HP-resolved timeouts, and raw/draw timeouts. They do not certify flight safety or aerodynamic superiority.",
        "",
        "## Go/No-Go Interpretation",
        "",
        (
            "- Systematic routing-opportunity threshold is met in: "
            + ", ".join(f"{row['opponent_stage']}/{row['initial_class']}" for row in systematic_routing)
            + "."
            if systematic_routing
            else "- No initial-class/opponent cell reaches the preregistered systematic routing-opportunity threshold."
        ),
        (
            "- Systematic specialist-library candidate threshold is met in: "
            + ", ".join(f"{row['opponent_stage']}/{row['initial_class']}" for row in systematic_library)
            + ". This supports only a subsequent low-level feasibility study."
            if systematic_library
            else "- No initial-class/opponent cell reaches the preregistered systematic specialist-library-candidate threshold; a four-skill retraining program is not justified by this audit alone."
        ),
        (
            "- Limited composition evidence is present in: "
            + ", ".join(f"{row['opponent_stage']}/{row['initial_class']}" for row in composition)
            + "."
            if composition
            else "- No initial-class/opponent cell reaches the preregistered limited-composition-evidence threshold."
        ),
        "- Any next research decision must preserve the split-specific evidence boundary and start from the episode-level tables, not aggregate pooled win rates.",
        "",
        "## What This Audit Does Not Establish",
        "",
        "- It does not establish statistical superiority, universal geometry recognition, or a universal five-skill requirement.",
        "- It does not identify a causal training defect from a single terminal category.",
        "- It does not replace an opponent capability card or a dedicated low-level feasibility experiment if either is later warranted.",
    ]
    return "\n".join(lines) + "\n"


def _build_chinese_report(
    pairing: Mapping[str, Any],
    initial_verification: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    mirror_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Write the thesis-facing interpretation without upgrading any claim."""

    class_summary = [
        row
        for row in summary_rows
        if row.get("initial_class") is not None
    ]
    systematic_library = [
        row for row in threshold_rows if row["systematic_library_gap_candidate"]
    ]
    systematic_routing = [
        row for row in threshold_rows if row["systematic_routing_opportunity"]
    ]
    composition = [row for row in threshold_rows if row["composition_evidence"]]
    mirror_flag_keys = sorted(
        {
            (row["opponent_stage"], row["initial_class"], row["method"])
            for row in mirror_rows
            if row["class_mirror_sensitivity_flag"]
        }
    )
    opponent_dependent_scenarios = len(
        {
            diagnostic["scenario"]
            for diagnostic in diagnostics
            if diagnostic["opponent_dependent"]
        }
    )
    systematic_library_keys = {
        (str(row["opponent_stage"]), str(row["initial_class"]))
        for row in systematic_library
    }
    library_cells = [
        diagnostic
        for diagnostic in diagnostics
        if "library_gap_candidate" in diagnostic["labels"]
        and (str(diagnostic["opponent_stage"]), str(diagnostic["initial_class"]))
        in systematic_library_keys
    ]
    terminal_counter = Counter(
        terminal
        for diagnostic in library_cells
        for terminal in (
            diagnostic["fixed_head_on_terminal_category"],
            diagnostic["fixed_crossing_terminal_category"],
        )
    )
    lines = [
        "# 硕士论文态势 Taxonomy 冻结式消融：Step 4 结果审计",
        "",
        "## 1. 审计目的与冻结边界",
        "",
        "本审计用于回答：当前仅含迎头与交叉两种冻结低层技能的层级接口，在不同初始相对几何态势下的不足更接近高层 routing、静态 Oracle 映射、低层技能库，还是对手条件依赖。",
        "",
        "- Source ID：`THESIS-TAXONOMY-ABLATION30-V1`；分析器版本："
        f"`{ANALYSIS_VERSION}`。",
        "- 证据范围：预注册 30 场景 × 4 种冻结方法 × expert/end-to-end 两个 opponent split = 240 条 JSBSim raw episodes。",
        "- 本步骤为只读分析：未训练、未调参、未更换 seed/场景，未改动 PPO、specialist、reward、VPP、guidance 或 PID。",
        "- 零 episode 的 `c049369` 初始化失败目录被保留为基础设施历史，但未作为本分析输入。",
        "",
        "## 2. 完整性与分类核验",
        "",
        f"- 四方法配对完整：{pairing['total_records']}/{pairing['expected_total_records']}；每个 `scenario × opponent` 均有四条同 seed 方法记录。",
        f"- 初始几何复算一致：{initial_verification['match_count']}/{initial_verification['record_count']}。四象限由首两帧的位置差分速度与 LOS 夹角计算；crossing-entry 仅要求显式 `transition`，不伪装为第五个象限。",
        "- 逐 step 统计中，`pre_merge`、`post_merge` 与 post-merge 且重新闭合的 `re_entry` 是描述性阶段标签，不是额外训练任务。",
        "- 结果绝不跨 opponent split 池化；每类只有 6 个显式场景，以下为配对工程诊断而非显著性或普适性结论。",
        "",
        "## 3. 按初始态势的冻结结果",
        "",
        _markdown_table(
            class_summary,
            (
                ("opponent_stage", "对手 split"),
                ("initial_class", "初始态势"),
                ("method", "方法"),
                ("n_total", "N总"),
                ("n_resolved", "N判定"),
                ("wins", "胜"),
                ("losses", "负"),
                ("win_rate_resolved", "判定胜率"),
                ("ego_crash_or_oob", "己方 crash/OOB"),
            ),
        ),
        "",
        "`N判定` 仅包含 win/loss；draw 或 unresolved 不进入胜率分母。因此 `5/5` 与 `N总=6` 并不矛盾。终端语义详见 `terminal_reason_summary.json`，不得把胜率直接解释为击杀率或飞控安全认证。",
        "",
        "## 4. Routing、技能库与组合标签",
        "",
        "- `routing_opportunity`：PPO 负而至少一个固定技能胜，表示现有技能库在该单一情形可能可用、但当前高层组合没有取得该结果。",
        "- `library_gap_candidate`：两个固定技能均负，只能说明需要进一步排除场景可行性、共同执行链和终端机制；不能直接推断“必须新增技能”。",
        "- `composition_gain`：PPO 胜且两个固定技能均负；预注册门槛为同类至少 3/6 且无新增己方 crash/OOB。",
        "",
        _markdown_table(
            threshold_rows,
            (
                ("opponent_stage", "对手 split"),
                ("initial_class", "初始态势"),
                ("routing_opportunity_count", "Routing候选"),
                ("systematic_routing_opportunity", "系统routing"),
                ("library_gap_candidate_count", "技能库候选"),
                ("systematic_library_gap_candidate", "系统技能库"),
                ("composition_gain_count", "组合增益"),
                ("composition_evidence", "组合证据"),
                ("opponent_dependent_count", "对手依赖"),
            ),
        ),
        "",
        "## 5. 可操作的证据结论",
        "",
        (
            "- **expert/disadvantage：低层能力边界候选成立。** 两个固定技能同时失败 "
            f"{systematic_library[0]['library_gap_candidate_count']}/6，覆盖正负镜像和三个高度条件；其中固定技能终端类别的计数为 "
            + ", ".join(f"`{key}`={value}" for key, value in sorted(terminal_counter.items()))
            + "。这足以启动单独的低层 feasibility 审计，但不足以把 Advantage/Disadvantage/Neutral 各固化为一个新 specialist。"
            if systematic_library
            else "- **尚无系统性低层能力边界候选。** 当前证据不足以启动四技能库训练。"
        ),
        (
            "- **expert/head_on：routing 机会成立。** PPO 负而固定技能至少一者胜的情形达到 "
            f"{systematic_routing[0]['routing_opportunity_count']}/6，并覆盖左右镜像。它说明当前两技能库中已有可利用的行为；优先研究的是 mode entry/routing，而不是立刻训练新低层技能。"
            if systematic_routing
            else "- **未出现系统性 routing 机会。** 当前证据不支持将高层选择作为首要瓶颈。"
        ),
        (
            "- **动态组合的有限正证据门槛已达到。** "
            + ", ".join(f"{row['opponent_stage']}/{row['initial_class']}" for row in composition)
            + "。"
            if composition
            else "- **动态组合未达到有限正证据门槛。** 不能据此声称 PPO 的动态切换普遍优于固定技能。"
        ),
        f"- **对手条件不能省略。** {opponent_dependent_scenarios}/30 个同一物理场景在两个 opponent split 下的诊断标签不同；后续必须先制作 opponent capability card，并保持所有结论的 split 条件。",
        f"- **镜像敏感性需要单列。** {len(mirror_flag_keys)} 个 `split × 初始态势 × 方法` 组合在三个高度镜像对中至少两对胜负不同。它是坐标符号/训练偏置/相对航向耦合的审计线索，而不是删减镜像样本的理由。",
        "",
        "## 6. 对共享技能库方案的 Go/No-Go",
        "",
        "本轮不批准直接进入“4 个新技能 + 7 个 intent profiles”的完整训练。原因是 expert/head_on 已存在可用固定技能却未被 PPO 选中，而 expert/disadvantage 的双固定失败仍可能由共同低层执行、终端条件或对手能力造成。",
        "",
        "若继续研发，顺序应为：先对 expert/disadvantage 的候选逐回合检查 terminal reason、VPP 几何和飞控响应，再单独做低层 feasibility gate；并并行完成 opponent capability card。只有低层候选通过该专门 gate，才考虑共享技能库，而不是按五个初始标签机械增加五个专家。",
        "",
        "## 7. 本审计不能证明什么",
        "",
        "- 不能证明五类初始态势必须一一对应五个 specialist，也不能证明新增 specialist 一定提高总体空战能力。",
        "- 不能证明 PPO 具有普适几何识别能力、统计显著优越性或跨对手不变性。",
        "- 不能把 crash/OOB、对手 crash/OOB 或 timeout 单独解释为控制器失稳、气动优势或击杀能力。",
        "- 不能替代独立 opponent capability card 和任何后续低层机制/技能训练的冻结验证。",
    ]
    return "\n".join(lines) + "\n"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def analyze(
    *,
    expert_dir: Path,
    end_to_end_dir: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Analyze immutable formal outputs and write a new isolated artifact root."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis output: {output_dir}")
    manifest = _load_manifest(manifest_path)
    manifest_scenarios = manifest["by_name"]
    expert_records, expert_sources = _load_split("expert", expert_dir, manifest_scenarios)
    end_to_end_records, end_to_end_sources = _load_split(
        "end_to_end", end_to_end_dir, manifest_scenarios
    )
    records = [*expert_records, *end_to_end_records]
    expected_total = len(manifest_scenarios) * len(METHODS) * 2
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total} records; found {len(records)}")
    mismatch_records = [record for record in records if not record["initial_geometry_matches_manifest"]]
    if mismatch_records:
        examples = [
            f"{record['opponent_stage']}/{record['scenario']}/{record['method']}="
            f"{record['initial_geometry_observed']}"
            for record in mismatch_records[:5]
        ]
        raise ValueError("Initial taxonomy verification failed: " + ", ".join(examples))
    pairing = {
        "source_id": SOURCE_ID,
        "expected_total_records": expected_total,
        "total_records": len(records),
        "methods": list(METHODS),
        "scenarios_per_split": len(manifest_scenarios),
        "records_per_split": {
            "expert": len(expert_records),
            "end_to_end": len(end_to_end_records),
        },
        "status": "passed",
    }
    initial_verification = {
        "classifier": "first_two_trajectory_positions_and_velocities",
        "telemetry_alias_contract": {
            "aa_deg": "own-velocity-to-target-LOS angle for step-wise summaries",
            "ata_deg": "target-velocity-to-own-LOS angle for step-wise summaries",
        },
        "transition_band_deg": [85.0, 95.0],
        "record_count": len(records),
        "match_count": len(records) - len(mismatch_records),
        "mismatch_count": len(mismatch_records),
        "status": "passed",
        "records": [
            {
                key: record[key]
                for key in (
                    "opponent_stage",
                    "scenario",
                    "method",
                    "initial_class",
                    "initial_geometry_expected",
                    "initial_geometry_observed",
                    "initial_geometry_matches_manifest",
                    "initial_own_to_target_los_angle_deg",
                    "initial_target_velocity_to_own_los_angle_deg",
                    "initial_los_elevation_deg",
                    "initial_specific_energy_difference_m2_s2",
                )
            }
            for record in records
        ],
    }
    summary_rows = _result_summary(records, ("opponent_stage", "initial_class", "method"))
    method_task_rows = _result_summary(records, ("opponent_stage", "task", "method"))
    diagnostics = _pair_diagnostics(records)
    mirror_rows = _mirror_rows(records)
    threshold_rows = _threshold_rows(diagnostics)
    terminal_summary = _terminal_summary(records)
    telemetry_rows = _telemetry_summary_rows(records)
    capability_card, reference_pressure_rows = _opponent_capability_card(records, SOURCE_ID)
    feasibility_audit, feasibility_scenario_rows, feasibility_detail_rows = (
        _expert_disadvantage_feasibility_audit(records, diagnostics)
    )
    raw_audit_rows = [
        {
            **{
                key: value
                for key, value in record.items()
                if key
                not in {
                    "source_raw_path",
                    "source_raw_sha256",
                    "geometry_state_step_counts",
                    "requested_mode_step_counts",
                    "effective_mode_step_counts",
                    "effective_mode_constraint_reasons",
                    "effective_mode_step_fractions",
                    "requested_mode_step_fractions",
                    "guard_override_reasons",
                    "opponent_config",
                }
            },
            "geometry_state_step_counts": json.dumps(record["geometry_state_step_counts"], sort_keys=True),
            "requested_mode_step_counts": json.dumps(record["requested_mode_step_counts"], sort_keys=True),
            "effective_mode_step_counts": json.dumps(record["effective_mode_step_counts"], sort_keys=True),
            "effective_mode_constraint_reasons": ";".join(record["effective_mode_constraint_reasons"]),
            "effective_mode_step_fractions": json.dumps(
                record["effective_mode_step_fractions"], sort_keys=True
            ),
            "requested_mode_step_fractions": json.dumps(
                record["requested_mode_step_fractions"], sort_keys=True
            ),
            "guard_override_reasons": ";".join(record["guard_override_reasons"]),
            "opponent_config": json.dumps(_json_safe(record["opponent_config"]), sort_keys=True),
        }
        for record in records
    ]
    geometry_audit = {
        "source_id": SOURCE_ID,
        "analysis_version": ANALYSIS_VERSION,
        "taxonomy_contract": {
            "classes": list(INITIAL_CLASSES),
            "crossing_entry_expected_geometry": "transition",
            "classification_input": "geometry only; scenario names are excluded",
            "phase_rule": "post-merge closing range-rate is labeled re_entry",
        },
        "episode_records": records,
        "initial_class_method_summary": summary_rows,
        "threshold_summary": threshold_rows,
        "library_gap_candidate_raw_audits": [
            diagnostic
            for diagnostic in diagnostics
            if "library_gap_candidate" in diagnostic["labels"]
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "episode_pairing_verification.json", pairing)
    _write_json(output_dir / "taxonomy_initial_state_verification.json", initial_verification)
    _write_json(
        output_dir / "method_task_summary.json",
        {
            "source_id": SOURCE_ID,
            "analysis_version": ANALYSIS_VERSION,
            "rows": method_task_rows,
            "initial_class_rows": summary_rows,
        },
    )
    _write_json(output_dir / "terminal_reason_summary.json", terminal_summary)
    _write_json(
        output_dir / "telemetry_summary_by_method.json",
        {"source_id": SOURCE_ID, "analysis_version": ANALYSIS_VERSION, "rows": telemetry_rows},
    )
    _write_csv(output_dir / "telemetry_summary_by_method.csv", telemetry_rows)
    _write_json(output_dir / "opponent_capability_card.json", capability_card)
    (output_dir / "opponent_capability_card_zh.md").write_text(
        _build_opponent_capability_markdown(capability_card), encoding="utf-8"
    )
    _write_csv(output_dir / "opponent_reference_pressure.csv", reference_pressure_rows)
    _write_json(output_dir / "expert_disadvantage_feasibility_audit.json", feasibility_audit)
    _write_csv(
        output_dir / "expert_disadvantage_feasibility_audit_summary.csv",
        feasibility_scenario_rows,
    )
    _write_csv(
        output_dir / "expert_disadvantage_feasibility_audit.csv", feasibility_detail_rows
    )
    (output_dir / "expert_disadvantage_feasibility_audit.md").write_text(
        _build_feasibility_markdown(feasibility_audit), encoding="utf-8"
    )
    _write_json(output_dir / "geometry_taxonomy_audit.json", geometry_audit)
    _write_csv(output_dir / "geometry_taxonomy_audit.csv", raw_audit_rows)
    _write_csv(
        output_dir / "diagnostic_label_by_episode.csv",
        [
            {
                **{key: value for key, value in diagnostic.items() if key != "library_gap_raw_geometry_audit"},
                "library_gap_raw_geometry_audit": json.dumps(
                    _json_safe(diagnostic["library_gap_raw_geometry_audit"]), sort_keys=True
                )
                if diagnostic["library_gap_raw_geometry_audit"] is not None
                else "",
            }
            for diagnostic in diagnostics
        ],
    )
    _write_csv(output_dir / "initial_class_opponent_summary.csv", summary_rows)
    _write_csv(output_dir / "mirror_pair_summary.csv", mirror_rows)
    _write_csv(output_dir / "terminal_reason_summary.csv", terminal_summary["rows"])
    _write_csv(output_dir / "diagnostic_threshold_summary.csv", threshold_rows)
    report = _build_report(
        pairing,
        initial_verification,
        summary_rows,
        diagnostics,
        threshold_rows,
        mirror_rows,
    )
    (output_dir / "geometry_taxonomy_audit.md").write_text(report, encoding="utf-8")
    (output_dir / "thesis_taxonomy_ablation30_result_zh.md").write_text(
        _build_chinese_report(
            pairing,
            initial_verification,
            summary_rows,
            diagnostics,
            threshold_rows,
            mirror_rows,
        ),
        encoding="utf-8",
    )
    source_manifest = {
        "source_id": SOURCE_ID,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_git_sha": _git_sha(),
        "analysis_code_sha256": _file_sha256(Path(__file__)),
        "taxonomy_classifier_sha256": _file_sha256(
            ROOT / "src" / "uav_vpp_guidance" / "evaluation" / "engagement_geometry_taxonomy.py"
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "input_runs": {"expert": expert_sources, "end_to_end": end_to_end_sources},
        "excluded_invalid_history": ["expert-c049369-20260713 (zero-episode initialization failure)"],
        "generated_output_sha256": {
            path.name: _file_sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    _write_json(output_dir / "artifact_source_hash_manifest.json", source_manifest)
    return {
        "output_dir": str(output_dir),
        "record_count": len(records),
        "diagnostic_cells": len(diagnostics),
        "initial_verification": initial_verification,
        "threshold_rows": threshold_rows,
        "telemetry_summary_rows": len(telemetry_rows),
        "feasibility_candidate_count": feasibility_audit["candidate_scenario_count"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dir", type=Path, required=True)
    parser.add_argument("--end-to-end-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config" / "experiment" / "manifests" / "thesis_taxonomy_ablation30_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        expert_dir=args.expert_dir.resolve(),
        end_to_end_dir=args.end_to_end_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "record_count": result["record_count"],
                "diagnostic_cells": result["diagnostic_cells"],
                "initial_match_count": result["initial_verification"]["match_count"],
                "initial_mismatch_count": result["initial_verification"]["mismatch_count"],
                "threshold_rows": result["threshold_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
