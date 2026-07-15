"""Pure validation for V18's non-learning phase-telemetry records."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-FIVE-STATE-V18-PHASE-TELEMETRY-AUDIT"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REQUIRED_STEP_FIELDS = (
    "base_observation_16d", "raw_range_m", "raw_range_rate_mps", "ata_rad", "aa_rad",
    "los_azimuth_rad", "los_elevation_rad", "own_speed_mps", "target_speed_mps",
    "own_altitude_m", "target_altitude_m", "own_specific_energy_j_per_kg",
    "target_specific_energy_j_per_kg", "dynamic_taxonomy", "phase",
    "first_pass_complete", "vpp_world_offset_m", "guidance_command_normalized",
    "actual_nz_g", "saturation_flags",
)


class V18PhaseTelemetryError(ValueError):
    """Raised when a V18 record cannot support the preregistered audit."""


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_vector(value: Any, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(_finite_scalar(item) for item in value)


def validate_step(step: Mapping[str, Any]) -> None:
    """Fail closed on incomplete or non-finite telemetry."""

    missing = [key for key in REQUIRED_STEP_FIELDS if key not in step]
    if missing:
        raise V18PhaseTelemetryError(f"telemetry is missing required fields: {missing}")
    if not _finite_vector(step["base_observation_16d"], 16):
        raise V18PhaseTelemetryError("base observation is not a finite 16-D vector")
    for key in REQUIRED_STEP_FIELDS:
        if key in {"base_observation_16d", "dynamic_taxonomy", "phase", "first_pass_complete", "saturation_flags", "vpp_world_offset_m", "guidance_command_normalized"}:
            continue
        if not _finite_scalar(step[key]):
            raise V18PhaseTelemetryError(f"telemetry field is non-finite: {key}")
    if str(step["dynamic_taxonomy"]) not in {"advantage", "head_on", "disadvantage", "neutral", "transition", "unknown"}:
        raise V18PhaseTelemetryError("dynamic taxonomy is invalid")
    if str(step["phase"]) not in {"pre_merge", "post_merge", "re_entry"}:
        raise V18PhaseTelemetryError("phase is invalid")
    if not _finite_vector(step["vpp_world_offset_m"], 3) or not _finite_vector(step["guidance_command_normalized"], 3):
        raise V18PhaseTelemetryError("VPP or guidance telemetry is invalid")
    flags = step["saturation_flags"]
    if not isinstance(flags, Mapping) or set(flags) != {"nz", "roll_rate", "throttle"} or not all(isinstance(value, bool) for value in flags.values()):
        raise V18PhaseTelemetryError("saturation telemetry is invalid")


def summarize_records(
    records: Sequence[Mapping[str, Any]], *, expected: Mapping[str, Mapping[str, int]], records_per_opponent: int = 24
) -> dict[str, Any]:
    """Verify exact V17 phase replay plus complete V18 telemetry, per opponent."""

    grouped = {opponent: [] for opponent in OPPONENTS}
    seen: set[tuple[str, str]] = set()
    for record in records:
        opponent = str(record.get("opponent"))
        pair_key = str(record.get("pair_key"))
        if str(record.get("source_id")) != SOURCE_ID or opponent not in grouped or not pair_key:
            raise V18PhaseTelemetryError("record identity is outside the V18 contract")
        if (opponent, pair_key) in seen:
            raise V18PhaseTelemetryError("duplicate opponent/pair record")
        seen.add((opponent, pair_key))
        steps = record.get("steps")
        if not isinstance(steps, list) or not steps:
            raise V18PhaseTelemetryError("record contains no telemetry steps")
        for step in steps:
            if not isinstance(step, Mapping):
                raise V18PhaseTelemetryError("telemetry step is not a mapping")
            validate_step(step)
        grouped[opponent].append(record)

    matrix: dict[str, Any] = {}
    for opponent, group in grouped.items():
        handoffs = sum(bool(row.get("handoff_reached")) for row in group)
        valid_path_steps = sum(int(row.get("window_steps", 0)) for row in group)
        phase_steps = Counter(str(step["phase"]) for row in group for step in row["steps"])
        actual = {"handoff_reached": handoffs, "valid_path_context_steps": valid_path_steps}
        expected_cell = dict(expected[opponent])
        matrix[opponent] = {
            "records": len(group),
            "actual": actual,
            "expected_v17": expected_cell,
            "phase_steps": dict(sorted(phase_steps.items())),
            "phase_replay_matches_v17": actual == expected_cell,
            "telemetry_complete": True,
        }
        matrix[opponent]["passes"] = len(group) == records_per_opponent and matrix[opponent]["phase_replay_matches_v17"]
    return {"source_id": SOURCE_ID, "records": len(records), "expected_records": records_per_opponent * len(OPPONENTS), "matrix": matrix, "passes": len(records) == records_per_opponent * len(OPPONENTS) and all(cell["passes"] for cell in matrix.values())}
