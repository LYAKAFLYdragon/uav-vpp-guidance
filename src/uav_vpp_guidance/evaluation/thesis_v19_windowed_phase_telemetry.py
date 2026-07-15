"""Validation for V19's predictor-warm-up and analysis-ready telemetry windows."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-FIVE-STATE-V19-WINDOWED-PHASE-TELEMETRY-AUDIT"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


class V19WindowedTelemetryError(ValueError):
    """Raised when a V19 record violates the windowed observability contract."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_step(step: Mapping[str, Any]) -> str:
    required = ("base_observation_16d", "phase", "first_pass_complete", "history_length", "prediction_valid", "prediction_fallback")
    if any(key not in step for key in required):
        raise V19WindowedTelemetryError("step is missing a window-contract field")
    base = step["base_observation_16d"]
    if not isinstance(base, list) or len(base) != 16 or not all(_finite(item) for item in base):
        raise V19WindowedTelemetryError("step lacks a finite 16-D base observation")
    if not isinstance(step["history_length"], int) or not 0 <= step["history_length"] <= 10:
        raise V19WindowedTelemetryError("history length is invalid")
    if bool(step["prediction_fallback"]):
        raise V19WindowedTelemetryError("prediction fallback is prohibited in both windows")
    ready = step["history_length"] == 10 and bool(step["prediction_valid"])
    if not ready:
        return "warmup"
    for key in ("own_specific_energy_j_per_kg", "target_specific_energy_j_per_kg", "actual_nz_g"):
        if not _finite(step.get(key)):
            raise V19WindowedTelemetryError("analysis-ready step lacks required finite telemetry")
    return "analysis_ready"


def summarize_records(records: Sequence[Mapping[str, Any]], *, expected: Mapping[str, Mapping[str, int]], records_per_opponent: int = 24, minimum_ready_steps: int = 20) -> dict[str, Any]:
    """Apply unpooled V19 gates without treating warm-up as a predictor failure."""

    groups = {opponent: [] for opponent in OPPONENTS}
    for record in records:
        opponent = str(record.get("opponent"))
        if str(record.get("source_id")) != SOURCE_ID or opponent not in groups or not isinstance(record.get("steps"), list):
            raise V19WindowedTelemetryError("record identity or telemetry is invalid")
        groups[opponent].append(record)
    matrix: dict[str, Any] = {}
    for opponent, group in groups.items():
        windows = Counter(_valid_step(step) for record in group for step in record["steps"])
        actual = {"handoff_reached": sum(bool(row.get("handoff_reached")) for row in group), "valid_path_context_steps": sum(int(row.get("window_steps", 0)) for row in group)}
        matrix[opponent] = {"records": len(group), "actual": actual, "expected_v17": dict(expected[opponent]), "window_steps": dict(windows), "phase_replay_matches_v17": actual == dict(expected[opponent])}
        matrix[opponent]["passes"] = len(group) == records_per_opponent and matrix[opponent]["phase_replay_matches_v17"] and windows["analysis_ready"] >= minimum_ready_steps
    return {"source_id": SOURCE_ID, "records": len(records), "expected_records": records_per_opponent * len(OPPONENTS), "matrix": matrix, "passes": len(records) == records_per_opponent * len(OPPONENTS) and all(cell["passes"] for cell in matrix.values())}
