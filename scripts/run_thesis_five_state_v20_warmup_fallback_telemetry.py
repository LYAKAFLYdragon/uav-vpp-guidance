#!/usr/bin/env python3
"""V20 wrapper: admit only the preregistered predictor warm-up fallback."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import run_thesis_five_state_v19_windowed_phase_telemetry as base
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import classify_relative_geometry


SOURCE_ID = "THESIS-FIVE-STATE-V20-WARMUP-FALLBACK-TELEMETRY-AUDIT"


def _window(step: Mapping[str, Any]) -> str:
    history = int(step["history_length"])
    valid, fallback = bool(step["prediction_valid"]), bool(step["prediction_fallback"])
    warmup = history < 10 or not valid
    if fallback and (not warmup or step.get("prediction_fallback_reason") != "fallback_to_current_target" or step.get("prediction_fallback_phase") != "configured_current_target"):
        raise RuntimeError("V20 observed an unregistered predictor fallback")
    return "analysis_ready" if history == 10 and valid and not fallback else "warmup"


def _summary(records: Sequence[Mapping[str, Any]], *, expected: Mapping[str, Mapping[str, int]], records_per_opponent: int, minimum_ready_steps: int) -> dict[str, Any]:
    groups = {name: [] for name in base.OPPONENTS}
    for record in records:
        if record.get("source_id") != SOURCE_ID or record.get("opponent") not in groups: raise RuntimeError("V20 record identity mismatch")
        groups[record["opponent"]].append(record)
    matrix = {}
    for opponent, rows in groups.items():
        windows = Counter(_window(step) for row in rows for step in row["steps"])
        actual = {"handoff_reached": sum(bool(row.get("handoff_reached")) for row in rows), "valid_path_context_steps": sum(int(row.get("window_steps", 0)) for row in rows)}
        matrix[opponent] = {"records": len(rows), "actual": actual, "expected_v17": dict(expected[opponent]), "window_steps": dict(windows), "phase_replay_matches_v17": actual == dict(expected[opponent])}
        matrix[opponent]["passes"] = len(rows) == records_per_opponent and matrix[opponent]["phase_replay_matches_v17"] and windows["analysis_ready"] >= minimum_ready_steps
    return {"source_id": SOURCE_ID, "records": len(records), "expected_records": records_per_opponent * len(base.OPPONENTS), "matrix": matrix, "passes": all(cell["passes"] for cell in matrix.values()) and len(records) == records_per_opponent * len(base.OPPONENTS)}


class V20Episode(base.WindowedTelemetryEpisode):
    def _telemetry_step(self) -> dict[str, Any]:
        if self.base is None: raise RuntimeError("V20 telemetry without observation")
        base_vector, info = self.base, self.last_info
        if bool(info.get("backend_fallback_occurred", False)): raise RuntimeError("V20 backend fallback")
        ata, aa = math.atan2(float(base_vector[8]), float(base_vector[9])), math.atan2(float(base_vector[10]), float(base_vector[11]))
        own_speed, target_speed, own_altitude, target_altitude = (float(base_vector[index]) for index in (12, 13, 14, 15)); own = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
        step = {"step": int(self.total_steps), "base_observation_16d": list(base_vector), "phase": self._phase(), "first_pass_complete": bool(self.phase_tracker.first_pass_complete), "history_length": len(self.history), "prediction_valid": bool(info.get("prediction_valid", False)), "prediction_fallback": bool(info.get("prediction_fallback", False)), "prediction_fallback_reason": info.get("prediction_fallback_reason"), "prediction_fallback_phase": info.get("prediction_fallback_phase"), "own_specific_energy_j_per_kg": .5*own_speed**2+9.80665*own_altitude, "target_specific_energy_j_per_kg": .5*target_speed**2+9.80665*target_altitude, "actual_nz_g": float(own.get("nz_g", info.get("nz_g"))), "dynamic_taxonomy": classify_relative_geometry(math.degrees(ata), math.degrees(aa)), "vpp_world_offset_m": self._finite_vector(info.get("vp_world_offset"), "vp_world_offset"), "guidance_command_normalized": self._guidance_vector(info)}
        _window(step); return step


base.SOURCE_ID = SOURCE_ID
base.WindowedTelemetryEpisode = V20Episode
base.summarize_records = _summary

if __name__ == "__main__": raise SystemExit(base.main())
