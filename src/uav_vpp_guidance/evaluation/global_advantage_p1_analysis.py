"""Read-only analysis for global-advantage P1 run-in evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


class GlobalAdvantageP1AnalysisError(ValueError):
    """Raised when P1 evidence is incomplete or tampered with."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_p1_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify each raw artifact and summarize the frozen P1 gate evidence."""

    gate = manifest.get("gate")
    episodes = manifest.get("episodes")
    artifacts = manifest.get("artifacts")
    if not isinstance(gate, Mapping) or not isinstance(episodes, Sequence):
        raise GlobalAdvantageP1AnalysisError("P1 manifest is missing gate or episodes")
    if not isinstance(artifacts, Sequence) or len(artifacts) != len(episodes):
        raise GlobalAdvantageP1AnalysisError("P1 artifact/episode count mismatch")
    artifact_by_path = {str(item["path"]): item for item in artifacts}
    rows: list[dict[str, Any]] = []
    reset_payload_presence: list[bool] = []
    for episode in episodes:
        path = Path(str(episode["artifact_path"]))
        expected = str(episode["artifact_sha256"])
        if not path.is_file() or sha256_file(path) != expected:
            raise GlobalAdvantageP1AnalysisError(f"episode artifact hash mismatch: {path}")
        listed = artifact_by_path.get(str(path))
        if listed is None or str(listed.get("sha256")) != expected:
            raise GlobalAdvantageP1AnalysisError(f"artifact manifest mismatch: {path}")
        raw_episode = json.loads(path.read_text(encoding="utf-8"))
        header = raw_episode.get("header", {})
        if header.get("scenario_name") != episode.get("scenario_name"):
            raise GlobalAdvantageP1AnalysisError(f"episode header mismatch: {path}")
        metadata = header.get("scenario_metadata", {})
        reset_payload_presence.append(
            "observation_vector" in raw_episode.get("reset", {})
        )
        rows.append(
            {
                "opponent": str(episode["opponent"]),
                "scenario_signature": str(episode["scenario_signature"]),
                "scenario_name": str(episode["scenario_name"]),
                "initial_class": metadata.get("initial_class"),
                "height_condition": metadata.get("height_condition"),
                "mirror_sign": metadata.get("mirror_sign"),
                "repeat_index": int(episode["repeat_index"]),
                "order_variant": str(episode["order_variant"]),
                "reset_state_sha256": str(episode["reset_state_sha256"]),
                "trajectory_sha256": str(episode["trajectory_sha256"]),
                "boundary_state_sha256": str(episode["boundary_state_sha256"]),
                "terminal_reason": str(episode["terminal_reason"]),
                "step_count": int(episode["step_count"]),
                "telemetry_complete": bool(episode["telemetry_complete"]),
                "no_backend_or_prediction_fallback": bool(
                    episode["no_backend_or_prediction_fallback"]
                ),
                "prediction_warmup_steps": int(episode.get("prediction_warmup_steps", 0)),
                "artifact_path": str(path),
                "artifact_sha256": expected,
            }
        )
    comparisons = list(gate.get("comparison_rows", []))
    if len(comparisons) * 3 != len(rows):
        raise GlobalAdvantageP1AnalysisError("P1 repeat/comparison cardinality mismatch")
    per_opponent: dict[str, dict[str, Any]] = {}
    for opponent in sorted({row["opponent"] for row in rows}):
        comparison_rows = [row for row in comparisons if row.get("opponent") == opponent]
        first_diffs = Counter(
            row.get("first_trajectory_difference_step")
            for row in comparison_rows
            if row.get("first_trajectory_difference_step") is not None
        )
        per_opponent[opponent] = {
            "comparison_cells": len(comparison_rows),
            "passed_cells": sum(bool(row.get("passed")) for row in comparison_rows),
            "reset_state_equal_cells": sum(
                bool(row.get("reset_state_equal")) for row in comparison_rows
            ),
            "trajectory_equal_cells": sum(
                bool(row.get("trajectory_equal")) for row in comparison_rows
            ),
            "boundary_state_equal_cells": sum(
                bool(row.get("boundary_state_equal")) for row in comparison_rows
            ),
            "terminal_equal_cells": sum(
                bool(row.get("terminal_equal")) for row in comparison_rows
            ),
            "telemetry_complete_cells": sum(
                bool(row.get("telemetry_complete")) for row in comparison_rows
            ),
            "fallback_free_cells": sum(
                bool(row.get("no_backend_or_prediction_fallback"))
                for row in comparison_rows
            ),
            "first_difference_step_distribution": dict(sorted(first_diffs.items())),
        }
    reset_observation_full_payload_present = bool(reset_payload_presence) and all(
        reset_payload_presence
    )
    return {
        "source_id": manifest.get("source_id"),
        "mode": "read_only_posthoc_audit",
        "artifact_count": len(rows),
        "comparison_cell_count": len(comparisons),
        "gate_passed": bool(gate.get("passed")),
        "gate_failure_count": int(gate.get("failure_count", 0)),
        "per_opponent": per_opponent,
        "rows": rows,
        "findings": {
            "reset_base_state_reproducible": all(
                item["reset_state_equal_cells"] == item["comparison_cells"]
                for item in per_opponent.values()
            ),
            "full_trajectory_reproducible": all(
                item["trajectory_equal_cells"] == item["comparison_cells"]
                for item in per_opponent.values()
            ),
            "telemetry_complete": all(
                item["telemetry_complete_cells"] == item["comparison_cells"]
                for item in per_opponent.values()
            ),
            "hard_fallback_absent": all(
                item["fallback_free_cells"] == item["comparison_cells"]
                for item in per_opponent.values()
            ),
            "full_reset_observation_payload_persisted": reset_observation_full_payload_present,
        },
        "verdict": (
            "runin_protocol_not_reproducible_do_not_train"
            if not bool(gate.get("passed"))
            else "runin_protocol_reproducible"
        ),
        "next_allowed_action": (
            "design_fresh_environment_per_episode_r3_with_full_reset_input_and_state_hashes"
            if not bool(gate.get("passed"))
            else "freeze_p1_and_request_p2_authorization"
        ),
    }
