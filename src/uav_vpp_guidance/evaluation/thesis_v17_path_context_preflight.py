"""V17 record-universe and per-opponent sampler-readiness gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-FIVE-STATE-PATH-CONTEXT-PREFLIGHT-V17"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


def is_ego_failure(reason: Any) -> bool:
    normalized = str(reason or "").lower()
    return not normalized.startswith("target_") and (
        "crash" in normalized or "out_of_bounds" in normalized
    )


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios_per_opponent: int,
    min_qualifying_episodes: int,
    min_valid_steps: int,
    min_mirrors: int,
    min_signatures: int,
) -> dict[str, Any]:
    """Apply V17's unpooled, data-contract-only readiness gate."""

    expected_records = len(OPPONENTS) * scenarios_per_opponent
    by_opponent = {opponent: [] for opponent in OPPONENTS}
    seen: set[tuple[str, str]] = set()
    for record in records:
        opponent = str(record.get("opponent"))
        pair_key = str(record.get("pair_key"))
        if opponent not in by_opponent:
            raise ValueError("V17 record is outside the frozen opponent registry")
        if (opponent, pair_key) in seen:
            raise ValueError("V17 record universe contains a duplicate")
        seen.add((opponent, pair_key))
        by_opponent[opponent].append(record)

    matrix: dict[str, dict[str, Any]] = {}
    for opponent, group in by_opponent.items():
        qualifying = [
            row
            for row in group
            if bool(row.get("handoff_reached"))
            and bool(row.get("contract_passed"))
            and int(row.get("window_steps", 0)) >= 20
        ]
        steps = [step for row in qualifying for step in row.get("window", [])]
        mirrors = {str(row.get("mirror_sign")) for row in qualifying}
        signatures = {str(row.get("scenario_signature")) for row in qualifying}
        anchors = {int(step.get("selected_anchor_index", -1)) for step in steps}
        finite_context = all(bool(step.get("path_context_finite")) for step in steps)
        finite_action = all(bool(step.get("raw_action_finite")) for step in steps)
        no_fallback = all(
            bool(step.get("prediction_valid")) and not bool(step.get("prediction_fallback"))
            for step in steps
        )
        ego_failures = sum(bool(row.get("ego_failure")) for row in group)
        cell = {
            "records": len(group),
            "qualifying_episodes": len(qualifying),
            "valid_path_context_steps": len(steps),
            "mirrors": sorted(mirrors),
            "scenario_signatures": len(signatures),
            "anchors_observed": sorted(anchors),
            "path_context_finite": finite_context,
            "raw_action_finite": finite_action,
            "prediction_without_fallback": no_fallback,
            "ego_crash_oob_count": ego_failures,
        }
        cell["passes"] = (
            cell["records"] == scenarios_per_opponent
            and cell["qualifying_episodes"] >= min_qualifying_episodes
            and cell["valid_path_context_steps"] >= min_valid_steps
            and len(mirrors) >= min_mirrors
            and len(signatures) >= min_signatures
            and {0, 2}.issubset(anchors)
            and finite_context
            and finite_action
            and no_fallback
            and ego_failures == 0
        )
        matrix[opponent] = cell
    return {
        "records": len(records),
        "expected_records": expected_records,
        "matrix": matrix,
        "passes": len(records) == expected_records and all(cell["passes"] for cell in matrix.values()),
    }
