"""Read-only phase-occupancy attribution for V17 path-context records."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-FIVE-STATE-PATH-CONTEXT-PREFLIGHT-V17"
PHASES = frozenset({"pre_merge", "post_merge", "re_entry"})


class V17PhaseOccupancyAuditError(ValueError):
    """Raised when V17 records cannot support a phase-occupancy attribution."""


def summarize_phase_occupancy(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize first-pass, phase occupancy, and handoff without inferring cause."""

    by_opponent: dict[str, list[Mapping[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for record in records:
        if str(record.get("source_id")) != SOURCE_ID:
            raise V17PhaseOccupancyAuditError("record source ID does not match V17")
        opponent = str(record.get("opponent") or "")
        pair_key = str(record.get("pair_key") or "")
        trace = record.get("phase_trace")
        if not opponent or not pair_key or not isinstance(trace, list) or not trace:
            raise V17PhaseOccupancyAuditError("record lacks identity or phase trace")
        if (opponent, pair_key) in seen:
            raise V17PhaseOccupancyAuditError("duplicate opponent/pair key")
        seen.add((opponent, pair_key))
        by_opponent.setdefault(opponent, []).append(record)

    matrix: dict[str, dict[str, Any]] = {}
    for opponent, group in sorted(by_opponent.items()):
        phase_steps: Counter[str] = Counter()
        terminal_reasons: Counter[str] = Counter()
        handoff_reached = 0
        first_pass_latched = 0
        episodes_with_post_merge = 0
        episodes_with_re_entry = 0
        first_pass_without_post_merge = 0
        candidates: list[dict[str, Any]] = []

        for record in group:
            trace = record["phase_trace"]
            phases = [str(step.get("phase") or "") for step in trace]
            if any(phase not in PHASES for phase in phases):
                raise V17PhaseOccupancyAuditError("phase trace contains an unknown phase")
            latched = any(bool(step.get("first_pass_complete")) for step in trace)
            has_post_merge = "post_merge" in phases
            has_re_entry = "re_entry" in phases
            handoff = bool(record.get("handoff_reached"))
            phase_steps.update(phases)
            terminal_reasons[str(record.get("terminal_reason") or "unknown")] += 1
            handoff_reached += int(handoff)
            first_pass_latched += int(latched)
            episodes_with_post_merge += int(has_post_merge)
            episodes_with_re_entry += int(has_re_entry)
            first_pass_without_post_merge += int(latched and not has_post_merge)
            candidates.append(
                {
                    "pair_key": str(record["pair_key"]),
                    "handoff_reached": handoff,
                    "window_steps": int(record.get("window_steps", 0)),
                    "first_pass_latched": latched,
                    "post_merge_observed": has_post_merge,
                    "re_entry_observed": has_re_entry,
                    "terminal_reason": str(record.get("terminal_reason") or "unknown"),
                }
            )

        matrix[opponent] = {
            "records": len(group),
            "handoff_reached": handoff_reached,
            "first_pass_latched": first_pass_latched,
            "episodes_with_post_merge": episodes_with_post_merge,
            "episodes_with_re_entry": episodes_with_re_entry,
            "first_pass_without_post_merge": first_pass_without_post_merge,
            "phase_steps": dict(sorted(phase_steps.items())),
            "terminal_reasons": dict(sorted(terminal_reasons.items())),
            "records_by_pair": sorted(candidates, key=lambda item: item["pair_key"]),
        }
    return {"source_id": SOURCE_ID, "matrix": matrix}
