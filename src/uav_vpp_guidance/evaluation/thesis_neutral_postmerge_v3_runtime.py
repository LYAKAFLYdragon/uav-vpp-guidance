"""Runtime-record serializer for the future V3 non-learning feasibility run."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v3_contract import (
    SOURCE_ID,
    NeutralPostMergeV3ContractError,
    episode_identity,
)


def runtime_record(
    *,
    opponent: str,
    scenario: Mapping[str, Any],
    result: Mapping[str, Any],
    ledger: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize one frozen-baseline episode without admitting paired scores.

    The V1 engine supplies the physical episode; V3 owns the persistent
    identity and fail-closed record contract.
    """

    identity = episode_identity(scenario)
    if identity["kind"] != "evaluation" or identity["paired_delta_eligible"] is not True:
        raise NeutralPostMergeV3ContractError("V3 runtime records require an evaluation identity")
    if opponent not in ("expert", "end_to_end", "independent_ppo_vpp"):
        raise NeutralPostMergeV3ContractError("unexpected V3 opponent")
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        raise NeutralPostMergeV3ContractError("V3 runtime scenario has no metadata")
    required_result = (
        "handoff_reached",
        "valid_target_steps",
        "qualifying",
        "telemetry_complete",
        "terminal_reason",
    )
    if any(key not in result for key in required_result):
        raise NeutralPostMergeV3ContractError("V3 runtime result lacks required fields")
    summary = ledger.get("summary") if isinstance(ledger.get("summary"), Mapping) else None
    if summary is None or summary.get("no_backend_fallback") is not True:
        raise NeutralPostMergeV3ContractError("V3 runtime ledger reports a backend fallback or lacks summary")
    if summary.get("full_post_handoff_telemetry") is not True:
        raise NeutralPostMergeV3ContractError("V3 runtime ledger lacks complete post-handoff telemetry")
    if bool(result["telemetry_complete"]) is not True:
        raise NeutralPostMergeV3ContractError("V3 runtime result is not telemetry-complete")
    for key in (
        "scenario_application_passed",
        "raw_si_phase_replay_passed",
        "initial_pre_merge_semantics_passed",
    ):
        if runtime_evidence.get(key) is not True:
            raise NeutralPostMergeV3ContractError(f"V3 runtime evidence failed: {key}")
    return {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "family_id": metadata.get("feasibility_family_id"),
        "pair_key": identity["key"],
        "metadata": copy.deepcopy(dict(metadata)),
        "result": {
            "handoff_reached": bool(result["handoff_reached"]),
            "valid_target_steps": int(result["valid_target_steps"]),
            "qualifying": bool(result["qualifying"]),
            "telemetry_complete": True,
            "terminal_reason": str(result["terminal_reason"]),
            "handoff_state_sha256": result.get("handoff_state_sha256"),
        },
        "runtime_evidence": {
            "scenario_application_passed": True,
            "raw_si_phase_replay_passed": True,
            "initial_pre_merge_semantics_passed": True,
        },
        "paired_delta_eligible": False,
        "role": "nonlearning_frozen_baseline_runtime_feasibility",
    }
