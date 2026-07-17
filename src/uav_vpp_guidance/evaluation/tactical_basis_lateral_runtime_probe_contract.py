"""Pure design-contract checks for the selected post-merge runtime probe."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-LATERAL-RUNTIME-PROBE-V1"
P2_SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1"
P2_SELECTED_VARIANT = "lateral_extension:lat+1:vert+1"
METHODS = ("fixed_head_on", "lateral_extension_lat_pos_vert_pos")
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REQUIRED_OPPONENTS = ("end_to_end", "independent_ppo_vpp")


class TacticalBasisLateralRuntimeProbeDesignError(ValueError):
    """Raised when the future runtime probe drifts from the frozen P2 result."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TacticalBasisLateralRuntimeProbeDesignError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TacticalBasisLateralRuntimeProbeDesignError(f"{name} must be a sequence")
    return value


def validate_design_spec(
    config: Mapping[str, Any],
    p2_summary: Mapping[str, Any],
    p2_verification: Mapping[str, Any],
    probe_manifest: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the design before any runtime policy adapter is implemented."""

    if config.get("source_id") != SOURCE_ID:
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe source ID drifted")
    if config.get("stage") != "paired_nonlearning_postmerge_runtime_probe":
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe stage drifted")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "execution_permitted",
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "high_level_ppo_training_permitted",
        "shared_skill_training_permitted",
        "guidance_pid_change_permitted",
        "action_envelope_expansion_permitted",
        "snapshot_restore_permitted",
        "future_state_injection_permitted",
        "history_padding_permitted",
    ):
        if authorization.get(key) is not False:
            raise TacticalBasisLateralRuntimeProbeDesignError(f"runtime probe must prohibit {key}")

    candidate = _mapping(config.get("candidate"), "candidate")
    if (
        candidate.get("p2_source_id") != P2_SOURCE_ID
        or candidate.get("p2_selected_variant") != P2_SELECTED_VARIANT
        or candidate.get("method_id") != METHODS[1]
        or int(candidate.get("program_steps", -1)) != 40
        or candidate.get("post_program_controller") != "frozen_fixed_head_on"
    ):
        raise TacticalBasisLateralRuntimeProbeDesignError("candidate is not the unique P2-selected program")
    if tuple(_sequence(_mapping(config.get("methods"), "methods").get("order"), "methods.order")) != METHODS:
        raise TacticalBasisLateralRuntimeProbeDesignError("method matrix is not the fixed paired comparison")
    opponents = _mapping(config.get("opponents"), "opponents")
    if tuple(_sequence(opponents.get("order"), "opponents.order")) != OPPONENTS:
        raise TacticalBasisLateralRuntimeProbeDesignError("opponent registry drifted")
    if tuple(_sequence(opponents.get("required_for_gate"), "opponents.required_for_gate")) != REQUIRED_OPPONENTS:
        raise TacticalBasisLateralRuntimeProbeDesignError("required opponents drifted")
    if opponents.get("pooled_gate") != "prohibited":
        raise TacticalBasisLateralRuntimeProbeDesignError("pooled gate must remain prohibited")

    if (
        p2_summary.get("source_id") != P2_SOURCE_ID
        or p2_summary.get("passes") is not True
        or p2_summary.get("selected_variant") != P2_SELECTED_VARIANT
        or p2_summary.get("next_action") != "one_selected_variant_runtime_probe_design_permitted"
        or p2_verification.get("source_id") != P2_SOURCE_ID
        or p2_verification.get("passes") is not True
        or int(p2_verification.get("rows_verified", -1)) != 136
    ):
        raise TacticalBasisLateralRuntimeProbeDesignError("P2 selection provenance is not valid")

    if probe_manifest.get("source_id") != SOURCE_ID + "-MANIFEST12":
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest source ID drifted")
    if probe_manifest.get("parent_manifest_sha256") != config["inputs"]["parent_manifest_sha256"]:
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest parent hash drifted")
    if parent_manifest.get("source_id") != probe_manifest.get("parent_manifest_source_id"):
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest parent source drifted")
    parent_by_name = {
        str(item.get("name")): item
        for item in _sequence(parent_manifest.get("scenarios"), "parent_manifest.scenarios")
        if isinstance(item, Mapping)
    }
    scenarios = _sequence(probe_manifest.get("scenarios"), "probe_manifest.scenarios")
    if len(scenarios) != 12 or len({str(item.get("name")) for item in scenarios if isinstance(item, Mapping)}) != 12:
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest must reference exactly 12 scenarios")
    for item in scenarios:
        entry = _mapping(item, "probe_manifest.scenario")
        name = str(entry.get("name") or "")
        parent = parent_by_name.get(name)
        if parent is None:
            raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe scenario is absent from the frozen parent")
        parent_metadata = _mapping(parent.get("metadata"), "parent_manifest.metadata")
        if int(entry.get("scenario_seed", -1)) != int(parent_metadata.get("scenario_seed", -2)):
            raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe scenario seed differs from the frozen parent")

    contract = _mapping(config.get("contract"), "contract")
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("action_semantics") != "tactical_basis_v1"
        or contract.get("reset_at_handoff") is not False
        or contract.get("snapshot_restore") is not False
        or contract.get("future_state_injection") is not False
        or contract.get("history_padding") is not False
        or contract.get("routing_enabled") is not False
        or tuple(_sequence(contract.get("target_phases"), "contract.target_phases")) != ("post_merge",)
        or contract.get("target_dynamic_state") != "disadvantage"
    ):
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime causal contract drifted")
    metrics = _mapping(config.get("metrics"), "metrics")
    if int(metrics.get("response_window_steps", -1)) != 40:
        raise TacticalBasisLateralRuntimeProbeDesignError("response window must equal the selected program")
    if float(_mapping(metrics.get("primary_pressure"), "metrics.primary_pressure").get("candidate_minus_baseline_max", 1.0)) >= 0.0:
        raise TacticalBasisLateralRuntimeProbeDesignError("primary pressure metric must require a reduction")
    secondary = _mapping(metrics.get("secondary_geometry"), "metrics.secondary_geometry")
    if not all(float(secondary.get(key, 0.0)) > 0.0 for key in ("range_rate_mean_min_mps", "end_range_min_m", "end_specific_energy_min_m")):
        raise TacticalBasisLateralRuntimeProbeDesignError("secondary geometry MDEs must be positive")
    gates = _mapping(config.get("gates"), "gates")
    if int(gates.get("min_qualifying_pairs_per_required_opponent", -1)) < 4 or int(gates.get("min_distinct_signatures_per_required_opponent", -1)) < 4:
        raise TacticalBasisLateralRuntimeProbeDesignError("pair coverage gate is weaker than the P2 provenance gate")

    return {
        "source_id": SOURCE_ID,
        "candidate_variant": P2_SELECTED_VARIANT,
        "methods": list(METHODS),
        "opponents": list(OPPONENTS),
        "required_opponents": list(REQUIRED_OPPONENTS),
        "scenarios": len(scenarios),
        "planned_pairs": len(OPPONENTS) * len(scenarios),
        "planned_episodes": len(METHODS) * len(OPPONENTS) * len(scenarios),
        "execution_permitted": False,
    }
