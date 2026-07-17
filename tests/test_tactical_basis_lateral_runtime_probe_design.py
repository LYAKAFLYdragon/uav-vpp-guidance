from __future__ import annotations

import copy

import pytest

from uav_vpp_guidance.evaluation.tactical_basis_lateral_runtime_probe_contract import (
    SOURCE_ID,
    TacticalBasisLateralRuntimeProbeDesignError,
    validate_design_spec,
)


def _parent() -> dict:
    return {
        "source_id": "THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1",
        "scenarios": [
            {"name": f"scenario-{index}", "metadata": {"scenario_seed": 100 + index}}
            for index in range(12)
        ],
    }


def _manifest() -> dict:
    return {
        "source_id": SOURCE_ID + "-MANIFEST12",
        "parent_manifest_source_id": "THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1",
        "parent_manifest_sha256": "parent-hash",
        "scenarios": [{"name": f"scenario-{index}", "scenario_seed": 100 + index} for index in range(12)],
    }


def _config() -> dict:
    return {
        "source_id": SOURCE_ID,
        "stage": "paired_nonlearning_postmerge_runtime_probe",
        "authorization": {
            "execution_permitted": False,
            "training_permitted": False,
            "tuning_permitted": False,
            "reward_change_permitted": False,
            "high_level_ppo_training_permitted": False,
            "shared_skill_training_permitted": False,
            "guidance_pid_change_permitted": False,
            "action_envelope_expansion_permitted": False,
            "snapshot_restore_permitted": False,
            "future_state_injection_permitted": False,
            "history_padding_permitted": False,
        },
        "inputs": {"parent_manifest_sha256": "parent-hash"},
        "candidate": {
            "p2_source_id": "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1",
            "p2_selected_variant": "lateral_extension:lat+1:vert+1",
            "method_id": "lateral_extension_lat_pos_vert_pos",
            "program_steps": 40,
            "post_program_controller": "frozen_fixed_head_on",
        },
        "methods": {"order": ["fixed_head_on", "lateral_extension_lat_pos_vert_pos"]},
        "opponents": {
            "order": ["expert", "end_to_end", "independent_ppo_vpp"],
            "required_for_gate": ["end_to_end", "independent_ppo_vpp"],
            "pooled_gate": "prohibited",
        },
        "contract": {
            "backend": "jsbsim",
            "strict_backend": True,
            "action_semantics": "tactical_basis_v1",
            "reset_at_handoff": False,
            "snapshot_restore": False,
            "future_state_injection": False,
            "history_padding": False,
            "routing_enabled": False,
            "target_phases": ["post_merge"],
            "target_dynamic_state": "disadvantage",
        },
        "metrics": {
            "response_window_steps": 40,
            "primary_pressure": {"candidate_minus_baseline_max": -0.05},
            "secondary_geometry": {
                "range_rate_mean_min_mps": 5.0,
                "end_range_min_m": 50.0,
                "end_specific_energy_min_m": 25.0,
            },
        },
        "gates": {
            "min_qualifying_pairs_per_required_opponent": 4,
            "min_distinct_signatures_per_required_opponent": 4,
        },
    }


def _p2_summary() -> dict:
    return {
        "source_id": "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1",
        "passes": True,
        "selected_variant": "lateral_extension:lat+1:vert+1",
        "next_action": "one_selected_variant_runtime_probe_design_permitted",
    }


def _p2_verification() -> dict:
    return {
        "source_id": "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1",
        "passes": True,
        "rows_verified": 136,
    }


def test_design_freezes_the_unique_p2_candidate_and_paired_matrix() -> None:
    result = validate_design_spec(_config(), _p2_summary(), _p2_verification(), _manifest(), _parent())

    assert result["candidate_variant"] == "lateral_extension:lat+1:vert+1"
    assert result["planned_pairs"] == 36
    assert result["planned_episodes"] == 72
    assert result["execution_permitted"] is False


def test_design_rejects_a_post_hoc_template_substitution() -> None:
    config = copy.deepcopy(_config())
    config["candidate"]["p2_selected_variant"] = "vertical_extension:lat+1:vert+1"

    with pytest.raises(TacticalBasisLateralRuntimeProbeDesignError, match="unique P2-selected"):
        validate_design_spec(config, _p2_summary(), _p2_verification(), _manifest(), _parent())
