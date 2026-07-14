from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "src" / "uav_vpp_guidance" / "evaluation" / "phase_reachability_handoff_contract.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_phase_reachability_v2.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _record(step: int) -> dict:
    return {
        "backend": "jsbsim",
        "backend_fallback_occurred": False,
        "reset_occurred": False,
        "future_state_injected": False,
        "first_pass_complete": True,
        "step": step,
        "episode_lineage_id": "episode-1",
        "jsbsim_state_lineage_id": "jsbsim-1",
        "observation_schema_sha256": "obs",
        "history_sha256": "history",
        "prediction_checkpoint_sha256": "predictor",
        "prediction_valid": True,
        "vpp_action_sha256": "vpp",
        "guidance_state_sha256": "guidance",
        "pid_state_sha256": "pid",
    }


def _sampler_record(step: int) -> dict:
    record = _record(step)
    record["parent_step"] = step - 1
    for field in ("history_sha256", "vpp_action_sha256", "guidance_state_sha256", "pid_state_sha256"):
        record[f"parent_{field}"] = _record(step - 1)[field]
    return record


def test_v2_design_preflight_is_nonexecuting_and_locks_v1_negative_evidence():
    preflight = _load(PREFLIGHT, "phase_reachability_v2_preflight")
    plan = preflight.validate_design(preflight.DEFAULT_CONFIG)
    assert plan["mode"] == "design_validation_only"
    assert plan["execution_permitted"] is False
    assert plan["opponents"] == ("expert", "end_to_end", "independent_ppo_vpp")


def test_handoff_contract_rejects_state_discontinuity():
    contract = _load(CONTRACT, "phase_reachability_contract")
    boundary = _record(41)
    sampler = _sampler_record(42)
    sampler["history_sha256"] = "new-history-at-k-plus-1"
    contract.validate_handoff_continuity(boundary, sampler)
    discontinuous = _sampler_record(42)
    discontinuous["parent_history_sha256"] = "wrong-parent"
    with pytest.raises(contract.PhaseReachabilityContractError, match="parent hash mismatch: history_sha256"):
        contract.validate_handoff_continuity(boundary, discontinuous)


def test_observe_only_sidecar_records_parent_hashes_without_mutating_action():
    contract = _load(CONTRACT, "phase_reachability_contract_sidecar")
    sidecar = contract.PhaseReachabilityContinuitySidecar(
        run_id="run",
        method_name="legacy_static_oracle_task_gate",
        task_name="head_on",
        seed=3,
        episode=1,
        scenario_name="scenario",
        observation_schema={"dim": 19, "feature_names": ["range_m"]},
        prediction_checkpoint_sha256="predictor",
        environment_episode=1,
    )
    action = [0.1, -0.2, 0.3]
    info = {
        "backend": "jsbsim",
        "backend_fallback_occurred": False,
        "current_step": 12,
        "first_pass_complete": True,
        "prediction_valid": True,
        "own_state": {"position_m": [1.0, 2.0, 3.0]},
        "target_state": {"position_m": [4.0, 5.0, 6.0]},
    }
    sidecar.observe_after_step(
        step=12,
        observation={"observation_vector": [1.0, 2.0]},
        action=action,
        info=info,
        agent=object(),
        environment_episode=1,
    )
    assert action == [0.1, -0.2, 0.3]
    info_next = dict(info, current_step=13)
    sidecar.observe_after_step(
        step=13,
        observation={"observation_vector": [1.1, 2.1]},
        action=action,
        info=info_next,
        agent=object(),
        environment_episode=1,
    )
    ledger = sidecar.ledger()
    assert ledger["continuity_valid"] is True
    assert ledger["sampler_observation"]["parent_history_sha256"] == ledger["run_in_boundary"]["history_sha256"]


def test_sidecar_rejects_reset_or_backend_fallback():
    contract = _load(CONTRACT, "phase_reachability_contract_rejection")
    sidecar = contract.PhaseReachabilityContinuitySidecar(
        run_id="run", method_name="oracle", task_name="head_on", seed=1, episode=1,
        scenario_name="scenario", observation_schema={"dim": 19},
        prediction_checkpoint_sha256="predictor", environment_episode=1,
    )
    sidecar.observe_after_step(
        step=4,
        observation={"observation_vector": [0.0]}, action=[0.0, 0.0, 0.0], agent=object(),
        environment_episode=2,
        info={"backend": "jsbsim", "backend_fallback_occurred": False, "current_step": 4, "first_pass_complete": True},
    )
    assert "episode lineage changed" in sidecar.ledger()["continuity_error"]


def test_handoff_contract_rejects_backend_fallback_and_future_state_injection():
    contract = _load(CONTRACT, "phase_reachability_contract_safety")
    boundary = _record(20)
    fallback_sampler = _sampler_record(21)
    fallback_sampler["backend_fallback_occurred"] = True
    with pytest.raises(contract.PhaseReachabilityContractError, match="strict JSBSim"):
        contract.validate_handoff_continuity(boundary, fallback_sampler)
    injected_sampler = _sampler_record(21)
    injected_sampler["future_state_injected"] = True
    with pytest.raises(contract.PhaseReachabilityContractError, match="future-state injection"):
        contract.validate_handoff_continuity(boundary, injected_sampler)
