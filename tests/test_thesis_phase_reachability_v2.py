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


def test_v2_design_preflight_is_nonexecuting_and_locks_v1_negative_evidence():
    preflight = _load(PREFLIGHT, "phase_reachability_v2_preflight")
    plan = preflight.validate_design(preflight.DEFAULT_CONFIG)
    assert plan["mode"] == "design_validation_only"
    assert plan["execution_permitted"] is False
    assert plan["opponents"] == ("expert", "end_to_end", "independent_ppo_vpp")


def test_handoff_contract_rejects_state_discontinuity():
    contract = _load(CONTRACT, "phase_reachability_contract")
    contract.validate_handoff_continuity(_record(41), _record(42))
    discontinuous = _record(42)
    discontinuous["history_sha256"] = "new-history"
    with pytest.raises(contract.PhaseReachabilityContractError, match="history_sha256"):
        contract.validate_handoff_continuity(_record(41), discontinuous)
