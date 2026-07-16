from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from uav_vpp_guidance.evaluation.thesis_runtime_smoke import RuntimeSmokeContractError, summarize_records


ROOT = Path(__file__).resolve().parent.parent


def _record(opponent: str) -> dict:
    return {
        "opponent": opponent,
        "handoff_reached": True,
        "handoff_state_hash": f"{opponent}:handoff",
        "released": True,
        "warmup_control_isolated": True,
        "post_release_prediction_fallback": False,
        "observer_trace": [{"step": 0}],
        "anchor_switches": [
            {"step": 0, "mode": "current_target", "reason": "v22_observer_warmup"},
            {"step": 10, "mode": "predicted_target", "reason": "v22_first_valid_observer_receipt"},
        ],
        "phase_trace": [
            {"phase": "pre_merge", "first_pass_complete": False},
            {"phase": "re_entry", "first_pass_complete": True},
        ],
        "runtime_trace": [
            {
                "step": 1,
                "phase": "re_entry",
                "raw_range_m": 1000.0,
                "raw_range_rate_mps": 10.0,
                "vp_world_offset_m": [1.0, 2.0, 3.0],
                "guidance_command_normalized": [0.1, 0.2, 0.3],
                "aoa_rad": 0.1,
                "actual_nz_g": 1.0,
                "runtime_vpp_anchor_mode": "predicted_target",
            }
        ],
    }


def test_runtime_smoke_requires_every_opponent_and_complete_telemetry():
    summary = summarize_records([_record("expert"), _record("end_to_end"), _record("independent_ppo_vpp")])
    assert summary["passes"] is True


def test_runtime_smoke_rejects_missing_observer_trace():
    broken = _record("expert")
    broken["observer_trace"] = []
    with pytest.raises(RuntimeSmokeContractError, match="observer trace"):
        summarize_records([broken, _record("end_to_end"), _record("independent_ppo_vpp")])


def test_runtime_smoke_runner_keeps_a_non_source_provenance_boundary():
    spec = importlib.util.spec_from_file_location("runtime_smoke_runner", ROOT / "scripts" / "run_thesis_runtime_smoke.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.SMOKE_ID.endswith("NOT-A-SOURCE-ID")
