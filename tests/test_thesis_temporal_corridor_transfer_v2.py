from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_temporal_corridor_transfer_v2 as runner
import run_thesis_temporal_corridor_transfer_v2_smoke as smoke
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer_v2 import summarize_records


def _record(method: str, opponent: str, key: str) -> dict:
    window = []
    for _ in range(20):
        receipt = None if method == "frozen_fixed_head_on" else {"projection_residual_norm_m": 0.0, "hard_bound_active": False}
        window.append(
            {
                "after_component_errors": [0.0] * 6,
                "after_weighted_error": 0.0,
                "temporal_corridor_receipt": receipt,
            }
        )
    return {
        "method": method,
        "opponent": opponent,
        "pair_key": key,
        "handoff_reached": True,
        "contract_passed": True,
        "window_steps": 20,
        "handoff_state_hash": f"{opponent}:{key}",
        "ego_failure": False,
        "window": window,
    }


def test_v2_transfer_gate_requires_two_unpooled_candidates() -> None:
    rows = [
        _record(method, opponent, str(index))
        for method in ("frozen_fixed_head_on", "moderate_left", "moderate_right")
        for opponent in ("expert", "end_to_end", "independent_ppo_vpp")
        for index in range(4)
    ]
    gates = {
        "min_qualifying_pairs_per_opponent": 4,
        "max_mean_projection_residual_m": 25.0,
        "hard_bound_steps_max": 0,
        "ego_crash_oob_delta_vs_reference_max": 0.05,
        "weighted_error_delta_vs_reference_max": -0.01,
        "min_improved_components": 2,
        "improved_component_delta_max": -0.01,
        "component_worsening_delta_max": 0.02,
        "candidate_opponents_required": 2,
    }
    for row in rows:
        if row["method"] == "moderate_left":
            for step in row["window"]:
                step["after_component_errors"] = [-0.02, -0.02, 0.0, 0.0, 0.0, 0.0]
                step["after_weighted_error"] = -0.02
    assert summarize_records(rows, scenarios_per_opponent=4, window_steps=20, gates=gates)["passes"] is True


def test_v2_requires_execution_runtime_mapping() -> None:
    design_path = ROOT / "config" / "experiment" / "thesis_temporal_corridor_transfer_v2_design.yaml"
    design = runner._config(design_path)
    assert runner.environment_config(design) is design

    missing = copy.deepcopy(design)
    missing.pop("execution")
    with pytest.raises(runner.TemporalCorridorTransferV2Error, match="explicit inputs and execution"):
        runner.environment_config(missing)

    mismatched = copy.deepcopy(design)
    mismatched["execution"]["runtime_template"] = "config/experiment/manifests/thesis_temporal_corridor_transfer_v2_dev12.yaml"
    with pytest.raises(runner.TemporalCorridorTransferV2Error, match="does not match"):
        runner.environment_config(mismatched)


def test_smoke_prediction_receipt_is_json_safe() -> None:
    receipt = {"predicted_target_position_neu": smoke._finite_vector(np.asarray([1.0, 2.0, 3.0]), "prediction")}
    assert json.loads(json.dumps(receipt))["predicted_target_position_neu"] == [1.0, 2.0, 3.0]


def test_v2_requires_a_hash_bound_passed_smoke_gate(tmp_path: Path) -> None:
    gate_path = tmp_path / "smoke_gate.json"
    gate = {
        "smoke_id": "ENGINEERING-TEMPORAL-CORRIDOR-TRANSFER-V2-BUILD-SMOKE-NOT-A-SOURCE-ID",
        "source_id_under_test": runner.SOURCE_ID,
        "records": 3,
        "all_opponents_present": True,
        "all_telemetry_passed": True,
        "passes": True,
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    execution = {
        "required_smoke_gate": str(gate_path),
        "required_smoke_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
    }
    runner.validate_required_smoke_gate(execution)

    execution["required_smoke_gate_sha256"] = "0" * 64
    with pytest.raises(runner.TemporalCorridorTransferV2Error, match="hash mismatch"):
        runner.validate_required_smoke_gate(execution)
