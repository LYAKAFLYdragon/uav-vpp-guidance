from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent


def _module():
    path = ROOT / "scripts" / "verify_thesis_disadvantage_capability_plan2.py"
    spec = importlib.util.spec_from_file_location("plan2_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _point(step: int, *, method: str, candidate: bool, motif: str | None) -> dict:
    active = candidate and step > 10
    is_range = motif == "range_extension_los"
    is_vertical = motif == "vertical_separation"
    is_lateral = motif == "lateral_break_same_side"
    altitude = 5200.0 if is_vertical and active else 5000.0
    point = {
        "step": step,
        "ego_pos_x": float(step),
        "ego_pos_y": 0.0,
        "ego_pos_z": altitude,
        "target_pos_x": 1000.0 + float(step),
        "target_pos_y": 0.0,
        "target_pos_z": 5000.0,
        "vpp_action": [0.0, 0.0, 0.0],
        "prediction_valid": bool(step > 9),
        "prediction_fallback": bool(step <= 9),
        "world_offset_x": 200.0 if active else 0.0,
        "world_offset_y": 0.0,
        "world_offset_z": 0.0,
        "nz_cmd": 1.2 if active else 1.0,
        "roll_rate_cmd": 0.1 if active else 0.0,
        "throttle_cmd": 0.8,
        "target_in_attack_zone": not (is_lateral and active),
        "range_rate_mps": 10.0 if is_range and active else 0.0,
        "speed_mps": 250.0,
        "target_speed_mps": 250.0,
        "altitude_m": altitude,
        "target_altitude_m": 5000.0,
        "nz_saturated": False,
        "roll_rate_saturated": False,
        "throttle_saturated": False,
    }
    if active:
        point["agent_step_metadata"] = {
            "motif_probe_phase": "motif_intervention",
            "motif_probe_name": motif,
            "motif_probe_global_step": step,
            "motif_probe_intervention_step": step - 10,
            "motif_probe_uses_future_state": False,
            "motif_probe_receipt": {
                "motif": motif,
                "basis_rank": 3,
                "basis_condition_number": 2.0,
                "actual_world_delta_m": [200.0, 0.0, 0.0],
                "projection_residual_norm_m": 0.0,
                "action_clipped": False,
            },
        }
    return point


def _record(scenario: str, method: str, opponent: str) -> dict:
    candidate = method in {"lateral_break_same_side", "range_extension_los", "vertical_separation"}
    motif = method if candidate else None
    return {
        "scenario": scenario,
        "controller": method,
        "opponent_stage": opponent,
        "backend": "jsbsim",
        "strict_backend": True,
        "backend_fallback_occurred": False,
        "combat_reason": "timeout",
        "trajectory": [_point(step, method=method, candidate=candidate, motif=motif) for step in range(1, 31)],
    }


def _write_fixture(tmp_path: Path):
    module = _module()
    manifest = module.yaml.safe_load(
        (ROOT / "config/experiment/manifests/thesis_disadvantage_capability_plan2_v1.yaml").read_text(encoding="utf-8")
    )
    scenarios = [item["name"] for item in manifest["scenarios"]]
    for opponent in module.OPPONENTS:
        run = tmp_path / f"{module.SOURCE_ID}-{opponent}"
        (run / "aggregate").mkdir(parents=True)
        (run / "failures.json").write_text('{"failures": []}\n', encoding="utf-8")
        records = [_record(scenario, method, opponent) for scenario in scenarios for method in module.METHODS]
        (run / "aggregate" / "episode_records.json").write_text(
            json.dumps({"episodes": records}), encoding="utf-8"
        )
    return module


def test_plan2_verifier_accepts_complete_cross_opponent_candidates(tmp_path: Path):
    module = _write_fixture(tmp_path)

    result = module.verify(ROOT / "config/experiment/thesis_disadvantage_capability_plan2_v1.yaml", tmp_path)

    assert result["total_records"] == 180
    assert set(result["candidate_motifs"]) == set(module.MOTIFS)
    assert result["decision"] == "candidate_low_level_motif_exists"
