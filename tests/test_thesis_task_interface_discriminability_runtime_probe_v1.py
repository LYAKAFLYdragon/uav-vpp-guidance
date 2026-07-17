from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_thesis_task_interface_discriminability_runtime_probe_v1.py"
VERIFIER = ROOT / "scripts" / "verify_thesis_task_interface_discriminability_runtime_probe_v1.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_plan_is_the_exact_paired_48_record_matrix() -> None:
    module = _module("task_interface_runtime_runner", RUNNER)
    plan = module.build_plan()
    assert len(plan) == 48
    assert {(row["opponent"], row["method"], row["context"]) for row in plan} == {
        (opponent, method, context)
        for opponent in module.OPPONENTS
        for method in module.METHODS
        for context in module.CONTEXTS
    }
    assert {row["physical_scenario_id"] for row in plan}


def test_first_difference_accepts_mode_or_action_change() -> None:
    module = _module("task_interface_runtime_verifier", VERIFIER)
    base = {"trajectory": [{"commander_mode_id": 0, "vpp_action": [0.0, 0.0, 0.0]}]}
    mode_changed = {"trajectory": [{"commander_mode_id": 1, "vpp_action": [0.0, 0.0, 0.0]}]}
    action_changed = {"trajectory": [{"commander_mode_id": 0, "vpp_action": [0.0, 0.1, 0.0]}]}
    assert module._first_difference(base, mode_changed) is True
    assert module._first_difference(base, action_changed) is True
    assert module._first_difference(base, base) is False


def test_initial_observation_accepts_flat_and_nested_runtime_receipts() -> None:
    module = _module("task_interface_runtime_verifier", VERIFIER)
    expected = {"feature_names": ["range_m", "is_crossing"], "vector": [0.2, 1.0]}
    assert module._initial_observation(expected) == expected
    assert module._initial_observation({"initial_observation": expected}) == expected


def test_same_except_task_bit_accepts_flat_runtime_receipts() -> None:
    module = _module("task_interface_runtime_verifier", VERIFIER)
    reset = {"own_position_m": [0.0, 0.0, 0.0], "target_position_m": [1.0, 0.0, 0.0]}
    base = {"physical_reset_receipt": reset, "feature_names": ["range_m", "is_crossing"], "vector": [0.2, 0.0]}
    crossing = {"physical_reset_receipt": reset, "feature_names": ["range_m", "is_crossing"], "vector": [0.2, 1.0]}
    module._same_except_task_bit(base, crossing, "flat")


def test_trajectory_accepts_episode_level_backend_provenance() -> None:
    module = _module("task_interface_runtime_verifier", VERIFIER)
    trajectory = []
    for index in range(20):
        trajectory.append(
            {
                "vpp_action": [0.0, 0.0, 0.0],
                "own_pos_m": [0.0, 0.0, 0.0],
                "target_pos_m": [1.0, 0.0, 0.0],
                "aoa_deg": 0.0,
                "nz_cmd": 1.0,
                "nz_actual": 1.0,
                "prediction_valid": index >= 9,
                "prediction_fallback": index < 9,
            }
        )
    module._verify_trajectory({"trajectory": trajectory}, "episode-level-backend")
