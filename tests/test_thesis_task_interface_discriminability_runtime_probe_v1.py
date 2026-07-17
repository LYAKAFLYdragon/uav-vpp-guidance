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
