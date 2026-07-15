from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_thesis_global_advantage_p2_b3_phase_observability.py"
CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b3_phase_observability.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("p2_b3_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario() -> dict:
    return {
        "own_init": {"position_m": [0.0, 0.0, 5200.0], "velocity_mps": 250.0, "heading_deg": 0.0},
        "target_init": {"position_m": [3000.0, 0.0, 5200.0], "velocity_mps": 240.0, "heading_deg": 180.0},
    }


def _states() -> dict:
    return {
        "own": {
            "position_neu": [0.0, 0.0, 5200.0],
            "velocity_vector_mps": [250.0, 0.0, 0.0],
            "speed_mps": 250.0,
            "altitude_m": 5200.0,
        },
        "target": {
            "position_neu": [3000.0, 0.0, 5200.0],
            "velocity_vector_mps": [-240.0, 0.0, 0.0],
            "speed_mps": 240.0,
            "altitude_m": 5200.0,
        },
    }


def test_scenario_application_receipt_accepts_matching_raw_jsbsim_state():
    runner = _module()
    receipt = runner.scenario_application_receipt(
        _scenario(),
        _states(),
        {"position_error_m": 1.0, "velocity_error_mps": 1.0, "heading_error_deg": 1.0, "altitude_error_m": 1.0},
    )
    assert receipt["passed"] is True
    assert receipt["requested_range_m"] == 3000.0
    assert receipt["requested_range_rate_mps"] == -490.0


def test_phase_receipts_are_replayable_and_tamper_fails_closed():
    runner = _module()
    tracker = runner.PhaseTracker()
    first = runner.phase_receipt(tracker, {"range_m": 3000.0, "range_rate_mps": -490.0})
    second = runner.phase_receipt(tracker, {"range_m": 900.0, "range_rate_mps": -100.0})
    assert first["phase_after_update"] == "pre_merge"
    assert second["phase_after_update"] == "post_merge"
    assert runner.replay_phase_receipts([first, second]) is True
    invalid = copy.deepcopy(second)
    invalid["phase_after_update"] = "pre_merge"
    assert runner.replay_phase_receipts([first, invalid]) is False


def test_default_config_validates_without_execution_authorization(tmp_path: Path):
    runner = _module()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["authorization"]["execution_permitted"] = False
    path = tmp_path / "p2_b3.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config, manifest, _runtime, _registry, scenarios = runner._validate_sources(path)
    assert config["authorization"]["execution_permitted"] is False
    assert manifest["source_id"] == runner.MANIFEST_SOURCE_ID
    assert len(scenarios) == 30


def test_phase_receipt_rejects_nonfinite_input():
    runner = _module()
    with pytest.raises(runner.PhaseObservabilityError, match="must be finite"):
        runner.phase_receipt(runner.PhaseTracker(), {"range_m": float("nan"), "range_rate_mps": 0.0})


def test_execution_authorization_requires_frozen_ancestor_and_exact_code_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _module()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    authorization = payload["authorization"]
    authorization["execution_permitted"] = True
    authorization["required_implementation_git_sha"] = "a" * 40
    code_path = ROOT / "scripts" / "run_thesis_global_advantage_p2_b3_phase_observability.py"
    authorization["authorized_code_files"] = [
        {
            "path": str(code_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        }
    ]
    config_path = tmp_path / "authorized_p2_b3.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(runner, "_git_is_ancestor", lambda commit: commit == "a" * 40)
    monkeypatch.setattr(runner, "_git_changed_paths_since", lambda commit: set())
    monkeypatch.setattr(
        runner,
        "_git_value",
        lambda *args: "" if args == ("status", "--porcelain") else "test",
    )

    runner._validate_sources(config_path)

    payload["authorization"]["authorized_code_files"][0]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(runner.PhaseObservabilityError, match="authorized code file.*SHA mismatch"):
        runner._validate_sources(config_path)
