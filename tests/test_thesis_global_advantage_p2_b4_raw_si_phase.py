from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_thesis_global_advantage_p2_b4_raw_si_phase.py"
CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b4_raw_si_phase.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("p2_b4_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_si_input_is_separate_from_normalized_policy_diagnostic():
    runner = _module()
    observation = {
        "relative_state": {"range_m": 4000.0, "range_rate_mps": -100.0},
        "observation_schema": {"feature_names": [
            "range_m", "range_rate_mps", "altitude_diff_m", "speed_diff_mps",
            "los_azimuth_sin", "los_azimuth_cos", "los_elevation_sin", "los_elevation_cos",
            "ata_sin", "ata_cos", "aa_sin", "aa_cos", "own_speed", "target_speed",
            "own_altitude", "target_altitude"
        ]},
        "observation_vector": [0.8, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.7, 0.6, 0.5, 0.5],
    }
    raw = runner.raw_si_phase_input(observation)
    normalized = runner.normalized_policy_diagnostic(observation)

    assert raw == {
        "range_m": 4000.0,
        "range_rate_mps": -100.0,
        "unit_contract": "raw_SI_from_observation.relative_state",
    }
    assert normalized == {"range_m_normalized": 0.8, "range_rate_normalized": -0.5}
    assert runner._normalization_consistent(
        raw,
        normalized,
        {"range_m_scale": 5000.0, "range_rate_mps_scale": 200.0, "range_tolerance_m": 1.0, "range_rate_tolerance_mps": 1.0},
    )


def test_raw_si_config_validates_without_execution_authorization(tmp_path: Path):
    runner = _module()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["authorization"]["execution_permitted"] = False
    payload["authorization"]["required_implementation_git_sha"] = None
    payload["authorization"]["authorized_code_files"] = []
    path = tmp_path / "p2_b4.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config, manifest, _runtime, _registry, scenarios = runner._validate_sources(path)
    assert config["authorization"]["execution_permitted"] is False
    assert manifest["source_id"] == runner.MANIFEST_SOURCE_ID
    assert len(scenarios) == 30


def test_execution_authorization_requires_frozen_ancestor_and_exact_code_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _module()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["authorization"].update(
        {
            "execution_permitted": True,
            "required_implementation_git_sha": "a" * 40,
            "authorized_code_files": [
                {
                    "path": "scripts/run_thesis_global_advantage_p2_b4_raw_si_phase.py",
                    "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                }
            ],
        }
    )
    config_path = tmp_path / "authorized_p2_b4.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(runner.b3, "_git_is_ancestor", lambda commit: commit == "a" * 40)
    monkeypatch.setattr(runner.b3, "_git_changed_paths_since", lambda commit: set())
    monkeypatch.setattr(
        runner.b3,
        "_git_value",
        lambda *args: "" if args == ("status", "--porcelain") else "test",
    )

    runner._validate_sources(config_path)
    payload["authorization"]["authorized_code_files"][0]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(runner.RawSIPhaseError, match="authorised code file.*SHA mismatch"):
        runner._validate_sources(config_path)
