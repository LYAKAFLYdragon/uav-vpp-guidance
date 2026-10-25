from pathlib import Path
import importlib.util
import json

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/experiment/mild_disadvantage_boundary_scan_v1.yaml"
CONFIG_V2_PATH = REPO_ROOT / "config/experiment/mild_disadvantage_boundary_scan_v2.yaml"
CONFIG_V3_PATH = REPO_ROOT / "config/experiment/mild_disadvantage_boundary_scan_v3.yaml"
CONFIG_V4_PATH = REPO_ROOT / "config/experiment/mild_disadvantage_boundary_scan_v4.yaml"
CONFIG_V5_PATH = REPO_ROOT / "config/experiment/mild_disadvantage_boundary_scan_v5.yaml"
VERIFY_PATH = REPO_ROOT / "scripts/verify_mild_disadvantage_boundary_scan_v1.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_mild_disadvantage_boundary_scan_v1.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("boundary_verifier", VERIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_runner():
    spec = importlib.util.spec_from_file_location("boundary_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scan_is_two_one_factor_axes_and_training_locked():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V1-20260903"
    assert config["protocol"]["training_started"] is False
    assert config["protocol"]["training_permitted"] is False
    assert set(config["scan"]["axes"]) == {"rearward_offset_m", "target_speed_mps"}
    assert len(config["scan"]["scenarios"]) == 9


def test_v2_scan_moves_to_the_near_neutral_boundary():
    config = yaml.safe_load(CONFIG_V2_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V2-20260903"
    assert config["protocol"]["training_permitted"] is False
    assert config["scan"]["axes"]["rearward_offset_m"]["levels"] == [100.0, 50.0, 0.0, -50.0, -100.0]
    assert config["scan"]["axes"]["target_speed_mps"]["levels"] == [185.0, 195.0, 205.0, 215.0, 225.0]
    assert len(config["scan"]["scenarios"]) == 10


def test_v3_scan_is_a_heading_only_disadvantage_probe():
    config = yaml.safe_load(CONFIG_V3_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V3-20260903"
    assert config["scan"]["axes"]["target_heading_deg"]["levels"] == [20.0, 60.0, 100.0, 140.0, 180.0, 220.0, 260.0]
    assert len(config["scan"]["scenarios"]) == 7


def test_verifier_accepts_v3_source_id(tmp_path):
    verifier = _load_verifier()
    manifest = {
        "source_id": verifier.SOURCE_ID_V3,
        "training_started": False,
        "training_permitted": False,
        "backend": "jsbsim",
        "opponents": list(verifier.EXPECTED_OPPONENTS),
        "scenarios": {name: {} for name in verifier.EXPECTED_SCENARIOS_V3},
        "raw_episode_count": 14,
        "runs": [],
    }
    (tmp_path / "pilot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = verifier.verify(tmp_path)
    assert "wrong source_id" not in result["errors"]


def test_v4_scan_connects_neutral_success_to_disadvantage_failure():
    config = yaml.safe_load(CONFIG_V4_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V4-20260903"
    assert config["protocol"]["training_permitted"] is False
    assert config["scan"]["axes"]["target_x_m"]["levels"] == [1900.0, 1000.0, 500.0, 0.0, -300.0]
    assert len(config["scan"]["scenarios"]) == 5
    for scenario in config["scan"]["scenarios"].values():
        assert scenario["target_init"]["position_m"][1:] == [200.0, 5000.0]
        assert scenario["target_init"]["velocity_mps"] == 200.0
        assert scenario["target_init"]["heading_deg"] == 175.0


def test_boundary_summary_uses_adjacent_transition_not_global_extremes():
    runner = _load_runner()
    scan = {
        "protocol": {"opponents": ["expert"]},
        "scan": {"axes": {"target_x_m": {}}},
    }
    rows = [
        {"opponent": "expert", "axis": "target_x_m", "parameter_value": "-300", "successes": "0"},
        {"opponent": "expert", "axis": "target_x_m", "parameter_value": "500", "successes": "0"},
        {"opponent": "expert", "axis": "target_x_m", "parameter_value": "1000", "successes": "1"},
        {"opponent": "expert", "axis": "target_x_m", "parameter_value": "1900", "successes": "1"},
    ]
    result = runner._boundary_analysis(scan, rows)
    bracket = result["expert"]["target_x_m"]["success_to_failure_bracket"]
    assert bracket["lower_level"] == 500.0
    assert bracket["upper_level"] == 1000.0
    assert bracket["success_level"] == 1000.0
    assert bracket["failure_level"] == 500.0


def test_v5_scan_refines_the_v4_longitudinal_bracket():
    config = yaml.safe_load(CONFIG_V5_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V5-20260903"
    assert config["protocol"]["training_started"] is False
    assert config["protocol"]["training_permitted"] is False
    assert config["scan"]["axes"]["target_x_m"]["levels"] == [1900.0, 1000.0, 875.0, 750.0, 625.0, 500.0, 0.0, -300.0]
    for name in ("target_x_875", "target_x_750", "target_x_625"):
        assert config["scan"]["scenarios"][name]["target_init"]["position_m"][0] in (875.0, 750.0, 625.0)


def test_verifier_rejects_missing_manifest(tmp_path):
    verifier = _load_verifier()
    result = verifier.verify(tmp_path)
    assert result["ok"] is False
    assert "missing" in result["errors"][0]


def test_verifier_rejects_wrong_episode_count(tmp_path):
    verifier = _load_verifier()
    manifest = {
        "source_id": verifier.SOURCE_ID,
        "training_started": False,
        "training_permitted": False,
        "backend": "jsbsim",
        "opponents": list(verifier.EXPECTED_OPPONENTS),
        "scenarios": {name: {} for name in verifier.EXPECTED_SCENARIOS},
        "raw_episode_count": 1,
        "runs": [],
    }
    (tmp_path / "pilot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = verifier.verify(tmp_path)
    assert result["ok"] is False
    assert any("18" in error or "run manifest" in error for error in result["errors"])
