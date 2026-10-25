from pathlib import Path
import importlib.util
import json

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/experiment/simplified_scenario_feasibility_pilot_v1.yaml"
VERIFY_PATH = REPO_ROOT / "scripts/verify_simplified_scenario_feasibility_pilot_v1.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("simplified_verifier", VERIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simplified_pilot_config_is_locked_and_distinctly_named():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-SIMPLIFIED-SCENARIO-FEASIBILITY-PILOT-V1-20260903"
    assert config["protocol"]["training_started"] is False
    assert config["protocol"]["training_permitted"] is False
    assert config["protocol"]["backend"] == "jsbsim"
    assert config["protocol"]["scenarios"] == [
        "head_on_reference",
        "mild_crossing_left",
        "mild_crossing_right",
        "mild_neutral",
        "mild_disadvantage",
    ]
    assert config["methods"]["h0_current_geometry"]["training_allowed"] is False


def test_verifier_rejects_missing_pilot_manifest(tmp_path):
    verifier = _load_verifier()
    result = verifier.verify(tmp_path)
    assert result["ok"] is False
    assert "missing" in result["errors"][0]


def test_verifier_accepts_minimal_valid_shape(tmp_path):
    verifier = _load_verifier()
    root = tmp_path
    manifest = {
        "source_id": verifier.SOURCE_ID,
        "phase": "pilot",
        "training_started": False,
        "training_permitted": False,
        "opponents": list(verifier.EXPECTED_OPPONENTS),
        "scenarios": list(verifier.EXPECTED_SCENARIOS),
        "backend": "jsbsim",
        "raw_episode_count": 10,
        "runs": [],
    }
    (root / "pilot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "metrics_by_scenario.csv").write_text("x\n" + "x\n" * 10, encoding="utf-8")
    result = verifier.verify(root)
    assert result["ok"] is False
    assert any("opponent" in error or "scenario summary" in error for error in result["errors"])
