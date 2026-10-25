from pathlib import Path
import importlib.util
import json

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/experiment/mild_disadvantage_capability_envelope_v1.yaml"
VERIFY_PATH = REPO_ROOT / "scripts/verify_mild_disadvantage_capability_envelope_v1.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("capability_envelope_verifier", VERIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_envelope_config_is_a_frozen_4x4_three_repeat_grid():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-CAPABILITY-ENVELOPE-V1-20260903"
    assert config["protocol"]["training_started"] is False
    assert config["protocol"]["training_permitted"] is False
    assert config["scan"]["axes"]["target_x_m"]["levels"] == [625.0, 675.0, 725.0, 750.0]
    assert config["scan"]["axes"]["target_y_m"]["levels"] == [100.0, 150.0, 200.0, 250.0]
    assert config["scan"]["repetitions_per_cell"] == 3
    assert len(config["scan"]["scenarios"]) == 16


def test_verifier_has_expected_grid_and_opponents():
    verifier = _load_verifier()
    assert len(verifier.EXPECTED_SCENARIOS) == 16
    assert verifier.EXPECTED_OPPONENTS == ("expert", "end_to_end")
    assert verifier.REPETITIONS_PER_CELL == 3


def test_verifier_rejects_incomplete_manifest(tmp_path):
    verifier = _load_verifier()
    manifest = {
        "source_id": verifier.SOURCE_ID,
        "training_started": False,
        "training_permitted": False,
        "backend": "jsbsim",
        "opponents": list(verifier.EXPECTED_OPPONENTS),
        "scenarios": {name: {} for name in verifier.EXPECTED_SCENARIOS},
        "repetitions_per_cell": verifier.REPETITIONS_PER_CELL,
        "raw_episode_count": 1,
        "runs": [],
    }
    (tmp_path / "pilot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = verifier.verify(tmp_path)
    assert result["ok"] is False
    assert any("raw episodes" in error or "run manifest" in error for error in result["errors"])
