from pathlib import Path
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_mild_disadvantage_envelope_recovery_matched_probe_v1.py"
VERIFY_PATH = REPO_ROOT / "scripts/verify_mild_disadvantage_envelope_recovery_matched_probe_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matched_probe_restores_crossing_contract():
    runner = _load(RUNNER_PATH, "matched_recovery_probe_runner")
    spec = runner._load(runner.DEFAULT_CONFIG)
    assert spec["experiment"]["source_id"] == runner.SOURCE_ID
    assert spec["matched_contract"]["action_semantics"] == "tactical_basis_v1"
    assert spec["matched_contract"]["anchor_mode"] == "predicted_target"
    assert spec["matched_contract"]["trajectory_prediction_enabled"] is True
    assert spec["matched_contract"]["task_name"] == "crossing_feasible"


def test_matched_verifier_rejects_missing_output():
    verifier = _load(VERIFY_PATH, "matched_recovery_probe_verifier")
    result = verifier.verify(REPO_ROOT / "outputs" / "__missing_matched_recovery_probe__")
    assert result["ok"] is False
    assert result["errors"]
