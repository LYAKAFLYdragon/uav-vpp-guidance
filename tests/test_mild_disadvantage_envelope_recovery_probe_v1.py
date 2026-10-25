from pathlib import Path
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_mild_disadvantage_envelope_recovery_probe_v1.py"
VERIFY_PATH = REPO_ROOT / "scripts/verify_mild_disadvantage_envelope_recovery_probe_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_is_locked_to_non_learning_and_exact_episode_design():
    runner = _load(RUNNER_PATH, "recovery_probe_runner")
    spec = runner._load(runner.DEFAULT_CONFIG)
    assert spec["experiment"]["source_id"] == runner.SOURCE_ID
    assert spec["probe"]["repetitions_per_candidate_and_opponent"] == 3
    assert tuple(spec["probe"]["opponents"]) == ("expert", "end_to_end")
    assert tuple(spec["probe"]["candidates"]) == ("x625_y200", "x750_y250")
    assert spec["protocol"]["training_started"] is False
    assert spec["protocol"]["training_permitted"] is False
    assert spec["protocol"]["strict_backend"] is True


def test_body_frame_transform_preserves_initial_axes():
    runner = _load(RUNNER_PATH, "recovery_probe_runner_axes")
    own = {"position_m": [0.0, 0.0, 5000.0], "velocity_vector_mps": [200.0, 0.0, 0.0]}
    target = {"position_m": [625.0, 200.0, 5000.0], "velocity_vector_mps": [-200.0, 0.0, 0.0]}
    assert runner._body_relative_position(own, target) == (625.0, 200.0)


def test_verifier_rejects_missing_probe():
    verifier = _load(VERIFY_PATH, "recovery_probe_verifier")
    result = verifier.verify(REPO_ROOT / "outputs" / "__missing_recovery_probe__")
    assert result["ok"] is False
    assert result["errors"]
