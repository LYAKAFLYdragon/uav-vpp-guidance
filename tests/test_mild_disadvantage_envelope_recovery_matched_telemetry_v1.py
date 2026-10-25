from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py"
VERIFY = REPO_ROOT / "scripts/verify_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py"


def _load(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_state_hash_is_canonical_and_stable():
    runner = _load(RUNNER, "matched_telemetry_runner")
    payload = {"b": [2, 1], "a": {"x": 1.0}}
    assert runner._state_hash(payload) == runner._state_hash({"a": {"x": 1.0}, "b": [2, 1]})
    assert len(runner._state_hash(payload)) == 64


def test_verifier_rejects_missing_output():
    verifier = _load(VERIFY, "matched_telemetry_verifier")
    result = verifier.verify(REPO_ROOT / "outputs" / "__missing_matched_telemetry_probe__")
    assert result["ok"] is False
    assert result["errors"]
