from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_thesis_global_advantage_p2_physical_preflight.py"
CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_physical_preflight.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("p2_physical_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finite_vector_rejects_nonfinite_or_wrong_shape():
    runner = _module()
    assert runner._finite_vector([0.1, 0.2, 0.3], "action") == [0.1, 0.2, 0.3]
    for action in ([0.1, 0.2], [0.1, np.nan, 0.3]):
        try:
            runner._finite_vector(action, "action")
        except runner.PreflightError:
            pass
        else:
            raise AssertionError("invalid action must fail closed")


def test_default_config_validates_sources_without_execution_authorization():
    runner = _module()
    config, manifest, _runtime, _registry, scenarios = runner._validate_sources(CONFIG)
    assert config["authorization"]["execution_permitted"] is False
    assert manifest["source_id"] == "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1"
    assert len(scenarios) == 60


def test_execution_authorization_requires_frozen_ancestor_and_exact_code_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    authorization["execution_permitted"] = True
    authorization["required_implementation_git_sha"] = "a" * 40
    code_path = ROOT / "scripts" / "run_thesis_global_advantage_p2_physical_preflight.py"
    authorization["authorized_code_files"] = [
        {
            "path": str(code_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        }
    ]
    config_path = tmp_path / "authorized_p2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(runner, "_git_is_ancestor", lambda commit: commit == "a" * 40)
    monkeypatch.setattr(runner, "_git_changed_paths_since", lambda commit: set())
    monkeypatch.setattr(
        runner,
        "_git_value",
        lambda *args: "" if args == ("status", "--porcelain") else "test",
    )

    runner._validate_sources(config_path)

    invalid = copy.deepcopy(config)
    invalid["authorization"]["authorized_code_files"][0]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(runner.PreflightError, match="authorized code file.*SHA mismatch"):
        runner._validate_sources(config_path)


def test_source_hash_mismatch_fails_closed(tmp_path: Path):
    runner = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["inputs"]["manifest_sha256"] = "0" * 64
    config_path = tmp_path / "tampered_p2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(runner.PreflightError, match="preflight manifest SHA mismatch"):
        runner._validate_sources(config_path)
