from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_thesis_phase_reachability_v2_manifest.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_phase_reachability_v2_probe.py"
RUNNER = ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_phase_reachability_v2_run_in_handoff_r1.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_manifest_is_independent_and_starts_outside_merge_range():
    builder = _load(BUILDER, "phase_reachability_v2_builder")
    manifest = builder.build_manifest()
    assert manifest["source_id"] == builder.SOURCE_ID
    assert manifest["independence"]["v1_manifest_reused"] is False
    assert len(manifest["scenarios"]) == 12
    assert all(item["metadata"]["taxonomy_geometry_state"] == "disadvantage" for item in manifest["scenarios"])
    assert all(float(item["metadata"]["initial_range_m"]) > 1000.0 for item in manifest["scenarios"])


def test_v2_preflight_and_runner_sidecar_setting_are_explicitly_authorised_only_for_probe():
    preflight = _load(PREFLIGHT, "phase_reachability_v2_probe_preflight")
    result = preflight.validate_probe(CONFIG)
    assert result["execution_permitted"] is True
    assert result["training_permitted"] is False
    runner = _load(RUNNER, "phase_reachability_v2_runner")
    config = runner._load_config_with_includes(CONFIG)
    setting = runner._phase_reachability_sidecar_settings(config)
    assert setting == {"prediction_checkpoint_sha256": "96bc8e5e1aebd32ca6a528fb3149cfd58aeff276c18bca66388d650f00495931"}
