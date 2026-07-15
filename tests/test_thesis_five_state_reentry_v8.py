from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_thesis_five_state_reentry_v8_manifest.py"
RUNNER = ROOT / "scripts" / "run_thesis_five_state_reentry_v8.py"
MANIFEST = ROOT / "config/experiment/manifests/thesis_five_state_reentry_v8_phase_contract12.yaml"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _step(range_normalized: float, rate_normalized: float, first_pass: bool, has_66d: bool) -> dict:
    base = np.zeros(16, dtype=float)
    base[0] = range_normalized
    base[1] = rate_normalized
    return {
        "base_observation_16d": base.tolist(),
        "raw_si_first_pass_latched": first_pass,
        "observation_66d": [0.0] * 66 if has_66d else None,
    }


def test_v8_manifest_is_rebuildable_and_records_v6_v7_disjointness():
    builder = _load("v8_manifest_builder", BUILDER)
    v6 = builder.v6
    sources = {label: v6._load(path) for label, path in v6.v5.DISJOINT_MANIFESTS}
    sources["v6"] = v6._load(ROOT / "config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml")
    sources["v7"] = v6._load(ROOT / "config/experiment/manifests/thesis_five_state_reentry_v7_teacher12.yaml")
    rebuilt = builder.build_manifest(sources)
    tracked = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert rebuilt["integrity"]["payload_sha256"] == tracked["integrity"]["payload_sha256"]
    assert len(tracked["scenarios"]) == 12
    assert tracked["disjointness"]["v6"]["physical_intersection_count"] == 0
    assert tracked["disjointness"]["v7"]["physical_intersection_count"] == 0


def test_v8_relabel_uses_physical_units_and_requires_first_pass_before_reentry():
    runner = _load("v8_runner", RUNNER)
    ledger = {
        "steps": [
            _step(0.24, -1.0, False, False),
            _step(0.18, -1.0, False, False),
            _step(0.14, -0.5, False, False),
            _step(0.12, -0.2, False, False),
            _step(0.14, 0.4, False, True),
            _step(0.18, 0.5, True, True),
            _step(0.26, -0.1, True, True),
            _step(0.24, -0.3, True, True),
        ]
    }
    summary = runner.relabel_physical_phase(ledger, {
        "closing_rate_threshold_mps": 0.0,
        "opening_rate_threshold_mps": 0.0,
        "minimum_closing_steps": 3,
        "minimum_opening_steps": 2,
    })

    assert ledger["steps"][2]["v8_phase"] == "post_merge"
    assert ledger["steps"][7]["v8_phase"] == "re_entry"
    assert ledger["steps"][7]["v8_raw_range_rate_mps"] == -60.0
    assert summary["reentry_66d_steps"] == 1
