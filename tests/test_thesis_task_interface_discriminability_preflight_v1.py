from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_thesis_task_interface_discriminability_manifest_v1.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_task_interface_discriminability_v1.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_creates_identical_physical_pairs_with_only_context_metadata_changed() -> None:
    module = _module("task_interface_builder", BUILDER)
    manifest = module.build()
    rows = manifest["scenarios"]
    assert len(rows) == 8
    grouped = {}
    for row in rows:
        grouped.setdefault(row["metadata"]["context_pair_id"], []).append(row)
    assert len(grouped) == 4
    for pair in grouped.values():
        assert len(pair) == 2
        assert pair[0]["own_init"] == pair[1]["own_init"]
        assert pair[0]["target_init"] == pair[1]["target_init"]
        assert {row["metadata"]["ppo_task_bit"] for row in pair} == {0, 1}


def test_static_preflight_accepts_the_design_without_authorizing_execution() -> None:
    module = _module("task_interface_preflight", PREFLIGHT)
    receipt = module.run_preflight()
    assert receipt["verdict"] == "pass"
    assert receipt["runtime_probe_design_ready"] is True
    assert receipt["execution_authorized"] is False
    assert len(receipt["config_cells"]) == 12
    assert {(row["max_high_level_steps"], row["high_level_dt_s"]) for row in receipt["config_cells"]} == {(20, 0.2)}
    assert set(tuple(value) for value in receipt["context_config_differences"].values()) == {
        ("provenance.config_overrides", "task.name", "task.scenarios")
    }
