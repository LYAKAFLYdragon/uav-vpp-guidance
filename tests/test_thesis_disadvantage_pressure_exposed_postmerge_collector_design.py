from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_thesis_disadvantage_pressure_exposed_postmerge_collector_v1.py"
CONFIG = ROOT / "config" / "experiment" / "thesis_disadvantage_pressure_exposed_postmerge_collector_v1.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("pressure_collector", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pressure_collector_design_is_frozen_and_not_authorized() -> None:
    design = _module().validate_design(CONFIG)
    assert design["source_id"] == "THESIS-DISADVANTAGE-PRESSURE-EXPOSED-POSTMERGE-COLLECTOR-V1"
    assert design["execution_permitted"] is False
    assert design["planned_records"] == 48
    assert design["gate"]["required_window_steps"] == 40
    assert design["gate"]["required_opponents"] == ["expert", "end_to_end", "independent_ppo_vpp"]
