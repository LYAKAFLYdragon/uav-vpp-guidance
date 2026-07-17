from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_disadvantage_pressure_exposed_m0m4_v1.py"
RUNNER = ROOT / "scripts" / "run_thesis_disadvantage_pressure_exposed_m0m4_v1.py"
VERIFIER = ROOT / "scripts" / "verify_thesis_disadvantage_pressure_exposed_m0m4_v1.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_accepts_the_frozen_pressure_exposed_source(tmp_path: Path) -> None:
    module = _module("pressure_m0m4_preflight", PREFLIGHT)
    module.MIN_FREE_BYTES = 0
    receipt = module.run_preflight(output_root=tmp_path / "formal_output")
    assert receipt["verdict"] == "pass"
    assert len(receipt["config_cells"]) == 15
    assert receipt["telemetry_prerequisite"]["gate"]["passes"] is True


def test_runner_plan_is_the_exact_three_by_sixteen_by_five_matrix(tmp_path: Path) -> None:
    module = _module("pressure_m0m4_runner", RUNNER)
    plan = module.build_plan(output_root=tmp_path / "formal_output")
    assert len(plan) == 3
    assert sum(item["expected_records"] for item in plan) == 240
    assert all(item["expected_records"] == 80 for item in plan)
    assert all("--tasks" in item["command"] and "head_on" in item["command"] for item in plan)


def test_verifier_requires_complete_physical_telemetry() -> None:
    module = _module("pressure_m0m4_verifier", VERIFIER)
    good = {
        "vpp_action": [0.0, 0.1, -0.1],
        "own_pos_m": [0.0, 0.0, 5000.0],
        "target_pos_m": [1000.0, 0.0, 5200.0],
        "range_m": 1020.0,
        "aa_deg": 160.0,
        "ata_deg": 20.0,
        "aoa_deg": 5.0,
        "nz_cmd": 2.0,
        "nz_actual": 1.2,
        "tactical_basis_world_offset_x": 100.0,
        "tactical_basis_world_offset_y": 0.0,
        "tactical_basis_world_offset_z": 20.0,
        "guidance_telemetry": {"effective_guidance_mode": "los_rate"},
        "clipping": {"nz_saturated": False, "roll_rate_saturated": False, "throttle_saturated": False, "any_saturated": False},
    }
    assert module._verify_step(good, "test") == "disadvantage"
    bad = dict(good)
    bad.pop("nz_actual")
    try:
        module._verify_step(bad, "test")
    except module.VerificationError:
        pass
    else:  # pragma: no cover - explicit gate failure assertion
        raise AssertionError("missing n_z response must reject the record")
