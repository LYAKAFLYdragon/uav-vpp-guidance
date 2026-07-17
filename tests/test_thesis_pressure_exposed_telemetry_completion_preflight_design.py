from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_thesis_pressure_exposed_telemetry_completion_preflight_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("telemetry_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_gate_accepts_complete_post_warmup_physical_telemetry() -> None:
    module = _module()
    row = {
        "backend": "jsbsim", "backend_fallback_occurred": False,
        "range_m": 1000.0, "range_rate_mps": 10.0, "aa_deg": 150.0, "ata_deg": 30.0,
        "own_speed_mps": 250.0, "target_speed_mps": 280.0,
        "own_altitude_m": 5000.0, "target_altitude_m": 5500.0, "own_specific_energy_m": 8186.0,
        "observation_66d_valid": True, "vpp_action_valid": True, "vp_bias_valid": True,
        "guidance_command_present": True, "aoa_present": True, "nz_command_present": True,
        "nz_actual_g": 1.0,
    }
    assert module._trace_passes([dict(row) for _ in range(10)]) is True


def test_trace_gate_rejects_missing_raw_nz_response() -> None:
    module = _module()
    row = {
        "backend": "jsbsim", "backend_fallback_occurred": False,
        "range_m": 1000.0, "range_rate_mps": 10.0, "aa_deg": 150.0, "ata_deg": 30.0,
        "own_speed_mps": 250.0, "target_speed_mps": 280.0,
        "own_altitude_m": 5000.0, "target_altitude_m": 5500.0, "own_specific_energy_m": 8186.0,
        "observation_66d_valid": True, "vpp_action_valid": True, "vp_bias_valid": True,
        "guidance_command_present": True, "aoa_present": True, "nz_command_present": True,
        "nz_actual_g": None,
    }
    assert module._trace_passes([dict(row) for _ in range(10)]) is False
