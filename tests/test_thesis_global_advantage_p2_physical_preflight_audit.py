from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analyze_thesis_global_advantage_p2_physical_preflight.py"


def _module():
    spec = importlib.util.spec_from_file_location("p2_physical_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(*, phase: str, first_pass: bool, observable: bool) -> dict:
    step = {
        "phase": phase,
        "action": [0.0, 0.0, 0.0],
        "own": {"speed_mps": 200.0, "altitude_m": 5000.0, "nz_g": 1.0},
        "target": {"speed_mps": 200.0, "altitude_m": 5000.0, "nz_g": 1.0},
        "backend_fallback": False,
        "prediction_fallback": False,
    }
    if observable:
        step.update({"range_m": 3000.0, "range_rate_mps": -50.0})
    return {
        "source_id": "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT-V1",
        "opponent": "expert",
        "scenario_name": "s0",
        "steps": [step] * 20,
        "step_count": 20,
        "valid": True,
        "first_pass_observed": first_pass,
        "terminal_reason": "horizon",
    }


def test_audit_marks_post_merge_only_and_unobservable_phase_as_no_go():
    module = _module()
    manifest = {
        "scenarios": [
            {
                "name": f"s{index}",
                "metadata": {"initial_class": "advantage", "phase_at_reset": "pre_merge"},
            }
            for index in range(60)
        ]
    }
    records = []
    for opponent in module.OPPONENTS:
        for index in range(60):
            record = _record(phase="post_merge", first_pass=False, observable=False)
            record["opponent"] = opponent
            record["scenario_name"] = f"s{index}"
            records.append(record)

    audit = module.audit_records(manifest, records)

    assert audit["physical_reachability"]["passed"] is True
    assert audit["phase_observability"]["passed"] is False
    assert audit["phase_observability"]["initial_phase_match_count"] == 0
    assert audit["decision"] == "physical_pass_phase_observability_no_go"


def test_audit_rejects_record_from_unknown_manifest_scenario():
    module = _module()
    manifest = {
        "scenarios": [
            {
                "name": f"s{index}",
                "metadata": {"initial_class": "advantage", "phase_at_reset": "pre_merge"},
            }
            for index in range(60)
        ]
    }
    record = _record(phase="pre_merge", first_pass=False, observable=True)
    record["scenario_name"] = "not_frozen"

    try:
        module.audit_records(manifest, [record])
    except module.AuditError as error:
        assert "absent from frozen manifest" in str(error)
    else:
        raise AssertionError("unknown scenario must fail closed")
