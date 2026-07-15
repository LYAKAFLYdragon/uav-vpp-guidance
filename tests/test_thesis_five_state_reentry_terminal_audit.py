from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analyze_thesis_five_state_reentry_terminal_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("terminal_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(*, terminal: str, first_pass_step: int | None, saturation: bool = False):
    steps = []
    for index, rate in enumerate((-0.4, -0.3, 0.2, -0.2, -0.2)):
        steps.append({
            "step": index + 1,
            "phase": "post_merge",
            "raw_si_first_pass_latched": first_pass_step is not None and index >= first_pass_step,
            "raw_si_phase_input": {"range_m": 0.2 + 0.1 * index, "range_rate_mps": rate},
            "own_state": {"sim_time": 0.2 * (index + 1)},
            "actual_nz_g": 2.0,
            "nz_cmd": 2.0,
            "nz_saturated": saturation,
            "roll_rate_saturated": False,
            "throttle_saturated": False,
            "observation_66d": [0.0] * 66,
        })
    return {
        "record": {"opponent": "expert", "controller": "frozen_fixed_head_on", "initial_state": "head_on", "result": {"terminal_reason": terminal}},
        "ledger": {"steps": steps},
    }


def test_summary_uses_frozen_observation_scaling_and_detects_reapproach():
    module = _load()
    row = module.summarize_payload("V6", Path("episode.json"), _payload(terminal="timeout", first_pass_step=2))

    assert row["first_pass_reached"] is True
    assert row["remaining_steps_after_first_pass"] == 2
    assert row["min_range_after_first_pass_m"] == 2000.0
    assert row["max_separation_after_first_pass_m"] == 3000.0000000000005
    assert row["longest_reapproach_steps"] == 2
    assert row["ledger_reentry_66d_steps"] == 0
    assert row["physical_reentry_candidate_66d_steps"] == 2


def test_attribution_requires_time_and_safety_exclusion_before_geometry_behavior():
    module = _load()
    adequate = [{
        "study": "V6", "first_pass_reached": True, "post_pass_seconds": 12.0,
        "ledger_reentry_66d_steps": 0, "physical_reentry_candidate_66d_steps": 0,
        "saturated_post_pass_steps": 0, "ego_crash_or_oob": False,
    }]
    assert module.attribution_v6_v7(adequate)["verdict"] == "geometry/behavior"

    short = [dict(adequate[0], post_pass_seconds=2.0)]
    assert module.attribution_v6_v7(short)["verdict"] == "terminal/horizon"

    saturated = [dict(adequate[0], saturated_post_pass_steps=1)]
    assert module.attribution_v6_v7(saturated)["verdict"] == "execution/safety"

    phase_contract_error = [dict(adequate[0], physical_reentry_candidate_66d_steps=1)]
    assert module.attribution_v6_v7(phase_contract_error)["verdict"] == "phase-semantics/unit-contract"
