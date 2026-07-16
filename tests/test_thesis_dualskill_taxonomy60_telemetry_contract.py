from __future__ import annotations

import math

import numpy as np
import pytest

from uav_vpp_guidance.evaluation.recorders import EpisodeRecorder


def test_formal_telemetry_record_preserves_vpp_and_execution_chain_fields():
    recorder = EpisodeRecorder(
        run_id="taxonomy60-test",
        task="head_on",
        controller="fixed_head_on",
        seed=7,
        episode=0,
        config={},
        config_sha256="test",
        git_commit="test",
    )
    recorder.record_step(
        step=1,
        time_s=0.2,
        own_state={
            "position_m": [0.0, 0.0, 5000.0],
            "alpha_rad": math.radians(4.0),
            "nz_g": 1.1,
        },
        target_state={"position_m": [1000.0, 0.0, 5000.0]},
        info={
            "nz_cmd": 1.2,
            "roll_rate_cmd": 0.3,
            "throttle_cmd": 0.7,
            "effective_guidance_mode": "los_rate",
            "nz_saturated": False,
            "roll_rate_saturated": True,
            "throttle_saturated": False,
            "saturation_flag": True,
        },
        reward=0.0,
        action=np.asarray([0.2, -0.4, 0.6], dtype=np.float32),
    )

    point = recorder.trajectory[0]
    assert point["vpp_action"] == pytest.approx([0.2, -0.4, 0.6])
    assert point["aoa_rad"] == pytest.approx(math.radians(4.0))
    assert point["aoa_deg"] == pytest.approx(4.0)
    assert point["nz_actual"] == pytest.approx(1.1)
    assert point["guidance_telemetry"] == {
        "effective_guidance_mode": "los_rate",
        "nz_cmd": pytest.approx(1.2),
        "roll_rate_cmd": pytest.approx(0.3),
        "throttle_cmd": pytest.approx(0.7),
        "nz_actual": pytest.approx(1.1),
    }
    assert point["clipping"] == {
        "nz_saturated": False,
        "roll_rate_saturated": True,
        "throttle_saturated": False,
        "any_saturated": True,
    }

    episode = recorder.finalize(
        steps=1,
        total_time_s=0.2,
        total_reward=0.0,
        termination_reason="timeout",
        success=False,
        final_position_m=[0.0, 0.0, 5000.0],
        final_speed_mps=200.0,
        final_altitude_m=5000.0,
    )
    assert episode["terminal_reason"] == "timeout"
    assert episode["outcome"] is None
