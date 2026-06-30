from __future__ import annotations

import numpy as np

from uav_vpp_guidance.virtual_point.no_vpp_guidance import NoVPPGuidance


def test_no_vpp_guidance_accepts_extended_override_kwargs():
    guidance = NoVPPGuidance()
    target_state = {
        "position_neu": np.array([1000.0, 200.0, 1500.0], dtype=np.float64),
    }

    virtual_point, info = guidance.action_to_virtual_point(
        action=np.array([0.1, -0.2, 0.3], dtype=np.float32),
        own_state={},
        target_state=target_state,
        predicted_target_blend_override=0.25,
        predicted_target_forward_scale_override=0.5,
        offensive_anchor_blend_override=0.1,
        offensive_anchor_longitudinal_blend_override=0.2,
        offensive_anchor_lateral_blend_override=0.3,
        offensive_anchor_lateral_world_offset_override=np.array(
            [10.0, 20.0, 0.0],
            dtype=np.float64,
        ),
        longitudinal_scale_override=0.4,
        lateral_scale_override=0.5,
        return_info=True,
        unexpected_future_override="ignored",
    )

    np.testing.assert_allclose(
        virtual_point["position"],
        target_state["position_neu"],
    )
    np.testing.assert_allclose(
        virtual_point["offset"],
        np.zeros(3, dtype=np.float64),
    )
    assert info["anchor_mode"] == "current_target"
    assert info["prediction_info"]["vpp_enabled"] is False
