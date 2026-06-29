"""
Unit tests for virtual point generation.
"""

import numpy as np
import pytest
from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator
from uav_vpp_guidance.virtual_point.coordinate_transform import (
    _signed_horizontal_angle,
    offset_to_world,
    world_to_offset_frame,
)
from uav_vpp_guidance.virtual_point.smoother import VirtualPointSmoother


class TestVirtualPointGenerator:
    def test_init(self):
        config = {"action_dim": 5}
        gen = VirtualPointGenerator(config)
        assert gen.action_dim == 5

    def test_invalid_offensive_anchor_lateral_sign_mode_raises(self):
        with pytest.raises(ValueError, match="offensive_anchor_lateral_sign_mode"):
            VirtualPointGenerator(
                {
                    "action_dim": 3,
                    "offensive_anchor_lateral_sign_mode": "diagonal",
                }
            )

    def test_action_to_virtual_point_current_target(self):
        config = {"action_dim": 5}
        gen = VirtualPointGenerator(config)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 0.0])}
        vp = gen.action_to_virtual_point(action, own_state, target_state)
        assert np.allclose(vp["position"], np.array([1000.0, 0.0, 0.0]))

    def test_get_target_position_with_position_m(self):
        """_get_target_position 应支持 position_m 字段。"""
        target_state = {"position_m": np.array([500.0, 200.0, 300.0])}
        pos = VirtualPointGenerator._get_target_position(target_state)
        assert np.allclose(pos, np.array([500.0, 200.0, 300.0]))

    def test_get_target_position_priority_neu_over_m(self):
        """position_neu 优先于 position_m。"""
        target_state = {
            "position_neu": np.array([1.0, 2.0, 3.0]),
            "position_m": np.array([4.0, 5.0, 6.0]),
        }
        pos = VirtualPointGenerator._get_target_position(target_state)
        assert np.allclose(pos, np.array([1.0, 2.0, 3.0]))

    def test_constant_velocity_with_velocity_vector_mps(self):
        """constant_velocity 应支持 velocity_vector_mps (NEU)。"""
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_vector_mps": np.array([50.0, 0.0, 10.0]),
        }
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1100.0, 0.0, 20.0]))

    def test_constant_velocity_with_velocity_ned_converts_to_neu(self):
        """velocity_ned [vn, ve, vd] 应转换为 NEU [vn, ve, -vd]。"""
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_ned": np.array([50.0, 0.0, -10.0]),  # vd=-10 -> 向上 10
        }
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1100.0, 0.0, 20.0]))

    def test_constant_velocity_fallback_to_velocity(self):
        """velocity 字段作为 fallback。"""
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity": np.array([30.0, 0.0, 5.0]),
        }
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1060.0, 0.0, 10.0]))

    def test_constant_velocity_missing_velocity_returns_current_pos(self):
        """缺少速度信息时返回当前位置。"""
        target_state = {"position_neu": np.array([1000.0, 0.0, 0.0])}
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1000.0, 0.0, 0.0]))


    def test_predicted_target_blend_default_preserves_future_anchor(self):
        gen = VirtualPointGenerator({"action_dim": 3})
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}
        predicted_target_position = np.array([1400.0, 0.0, 5000.0])

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=predicted_target_position,
            return_info=True,
        )

        assert info["predicted_target_blend"] == 1.0
        assert np.allclose(info["anchor_pos"], predicted_target_position)
        assert np.allclose(vp["position"], predicted_target_position)

    def test_predicted_target_blend_by_task_interpolates_anchor(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "predicted_target_blend_by_task": {"head_on": 0.25},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}
        predicted_target_position = np.array([1400.0, 0.0, 5000.0])

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=predicted_target_position,
            return_info=True,
        )

        expected_anchor = np.array([1100.0, 0.0, 5000.0])
        assert info["predicted_target_blend"] == 0.25
        assert np.allclose(info["unblended_anchor_pos"], predicted_target_position)
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_predicted_target_blend_override_applies_only_to_this_call(self):
        gen = VirtualPointGenerator({"action_dim": 3}, task_name="head_on")
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}
        predicted_target_position = np.array([1400.0, 0.0, 5000.0])

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=predicted_target_position,
            predicted_target_blend_override=0.5,
            return_info=True,
        )

        expected_anchor = np.array([1200.0, 0.0, 5000.0])
        assert info["predicted_target_blend"] == 0.5
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)
        assert gen.predicted_target_blend == 1.0

    def test_predicted_target_forward_scale_by_task_applies_before_override(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "predicted_target_forward_scale_by_task": {"head_on": 0.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=np.array([800.0, 400.0, 5000.0]),
            return_info=True,
        )

        assert info["predicted_target_forward_scale"] == 0.0
        assert np.allclose(
            info["pre_predicted_target_forward_scale_anchor_pos"],
            np.array([800.0, 400.0, 5000.0]),
        )
        assert np.allclose(info["anchor_pos"], np.array([800.0, 200.0, 5000.0]))
        assert np.allclose(vp["position"], np.array([800.0, 200.0, 5000.0]))

    def test_invalid_predicted_target_blend_raises(self):
        try:
            VirtualPointGenerator(
                {
                    "action_dim": 3,
                    "predicted_target_blend": 1.2,
                }
            )
        except ValueError as exc:
            assert "predicted_target_blend" in str(exc)
        else:
            raise AssertionError("expected invalid predicted_target_blend to raise")

    def test_offensive_position_anchor_sits_behind_target_velocity(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="offensive_position",
            return_info=True,
        )

        expected_anchor = np.array([1000.0, -600.0, 5000.0])
        assert info["anchor_mode"] == "offensive_position"
        assert info["offensive_anchor_longitudinal_m"] == 800.0
        assert np.allclose(info["offensive_anchor_offset"], np.array([-800.0, 0.0, 0.0]))
        assert np.allclose(info["offensive_anchor_world_offset"], np.array([0.0, -800.0, 0.0]))
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_offensive_position_anchor_can_apply_high_side_vertical_offset(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_vertical_m_by_task": {"head_on": 500.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="offensive_position",
            return_info=True,
        )

        expected_anchor = np.array([1000.0, -600.0, 5500.0])
        assert info["offensive_anchor_vertical_m"] == 500.0
        assert np.allclose(
            info["offensive_anchor_offset"],
            np.array([-800.0, 0.0, 500.0]),
        )
        assert np.allclose(
            info["offensive_anchor_world_offset"],
            np.array([0.0, -800.0, 500.0]),
        )
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_offensive_anchor_blend_softens_predicted_target_anchor(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=np.array([1000.0, 400.0, 5000.0]),
            offensive_anchor_blend_override=0.25,
            return_info=True,
        )

        assert info["anchor_mode"] == "predicted_target"
        assert info["offensive_anchor_blend"] == 0.25
        assert np.allclose(
            info["pre_offensive_blend_anchor_pos"],
            np.array([1000.0, 400.0, 5000.0]),
        )
        assert np.allclose(
            info["offensive_anchor_pos"],
            np.array([1000.0, -600.0, 5000.0]),
        )
        assert np.allclose(
            info["anchor_pos"],
            np.array([1000.0, 150.0, 5000.0]),
        )
        assert np.allclose(vp["position"], np.array([1000.0, 150.0, 5000.0]))
        assert info["offensive_anchor_longitudinal_blend"] == 0.25
        assert info["offensive_anchor_lateral_blend"] == 0.25

    def test_offensive_anchor_blend_can_decompose_longitudinal_and_lateral_strength(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
                "offensive_anchor_frame_by_task": {"head_on": "target_velocity"},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([1100.0, 400.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=np.array([1300.0, 400.0, 5000.0]),
            offensive_anchor_blend_override=0.0,
            offensive_anchor_longitudinal_blend_override=1.0,
            offensive_anchor_lateral_blend_override=0.25,
            return_info=True,
        )

        assert info["offensive_anchor_blend"] == 0.0
        assert info["offensive_anchor_longitudinal_blend"] == 1.0
        assert info["offensive_anchor_lateral_blend"] == 0.25
        assert np.allclose(
            info["pre_offensive_blend_anchor_local"],
            np.array([200.0, -300.0, 0.0]),
        )
        assert np.allclose(
            info["post_offensive_blend_anchor_local"],
            np.array([-800.0, -300.0, 0.0]),
        )
        assert np.allclose(info["anchor_pos"], np.array([1300.0, -600.0, 5000.0]))
        assert np.allclose(vp["position"], np.array([1300.0, -600.0, 5000.0]))

    def test_offensive_position_anchor_can_use_same_side_rear_quarter_offset(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([1300.0, 200.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="offensive_position",
            return_info=True,
        )

        expected_anchor = np.array([1300.0, -600.0, 5000.0])
        assert info["offensive_anchor_longitudinal_m"] == 800.0
        assert info["offensive_anchor_lateral_m"] == 300.0
        assert info["offensive_anchor_lateral_sign_mode"] == "same_side"
        assert info["offensive_anchor_lateral_sign"] == pytest.approx(-1.0)
        assert np.allclose(
            info["offensive_anchor_offset"],
            np.array([-800.0, -300.0, 0.0]),
        )
        assert np.allclose(
            info["offensive_anchor_world_offset"],
            np.array([300.0, -800.0, 0.0]),
        )
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_offensive_position_anchor_can_force_fixed_positive_rear_quarter_side(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
                "offensive_anchor_lateral_sign_mode_by_task": {
                    "head_on": "fixed_positive"
                },
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([1300.0, 200.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="offensive_position",
            return_info=True,
        )

        expected_anchor = np.array([700.0, -600.0, 5000.0])
        assert info["offensive_anchor_lateral_sign_mode"] == "fixed_positive"
        assert info["offensive_anchor_lateral_sign"] == pytest.approx(1.0)
        assert np.allclose(
            info["offensive_anchor_offset"],
            np.array([-800.0, 300.0, 0.0]),
        )
        assert np.allclose(
            info["offensive_anchor_world_offset"],
            np.array([-300.0, -800.0, 0.0]),
        )
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_offensive_anchor_blend_can_use_separate_lateral_frame(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
                "offensive_anchor_frame_by_task": {"head_on": "target_velocity"},
                "offensive_anchor_lateral_frame_by_task": {"head_on": "encounter"},
                "offensive_anchor_lateral_sign_mode_by_task": {
                    "head_on": "fixed_positive"
                },
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=np.array([1300.0, 0.0, 5000.0]),
            offensive_anchor_blend_override=0.0,
            offensive_anchor_longitudinal_blend_override=0.0,
            offensive_anchor_lateral_blend_override=1.0,
            return_info=True,
        )

        predicted_world_offset = np.array([300.0, 0.0, 0.0], dtype=np.float64)
        predicted_local = np.array(
            world_to_offset_frame(
                predicted_world_offset,
                "target_velocity",
                own_state=own_state,
                target_state=target_state,
            ),
            dtype=np.float64,
        )
        expected_world_offset = offset_to_world(
            np.array([predicted_local[0], 0.0, predicted_local[2]], dtype=np.float64),
            "target_velocity",
            own_state=own_state,
            target_state=target_state,
        ) + offset_to_world(
            np.array([0.0, 300.0, 0.0], dtype=np.float64),
            "encounter",
            own_state=own_state,
            target_state=target_state,
        )
        expected_anchor = np.array(target_state["position_neu"], dtype=np.float64) + expected_world_offset

        assert info["offensive_anchor_frame"] == "target_velocity"
        assert info["offensive_anchor_lateral_frame"] == "encounter"
        assert info["offensive_anchor_lateral_sign_mode"] == "fixed_positive"
        assert np.allclose(
            info["pre_offensive_blend_anchor_lateral_local"],
            world_to_offset_frame(
                predicted_world_offset,
                "encounter",
                own_state=own_state,
                target_state=target_state,
            ),
        )
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_offensive_anchor_lateral_world_offset_override_preserves_world_body(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
                "offensive_anchor_frame_by_task": {"head_on": "target_velocity"},
                "offensive_anchor_lateral_frame_by_task": {"head_on": "encounter"},
                "offensive_anchor_lateral_sign_mode_by_task": {
                    "head_on": "fixed_positive"
                },
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        first_target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }
        second_target_state = {
            "position_neu": np.array([0.0, 1000.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        (
            _first_anchor_pos,
            _first_anchor_offset,
            _first_anchor_world_offset,
            _first_anchor_lateral_local_offset,
            first_lateral_world_offset,
            _first_lateral_sign,
        ) = gen.compute_offensive_anchor_components(
            first_target_state,
            own_state=own_state,
        )
        (
            _second_anchor_pos,
            _second_anchor_offset,
            _second_anchor_world_offset,
            _second_anchor_lateral_local_offset,
            second_lateral_world_offset,
            _second_lateral_sign,
        ) = gen.compute_offensive_anchor_components(
            second_target_state,
            own_state=own_state,
        )
        (
            overridden_anchor_pos,
            _overridden_anchor_offset,
            overridden_anchor_world_offset,
            overridden_lateral_local_offset,
            overridden_lateral_world_offset,
            overridden_lateral_sign,
        ) = gen.compute_offensive_anchor_components(
            second_target_state,
            own_state=own_state,
            lateral_world_offset_override=first_lateral_world_offset,
        )

        expected_longitudinal_world_offset = offset_to_world(
            np.array([-800.0, 0.0, 0.0], dtype=np.float64),
            "target_velocity",
            own_state=own_state,
            target_state=second_target_state,
        )
        expected_anchor = (
            np.asarray(second_target_state["position_neu"], dtype=np.float64)
            + expected_longitudinal_world_offset
            + first_lateral_world_offset
        )

        assert not np.allclose(
            second_lateral_world_offset,
            first_lateral_world_offset,
        )
        assert np.allclose(
            overridden_lateral_world_offset,
            first_lateral_world_offset,
        )
        assert np.allclose(
            overridden_anchor_world_offset,
            expected_longitudinal_world_offset + first_lateral_world_offset,
        )
        assert np.allclose(overridden_anchor_pos, expected_anchor)
        assert np.allclose(
            overridden_lateral_local_offset,
            world_to_offset_frame(
                first_lateral_world_offset,
                "encounter",
                own_state=own_state,
                target_state=second_target_state,
            ),
        )
        assert overridden_lateral_sign == pytest.approx(
            np.sign(overridden_lateral_local_offset[1])
        )

    def test_predicted_target_forward_scale_only_scales_longitudinal_component(self):
        gen = VirtualPointGenerator({"action_dim": 3}, task_name="head_on")
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="predicted_target",
            predicted_target_position=np.array([800.0, 400.0, 5000.0]),
            predicted_target_forward_scale_override=0.0,
            return_info=True,
        )

        assert info["predicted_target_forward_scale"] == 0.0
        assert np.allclose(
            info["pre_predicted_target_forward_scale_anchor_pos"],
            np.array([800.0, 400.0, 5000.0]),
        )
        assert np.allclose(info["anchor_pos"], np.array([800.0, 200.0, 5000.0]))
        assert np.allclose(vp["position"], np.array([800.0, 200.0, 5000.0]))

    def test_lateral_scale_by_task_scales_local_lateral_offset(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "d_long_range": [-100.0, 100.0],
                "d_lat_range": [-100.0, 100.0],
                "d_vert_range": [-10.0, 10.0],
                "lateral_scale_by_task": {"head_on": 0.25},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}

        vp, info = gen.action_to_virtual_point(
            np.array([0.0, 1.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["lateral_scale"] == 0.25
        assert np.allclose(info["offset"], np.array([0.0, 25.0, 0.0]))
        assert np.allclose(info["world_offset"], np.array([0.0, 25.0, 0.0]))
        assert np.allclose(vp["position"], np.array([1000.0, 25.0, 5000.0]))

    def test_longitudinal_scale_by_task_scales_local_longitudinal_offset(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "d_long_range": [-100.0, 100.0],
                "d_lat_range": [-100.0, 100.0],
                "d_vert_range": [-10.0, 10.0],
                "longitudinal_scale_by_task": {"head_on": 0.25},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}

        vp, info = gen.action_to_virtual_point(
            np.array([1.0, 0.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["longitudinal_scale"] == 0.25
        assert np.allclose(info["offset"], np.array([25.0, 0.0, 0.0]))
        assert np.allclose(info["world_offset"], np.array([25.0, 0.0, 0.0]))
        assert np.allclose(vp["position"], np.array([1025.0, 0.0, 5000.0]))

    def test_scale_overrides_apply_only_to_current_call(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "d_long_range": [-100.0, 100.0],
                "d_lat_range": [-100.0, 100.0],
                "d_vert_range": [-10.0, 10.0],
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}

        vp, info = gen.action_to_virtual_point(
            np.array([1.0, 1.0, 0.0]),
            own_state,
            target_state,
            longitudinal_scale_override=0.0,
            lateral_scale_override=0.25,
            return_info=True,
        )

        assert info["longitudinal_scale"] == 0.0
        assert info["lateral_scale"] == 0.25
        assert np.allclose(info["offset"], np.array([0.0, 25.0, 0.0]))
        assert np.allclose(vp["position"], np.array([1000.0, 25.0, 5000.0]))

        vp, info = gen.action_to_virtual_point(
            np.array([1.0, 1.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["longitudinal_scale"] == 1.0
        assert info["lateral_scale"] == 1.0
        assert np.allclose(info["offset"], np.array([100.0, 100.0, 0.0]))
        assert np.allclose(vp["position"], np.array([1100.0, 100.0, 5000.0]))

    def test_invalid_lateral_scale_raises(self):
        try:
            VirtualPointGenerator(
                {
                    "action_dim": 3,
                    "lateral_scale": -0.1,
                }
            )
        except ValueError as exc:
            assert "lateral_scale" in str(exc)
        else:
            raise AssertionError("expected invalid lateral_scale to raise")


class TestVirtualPointOffsetFrames:
    def test_offset_frame_omitted_matches_world_neu(self):
        config = {
            "action_dim": 3,
            "d_long_range": [-100.0, 100.0],
            "d_lat_range": [-50.0, 50.0],
            "d_vert_range": [-10.0, 10.0],
        }
        world_config = {**config, "offset_frame": "world_neu"}
        action = np.array([0.2, -0.4, 0.6])
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([1000.0, 200.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }

        legacy_vp, legacy_info = VirtualPointGenerator(config).action_to_virtual_point(
            action,
            own_state,
            target_state,
            return_info=True,
        )
        world_vp, world_info = VirtualPointGenerator(world_config).action_to_virtual_point(
            action,
            own_state,
            target_state,
            return_info=True,
        )

        assert np.allclose(legacy_vp["position"], world_vp["position"])
        assert np.allclose(legacy_info["offset"], world_info["offset"])
        assert np.allclose(legacy_info["world_offset"], legacy_info["offset"])
        assert legacy_info["offset_frame"] == "world_neu"

    def test_target_velocity_offset_frame_rotates_canonical_east_heading(self):
        target_state = {
            "position_neu": np.zeros(3),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }
        assert np.allclose(
            offset_to_world(
                [100.0, 0.0, 0.0],
                "target_velocity",
                target_state=target_state,
            ),
            np.array([0.0, 100.0, 0.0]),
        )
        assert np.allclose(
            offset_to_world(
                [0.0, 100.0, 0.0],
                "target_velocity",
                target_state=target_state,
            ),
            np.array([-100.0, 0.0, 0.0]),
        )

    def test_los_relative_offset_frame_uses_own_to_target_los(self):
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {"position_neu": np.array([0.0, 1000.0, 0.0])}
        assert np.allclose(
            offset_to_world(
                [100.0, 0.0, 0.0],
                "los_relative",
                own_state=own_state,
                target_state=target_state,
            ),
            np.array([0.0, 100.0, 0.0]),
        )
        assert np.allclose(
            offset_to_world(
                [0.0, 100.0, 0.0],
                "los_relative",
                own_state=own_state,
                target_state=target_state,
            ),
            np.array([-100.0, 0.0, 0.0]),
        )

    def test_encounter_offset_frame_uses_los_target_velocity_bisector(self):
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }
        diagonal = 100.0 / np.sqrt(2.0)
        assert np.allclose(
            offset_to_world(
                [100.0, 0.0, 0.0],
                "encounter",
                own_state=own_state,
                target_state=target_state,
            ),
            np.array([diagonal, diagonal, 0.0]),
        )
        assert np.allclose(
            offset_to_world(
                [0.0, 100.0, 0.0],
                "encounter",
                own_state=own_state,
                target_state=target_state,
            ),
            np.array([-diagonal, diagonal, 0.0]),
        )

    def test_encounter_stable_clamps_large_bisector_rotation_toward_target_velocity(self):
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([1000.0, -1000.0, 0.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }
        diagonal = 100.0 / np.sqrt(2.0)
        target_velocity_world = np.array([0.0, 100.0, 0.0])

        encounter_world = offset_to_world(
            [100.0, 0.0, 0.0],
            "encounter",
            own_state=own_state,
            target_state=target_state,
        )
        stable_world = offset_to_world(
            [100.0, 0.0, 0.0],
            "encounter_stable",
            own_state=own_state,
            target_state=target_state,
        )

        encounter_heading_error_deg = abs(
            np.rad2deg(
                _signed_horizontal_angle(target_velocity_world, encounter_world)
            )
        )
        stable_heading_error_deg = abs(
            np.rad2deg(
                _signed_horizontal_angle(target_velocity_world, stable_world)
            )
        )

        assert stable_heading_error_deg < encounter_heading_error_deg
        assert stable_heading_error_deg == pytest.approx(45.0)
        assert np.allclose(stable_world, np.array([diagonal, diagonal, 0.0]))

    def test_encounter_stable_accepts_custom_max_heading_delta(self):
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([1000.0, -1000.0, 0.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }
        target_velocity_world = np.array([0.0, 100.0, 0.0])

        stable_world = offset_to_world(
            [100.0, 0.0, 0.0],
            "encounter_stable",
            own_state=own_state,
            target_state=target_state,
            encounter_stable_max_heading_delta_deg=20.0,
        )
        stable_heading_error_deg = abs(
            np.rad2deg(
                _signed_horizontal_angle(target_velocity_world, stable_world)
            )
        )

        assert stable_heading_error_deg == pytest.approx(20.0)

    def test_generator_target_velocity_frame_preserves_local_offset_metadata(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offset_frame": "target_velocity",
                "d_long_range": [-100.0, 100.0],
                "d_lat_range": [-50.0, 50.0],
                "d_vert_range": [-10.0, 10.0],
            }
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.array([1.0, 0.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["offset_frame"] == "target_velocity"
        assert np.allclose(info["offset"], np.array([100.0, 0.0, 0.0]))
        assert np.allclose(info["world_offset"], np.array([0.0, 100.0, 0.0]))
        assert np.allclose(vp["position"], np.array([1000.0, 100.0, 5000.0]))

    def test_task_specific_offset_frame_overrides_default(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offset_frame": "world_neu",
                "offset_frame_by_task": {"head_on": "target_velocity"},
                "d_long_range": [-100.0, 100.0],
                "d_lat_range": [-50.0, 50.0],
                "d_vert_range": [-10.0, 10.0],
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.array([1.0, 0.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["offset_frame"] == "target_velocity"
        assert np.allclose(info["world_offset"], np.array([0.0, 100.0, 0.0]))
        assert np.allclose(vp["position"], np.array([1000.0, 100.0, 5000.0]))

    def test_task_specific_offset_frame_falls_back_to_default_for_other_tasks(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offset_frame": "world_neu",
                "offset_frame_by_task": {"head_on": "target_velocity"},
                "d_long_range": [-100.0, 100.0],
                "d_lat_range": [-50.0, 50.0],
                "d_vert_range": [-10.0, 10.0],
            },
            task_name="crossing_feasible",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }

        _, info = gen.action_to_virtual_point(
            np.array([1.0, 0.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["offset_frame"] == "world_neu"
        assert np.allclose(info["world_offset"], np.array([100.0, 0.0, 0.0]))

    def test_offensive_position_anchor_can_use_encounter_frame(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_frame_by_task": {"head_on": "encounter"},
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="offensive_position",
            return_info=True,
        )

        expected_world_offset = offset_to_world(
            np.array([-800.0, 300.0, 0.0]),
            "encounter",
            own_state=own_state,
            target_state=target_state,
        )
        expected_anchor = (
            np.asarray(target_state["position_neu"], dtype=np.float64)
            + expected_world_offset
        )

        assert info["offensive_anchor_frame"] == "encounter"
        assert info["offensive_anchor_longitudinal_m"] == 800.0
        assert info["offensive_anchor_lateral_m"] == 300.0
        assert info["offensive_anchor_vertical_m"] == 0.0
        assert np.allclose(
            info["offensive_anchor_offset"],
            np.array([-800.0, 300.0, 0.0]),
        )
        assert np.allclose(info["offensive_anchor_world_offset"], expected_world_offset)
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_offensive_position_anchor_can_use_encounter_stable_frame(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_frame_by_task": {"head_on": "encounter_stable"},
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, -1000.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="offensive_position",
            return_info=True,
        )

        expected_world_offset = offset_to_world(
            np.array([-800.0, 300.0, 0.0]),
            "encounter_stable",
            own_state=own_state,
            target_state=target_state,
        )
        expected_anchor = (
            np.asarray(target_state["position_neu"], dtype=np.float64)
            + expected_world_offset
        )

        assert info["offensive_anchor_frame"] == "encounter_stable"
        assert np.allclose(
            info["offensive_anchor_offset"],
            np.array([-800.0, 300.0, 0.0]),
        )
        assert np.allclose(info["offensive_anchor_world_offset"], expected_world_offset)
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)

    def test_offensive_position_anchor_can_configure_encounter_stable_heading_delta(
        self,
    ):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "offensive_anchor_frame_by_task": {"head_on": "encounter_stable"},
                "offensive_anchor_lateral_frame_by_task": {
                    "head_on": "encounter_stable"
                },
                "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task": {
                    "head_on": 20.0
                },
                "offensive_anchor_longitudinal_m_by_task": {"head_on": 800.0},
                "offensive_anchor_lateral_m_by_task": {"head_on": 300.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, -1000.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        vp, info = gen.action_to_virtual_point(
            np.zeros(3),
            own_state,
            target_state,
            anchor_mode="offensive_position",
            return_info=True,
        )

        expected_world_offset = offset_to_world(
            np.array([-800.0, 300.0, 0.0]),
            "encounter_stable",
            own_state=own_state,
            target_state=target_state,
            encounter_stable_max_heading_delta_deg=20.0,
        )
        expected_anchor = (
            np.asarray(target_state["position_neu"], dtype=np.float64)
            + expected_world_offset
        )

        assert (
            info["offensive_anchor_encounter_stable_max_heading_delta_deg"]
            == pytest.approx(20.0)
        )
        assert np.allclose(info["offensive_anchor_world_offset"], expected_world_offset)
        assert np.allclose(info["anchor_pos"], expected_anchor)
        assert np.allclose(vp["position"], expected_anchor)


class TestTacticalBasisVirtualPoint:
    def test_action_semantics_default_is_cartesian_offset(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "d_long_range": [-100.0, 100.0],
                "d_lat_range": [-50.0, 50.0],
                "d_vert_range": [-10.0, 10.0],
            }
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}

        vp, info = gen.action_to_virtual_point(
            np.array([1.0, 0.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["action_semantics"] == "cartesian_offset"
        assert info["configured_action_semantics"] == "cartesian_offset"
        assert info["tactical_basis_enabled"] is False
        assert np.allclose(info["offset"], np.array([100.0, 0.0, 0.0]))
        assert np.allclose(vp["position"], np.array([1100.0, 0.0, 5000.0]))

    def test_tactical_basis_invalid_action_semantics_raises(self):
        with pytest.raises(ValueError, match="action_semantics"):
            VirtualPointGenerator(
                {
                    "action_dim": 3,
                    "action_semantics": "diagonal_magic",
                }
            )

    def test_tactical_basis_head_on_combines_three_basis_vectors(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "action_semantics": "tactical_basis_v1",
                "offset_frame": "world_neu",
                "tactical_basis_longitudinal_frame": "world_neu",
                "tactical_basis_lateral_frame": "world_neu",
                "tactical_basis_vertical_frame": "world_neu",
                "tactical_basis_lateral_sign_mode": "fixed_positive",
                "tactical_basis_lead_lag_extent_m_by_task": {"head_on": 100.0},
                "tactical_basis_inside_outside_extent_m_by_task": {"head_on": 50.0},
                "tactical_basis_climb_descent_extent_m_by_task": {"head_on": 25.0},
            },
            task_name="head_on",
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}

        _, info = gen.action_to_virtual_point(
            np.array([1.0, -1.0, 0.5]),
            own_state,
            target_state,
            return_info=True,
        )

        expected = np.array([100.0, -50.0, 12.5])
        assert info["tactical_basis_enabled"] is True
        assert np.allclose(info["tactical_basis_ll_world"], np.array([100.0, 0.0, 0.0]))
        assert np.allclose(info["tactical_basis_io_world"], np.array([0.0, 50.0, 0.0]))
        assert np.allclose(info["tactical_basis_cd_world"], np.array([0.0, 0.0, 25.0]))
        assert np.allclose(info["tactical_basis_world_offset"], expected)
        assert np.allclose(info["world_offset"], expected)

    def test_tactical_basis_lead_lag_uses_target_velocity_frame(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "action_semantics": "tactical_basis_v1",
                "offset_frame": "world_neu",
                "tactical_basis_longitudinal_frame": "target_velocity",
                "tactical_basis_lateral_frame": "world_neu",
                "tactical_basis_vertical_frame": "world_neu",
                "tactical_basis_lateral_sign_mode": "fixed_positive",
                "tactical_basis_lead_lag_extent_m": 400.0,
            }
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        _, info = gen.action_to_virtual_point(
            np.array([1.0, 0.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert info["tactical_basis_longitudinal_frame"] == "target_velocity"
        assert np.allclose(info["tactical_basis_ll_world"], np.array([0.0, 400.0, 0.0]))
        assert np.allclose(info["world_offset"], np.array([0.0, 400.0, 0.0]))

    def test_tactical_basis_inside_outside_uses_encounter_stable_lateral_frame(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "action_semantics": "tactical_basis_v1",
                "offset_frame": "world_neu",
                "tactical_basis_longitudinal_frame": "world_neu",
                "tactical_basis_lateral_frame": "encounter_stable",
                "tactical_basis_vertical_frame": "world_neu",
                "tactical_basis_lateral_sign_mode": "fixed_positive",
                "tactical_basis_inside_outside_extent_m": 300.0,
            }
        )
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }

        _, info = gen.action_to_virtual_point(
            np.array([0.0, 1.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        expected = offset_to_world(
            np.array([0.0, 300.0, 0.0]),
            "encounter_stable",
            own_state=own_state,
            target_state=target_state,
        )
        assert info["tactical_basis_lateral_frame"] == "encounter_stable"
        assert info["tactical_basis_lateral_sign_mode"] == "fixed_positive"
        assert info["tactical_basis_lateral_sign"] == pytest.approx(1.0)
        assert np.allclose(info["tactical_basis_io_world"], expected)
        assert np.allclose(info["world_offset"], expected)

    def test_tactical_basis_climb_descent_uses_world_up(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "action_semantics": "tactical_basis_v1",
                "offset_frame": "world_neu",
                "tactical_basis_longitudinal_frame": "world_neu",
                "tactical_basis_lateral_frame": "world_neu",
                "tactical_basis_vertical_frame": "world_neu",
                "tactical_basis_climb_descent_extent_m": 200.0,
            }
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}

        vp, info = gen.action_to_virtual_point(
            np.array([0.0, 0.0, 1.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert np.allclose(info["tactical_basis_cd_world"], np.array([0.0, 0.0, 200.0]))
        assert np.allclose(info["world_offset"], np.array([0.0, 0.0, 200.0]))
        assert np.allclose(vp["position"], np.array([1000.0, 0.0, 5200.0]))

    def test_tactical_basis_by_task_extents_apply_to_head_on_and_crossing(self):
        config = {
            "action_dim": 3,
            "action_semantics": "tactical_basis_v1",
            "offset_frame": "world_neu",
            "tactical_basis_longitudinal_frame": "world_neu",
            "tactical_basis_lateral_frame": "world_neu",
            "tactical_basis_vertical_frame": "world_neu",
            "tactical_basis_lateral_sign_mode": "fixed_positive",
            "tactical_basis_inside_outside_extent_m_by_task": {
                "head_on": 800.0,
                "crossing_feasible": 200.0,
            },
        }
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}

        _, head_on_info = VirtualPointGenerator(
            config,
            task_name="head_on",
        ).action_to_virtual_point(
            np.array([0.0, 1.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )
        _, crossing_info = VirtualPointGenerator(
            config,
            task_name="crossing_feasible",
        ).action_to_virtual_point(
            np.array([0.0, 1.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert head_on_info["tactical_basis_inside_outside_extent_m"] == 800.0
        assert crossing_info["tactical_basis_inside_outside_extent_m"] == 200.0
        assert np.allclose(head_on_info["world_offset"], np.array([0.0, 800.0, 0.0]))
        assert np.allclose(crossing_info["world_offset"], np.array([0.0, 200.0, 0.0]))

    def test_tactical_basis_returns_legacy_offset_projection_for_recorder_compat(self):
        gen = VirtualPointGenerator(
            {
                "action_dim": 3,
                "action_semantics": "tactical_basis_v1",
                "offset_frame": "target_velocity",
                "tactical_basis_longitudinal_frame": "world_neu",
                "tactical_basis_lateral_frame": "world_neu",
                "tactical_basis_vertical_frame": "world_neu",
                "tactical_basis_lateral_sign_mode": "fixed_positive",
                "tactical_basis_lead_lag_extent_m": 100.0,
            }
        )
        own_state = {"position_neu": np.zeros(3)}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 200.0, 0.0]),
        }

        _, info = gen.action_to_virtual_point(
            np.array([1.0, 0.0, 0.0]),
            own_state,
            target_state,
            return_info=True,
        )

        assert np.allclose(
            info["offset"],
            world_to_offset_frame(
                info["world_offset"],
                info["offset_frame"],
                own_state=own_state,
                target_state=target_state,
            ),
        )


class TestVirtualPointSmoother:
    def test_smooth_first_call(self):
        smoother = VirtualPointSmoother(alpha=0.3)
        point = np.array([1.0, 2.0, 3.0])
        out = smoother.smooth(point)
        assert np.allclose(out, point)

    def test_smooth_subsequent(self):
        smoother = VirtualPointSmoother(alpha=0.5)
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([3.0, 3.0, 3.0])
        smoother.smooth(p1)
        out = smoother.smooth(p2)
        expected = 0.5 * p2 + 0.5 * p1
        assert np.allclose(out, expected)
