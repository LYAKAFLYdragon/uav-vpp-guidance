"""Targeted tests for multi-waypoint capture and task command shaping."""

import numpy as np
import pytest

from uav_vpp_guidance.envs.multi_waypoint_tracking_env import MultiWaypointTrackingEnv


def _base_config():
    return {
        "backend": "simple",
        "env": {
            "use_jsbsim": False,
            "strict_backend": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "high_level_dt": 0.2,
            "max_high_level_steps": 64,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 20000.0,
            "xy_limit_m": 20000.0,
            "target_mode": "constant_velocity",
        },
        "virtual_point": {
            "enabled": True,
            "anchor_mode": "current_target",
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        },
        "trajectory_prediction": {"enabled": False},
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.4,
            "throttle_max": 0.9,
        },
        "reward": {
            "w_range": 0.5,
            "w_angle": 0.8,
            "w_energy": 0.2,
            "w_safety": 2.0,
            "w_saturation": 1.0,
            "w_smooth": 0.1,
            "terminal_success": 200.0,
            "terminal_failure": -200.0,
            "terminal_crash": -300.0,
            "min_altitude_m": 500.0,
        },
        "guidance": {
            "mode": "los_rate",
            "gains": {
                "k_los": 1.0,
                "k_pos": 0.5,
                "k_damp": 0.2,
                "k_roll": 1.0,
                "k_speed": 0.2,
                "alpha_filter": 0.3,
            },
        },
        "task": {
            "name": "multi_waypoint",
            "env_class": "MultiWaypointTrackingEnv",
            "multi_waypoint": {
                "n_waypoints": 2,
                "waypoint_distance_range_m": [1000.0, 1000.0],
                "target_speed_range_mps": [100.0, 100.0],
                "heading_delta_limit_deg": 0.0,
                "target_altitude_m": 5000.0,
                "reach_radius_m": 600.0,
                "capture_radius_m": 750.0,
                "segment_timeout_s": 1.0,
                "near_miss_capture": {
                    "enabled": True,
                    "radius_scale": 1.2,
                    "min_recede_m": 75.0,
                    "min_elapsed_s": 0.4,
                },
                "approach_guidance": {
                    "enabled": True,
                    "blend_start_range_m": 2800.0,
                    "static_range_m": 1500.0,
                },
                "command_shaping": {
                    "enabled": True,
                    "near_waypoint_roll_moderation_enabled": True,
                    "roll_moderation_range_m": 1200.0,
                    "roll_moderation_min_scale": 0.45,
                    "same_bank_guard_enabled": True,
                    "same_bank_guard_range_m": 2500.0,
                    "same_bank_guard_start_deg": 75.0,
                    "same_bank_guard_soft_cap_enabled": True,
                    "same_bank_guard_soft_cap_rate_cmd": 0.1,
                    "same_bank_guard_soft_cap_heading_error_deg": 120.0,
                    "same_bank_guard_soft_cap_max_recede_m": 100.0,
                    "overbank_recovery_enabled": True,
                    "roll_recovery_start_deg": 80.0,
                    "roll_recovery_rate_cmd": 1.5,
                },
            },
        },
    }


@pytest.fixture
def env():
    environment = MultiWaypointTrackingEnv(_base_config())
    yield environment
    environment.close()


def test_reset_observation_uses_synthetic_waypoint_target(env):
    obs = env.reset(seed=0)
    np.testing.assert_allclose(
        obs["target_state"]["position_m"],
        env.synthetic_target_state["position_m"],
    )


def test_approach_guidance_blends_target_to_static_waypoint(env):
    env.reset(seed=0)
    waypoint = env.waypoints[0]
    static_pos = np.asarray(waypoint["pos"], dtype=float)
    env.synthetic_target_state = env._make_target_state(waypoint)
    env.synthetic_target_state["position_m"] = static_pos + np.array([500.0, 0.0, 0.0])
    env.synthetic_target_state["position_neu"] = env.synthetic_target_state["position_m"]
    own_state = {
        "position_m": static_pos + np.array([1000.0, 0.0, 0.0]),
        "altitude_m": static_pos[2],
    }

    target = env._task_get_target_state({}, own_state)

    np.testing.assert_allclose(target["position_m"], static_pos)
    assert env._last_approach_guidance_info["multi_waypoint_static_waypoint_blend"] == pytest.approx(1.0)


def test_approach_guidance_is_opt_in_for_legacy_configs():
    cfg = _base_config()
    cfg["task"]["multi_waypoint"].pop("approach_guidance")
    environment = MultiWaypointTrackingEnv(cfg)
    try:
        environment.reset(seed=0)
        waypoint = environment.waypoints[0]
        static_pos = np.asarray(waypoint["pos"], dtype=float)
        synthetic_pos = static_pos + np.array([500.0, 0.0, 0.0])
        environment.synthetic_target_state = environment._make_target_state(waypoint)
        environment.synthetic_target_state["position_m"] = synthetic_pos
        environment.synthetic_target_state["position_neu"] = synthetic_pos
        own_state = {
            "position_m": static_pos + np.array([1000.0, 0.0, 0.0]),
            "altitude_m": static_pos[2],
        }

        target = environment._task_get_target_state({}, own_state)
    finally:
        environment.close()

    np.testing.assert_allclose(target["position_m"], synthetic_pos)
    assert environment._last_approach_guidance_info["multi_waypoint_approach_guidance_enabled"] is False


def test_waypoint_capture_uses_static_waypoint_range(env):
    env.reset(seed=0)
    reach_radius = env.task_cfg["reach_radius_m"]
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    own_state = {
        "position_m": waypoint_pos + np.array([0.5 * reach_radius, 0.0, 0.0]),
        "altitude_m": waypoint_pos[2],
    }
    info = {"range_m": 9999.0, "is_success": False, "reason": None}

    _, terminated, truncated, out = env._task_post_step(
        own_state,
        target_state_post={},
        info=info,
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    assert not terminated
    assert not truncated
    assert out["completed_waypoints"] == 1
    assert out["active_waypoint_index"] == 1
    assert out["backend_range_m"] == pytest.approx(9999.0)
    assert out["waypoint_range_m"] < reach_radius
    assert out["switch_events"][0]["reached"] is True


def test_waypoint_capture_allows_configured_capture_radius(env):
    env.reset(seed=0)
    reach_radius = env.task_cfg["reach_radius_m"]
    capture_radius = env.task_cfg["capture_radius_m"]
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    own_state = {
        "position_m": waypoint_pos + np.array([0.5 * (reach_radius + capture_radius), 0.0, 0.0]),
        "altitude_m": waypoint_pos[2],
    }
    info = {"range_m": 9999.0, "is_success": False, "reason": None}

    _, _, _, out = env._task_post_step(
        own_state,
        target_state_post={},
        info=info,
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    assert reach_radius < out["waypoint_range_m"] < capture_radius
    assert out["completed_waypoints"] == 1
    assert out["switch_events"][0]["capture_radius_m"] == pytest.approx(capture_radius)


def test_near_miss_passed_waypoint_capture_advances_segment(env):
    env.reset(seed=0)
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    env.segment_elapsed_s = 0.5
    env.segment_min_waypoint_range_m = 850.0
    env.segment_min_waypoint_range_step = 7
    own_state = {
        "position_m": waypoint_pos + np.array([940.0, 0.0, 0.0]),
        "altitude_m": waypoint_pos[2],
    }
    info = {"range_m": 9999.0, "is_success": False, "reason": None}

    _, terminated, truncated, out = env._task_post_step(
        own_state,
        target_state_post={},
        info=info,
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    assert not terminated
    assert not truncated
    assert out["completed_waypoints"] == 1
    assert out["active_waypoint_index"] == 1
    assert out["capture_reason"] == "near_miss_passed_waypoint"
    assert out["near_miss_capture_active"] is True
    assert out["segment_range_receded_m"] == pytest.approx(90.0)
    assert out["switch_events"][0]["from_idx"] == 0
    assert out["switch_events"][0]["capture_reason"] == "near_miss_passed_waypoint"


def test_near_miss_capture_is_opt_in_for_legacy_configs():
    cfg = _base_config()
    cfg["task"]["multi_waypoint"].pop("near_miss_capture")
    environment = MultiWaypointTrackingEnv(cfg)
    try:
        environment.reset(seed=0)
        waypoint_pos = np.asarray(environment.waypoints[0]["pos"], dtype=float)
        environment.segment_elapsed_s = 0.5
        environment.segment_min_waypoint_range_m = 850.0
        own_state = {
            "position_m": waypoint_pos + np.array([940.0, 0.0, 0.0]),
            "altitude_m": waypoint_pos[2],
        }
        info = {"range_m": 9999.0, "is_success": False, "reason": None}

        _, terminated, truncated, out = environment._task_post_step(
            own_state,
            target_state_post={},
            info=info,
            reward=0.0,
            terminated=False,
            truncated=False,
        )
    finally:
        environment.close()

    assert not terminated
    assert not truncated
    assert out["completed_waypoints"] == 0
    assert out["active_waypoint_index"] == 0
    assert out["near_miss_capture_enabled"] is False
    assert out["near_miss_capture_active"] is False
    assert out["switch_events"] == []


def test_segment_timeout_advances_without_completion_when_not_captured(env):
    env.reset(seed=0)
    env.segment_elapsed_s = env.task_cfg["segment_timeout_s"]
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    own_state = {
        "position_m": waypoint_pos + np.array([5000.0, 0.0, 0.0]),
        "altitude_m": waypoint_pos[2],
    }
    info = {"range_m": 5000.0, "is_success": False, "reason": None}

    _, terminated, truncated, out = env._task_post_step(
        own_state,
        target_state_post={},
        info=info,
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    assert not terminated
    assert not truncated
    assert out["completed_waypoints"] == 0
    assert out["active_waypoint_index"] == 1
    assert out["switch_events"][0]["timeout"] is True
    assert out["switch_events"][0]["reached"] is False


def test_near_waypoint_roll_moderation_leaves_far_command_unchanged(env):
    env.reset(seed=0)
    command = {"nz_cmd": 2.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)

    far_state = {
        "position_m": waypoint_pos + np.array([5000.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    far, far_meta = env._task_adjust_command(
        command,
        far_state,
        target_state={},
        rel_state={},
    )
    assert far == command
    assert far_meta["multi_waypoint_roll_moderation_active"] is False

    near_state = {"position_m": waypoint_pos.copy(), "roll_rad": 0.0}
    near, near_meta = env._task_adjust_command(
        command,
        near_state,
        target_state={},
        rel_state={},
    )
    assert near["roll_rate_cmd"] == pytest.approx(0.45)
    assert near_meta["multi_waypoint_roll_moderation_active"] is True
    assert near_meta["multi_waypoint_roll_recovery_active"] is False


def test_same_bank_guard_soft_caps_when_heading_error_supports_turn(env):
    env.reset(seed=0)
    command = {"nz_cmd": 2.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    own_state = {
        "position_m": waypoint_pos + np.array([2500.0, 0.0, 0.0]),
        "roll_rad": np.deg2rad(75.0),
        "yaw_rad": 0.0,
    }

    adjusted, meta = env._task_adjust_command(
        command,
        own_state,
        target_state={},
        rel_state={},
    )

    assert adjusted["roll_rate_cmd"] == pytest.approx(0.1)
    assert meta["multi_waypoint_same_bank_guard_active"] is True
    assert meta["multi_waypoint_same_bank_guard_soft_cap_active"] is True
    assert meta["multi_waypoint_same_bank_guard_action"] == "soft_cap"
    assert meta["multi_waypoint_roll_recovery_active"] is False
    assert meta["reset_command_filter"] is True


def test_same_bank_guard_zeroes_when_heading_error_does_not_support_turn(env):
    env.reset(seed=0)
    command = {"nz_cmd": 2.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    own_state = {
        "position_m": waypoint_pos + np.array([-2500.0, 0.0, 0.0]),
        "roll_rad": np.deg2rad(75.0),
        "yaw_rad": 0.0,
    }

    adjusted, meta = env._task_adjust_command(
        command,
        own_state,
        target_state={},
        rel_state={},
    )

    assert adjusted["roll_rate_cmd"] == pytest.approx(0.0)
    assert meta["multi_waypoint_same_bank_guard_active"] is True
    assert meta["multi_waypoint_same_bank_guard_soft_cap_active"] is False
    assert meta["multi_waypoint_same_bank_guard_action"] == "zero"


def test_same_bank_guard_keeps_hard_zero_when_soft_cap_disabled():
    cfg = _base_config()
    shaping = cfg["task"]["multi_waypoint"]["command_shaping"]
    shaping.pop("same_bank_guard_soft_cap_enabled")
    environment = MultiWaypointTrackingEnv(cfg)
    try:
        environment.reset(seed=0)
        command = {"nz_cmd": 2.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
        waypoint_pos = np.asarray(environment.waypoints[0]["pos"], dtype=float)
        own_state = {
            "position_m": waypoint_pos + np.array([2500.0, 0.0, 0.0]),
            "roll_rad": np.deg2rad(75.0),
            "yaw_rad": 0.0,
        }

        adjusted, meta = environment._task_adjust_command(
            command,
            own_state,
            target_state={},
            rel_state={},
        )
    finally:
        environment.close()

    assert adjusted["roll_rate_cmd"] == pytest.approx(0.0)
    assert meta["multi_waypoint_same_bank_guard_active"] is True
    assert meta["multi_waypoint_same_bank_guard_action"] == "zero"


def test_same_bank_guard_allows_commands_toward_level(env):
    env.reset(seed=0)
    command = {"nz_cmd": 2.0, "roll_rate_cmd": -1.0, "throttle_cmd": 0.7}
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    own_state = {
        "position_m": waypoint_pos + np.array([2500.0, 0.0, 0.0]),
        "roll_rad": np.deg2rad(75.0),
        "yaw_rad": 0.0,
    }

    adjusted, meta = env._task_adjust_command(
        command,
        own_state,
        target_state={},
        rel_state={},
    )

    assert adjusted["roll_rate_cmd"] == pytest.approx(-1.0)
    assert meta["multi_waypoint_same_bank_guard_active"] is False


@pytest.mark.parametrize(
    ("roll_deg", "expected_roll_rate"),
    [
        (100.0, -1.5),
        (-100.0, 1.5),
    ],
)
def test_overbank_recovery_commands_roll_toward_level(env, roll_deg, expected_roll_rate):
    env.reset(seed=0)
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    command = {"nz_cmd": 2.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
    own_state = {
        "position_m": waypoint_pos + np.array([5000.0, 0.0, 0.0]),
        "roll_rad": np.deg2rad(roll_deg),
    }

    adjusted, meta = env._task_adjust_command(
        command,
        own_state,
        target_state={},
        rel_state={},
    )

    assert adjusted["roll_rate_cmd"] == pytest.approx(expected_roll_rate)
    assert meta["multi_waypoint_roll_recovery_active"] is True
    assert meta["reset_command_filter"] is True


def test_command_override_bypasses_multi_waypoint_shaping(env):
    env.reset(seed=0)
    command = {"nz_cmd": 2.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
    waypoint_pos = np.asarray(env.waypoints[0]["pos"], dtype=float)
    own_state = {
        "position_m": waypoint_pos.copy(),
        "roll_rad": np.deg2rad(120.0),
    }

    adjusted, meta = env._task_adjust_command(
        command,
        own_state,
        target_state={},
        rel_state={},
        use_command_override=True,
    )

    assert adjusted == command
    assert meta["multi_waypoint_command_shaping_bypassed"] is True


def test_command_shaping_is_opt_in_for_legacy_configs():
    cfg = _base_config()
    cfg["task"]["multi_waypoint"].pop("command_shaping")
    environment = MultiWaypointTrackingEnv(cfg)
    try:
        environment.reset(seed=0)
        waypoint_pos = np.asarray(environment.waypoints[0]["pos"], dtype=float)
        command = {"nz_cmd": 2.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7}
        own_state = {
            "position_m": waypoint_pos.copy(),
            "roll_rad": np.deg2rad(120.0),
        }

        adjusted, meta = environment._task_adjust_command(
            command,
            own_state,
            target_state={},
            rel_state={},
        )
    finally:
        environment.close()

    assert adjusted == command
    assert meta["multi_waypoint_command_shaping_enabled"] is False
