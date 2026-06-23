import numpy as np

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


def _base_config(add_prediction_to_observation: bool):
    return {
        "env": {
            "use_jsbsim": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "max_high_level_steps": 32,
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.2,
            "hysteresis_range_m": 950.0,
            "hysteresis_ata_deg": 30.0,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 12000.0,
            "target_mode": "constant_velocity",
        },
        "virtual_point": {
            "anchor_mode": "predicted_target",
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
        },
        "trajectory_prediction": {
            "enabled": True,
            "predictor_type": "constant_velocity",
            "freeze_predictor_during_rl": True,
            "prediction": {
                "lookahead_time_s": 1.0,
                "output_mode": "absolute_position",
                "fallback_mode": "constant_velocity",
            },
            "history": {"history_len": 5, "padding_mode": "repeat_first"},
            "integration": {
                "anchor_mode": "predicted_target",
                "add_prediction_to_observation": add_prediction_to_observation,
            },
            "normalization": {
                "position_scale_m": 1000.0,
                "velocity_scale_mps": 300.0,
            },
        },
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
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
    }


def test_prediction_observation_flag_controls_observation_dimension():
    env_base = CloseRangeTrackingEnv(_base_config(False))
    env_aug = CloseRangeTrackingEnv(_base_config(True))
    try:
        base_obs = env_base.reset(seed=0)
        aug_obs = env_aug.reset(seed=0)

        assert base_obs["observation_vector"].shape[0] == 16
        assert aug_obs["observation_vector"].shape[0] == 30
        assert np.isfinite(aug_obs["observation_vector"]).all()

        aug_obs, *_ = env_aug.step(np.zeros(3))
        assert aug_obs["observation_vector"].shape[0] == 30
        assert np.isfinite(aug_obs["observation_vector"]).all()
    finally:
        env_base.close()
        env_aug.close()
