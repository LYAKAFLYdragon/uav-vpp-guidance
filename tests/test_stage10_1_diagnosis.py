"""
Stage 10.1 JSBSim divergence diagnosis tests.

Covers:
- telemetry schema contract
- command unit conversion checks
- termination reason taxonomy
- diagnosis runner smoke test
- JSBSim scenario position conversion
"""

import csv
import os
import tempfile

import numpy as np
import pytest

from uav_vpp_guidance.evaluation.jsbsim_diagnosis import (
    STEP_COLUMNS,
    EPISODE_COLUMNS,
    FAILURE_ROOT_CAUSE_COLUMNS,
    HoldController,
    DirectPNController,
    LowGainDirectController,
    compute_command_sanity,
    classify_failure_root_cause,
    _df_to_markdown,
)
from uav_vpp_guidance.flight_control.actuator_interface import JSBSimActuatorInterface
from uav_vpp_guidance.envs.jsbsim_env import neu2lla, lla2neu


class TestTelemetrySchemaContract:
    """Ensure the diagnosis telemetry schema is stable and complete."""

    def test_step_columns_contain_required_fields(self):
        required = {
            "episode_id", "method", "scenario", "seed", "step", "time_s",
            "range_m", "range_rate_mps",
            "own_x_m", "own_y_m", "own_z_m",
            "own_vx_mps", "own_vy_mps", "own_vz_mps",
            "target_x_m", "target_y_m", "target_z_m",
            "own_roll_rad", "own_pitch_rad", "own_yaw_rad",
            "own_speed_mps", "own_altitude_m", "own_nz_g",
            "nz_cmd", "roll_rate_cmd", "throttle_cmd",
            "elevator_cmd", "aileron_cmd", "rudder_cmd", "throttle_actual",
            "saturation_flag", "effective_guidance_mode", "virtual_point_source",
            "mode_switch_effective", "reason",
        }
        assert required.issubset(set(STEP_COLUMNS)), "STEP_COLUMNS missing required fields"

    def test_episode_columns_contain_required_fields(self):
        required = {
            "episode_id", "method", "scenario", "seed", "return", "length",
            "is_success", "is_crash", "is_timeout", "is_out_of_bounds", "reason",
            "min_range_m", "final_range_m", "mean_nz_cmd", "mean_roll_rate_cmd",
            "mean_throttle_cmd", "command_override",
        }
        assert required.issubset(set(EPISODE_COLUMNS)), "EPISODE_COLUMNS missing required fields"

    def test_failure_root_cause_columns(self):
        required = {"episode_id", "method", "scenario", "seed", "reason", "root_cause", "diagnosis_note"}
        assert required.issubset(set(FAILURE_ROOT_CAUSE_COLUMNS))


class TestCommandUnitConversion:
    """Verify JSBSim actuator mapping uses expected units and ranges."""

    def test_nz_to_elevator_gain(self):
        iface = JSBSimActuatorInterface({})
        # 7 g -> full elevator deflection
        assert iface.nz_gain == pytest.approx(1.0 / 7.0)

    def test_roll_rate_to_aileron_gain(self):
        iface = JSBSimActuatorInterface({})
        # 1.5 rad/s -> full aileron deflection
        assert iface.roll_rate_gain == pytest.approx(1.0 / 1.5)

    def test_command_to_jsbsim_properties_clips_to_normalised(self):
        iface = JSBSimActuatorInterface({})
        result = iface.command_to_jsbsim_properties({
            "nz_cmd": 14.0,  # 2x full elevator
            "roll_rate_cmd": 3.0,  # 2x full aileron
            "throttle_cmd": 1.5,
        })
        assert result["fcs/elevator-cmd-norm"] == pytest.approx(-1.0, abs=1e-6)
        assert result["fcs/aileron-cmd-norm"] == pytest.approx(1.0, abs=1e-6)
        assert result["fcs/throttle-cmd-norm"] == pytest.approx(1.0, abs=1e-6)
        assert result["saturation_flag"] is True

    def test_throttle_within_zero_one_for_valid_commands(self):
        iface = JSBSimActuatorInterface({})
        result = iface.command_to_jsbsim_properties({
            "nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.7
        })
        assert 0.0 <= result["fcs/throttle-cmd-norm"] <= 1.0

    def test_roll_rate_radians_not_degrees(self):
        """Roll rate command is in rad/s; 1.5 rad/s ~ 86 deg/s, not 1.5 deg/s."""
        iface = JSBSimActuatorInterface({})
        # A small command in rad/s should map to small aileron
        result_small = iface.command_to_jsbsim_properties({
            "nz_cmd": 1.0, "roll_rate_cmd": 0.1, "throttle_cmd": 0.7
        })
        assert abs(result_small["fcs/aileron-cmd-norm"]) < 0.1

        # If someone mistakenly passed deg/s (e.g. 30), it would saturate
        result_big = iface.command_to_jsbsim_properties({
            "nz_cmd": 1.0, "roll_rate_cmd": 30.0, "throttle_cmd": 0.7
        })
        assert result_big["saturation_flag"] is True


class TestBaselineControllers:
    """Unit tests for the baseline controllers used in diagnosis."""

    def test_hold_controller_returns_constant_command(self):
        ctrl = HoldController(nz_cmd=1.1, roll_rate_cmd=0.1, throttle_cmd=0.8)
        cmd = ctrl.compute({}, {}, {})
        assert cmd["nz_cmd"] == pytest.approx(1.1)
        assert cmd["roll_rate_cmd"] == pytest.approx(0.1)
        assert cmd["throttle_cmd"] == pytest.approx(0.8)

    def test_direct_pn_controller_has_expected_mode(self):
        cfg = {"gains": {"k_roll": 1.0, "k_speed": 0.2}, "params": {}}
        ctrl = DirectPNController(cfg)
        assert ctrl.guidance.mode == "proportional_navigation"

    def test_low_gain_direct_controller_scales_gains(self):
        cfg = {
            "gains": {"k_los": 1.0, "k_pos": 0.5, "k_damp": 0.2, "k_roll": 1.0, "k_speed": 0.2},
            "params": {},
        }
        ctrl = LowGainDirectController(cfg, gain_scale=0.3)
        assert ctrl._override_gains.k_los == pytest.approx(0.3)
        assert ctrl._override_gains.k_pos == pytest.approx(0.15)
        assert ctrl._override_gains.k_roll == pytest.approx(0.3)


class TestCommandSanityChecks:
    """Tests for compute_command_sanity and classify_failure_root_cause."""

    def test_empty_step_df_returns_false_flags(self):
        import pandas as pd
        sanity = compute_command_sanity(pd.DataFrame())
        assert sanity["nz_saturated"] is False
        assert sanity["roll_rate_saturated"] is False
        assert sanity["throttle_saturated"] is False

    def test_sanity_detects_saturated_elevator(self):
        import pandas as pd
        df = pd.DataFrame({
            "nz_cmd": [1.0, 1.0, 7.0],
            "roll_rate_cmd": [0.0, 0.0, 0.0],
            "throttle_cmd": [0.7, 0.7, 0.7],
            "elevator_cmd": [0.0, 0.0, -1.0],
            "aileron_cmd": [0.0, 0.0, 0.0],
            "own_altitude_m": [5000.0, 5000.0, 5000.0],
            "own_speed_mps": [200.0, 200.0, 200.0],
        })
        sanity = compute_command_sanity(df)
        assert bool(sanity["nz_saturated"]) is True
        assert bool(sanity["elevator_saturated"]) is True

    def test_throttle_out_of_range_detected(self):
        import pandas as pd
        df = pd.DataFrame({
            "nz_cmd": [1.0], "roll_rate_cmd": [0.0], "throttle_cmd": [1.2],
            "elevator_cmd": [0.0], "aileron_cmd": [0.0],
            "own_altitude_m": [5000.0], "own_speed_mps": [200.0],
        })
        sanity = compute_command_sanity(df)
        assert bool(sanity["throttle_saturated"]) is True
        assert bool(sanity["throttle_out_of_01"]) is True

    def test_taxonomy_success(self):
        row = {"reason": "success", "method": "hold"}
        cause, note = classify_failure_root_cause(row, {})
        assert cause == "success"

    def test_taxonomy_baseline_oob(self):
        row = {"reason": "out_of_bounds", "method": "direct_pn", "min_range_m": 6000.0}
        sanity = {"nz_saturated": False, "roll_rate_saturated": False,
                  "elevator_saturated": False, "aileron_saturated": False}
        cause, note = classify_failure_root_cause(row, sanity)
        assert cause == "baseline_oob"

    def test_taxonomy_ppo_saturation(self):
        row = {"reason": "out_of_bounds", "method": "no_prediction", "min_range_m": 2000.0}
        sanity = {"nz_saturated": True, "roll_rate_saturated": True,
                  "elevator_saturated": False, "aileron_saturated": False}
        cause, note = classify_failure_root_cause(row, sanity)
        assert cause == "ppo_control_saturation"


class TestMarkdownHelper:
    """Tests for the pandas-to-markdown helper."""

    def test_df_to_markdown_renders_header(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
        md = _df_to_markdown(df)
        assert "| a | b |" in md
        assert "---" in md
        assert "3.5000" in md

    def test_df_to_markdown_empty(self):
        import pandas as pd
        md = _df_to_markdown(pd.DataFrame())
        assert "empty" in md


class TestJSBSimScenarioPositionConversion:
    """Verify that scenario position_m is correctly converted to geodetic for JSBSim."""

    def test_neu2lla_roundtrip(self):
        try:
            import pymap3d  # noqa: F401
        except ImportError:
            pytest.skip("pymap3d not installed")

        lon0, lat0, alt0 = 120.0, 60.0, 0.0
        # Head-on scenario target offset: 2000 m north
        n, e, u = 2000.0, 0.0, 5000.0
        lon, lat, alt = neu2lla(n, e, u, lon0, lat0, alt0)
        n2, e2, u2 = lla2neu(lon, lat, alt, lon0, lat0, alt0)
        assert n2 == pytest.approx(n, abs=1e-3)
        assert e2 == pytest.approx(e, abs=1e-3)
        assert u2 == pytest.approx(u, abs=1e-3)

    def test_scenario_to_jsbsim_init_sets_longitude_latitude(self):
        """Regression test for Stage 10.1 position conversion bug."""
        try:
            import pymap3d  # noqa: F401
        except ImportError:
            pytest.skip("pymap3d not installed")

        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        config = {
            "backend": "jsbsim",
            "env": {
                "sim_freq": 60,
                "decision_freq": 5,
                "max_high_level_steps": 32,
                "aircraft_model": "f16",
                "legacy_project_root": "E:/CloseAirCombat_control",
                "origin": [120.0, 60.0, 0.0],
                "use_jsbsim": True,
            },
            "guidance": {"mode": "los_rate", "gains": {"k_los": 1.0}},
        }
        env = CloseRangeTrackingEnv(config)
        if env._backend != "jsbsim":
            pytest.skip("JSBSim backend not available")

        scenario = {
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
            "target_init": {"position_m": [2000.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 180.0},
        }
        init = env._scenario_to_jsbsim_init(scenario["target_init"])
        assert "ic/long-gc-deg" in init
        assert "ic/lat-geod-deg" in init
        # Target is 2000 m north of origin at latitude 60 deg:
        # ~1 deg latitude = 111320 m, so delta lat ~ 0.01796 deg
        assert init["ic/lat-geod-deg"] == pytest.approx(60.01796, abs=1e-4)
        env.close()


class TestDiagnosisRunnerSmoke:
    """Smoke test for the diagnosis runner using a minimal matrix."""

    def test_runner_produces_required_outputs(self, tmp_path):
        try:
            import pymap3d  # noqa: F401
        except ImportError:
            pytest.skip("pymap3d not installed")

        from uav_vpp_guidance.evaluation.jsbsim_diagnosis import Stage10DiagnosisRunner

        config = {
            "backend": "jsbsim",
            "env": {
                "sim_freq": 60,
                "decision_freq": 5,
                "max_high_level_steps": 32,
                "aircraft_model": "f16",
                "legacy_project_root": "E:/CloseAirCombat_control",
                "origin": [120.0, 60.0, 0.0],
                "use_jsbsim": True,
                "success_range_m": 900.0,
                "success_ata_deg": 25.0,
                "success_hold_time_s": 0.2,
                "hysteresis_range_m": 950.0,
                "hysteresis_ata_deg": 30.0,
                "min_altitude_m": 500.0,
                "max_altitude_m": 15000.0,
                "max_range_m": 8000.0,
            },
            "guidance": {
                "mode": "los_rate",
                "gains": {"k_los": 1.0, "k_pos": 0.5, "k_damp": 0.2, "k_roll": 1.0, "k_speed": 0.2, "alpha_filter": 0.3},
            },
            "limits": {
                "nz_min": -2.0, "nz_max": 7.0,
                "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                "throttle_min": 0.0, "throttle_max": 1.0,
            },
            "virtual_point": {
                "anchor_mode": "current_target",
                "action_dim": 3,
                "d_long_range": [-1500.0, 1500.0],
                "d_lat_range": [-800.0, 800.0],
                "d_vert_range": [-500.0, 500.0],
            },
            "trajectory_prediction": {"enabled": False},
            "reward": {"w_range": 0.0, "w_angle": 0.0, "w_energy": 0.0, "w_safety": 0.0, "w_saturation": 0.0, "w_smooth": 0.0,
                       "terminal_success": 0.0, "terminal_failure": 0.0, "terminal_crash": 0.0},
        }

        runner = Stage10DiagnosisRunner(
            config=config,
            methods=["hold"],
            scenarios=["smoke_head_on"],
            seeds=[0],
            output_dir=str(tmp_path),
        )
        runner.run()
        runner.save()

        assert os.path.exists(os.path.join(tmp_path, "raw_steps.csv"))
        assert os.path.exists(os.path.join(tmp_path, "raw_episodes.csv"))

        with open(os.path.join(tmp_path, "raw_steps.csv"), "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == set(STEP_COLUMNS)
            rows = list(reader)
            assert len(rows) > 0
            # Telemetry should include actual JSBSim state
            assert any(rows[0].get(col) not in ("", "nan") for col in ["own_altitude_m", "own_speed_mps", "own_nz_g"])

        with open(os.path.join(tmp_path, "raw_episodes.csv"), "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == set(EPISODE_COLUMNS)
            eps = list(reader)
            assert len(eps) == 1
            assert eps[0]["method"] == "hold"
