"""
Tests for scripts/eval_jsbsim_guidance_comparison.py exit-code behaviour.
"""

import subprocess
import sys
import os


# Path to the script under test
SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "eval_jsbsim_guidance_comparison.py",
)


def _write_temp_config(tmpdir, backend="simple", strict_backend=False):
    """Write a minimal config that forces the chosen backend."""
    cfg_path = os.path.join(tmpdir, "test_config.yaml")
    content = f"""
backend: {backend}
env:
  use_jsbsim: {str(backend == 'jsbsim').lower()}
  strict_backend: {str(strict_backend).lower()}
  aircraft_model: f16
  legacy_project_root: ""
  origin: [120.0, 60.0, 0.0]
  sim_freq: 60
  decision_freq: 5
  high_level_dt: 0.2
  low_level_dt: 0.0166667
  action_repeat: 12
  max_high_level_steps: 10
  target_mode: constant_velocity
  success_range_m: 900.0
  success_ata_deg: 25.0
  success_hold_time_s: 0.2
  hysteresis_range_m: 950.0
  hysteresis_ata_deg: 30.0
  min_altitude_m: 500.0
  max_altitude_m: 15000.0
  max_range_m: 8000.0
guidance:
  mode: los_rate
  use_gain_adapter: false
  gains:
    k_los: 1.0
    k_pos: 0.5
    k_damp: 0.2
    k_roll: 1.0
    k_speed: 0.2
    alpha_filter: 0.3
limits:
  nz_min: -2.0
  nz_max: 7.0
  roll_rate_min: -1.5
  roll_rate_max: 1.5
  throttle_min: 0.0
  throttle_max: 1.0
reward:
  w_range: 0.5
  w_angle: 0.8
  w_energy: 0.2
  w_safety: 2.0
  w_saturation: 1.0
  w_smooth: 0.1
  terminal_success: 200.0
  terminal_failure: -200.0
  terminal_crash: -300.0
  min_altitude_m: 500.0
training:
  total_steps: 10000
  rollout_steps: 512
  eval_interval: 2000
  save_interval: 5000
evaluation:
  episodes: 2
  scenario_types:
    - favorable
scenarios:
  favorable:
    name: favorable
    own_init:
      position_m: [0.0, 0.0, 5000.0]
      velocity_mps: 220.0
      heading_deg: 0.0
      pitch_deg: 0.0
      roll_deg: 0.0
    target_init:
      position_m: [2000.0, 0.0, 5000.0]
      velocity_mps: 180.0
      heading_deg: 0.0
      pitch_deg: 0.0
      roll_deg: 0.0
"""
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)
    return cfg_path


class TestComparisonExitCode:
    def test_backend_mismatch_returns_nonzero(self, tmpdir):
        """--require-backend jsbsim 但 config 使用 simple backend 时应非零退出。"""
        cfg = _write_temp_config(str(tmpdir), backend="simple", strict_backend=False)
        result = subprocess.run(
            [
                sys.executable,
                SCRIPT_PATH,
                "--config",
                cfg,
                "--seeds",
                "0",
                "--episodes",
                "1",
                "--require-backend",
                "jsbsim",
                "--output-dir",
                str(os.path.join(tmpdir, "out")),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit for backend mismatch.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
