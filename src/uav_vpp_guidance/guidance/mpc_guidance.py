"""Model-predictive guidance law for the simple backend.

This is a deliberately simple, single-shooting MPC that optimizes a *constant*
command over a short horizon.  It predicts the aircraft's motion using the same
point-mass kinematics as SimplePointMassEnv and minimizes a weighted combination
of final miss distance, terminal aspect error, control effort, and speed error.

It is intended as a strong classical baseline for comparison against learned
policies, not as a production flight controller.
"""

from typing import Dict, Any, Optional
import numpy as np

from uav_vpp_guidance.flight_control.command_limiter import effective_throttle_limits
from scipy.optimize import minimize


class MPCGuidance:
    """Constant-command receding-horizon MPC guidance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        params = config.get("params", {})
        limits = config.get("limits", {})

        self.horizon_steps = int(params.get("horizon_steps", 10))
        self.dt = float(params.get("dt", 0.2))

        self.nz_min = float(limits.get("nz_min", 1.0))
        self.nz_max = float(limits.get("nz_max", 5.0))
        self.roll_rate_max = float(limits.get("roll_rate_max", 1.5))
        self.throttle_min, self.throttle_max = effective_throttle_limits(limits)

        self.w_range = float(params.get("w_range", 1.0))
        self.w_ata = float(params.get("w_ata", 0.01))
        self.w_effort = float(params.get("w_effort", 1e-4))
        self.w_speed = float(params.get("w_speed", 0.1))
        self.speed_target = float(params.get("speed_target_mps", 200.0))

        # Kinematic gains matching the simple backend defaults.
        self._heading_rate_per_roll = 1.5
        self._pitch_rate_per_nz = 0.3
        self._accel_per_throttle = 20.0
        self._max_roll = np.deg2rad(params.get("max_roll_deg", 60.0))
        self._max_pitch = np.deg2rad(params.get("max_pitch_deg", 45.0))

        self._prev_command = np.array([1.0, 0.0, 0.5], dtype=np.float64)

    def reset(self):
        """Reset internal state."""
        self._prev_command = np.array([1.0, 0.0, 0.5], dtype=np.float64)

    def compute_command(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Compute [nz_cmd, roll_rate_cmd, throttle_cmd] via MPC."""
        # Reference point: virtual pursuit point (target when VPP is zero-offset).
        vp_pos = np.asarray(virtual_point.get("position_neu") or virtual_point.get("position_m"), dtype=np.float64)

        # Target state for prediction.
        tgt_pos = np.asarray(target_state.get("position_m", target_state.get("position_neu", vp_pos)), dtype=np.float64)
        tgt_vel = np.asarray(target_state.get("velocity_vector_mps", np.zeros(3)), dtype=np.float64)

        # Initial own state.
        own_pos = np.asarray(own_state.get("position_m", own_state.get("position_neu", np.zeros(3))), dtype=np.float64)
        own_vel = np.asarray(own_state.get("velocity_vector_mps", np.zeros(3)), dtype=np.float64)
        speed0 = float(np.linalg.norm(own_vel))
        if speed0 < 1.0:
            speed0 = 200.0
        heading0 = float(np.arctan2(own_vel[1], own_vel[0]))
        pitch0 = float(np.arcsin(np.clip(own_vel[2] / max(speed0, 1e-6), -1.0, 1.0)))
        roll0 = float(own_state.get("roll_rad", 0.0))

        bounds = [
            (self.nz_min, self.nz_max),
            (-self.roll_rate_max, self.roll_rate_max),
            (self.throttle_min, self.throttle_max),
        ]

        def _simulate(u):
            nz, roll_rate, throttle = u
            roll = roll0
            pitch = pitch0
            yaw = heading0
            speed = speed0
            pos = own_pos.copy()
            for _ in range(self.horizon_steps):
                roll += roll_rate * self.dt
                roll = np.clip(roll, -self._max_roll, self._max_roll)
                pitch += self._pitch_rate_per_nz * nz * self.dt
                pitch = np.clip(pitch, -self._max_pitch, self._max_pitch)
                yaw += self._heading_rate_per_roll * roll * self.dt
                speed += self._accel_per_throttle * (throttle - 0.5) * self.dt
                speed = np.clip(speed, 150.0, 400.0)
                vx = speed * np.cos(pitch) * np.cos(yaw)
                vy = speed * np.cos(pitch) * np.sin(yaw)
                vz = speed * np.sin(pitch)
                pos += np.array([vx, vy, vz]) * self.dt
            return pos, yaw, speed

        def _cost(u):
            pred_pos, pred_yaw, pred_speed = _simulate(u)
            tgt_pred = tgt_pos + tgt_vel * (self.horizon_steps * self.dt)
            rel = tgt_pred - pred_pos
            range_m = float(np.linalg.norm(rel))
            desired_yaw = float(np.arctan2(rel[1], rel[0]))
            yaw_err = float(np.arctan2(np.sin(desired_yaw - pred_yaw), np.cos(desired_yaw - pred_yaw)))
            ata = abs(yaw_err)
            speed_err = pred_speed - self.speed_target
            effort = (u[0] - 1.0) ** 2 + u[1] ** 2 + (u[2] - 0.5) ** 2
            return (
                self.w_range * range_m ** 2
                + self.w_ata * ata ** 2
                + self.w_effort * effort
                + self.w_speed * speed_err ** 2
            )

        try:
            res = minimize(
                _cost,
                self._prev_command,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 50, "ftol": 1e-3, "disp": False},
            )
            if res.success:
                u_opt = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            else:
                u_opt = self._fallback(own_pos, vp_pos, speed0, heading0)
        except Exception:
            u_opt = self._fallback(own_pos, vp_pos, speed0, heading0)

        self._prev_command = np.asarray(u_opt, dtype=np.float64)
        return {
            "nz_cmd": float(u_opt[0]),
            "roll_rate_cmd": float(u_opt[1]),
            "throttle_cmd": float(u_opt[2]),
        }

    @staticmethod
    def _fallback(own_pos, target_pos, speed, heading):
        """Simple geometric fallback: turn toward target, level nz, cruise throttle."""
        rel = target_pos - own_pos
        desired_heading = float(np.arctan2(rel[1], rel[0]))
        heading_err = float(np.arctan2(np.sin(desired_heading - heading), np.cos(desired_heading - heading)))
        roll_rate = np.clip(heading_err * 2.0, -1.5, 1.5)
        return np.array([1.0, roll_rate, 0.5], dtype=np.float64)
