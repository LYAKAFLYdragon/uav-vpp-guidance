"""
True 3D Proportional Navigation (PN) guidance law.

Estimates LOS rate via filtered numerical differentiation and produces
acceleration commands perpendicular to the LOS.

The acceleration is then decomposed into normal overload (nz_cmd) and
roll-rate command (roll_rate_cmd) compatible with the environment pipeline.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

from .los_rate_guidance import _extract_position, _extract_velocity, _normalize_angle

logger = logging.getLogger(__name__)


class ProportionalNavigationGuidance:
    """
    True 3D Proportional Navigation with LOS-rate filtering.

    Configurable parameters (under guidance.params in YAML):
        - navigation_constant: Effective navigation ratio N (default 3.0)
        - los_rate_filter_alpha: First-order filter alpha for LOS rate (default 0.3)
        - max_accel_mps2: Maximum commanded acceleration magnitude (default 100.0)
        - epsilon: Singularity protection (default 1.0e-6)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        gains = config.get("gains", {})
        params = config.get("params", {})

        self.k_los = float(gains.get("k_los", 1.0))  # compatibility
        self.k_pos = float(gains.get("k_pos", 0.5))
        self.k_damp = float(gains.get("k_damp", 0.2))
        self.k_roll = float(gains.get("k_roll", 1.0))
        self.k_speed = float(gains.get("k_speed", 0.2))

        self.navigation_constant = float(params.get("navigation_constant", 3.0))
        self.los_rate_filter_alpha = float(params.get("los_rate_filter_alpha", 0.3))
        self.max_accel_mps2 = float(params.get("max_accel_mps2", 100.0))
        self.epsilon = float(params.get("epsilon", 1.0e-6))
        self.base_throttle = float(params.get("base_throttle", 0.7))
        self.target_speed_mps = float(params.get("target_speed_mps", 250.0))
        self.speed_error_scale_mps = float(params.get("speed_error_scale_mps", 100.0))
        self.dt = float(params.get("dt", 0.2))
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        self.mode = "proportional_navigation"

        # Internal state for LOS rate filtering
        self._prev_los_vec: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._filtered_los_rate: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset LOS rate filter state."""
        self._prev_los_vec = None
        self._prev_time = None
        self._filtered_los_rate = None

    def compute_command(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Compute PN guidance commands.

        Args:
            own_state (dict): Own aircraft state with position and velocity.
            target_state (dict, optional): Reserved for future use.
            virtual_point (dict): Virtual pursuit point position.
            gains (GuidanceGains, optional): Gain override.

        Returns:
            dict: Command with keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
        """
        if target_state is not None:
            # PN does not currently use target_state directly (uses VP)
            pass

        # Resolve gains
        k_roll, k_speed = self._resolve_gains(gains)

        # Extract vectors
        own_pos = _extract_position(own_state)
        own_vel = _extract_velocity(own_state)
        vp_pos = _extract_position(virtual_point)

        # Relative geometry
        rel_pos = vp_pos - own_pos
        # Use target-relative closing velocity when available, else own velocity
        rel_vel = self._compute_relative_velocity(own_vel, target_state)
        distance = float(np.linalg.norm(rel_pos))

        # LOS unit vector
        if distance > self.epsilon:
            los_unit = rel_pos / distance
        else:
            los_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # LOS rate estimation (filtered)
        los_rate_vec = self._estimate_los_rate(los_unit)
        los_rate_mag = float(np.linalg.norm(los_rate_vec))

        # Closing velocity (negative of range rate)
        range_rate = float(np.dot(rel_vel, los_unit))
        closing_velocity = -range_rate
        if not np.isfinite(closing_velocity):
            closing_velocity = 0.0

        # PN acceleration: a = N * Vc * d(lambda)/dt  (perpendicular to LOS)
        if los_rate_mag > self.epsilon and abs(closing_velocity) > self.epsilon:
            accel_mag = self.navigation_constant * abs(closing_velocity) * los_rate_mag
            accel_mag = min(accel_mag, self.max_accel_mps2)
            # Acceleration direction is perpendicular to LOS, in the plane of LOS rate
            accel_dir = los_rate_vec / los_rate_mag
            accel = accel_mag * accel_dir
        else:
            accel = np.zeros(3, dtype=np.float64)

        # Decompose acceleration into horizontal (turn) and vertical (pull-up)
        accel_horizontal = accel.copy()
        accel_horizontal[2] = 0.0
        accel_h_mag = float(np.linalg.norm(accel_horizontal))

        # Heading error -> roll_rate_cmd
        own_speed = float(np.linalg.norm(own_vel))
        if own_speed > self.epsilon:
            own_heading = float(np.arctan2(own_vel[1], own_vel[0]))
        else:
            own_heading = 0.0

        los_horizontal = rel_pos.copy()
        los_horizontal[2] = 0.0
        los_h_dist = float(np.linalg.norm(los_horizontal))
        if los_h_dist > self.epsilon:
            los_heading = float(np.arctan2(los_horizontal[1], los_horizontal[0]))
        else:
            los_heading = own_heading

        heading_error = _normalize_angle(los_heading - own_heading)
        current_roll = float(own_state.get("roll_rad", 0.0))
        roll_rate_cmd = k_roll * heading_error - self.k_damp * current_roll

        # nz_cmd: map vertical acceleration component to g-load
        # a_z = g * (nz - 1)  => nz = 1 + a_z / g
        gravity = 9.80665
        nz_cmd = 1.0 + accel[2] / gravity
        # Add a small heading-turn coupling term (horizontal acceleration also needs lift)
        if own_speed > self.epsilon:
            nz_cmd += accel_h_mag / gravity

        # Throttle
        speed_error = self.target_speed_mps - own_speed
        throttle_cmd = self.base_throttle + k_speed * (
            speed_error / self.speed_error_scale_mps
        )
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

        return {
            "nz_cmd": float(nz_cmd),
            "roll_rate_cmd": float(roll_rate_cmd),
            "throttle_cmd": float(throttle_cmd),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_gains(self, gains: Optional[Any]) -> tuple:
        if gains is not None:
            return (
                float(getattr(gains, "k_roll", self.k_roll)),
                float(getattr(gains, "k_speed", self.k_speed)),
            )
        return (self.k_roll, self.k_speed)

    def _compute_relative_velocity(
        self, own_vel: np.ndarray, target_state: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Compute relative velocity (target - own) for closing velocity."""
        if target_state is None:
            return -own_vel
        try:
            target_vel = _extract_velocity(target_state)
            return target_vel - own_vel
        except (ValueError, TypeError):
            # Fallback if target has no velocity field
            return -own_vel

    def _estimate_los_rate(self, los_unit: np.ndarray) -> np.ndarray:
        """
        Estimate LOS rate via angle-based differentiation with first-order filtering.

        Uses azimuth / elevation angle differences (with wrapping) instead of
        raw vector subtraction to avoid pi-boundary jumps, then maps the angle
        rates back to Cartesian space so callers can continue using cross-product
        geometry.

        Returns LOS rate vector in rad/s (Cartesian).
        """
        if self._prev_los_vec is None:
            self._prev_los_vec = los_unit.copy()
            self._filtered_los_rate = np.zeros(3, dtype=np.float64)
            return self._filtered_los_rate

        dt_safe = max(self.dt, self.epsilon)

        # Previous angles
        prev_az = np.arctan2(self._prev_los_vec[1], self._prev_los_vec[0])
        prev_el = np.arcsin(np.clip(self._prev_los_vec[2], -1.0, 1.0))

        # Current angles
        curr_az = np.arctan2(los_unit[1], los_unit[0])
        curr_el = np.arcsin(np.clip(los_unit[2], -1.0, 1.0))

        # Wrapped angle differences
        d_az = _normalize_angle(curr_az - prev_az)
        d_el = curr_el - prev_el

        # Map angle rates back to Cartesian using spherical tangent vectors
        cos_az = np.cos(curr_az)
        sin_az = np.sin(curr_az)
        cos_el = np.cos(curr_el)
        sin_el = np.sin(curr_el)

        e_az = np.array([-sin_az * cos_el, cos_az * cos_el, 0.0], dtype=np.float64)
        e_el = np.array([-cos_az * sin_el, -sin_az * sin_el, cos_el], dtype=np.float64)

        raw_rate = (d_az * e_az + d_el * e_el) / dt_safe
        self._prev_los_vec = los_unit.copy()

        # First-order filter
        alpha = self.los_rate_filter_alpha
        self._filtered_los_rate = (
            alpha * raw_rate + (1.0 - alpha) * self._filtered_los_rate
        )
        return self._filtered_los_rate
