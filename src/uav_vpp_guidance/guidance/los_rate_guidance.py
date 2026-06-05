"""
LOS-rate-based guidance law with numerical stability protections.

Inputs:
- own aircraft state
- target aircraft state (reserved for future extensions, currently unused)
- virtual pursuit point
- guidance gains and parameters

Outputs:
- normal overload command (nz_cmd)
- roll-rate command (roll_rate_cmd)
- throttle command (throttle_cmd)

Physics:
1. Compute direction from own aircraft to virtual point;
2. heading_error drives roll_rate_cmd (roll changes heading);
3. elevation_error drives nz_cmd (pitch/normal overload changes altitude);
4. throttle_cmd maintains or adjusts approach speed.

Stability notes:
- All divisions use a safe epsilon (EPS = 1e-9).
- Elevation uses np.arctan2 instead of np.arcsin to avoid endpoint singularities.
- Heading error uses np.arctan2(sin, cos) for robust [-pi, pi] wrapping.
- Capture radius switches to level-flight hold when d < threshold.
- Internal clipping and optional filtering protect downstream actuators.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Global safe epsilon for all divisions and norm checks.
EPS = 1.0e-9

# Arcsin safe clamp margin to stay away from ±1.0 singularity.
SIN_CLIP_MARGIN = 1.0e-12


class LOSRateGuidance:
    """
    LOS-rate-based guidance law with numerical stability protections.

    All previously hard-coded magic numbers (distance scale, target speed,
    base throttle, epsilon) are now configurable via the guidance config.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config (dict): Guidance configuration dictionary. Expected keys:
                - gains: dict with k_los, k_pos, k_damp, k_roll, k_speed
                - params: dict with distance_scale_m, target_speed_mps,
                  speed_error_scale_mps, base_throttle, epsilon, base_nz,
                  capture_radius_m, enable_internal_clip, enable_internal_filter
                - limits: dict with nz_min, nz_max, roll_rate_min, roll_rate_max,
                  throttle_min, throttle_max
        """
        config = config or {}
        self._parse_config(config)
        self.prev_command: Optional[Dict[str, float]] = None
        self._warned_target_state_unused = False

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_config(self, config: Dict[str, Any]) -> None:
        """Extract gains and parameters from config with safe defaults."""
        gains = config.get("gains", {})
        self.k_los = float(gains.get("k_los", 1.0))
        self.k_pos = float(gains.get("k_pos", 0.5))
        self.k_damp = float(gains.get("k_damp", 0.2))
        self.k_roll = float(gains.get("k_roll", 1.0))
        self.k_speed = float(gains.get("k_speed", 0.2))

        params = config.get("params", {})
        self.distance_scale_m = float(params.get("distance_scale_m", 2000.0))
        self.target_speed_mps = float(params.get("target_speed_mps", 250.0))
        self.speed_error_scale_mps = float(params.get("speed_error_scale_mps", 100.0))
        self.base_throttle = float(params.get("base_throttle", 0.7))
        self.epsilon = float(params.get("epsilon", EPS))
        self.base_nz = float(params.get("base_nz", 1.0))
        self.capture_radius_m = float(params.get("capture_radius_m", 50.0))
        self.enable_internal_clip = bool(params.get("enable_internal_clip", True))
        self.enable_internal_filter = bool(params.get("enable_internal_filter", False))
        self.alpha_filter = float(gains.get("alpha_filter", 0.3))

        # Limits (prefer top-level limits block, fall back to defaults)
        limits = config.get("limits", {})
        self.nz_min = float(limits.get("nz_min", -2.0))
        self.nz_max = float(limits.get("nz_max", 7.0))
        self.roll_rate_min = float(limits.get("roll_rate_min", -1.5))
        self.roll_rate_max = float(limits.get("roll_rate_max", 1.5))
        self.throttle_min = float(limits.get("throttle_min", 0.0))
        self.throttle_max = float(limits.get("throttle_max", 1.0))

        # Sanity checks
        if self.distance_scale_m <= 0.0:
            raise ValueError(
                f"distance_scale_m must be positive, got {self.distance_scale_m}"
            )
        if self.speed_error_scale_mps <= 0.0:
            raise ValueError(
                f"speed_error_scale_mps must be positive, got {self.speed_error_scale_mps}"
            )
        self.mode = "los_rate"
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.capture_radius_m < 0.0:
            raise ValueError(
                f"capture_radius_m must be non-negative, got {self.capture_radius_m}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal guidance state."""
        self.prev_command = None
        self._warned_target_state_unused = False

    def compute_command(
        self,
        own_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]],
        virtual_point: Dict[str, Any],
        gains: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Compute guidance commands with numerical stability protections.

        Args:
            own_state (dict): Own aircraft state. Must contain a position field
                (position_neu / position_m / position) and a velocity field
                (velocity_ned / velocity_vector_mps / velocity).
            target_state (dict, optional): Target aircraft state. Reserved for
                future extensions; currently unused but logged once.
            virtual_point (dict): Virtual pursuit point state. Must contain a
                position field (position_neu / position_m / position).
            gains (GuidanceGains, optional): External gain override. If None,
                uses the gains from config.

        Returns:
            dict: Command dictionary with keys 'nz_cmd', 'roll_rate_cmd',
                'throttle_cmd'. All values are finite and clipped to limits.
        """
        # Log once that target_state is unused (helps during debugging)
        if target_state is not None and not self._warned_target_state_unused:
            logger.debug("target_state is provided but unused by LOSRateGuidance")
            self._warned_target_state_unused = True

        # Resolve gains
        k_los, k_pos, k_damp, k_roll, k_speed = self._resolve_gains(gains)

        # Validate and extract state vectors
        self._validate_state(own_state, "own_state")
        self._validate_state(virtual_point, "virtual_point")

        own_pos = _extract_position(own_state)
        vp_pos = _extract_position(virtual_point)
        own_vel = _extract_velocity(own_state)

        # ------------------------------------------------------------------
        # Relative geometry with safe epsilon
        # ------------------------------------------------------------------
        rel_pos = vp_pos - own_pos
        distance = float(np.linalg.norm(rel_pos))
        d_safe = max(distance, self.epsilon)

        # Heading of own aircraft
        own_speed = float(np.linalg.norm(own_vel))
        own_heading = self._compute_heading_from_velocity(own_vel, own_speed)

        # LOS heading in horizontal plane
        los_heading = self._compute_los_heading(rel_pos, own_heading)

        # 1. Heading error -> roll_rate_cmd
        heading_error = _stable_angle_diff(los_heading, own_heading)
        current_roll = float(own_state.get("roll_rad", 0.0))
        roll_rate_cmd = k_roll * heading_error - k_damp * current_roll

        # 2. Elevation -> nz_cmd (using arctan2 for stability)
        los_elevation = self._compute_stable_elevation(rel_pos, d_safe)
        nz_cmd = self._compute_nz_cmd(los_elevation, distance, k_los, k_pos)

        # 3. Speed / throttle
        throttle_cmd = self._compute_throttle_cmd(own_speed, k_speed)

        command = {
            "nz_cmd": float(nz_cmd),
            "roll_rate_cmd": float(roll_rate_cmd),
            "throttle_cmd": float(throttle_cmd),
        }

        # ------------------------------------------------------------------
        # Capture-radius protection: when very close to VPP, avoid singularity
        # by switching to level-flight hold (reduced roll, base nz, maintain speed).
        # ------------------------------------------------------------------
        if distance < self.capture_radius_m:
            capture_ratio = distance / self.capture_radius_m  # [0, 1)
            # Blend from capture hold to normal command as we exit the radius
            command["roll_rate_cmd"] = capture_ratio * command["roll_rate_cmd"]
            command["nz_cmd"] = (
                1.0 - capture_ratio
            ) * self.base_nz + capture_ratio * command["nz_cmd"]
            # Throttle remains as computed (speed hold)

        # ------------------------------------------------------------------
        # Internal clipping (defense in depth even if caller also clips)
        # ------------------------------------------------------------------
        if self.enable_internal_clip:
            command = self._apply_internal_limits(command)

        # ------------------------------------------------------------------
        # Internal first-order filtering (optional, for direct-call safety)
        # ------------------------------------------------------------------
        if self.enable_internal_filter:
            command = self._apply_internal_filter(command)

        # ------------------------------------------------------------------
        # NaN / Inf defense: if anything went wrong, fall back to safe defaults
        # ------------------------------------------------------------------
        if not _all_finite(command):
            logger.warning(
                "LOSRateGuidance produced non-finite command (distance=%.3e, "
                "own_speed=%.3e). Falling back to safe hold commands.",
                distance,
                own_speed,
            )
            command = {
                "nz_cmd": float(np.clip(self.base_nz, self.nz_min, self.nz_max)),
                "roll_rate_cmd": 0.0,
                "throttle_cmd": float(
                    np.clip(self.base_throttle, self.throttle_min, self.throttle_max)
                ),
            }

        self.prev_command = command
        return command

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_gains(self, gains: Optional[Any]) -> tuple:
        """Return resolved gain values (k_los, k_pos, k_damp, k_roll, k_speed)."""
        if gains is not None:
            return (
                float(getattr(gains, "k_los", self.k_los)),
                float(getattr(gains, "k_pos", self.k_pos)),
                float(getattr(gains, "k_damp", self.k_damp)),
                float(getattr(gains, "k_roll", self.k_roll)),
                float(getattr(gains, "k_speed", self.k_speed)),
            )
        return (self.k_los, self.k_pos, self.k_damp, self.k_roll, self.k_speed)

    def _validate_state(self, state: Dict[str, Any], name: str) -> None:
        """Validate that state dict contains at least a position field."""
        if not isinstance(state, dict):
            raise TypeError(f"{name} must be a dict, got {type(state).__name__}")
        for key in ("position_ned", "position_neu", "position_m", "position"):
            if state.get(key) is not None:
                return
        raise ValueError(
            f"{name} missing position field (expected one of: "
            "position_ned, position_neu, position_m, position)"
        )

    def _compute_heading_from_velocity(self, vel: np.ndarray, speed: float) -> float:
        """
        Compute heading from velocity vector.

        Returns 0.0 when speed is below epsilon (avoids atan2(0,0) ambiguity).
        """
        if speed > self.epsilon:
            return float(np.arctan2(vel[1], vel[0]))
        return 0.0

    def _compute_los_heading(
        self, rel_pos: np.ndarray, fallback_heading: float
    ) -> float:
        """
        Compute LOS heading in the horizontal plane.

        If the horizontal distance is below epsilon, returns fallback_heading
        to avoid singularity.
        """
        los_horizontal = rel_pos.copy()
        los_horizontal[2] = 0.0
        los_h_dist = float(np.linalg.norm(los_horizontal))
        if los_h_dist > self.epsilon:
            return float(np.arctan2(los_horizontal[1], los_horizontal[0]))
        return fallback_heading

    def _compute_stable_elevation(self, rel_pos: np.ndarray, distance: float) -> float:
        """
        Compute LOS elevation angle using arctan2 for numerical stability.

        Uses the horizontal distance as the real axis and vertical component
        as the imaginary axis. This avoids the arcsin endpoint singularity at
        ±pi/2 and naturally handles the full [-pi, pi] range.

        When distance is below epsilon, returns 0.0.
        """
        if distance <= self.epsilon:
            return 0.0
        d_horiz = float(np.linalg.norm(rel_pos[:2]))
        # arctan2(y, x) where y = vertical, x = horizontal
        return float(np.arctan2(rel_pos[2], max(d_horiz, self.epsilon)))

    def _compute_nz_cmd(
        self,
        los_elevation: float,
        distance: float,
        k_los: float,
        k_pos: float,
    ) -> float:
        """
        Compute normal overload command.

        nz = base_nz + k_los * elevation + k_pos * (distance / distance_scale)
        """
        proportional_term = k_pos * (distance / self.distance_scale_m)
        return self.base_nz + k_los * los_elevation + proportional_term

    def _compute_throttle_cmd(self, own_speed: float, k_speed: float) -> float:
        """
        Compute throttle command.

        throttle = base_throttle + k_speed * (speed_error / speed_error_scale)
        Clamped to [0, 1]. If own_speed is non-finite, returns base_throttle.
        """
        if not np.isfinite(own_speed):
            return float(
                np.clip(self.base_throttle, self.throttle_min, self.throttle_max)
            )
        speed_error = self.target_speed_mps - own_speed
        throttle = self.base_throttle + k_speed * (
            speed_error / self.speed_error_scale_mps
        )
        return float(np.clip(throttle, self.throttle_min, self.throttle_max))

    def _apply_internal_limits(self, command: Dict[str, float]) -> Dict[str, float]:
        """Clip commands to configured physical limits."""
        return {
            "nz_cmd": float(np.clip(command["nz_cmd"], self.nz_min, self.nz_max)),
            "roll_rate_cmd": float(
                np.clip(
                    command["roll_rate_cmd"], self.roll_rate_min, self.roll_rate_max
                )
            ),
            "throttle_cmd": float(
                np.clip(command["throttle_cmd"], self.throttle_min, self.throttle_max)
            ),
        }

    def _apply_internal_filter(self, command: Dict[str, float]) -> Dict[str, float]:
        """Apply first-order low-pass filter using alpha_filter."""
        if self.prev_command is None:
            return dict(command)
        alpha = self.alpha_filter
        return {
            "nz_cmd": alpha * command["nz_cmd"]
            + (1.0 - alpha) * self.prev_command["nz_cmd"],
            "roll_rate_cmd": alpha * command["roll_rate_cmd"]
            + (1.0 - alpha) * self.prev_command["roll_rate_cmd"],
            "throttle_cmd": alpha * command["throttle_cmd"]
            + (1.0 - alpha) * self.prev_command["throttle_cmd"],
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_position(state: Dict[str, Any]) -> np.ndarray:
    """Extract 3-D position vector from state dict.

    Tries keys in order: position_ned, position_neu, position_m, position.
    Raises ValueError if missing or wrong shape.
    """
    for key in ("position_ned", "position_neu", "position_m", "position"):
        pos = state.get(key)
        if pos is not None:
            arr = np.asarray(pos, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(
                    f"Position must be a 3-element vector, got shape {arr.shape}"
                )
            return arr
    raise ValueError(
        "State missing position field (expected one of: "
        "position_ned, position_neu, position_m, position)"
    )


def _extract_velocity(state: Dict[str, Any]) -> np.ndarray:
    """Extract 3-D velocity vector from state dict.

    Tries keys in order: velocity_ned, velocity_vector_mps, velocity.
    Raises ValueError if missing or wrong shape.
    """
    for key in ("velocity_ned", "velocity_vector_mps", "velocity"):
        vel = state.get(key)
        if vel is not None:
            arr = np.asarray(vel, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(
                    f"Velocity must be a 3-element vector, got shape {arr.shape}"
                )
            return arr
    raise ValueError(
        "State missing velocity field (expected one of: "
        "velocity_ned, velocity_vector_mps, velocity)"
    )


def _stable_angle_diff(a: float, b: float) -> float:
    """
    Compute the smallest signed difference between two angles (radians).

    Uses np.arctan2(sin(delta), cos(delta)) to robustly map the result to
    [-pi, pi] without the modulo/branching issues of _normalize_angle.

    Handles NaN by returning NaN.
    """
    delta = a - b
    if not np.isfinite(delta):
        return float(delta)
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]. Handles NaN by returning NaN."""
    if not np.isfinite(angle):
        return float(angle)
    angle = angle % (2.0 * np.pi)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return float(angle)


def _all_finite(command: Dict[str, float]) -> bool:
    """Check that all values in a command dict are finite."""
    return all(np.isfinite(v) for v in command.values())
