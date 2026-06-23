"""
True low-level PID controller family.

This module exposes the controller classes required by the flight-control
comparison tasks.  They are thin, named abstractions over the existing
PID implementations so that:

- ``BaselinePIDController`` is a genuine PID (not just a command filter).
- ``EnhancedPIDController`` is the full-featured closed-loop controller.
- ``GainScheduledPIDController`` schedules PID gains with flight condition.
- ``HybridPPOPIDAdapter`` documents the PPO+PID interface where the policy
  outputs an ``aggressiveness`` knob that scales PID gains.

All classes keep the same ``compute_actuator`` interface as the legacy
``LowLevelController`` for drop-in use in ``CloseRangeTrackingEnv``.
"""

from __future__ import annotations

import math

from .enhanced_low_level_controller import (
    EnhancedLowLevelController,
    GainScheduledEnhancedController,
)


class BaselinePIDController(EnhancedLowLevelController):
    """
    Baseline PID low-level controller.

    Uses proportional-integral-derivative feedback for nz and roll-rate
    tracking, but disables the advanced protections / cross-couplings so it
    serves as a clean baseline in ablations.  This is no longer the legacy
    feed-forward ``LowLevelController``; it is a real PID.
    """

    def __init__(self, config=None):
        config = config or {}
        defaults = {
            "use_pid": True,
            "use_rudder": False,
            "use_aoa_protection": False,
            "use_beta_suppression": False,
            "enable_speed_hold": False,
            "enable_stall_protection": False,
            "enable_dynamic_nz_limit": False,
            "enable_energy_boost": False,
            "enable_altitude_hold": False,
        }
        merged = {**defaults, **config}
        super().__init__(merged)


class EnhancedPIDController(EnhancedLowLevelController):
    """
    Enhanced PID low-level controller.

    Full-featured closed-loop controller with rudder coordination, AoA
    protection, stall protection, energy boost, and altitude hold.
    """


class RobustPIDController(EnhancedLowLevelController):
    """
    Robust fixed-gain PID controller for high-fidelity JSBSim evaluation.

    Combines the enhanced PID feedback loops with stronger protections:

    - rudder / beta suppression for lateral-directional coordination
    - angle-of-attack and stall protection
    - speed hold and energy boost
    - altitude hold with an increased gain
    - bank-angle protection to suppress spiral-dive instabilities

    This controller is intended for the flight-control comparison when the
    baseline PID is too stripped-down to fly the F-16 safely, while still
    remaining a fixed (non-learned) controller.
    """

    def __init__(self, config=None):
        config = config or {}
        defaults = {
            # Base behaviour mirrors EnhancedPIDController so small-crossing
            # intercepts keep their authority.
            "use_pid": True,
            "use_rudder": True,
            "use_aoa_protection": True,
            "use_beta_suppression": True,
            "enable_speed_hold": True,
            "enable_stall_protection": True,
            "enable_dynamic_nz_limit": True,
            "enable_energy_boost": True,
            "enable_altitude_hold": True,
            # Added bank-angle protection: allow transient high bank for lead
            # turns, but recover if the bank limit is exceeded for a sustained
            # interval (the signature of a spiral-dive instability).
            "enable_bank_angle_protection": True,
            "max_bank_rad": math.radians(55.0),
            "bank_violation_threshold": 25,
            "bank_protection_nz_increment": 0.2,
        }
        merged = {**defaults, **config}
        super().__init__(merged)


class GainScheduledPIDController(GainScheduledEnhancedController):
    """
    Gain-scheduled PID low-level controller.

    Extends ``EnhancedPIDController`` with dynamic gain scheduling based on
    dynamic pressure, altitude, and angle of attack.
    """


class HybridPPOPIDAdapter(EnhancedPIDController):
    """
    Adapter for PPO+PID hybrid control.

    The high-level PPO policy outputs a 4D action
    ``[dx, dy, dz, aggressiveness]``.  The environment extracts
    ``aggressiveness`` and passes it to this controller, which scales the
    underlying PID gains.  This class is a named marker that makes the
    PPO+PID chain explicit in configs and telemetry.
    """

    def __init__(self, config=None):
        config = config or {}
        # Ensure gain-scaling source is recorded for provenance.
        gs = config.get("gain_schedule", {})
        gs.setdefault("source", "aggressiveness")
        config["gain_schedule"] = gs
        super().__init__(config)
