"""
Canonical guidance-law interface.

This module documents the guidance law that should be referenced by the
paper and all paper-safe experiments. It is **not** a new implementation;
it re-exports ``LOSRateGuidance`` as the canonical command-chain and adds
a machine-readable specification of the inputs, outputs, and tunable gains.

Theoretical reports should describe the guidance law in terms of the
quantities defined here, not the idealized translational-acceleration model
used for analytical intuition.

Canonical configuration: ``config/canonical/guidance.yaml``
Canonical gain space:     ``config/canonical/gain_space.yaml``
"""

from .los_rate_guidance import LOSRateGuidance


#: Canonical guidance-law class. Any paper-safe experiment should use this
#: class (or a wrapper whose semantics are documented here).
CanonicalGuidance = LOSRateGuidance


#: Names of the guidance commands produced by the canonical law.
CANONICAL_COMMANDS = ("nz_cmd", "roll_rate_cmd", "throttle_cmd")


#: Names of the CEM-optimized gains. The order matches the canonical gain
#: space in ``config/canonical/gain_space.yaml``.
CANONICAL_OPTIMIZED_GAINS = (
    "k_los",
    "k_pos",
    "k_damp",
    "k_roll",
    "k_speed",
)


#: Names of the fixed guidance parameters (not optimized by CEM).
CANONICAL_FIXED_PARAMS = (
    "alpha_filter",
    "k_energy",
    "distance_scale_m",
    "target_speed_mps",
    "speed_error_scale_mps",
    "base_throttle",
    "base_nz",
    "capture_radius_m",
)


#: Human-readable specification of the canonical command computation.
#: These equations are implemented in ``LOSRateGuidance.compute_command``.
CANONICAL_SPEC = """
Canonical Guidance Law (LOSRateGuidance)
==========================================

Inputs:
  - own_state: own aircraft state (position, velocity, attitude)
  - target_state: reserved for future extensions; currently unused
  - virtual_point: VPP position (3-D spatial offset from target)
  - gains: CEM-optimized gain vector g = (k_los, k_pos, k_damp, k_roll, k_speed)

Outputs:
  - roll_rate_cmd = k_roll * heading_error - k_damp * current_roll
  - nz_cmd        = base_nz
                    + k_los  * los_elevation
                    + k_pos  * (distance / distance_scale_m)
  - throttle_cmd  = base_throttle
                    + k_speed * (speed_error / speed_error_scale_mps)

Fixed parameters (from config):
  - base_nz, base_throttle
  - distance_scale_m, speed_error_scale_mps
  - alpha_filter (command smoothing, if internal filter enabled)
  - k_energy (reserved, currently unused by LOSRateGuidance)

CEM search space:
  See config/canonical/gain_space.yaml.
"""


__all__ = [
    "CanonicalGuidance",
    "CANONICAL_COMMANDS",
    "CANONICAL_OPTIMIZED_GAINS",
    "CANONICAL_FIXED_PARAMS",
    "CANONICAL_SPEC",
]
