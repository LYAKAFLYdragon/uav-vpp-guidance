"""
Enhanced low-level controller with PID feedback and coordinated turn rudder.

Extends the baseline LowLevelController with:
- PID feedback loops for nz and roll_rate tracking
- Coordinated turn rudder control (beta suppression + turn coordination)
- Angle-of-attack protection
- Configurable command tracking gains

This controller is drop-in compatible with LowLevelController (same interface).
"""

import numpy as np
import math
from .actuator_interface import JSBSimActuatorInterface
from .command_limiter import effective_throttle_limits


class EnhancedLowLevelController:
    """
    Enhanced low-level controller with PID feedback loops.

    Translates guidance commands (nz, roll_rate, throttle) into actuator
    inputs with closed-loop tracking and flight-quality protections.

    Interface matches LowLevelController exactly for drop-in replacement.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Flight control configuration.
                Keys (all optional, with sensible defaults):
                - alpha_filter (float): first-order smoothing on command input
                - nz_gain (float): elevator scaling, default 1/9.0
                - roll_rate_gain (float): aileron scaling, default 1/3.0
                - rudder_gain (float): rudder scaling, default 1.0
                - rudder_from_roll (float): rudder-to-roll coupling gain, default 0.3
                - Kp_nz (float): nz proportional gain, default 0.15
                - Ki_nz (float): nz integral gain, default 0.02
                - Kd_nz (float): nz derivative gain, default 0.05
                - Kp_roll (float): roll rate proportional gain, default 0.3
                - Ki_roll (float): roll rate integral gain, default 0.05
                - Kd_roll (float): roll rate derivative gain, default 0.1
                - alpha_limit (float): angle-of-attack protection threshold (rad), default 0.5
                - beta_Kp (float): sideslip suppression gain, default 0.5
                - rudder_max (float): rudder command limit, default 0.5
                - elevator_max (float): elevator command limit, default 1.0
                - aileron_max (float): aileron command limit, default 1.0
                - integral_windup_limit (float): anti-windup limit, default 5.0
                - use_pid (bool): enable PID feedback, default True
                - use_rudder (bool): enable rudder control, default True
                - use_aoa_protection (bool): enable AoA protection, default True
                - actuator (dict): passed to JSBSimActuatorInterface
        """
        self.config = config or {}

        # Command smoothing (first-order filter)
        self.alpha = self.config.get("alpha_filter", 0.3)

        # Gain mappings (relaxed from baseline 1/7.0 and 1/1.5)
        self.nz_gain = self.config.get("nz_to_elevator_gain", 1.0 / 9.0)
        self.roll_rate_gain = self.config.get("roll_rate_to_aileron_gain", 1.0 / 3.0)
        self.rudder_gain = self.config.get("rudder_gain", 1.0)
        self.rudder_from_roll = self.config.get("rudder_from_roll_gain", 0.3)

        # PID gains (calibrated for gain-scheduled context)
        self.use_pid = self.config.get("use_pid", True)
        self.Kp_nz = self.config.get("Kp_nz", 0.22)
        self.Ki_nz = self.config.get("Ki_nz", 0.025)
        self.Kd_nz = self.config.get("Kd_nz", 0.06)

        self.Kp_roll = self.config.get("Kp_roll", 0.35)
        self.Ki_roll = self.config.get("Ki_roll", 0.06)
        self.Kd_roll = self.config.get("Kd_roll", 0.12)

        # Limits
        self.nz_min = self.config.get("nz_min", -2.0)
        self.nz_max = self.config.get("nz_max", 7.0)
        self.roll_rate_min = self.config.get("roll_rate_min", -1.5)
        self.roll_rate_max = self.config.get("roll_rate_max", 1.5)
        self.elevator_max = self.config.get("elevator_max", 1.0)
        self.aileron_max = self.config.get("aileron_max", 1.0)
        self.rudder_max = self.config.get("rudder_max", 0.5)
        self.throttle_min, self.throttle_max = effective_throttle_limits(self.config)
        self.alpha_limit = self.config.get("alpha_limit", 0.5)
        self.integral_windup_limit = self.config.get("integral_windup_limit", 5.0)
        self.use_conditional_integration = self.config.get(
            "use_conditional_integration", True
        )
        self.conditional_integration_threshold = self.config.get(
            "conditional_integration_threshold", 0.95
        )
        self.anti_windup_backcalc_gain = self.config.get(
            "anti_windup_backcalc_gain", 0.25
        )

        # Feature flags
        self.use_rudder = self.config.get("use_rudder", True)
        self.use_aoa_protection = self.config.get("use_aoa_protection", True)
        self.use_beta_suppression = self.config.get("use_beta_suppression", True)

        # Energy / speed / stall protection settings
        self.enable_speed_hold = self.config.get("enable_speed_hold", True)
        self.enable_stall_protection = self.config.get("enable_stall_protection", True)
        self.enable_dynamic_nz_limit = self.config.get("enable_dynamic_nz_limit", True)
        self.enable_energy_boost = self.config.get("enable_energy_boost", True)
        self.energy_boost_gain = self.config.get("energy_boost_gain", 0.03)

        self.target_speed_mps = self.config.get("target_speed_mps", 250.0)
        self.stall_speed_mps = self.config.get("stall_speed_mps", 150.0)
        self.recovery_speed_mps = self.config.get("recovery_speed_mps", 180.0)
        self.speed_kp = self.config.get("speed_kp", 0.6)
        self.speed_ki = self.config.get("speed_ki", 0.05)
        self.throttle_integral_limit = self.config.get("throttle_integral_limit", 0.3)
        self.control_dt = self.config.get("control_dt", 0.2)

        self.enable_altitude_hold = self.config.get("enable_altitude_hold", True)
        self.altitude_reference_m = self.config.get("altitude_reference_m", 5000.0)
        self.altitude_hold_gain = self.config.get("altitude_hold_gain", 0.002)

        # Optional bank-angle protection to prevent spiral-dive crashes when
        # guidance commands sustained high roll-rate (e.g. LOS-rate tail-chase).
        self.enable_bank_angle_protection = self.config.get(
            "enable_bank_angle_protection", False
        )
        self.max_bank_rad = self.config.get("max_bank_rad", math.radians(55.0))
        self.bank_protection_nz_increment = self.config.get(
            "bank_protection_nz_increment", 0.5
        )
        self.bank_protection_nz_increment_max = self.config.get(
            "bank_protection_nz_increment_max", 1.0
        )
        self.bank_taper_start_rad = self.config.get(
            "bank_taper_start_rad", math.radians(5.0)
        )
        # Sustained-bank detection: allow transient high bank for lead turns,
        # but recover if the limit is exceeded for many consecutive steps
        # (the signature of a spiral-dive instability).
        self.bank_violation_threshold = self.config.get(
            "bank_violation_threshold", 25
        )
        self._bank_violation_steps = 0

        # Actuator interface (we use our own mapping, but keep interface compatible)
        actuator_cfg = self.config.get("actuator", {})
        # Override with our gains
        actuator_cfg = {
            **actuator_cfg,
            "nz_to_elevator_gain": self.nz_gain,
            "roll_rate_to_aileron_gain": self.roll_rate_gain,
            "rudder_from_roll_gain": self.rudder_from_roll,
            "elevator_max": self.elevator_max,
            "aileron_max": self.aileron_max,
            "rudder_max": self.rudder_max,
            "throttle_min": self.throttle_min,
            "throttle_max": self.throttle_max,
        }
        self.actuator = JSBSimActuatorInterface(actuator_cfg)

        # Filter state (command smoothing)
        self._prev_nz = 1.0
        self._prev_roll_rate = 0.0
        self._prev_throttle = 0.7

        # PID state
        self._nz_integral = 0.0
        self._nz_prev_error = 0.0
        self._roll_integral = 0.0
        self._roll_prev_error = 0.0
        self._throttle_integral = 0.0

        # History for diagnostics
        self.command_history = []

    # APIC gain-delta mapping
    _APIC_GAIN_NAMES = ["Kp_nz", "Ki_nz", "Kd_nz", "Kp_roll", "Ki_roll", "Kd_roll"]

    def _effective_pid_gains(self, pid_gain_deltas=None):
        """
        Return the six effective PID gains.

        If ``pid_gain_deltas`` is provided, apply per-gain multiplicative
        adjustments around the current base gains:

            K_eff = K_base * (1 + range_k * delta_k),  delta_k in [-1, 1]

        ``pid_gain_deltas`` may be a dict keyed by gain name or a sequence in
        the canonical order [Kp_nz, Ki_nz, Kd_nz, Kp_roll, Ki_roll, Kd_roll].
        """
        base_gains = [
            self.Kp_nz,
            self.Ki_nz,
            self.Kd_nz,
            self.Kp_roll,
            self.Ki_roll,
            self.Kd_roll,
        ]
        if pid_gain_deltas is None:
            return base_gains

        apic_cfg = self.config.get("apic", {})
        ranges = apic_cfg.get("gain_ranges", {})
        default_range = float(apic_cfg.get("default_range", 0.3))
        effective = []
        for i, name in enumerate(self._APIC_GAIN_NAMES):
            if isinstance(pid_gain_deltas, dict):
                delta = float(pid_gain_deltas.get(name, 0.0))
            else:
                delta = float(pid_gain_deltas[i]) if i < len(pid_gain_deltas) else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))
            rng = float(ranges.get(name, default_range))
            effective.append(base_gains[i] * (1.0 + delta * rng))
        return effective

    def _stall_margin(self, speed: float) -> float:
        """
        Return a smooth margin in [0, 1].

        - 0.0 when speed <= stall_speed_mps (full protection)
        - 1.0 when speed >= recovery_speed_mps (no protection)
        - linear in between
        """
        if not np.isfinite(speed):
            return 0.0
        if speed >= self.recovery_speed_mps:
            return 1.0
        if speed <= self.stall_speed_mps:
            return 0.0
        return float(
            (speed - self.stall_speed_mps)
            / (self.recovery_speed_mps - self.stall_speed_mps + 1e-9)
        )

    def _integrate_with_anti_windup(
        self,
        *,
        error: float,
        integral: float,
        feedforward: float,
        non_integral_term: float,
        integral_gain: float,
        control_sign: float,
        control_min: float,
        control_max: float,
    ) -> float:
        """Update an integral term with conditional integration and back-calculation."""
        if not self.use_conditional_integration:
            return float(
                np.clip(
                    integral + error,
                    -self.integral_windup_limit,
                    self.integral_windup_limit,
                )
            )

        if abs(integral_gain) < 1e-9:
            return float(integral)

        candidate_integral = integral + error
        candidate_control = feedforward + control_sign * (
            non_integral_term + integral_gain * candidate_integral
        )
        candidate_clipped = float(np.clip(candidate_control, control_min, control_max))
        hard_saturated = abs(candidate_control - candidate_clipped) > 1e-6
        threshold = (
            max(abs(control_min), abs(control_max))
            * self.conditional_integration_threshold
        )
        near_saturated = abs(candidate_control) >= threshold

        next_integral = candidate_integral
        if hard_saturated or near_saturated:
            integral_control_delta = control_sign * integral_gain * error
            saturation_error = candidate_control - candidate_clipped
            if near_saturated and not hard_saturated:
                saturation_error = math.copysign(
                    abs(candidate_control) - threshold,
                    candidate_control,
                )
            drives_deeper = saturation_error * integral_control_delta > 0.0
            if drives_deeper:
                if hard_saturated:
                    next_integral = (
                        (candidate_clipped - feedforward) / control_sign
                        - non_integral_term
                    ) / integral_gain
                else:
                    next_integral = integral

        control = feedforward + control_sign * (
            non_integral_term + integral_gain * next_integral
        )
        control_clipped = float(np.clip(control, control_min, control_max))
        if hard_saturated and self.anti_windup_backcalc_gain > 0.0:
            next_integral += self.anti_windup_backcalc_gain * (
                control_clipped - control
            ) / (control_sign * integral_gain)

        return float(
            np.clip(
                next_integral,
                -self.integral_windup_limit,
                self.integral_windup_limit,
            )
        )

    def _apply_protection_arbitration(
        self,
        nz_filt: float,
        roll_rate_filt: float,
        *,
        speed: float,
        altitude: float,
        roll: float,
    ) -> tuple:
        """Apply stall, bank, and altitude protections in priority order."""
        flags = {
            "stall_limited": False,
            "bank_recovery": False,
            "altitude_hold": False,
        }

        stall_margin = self._stall_margin(speed)
        nz_upper = self.nz_max
        if self.enable_dynamic_nz_limit:
            nz_upper = 1.0 + stall_margin * (self.nz_max - 1.0)
            flags["stall_limited"] = nz_upper < self.nz_max - 1e-6

        nz_out = float(np.clip(nz_filt, self.nz_min, nz_upper))

        if self.enable_stall_protection:
            roll_scale = (
                0.0 if speed <= self.stall_speed_mps else max(0.1, stall_margin)
            )
            roll_rate_filt *= roll_scale
            roll_rate_filt = float(
                np.clip(roll_rate_filt, self.roll_rate_min, self.roll_rate_max)
            )

        bank_active = False
        if self.enable_bank_angle_protection:
            abs_roll = abs(roll)
            if abs_roll > self.max_bank_rad:
                self._bank_violation_steps += 1
                overbank = abs_roll - self.max_bank_rad
                sustained = self._bank_violation_steps > self.bank_violation_threshold
                immediate = overbank > math.radians(10.0)
                if sustained or immediate:
                    bank_active = True
                    flags["bank_recovery"] = True
                    roll_rate_filt = float(
                        -math.copysign(
                            min(overbank * 5.0, abs(self.roll_rate_max)),
                            roll,
                        )
                    )
                    bank_delta = self.bank_protection_nz_increment * (
                        overbank / math.radians(20.0)
                    )
                    bank_delta = min(bank_delta, self.bank_protection_nz_increment_max)
                    nz_out += bank_delta
            else:
                self._bank_violation_steps = 0

        if self.enable_altitude_hold:
            alt_error = self.altitude_reference_m - altitude
            alt_delta = self.altitude_hold_gain * alt_error
            if bank_active and alt_delta < 0.0:
                alt_delta = 0.0
            nz_out += alt_delta
            flags["altitude_hold"] = abs(alt_delta) > 1e-9

        nz_out = float(np.clip(nz_out, self.nz_min, nz_upper))
        return nz_out, roll_rate_filt, flags

    def reset(self):
        """Reset controller internal state."""
        self._prev_nz = 1.0
        self._prev_roll_rate = 0.0
        self._prev_throttle = 0.7
        self._nz_integral = 0.0
        self._nz_prev_error = 0.0
        self._roll_integral = 0.0
        self._roll_prev_error = 0.0
        self._throttle_integral = 0.0
        self._bank_violation_steps = 0
        self.command_history.clear()

    def compute_actuator(
        self,
        guidance_command: dict,
        aircraft_state: dict = None,
        aggressiveness: float = None,
        pid_gain_deltas=None,
    ) -> dict:
        """
        Compute actuator commands with PID feedback and coordinated turn.

        Args:
            guidance_command (dict): Keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
            aircraft_state (dict, optional): Current aircraft state from JSBSim.
                Expected keys: nz_g, p_rps, roll_rad, pitch_rad, alpha_rad, beta_rad,
                speed_mps, altitude_m.
            aggressiveness (float, optional): PPO-learned aggressiveness parameter
                in [-1, 1]. Controls PID gain scaling: -1 = conservative (0.5x),
                0 = normal, +1 = aggressive (1.5x). Provides a single interpretable
                knob for the policy to modulate controller response speed.
            pid_gain_deltas (dict or sequence, optional): APIC-style per-gain
                adjustments in [-1, 1] for [Kp_nz, Ki_nz, Kd_nz, Kp_roll, Ki_roll,
                Kd_roll]. If provided, it overrides the aggressiveness scaling.

        Returns:
            dict: Actuator command dictionary for JSBSim.
        """
        # --- Sanitize inputs ---
        nz_cmd = self._safe_float(guidance_command.get("nz_cmd", 1.0))
        roll_rate_cmd = self._safe_float(guidance_command.get("roll_rate_cmd", 0.0))
        throttle_cmd = self._safe_float(guidance_command.get("throttle_cmd", 0.7))

        # --- First-order filter on command input ---
        nz_filt = self.alpha * nz_cmd + (1.0 - self.alpha) * self._prev_nz
        roll_rate_filt = self.alpha * roll_rate_cmd + (1.0 - self.alpha) * self._prev_roll_rate
        throttle_filt = self.alpha * throttle_cmd + (1.0 - self.alpha) * self._prev_throttle
        self._prev_nz = nz_filt
        self._prev_roll_rate = roll_rate_filt
        self._prev_throttle = throttle_filt

        filtered = {
            "nz_cmd": float(nz_filt),
            "roll_rate_cmd": float(roll_rate_filt),
            "throttle_cmd": float(throttle_filt),
        }

        # --- Extract aircraft state (with safe defaults) ---
        if aircraft_state is None:
            aircraft_state = {}
        actual_nz = self._safe_float(aircraft_state.get("nz_g", 1.0))
        actual_roll_rate = self._safe_float(aircraft_state.get("p_rps", 0.0))
        roll = self._safe_float(aircraft_state.get("roll_rad", 0.0))
        pitch = self._safe_float(aircraft_state.get("pitch_rad", 0.0))
        alpha = self._safe_float(aircraft_state.get("alpha_rad", 0.0))
        beta = self._safe_float(aircraft_state.get("beta_rad", 0.0))
        speed = self._safe_float(aircraft_state.get("speed_mps", 250.0))
        altitude = self._safe_float(aircraft_state.get("altitude_m", 5000.0))

        # --- Flight-envelope protection arbitration ---
        # Priority: stall hard limits > bank recovery > altitude-hold correction.
        nz_filt, roll_rate_filt, protection_flags = self._apply_protection_arbitration(
            nz_filt,
            roll_rate_filt,
            speed=speed,
            altitude=altitude,
            roll=roll,
        )

        # --- APIC-style per-gain adaptation overrides aggressiveness ---
        if pid_gain_deltas is not None:
            (
                Kp_nz_eff,
                Ki_nz_eff,
                Kd_nz_eff,
                Kp_roll_eff,
                Ki_roll_eff,
                Kd_roll_eff,
            ) = self._effective_pid_gains(pid_gain_deltas)
        elif aggressiveness is not None:
            # PPO-learned aggressiveness: scale PID gains
            # aggressiveness ∈ [-1, 1] → gain_scale ∈ [0.5, 1.5]
            # +1 = aggressive (higher gains, faster response)
            # -1 = conservative (lower gains, smoother response)
            agg = float(np.clip(aggressiveness, -1.0, 1.0))
            gain_scale = 1.0 + agg * 0.5
            # Apply to current gains (may include gain-scheduling adjustments)
            Kp_nz_eff = self.Kp_nz * gain_scale
            Ki_nz_eff = self.Ki_nz * gain_scale
            Kd_nz_eff = self.Kd_nz * gain_scale
            Kp_roll_eff = self.Kp_roll * gain_scale
            Ki_roll_eff = self.Ki_roll * gain_scale
            Kd_roll_eff = self.Kd_roll * gain_scale
        else:
            Kp_nz_eff = self.Kp_nz
            Ki_nz_eff = self.Ki_nz
            Kd_nz_eff = self.Kd_nz
            Kp_roll_eff = self.Kp_roll
            Ki_roll_eff = self.Ki_roll
            Kd_roll_eff = self.Kd_roll

        # --- Elevator: PID + feedforward for nz tracking ---
        nz_error = nz_filt - actual_nz
        nz_derivative = nz_error - self._nz_prev_error

        if self.use_pid:
            elevator_pd = Kp_nz_eff * nz_error + Kd_nz_eff * nz_derivative
        else:
            elevator_pd = 0.0

        # Feedforward: expected elevator for the commanded nz (inverted plant model)
        # NOTE: elevator is negative for positive nz (pull-up)
        elevator_ff = -nz_filt * self.nz_gain
        if self.use_pid:
            self._nz_integral = self._integrate_with_anti_windup(
                error=nz_error,
                integral=self._nz_integral,
                feedforward=elevator_ff,
                non_integral_term=elevator_pd,
                integral_gain=Ki_nz_eff,
                control_sign=-1.0,
                control_min=-self.elevator_max,
                control_max=self.elevator_max,
            )
            elevator_pid = elevator_pd + Ki_nz_eff * self._nz_integral
        else:
            elevator_pid = 0.0
        # PID correction: if actual_nz < nz_cmd, need MORE negative elevator (more pull-up)
        # nz_error = nz_cmd - actual_nz > 0  =>  need to subtract elevator_pid
        elevator = elevator_ff - elevator_pid
        elevator = np.clip(elevator, -self.elevator_max, self.elevator_max)
        self._nz_prev_error = nz_error

        # --- Angle-of-attack protection ---
        if self.use_aoa_protection and abs(alpha) > self.alpha_limit:
            # Reduce elevator authority when AoA is high to prevent stall
            aoa_excess = abs(alpha) - self.alpha_limit
            aoa_scale = max(0.0, 1.0 - aoa_excess * 2.0)  # Gradual reduction
            elevator = np.clip(elevator, -self.elevator_max * aoa_scale, self.elevator_max * aoa_scale)

        # --- Aileron: PID + feedforward for roll rate tracking ---
        roll_rate_error = roll_rate_filt - actual_roll_rate
        roll_derivative = roll_rate_error - self._roll_prev_error

        if self.use_pid:
            aileron_pd = Kp_roll_eff * roll_rate_error + Kd_roll_eff * roll_derivative
        else:
            aileron_pd = 0.0

        # Feedforward: expected aileron for the commanded roll rate
        aileron_ff = roll_rate_filt * self.roll_rate_gain
        if self.use_pid:
            self._roll_integral = self._integrate_with_anti_windup(
                error=roll_rate_error,
                integral=self._roll_integral,
                feedforward=aileron_ff,
                non_integral_term=aileron_pd,
                integral_gain=Ki_roll_eff,
                control_sign=1.0,
                control_min=-self.aileron_max,
                control_max=self.aileron_max,
            )
            aileron_pid = aileron_pd + Ki_roll_eff * self._roll_integral
        else:
            aileron_pid = 0.0
        aileron = aileron_ff + aileron_pid
        aileron = np.clip(aileron, -self.aileron_max, self.aileron_max)
        self._roll_prev_error = roll_rate_error

        # Hard roll-recovery override only for sustained or extreme overbank.
        if self.enable_bank_angle_protection and abs(roll) > self.max_bank_rad:
            overbank = abs(roll) - self.max_bank_rad
            sustained = (
                self._bank_violation_steps > self.bank_violation_threshold
            )
            immediate = overbank > math.radians(10.0)
            if sustained or immediate:
                recovery_threshold = math.radians(5.0)
                if overbank > recovery_threshold:
                    override_mag = self.aileron_max
                else:
                    override_mag = self.aileron_max * (overbank / recovery_threshold)
                aileron = -math.copysign(override_mag, roll)
                # Dump roll integrator so the PID does not fight the recovery.
                self._roll_integral = 0.0
                self._roll_prev_error = 0.0

        # --- Rudder: coordinated turn + sideslip suppression ---
        if self.use_rudder:
            # Term 1: suppress sideslip (beta -> 0)
            beta_suppression = 0.0
            if self.use_beta_suppression:
                beta_Kp = self.config.get("beta_Kp", 0.5)
                beta_suppression = -beta_Kp * beta

            # Term 2: turn coordination (opposite aileron-induced adverse yaw)
            # Positive roll rate -> negative rudder to counter adverse yaw
            turn_coordination = -self.rudder_from_roll * actual_roll_rate

            # Term 3: pro-verse rudder to assist roll (small gain)
            roll_assist = 0.1 * roll_rate_filt

            rudder = beta_suppression + turn_coordination + roll_assist
            rudder = np.clip(rudder, -self.rudder_max, self.rudder_max)
        else:
            rudder = 0.0

        # --- Throttle: feedforward + speed-hold PI + stall recovery ---
        speed_error = self.target_speed_mps - speed

        # Integral term for speed hold
        self._throttle_integral += speed_error * self.control_dt
        self._throttle_integral = float(
            np.clip(self._throttle_integral, -self.throttle_integral_limit, self.throttle_integral_limit)
        )

        # Speed-hold correction
        throttle_correction = 0.0
        if self.enable_speed_hold:
            throttle_correction = (
                self.speed_kp * speed_error / self.target_speed_mps
                + self.speed_ki * self._throttle_integral
            )

        # Energy boost: high-g manoeuvres need more thrust
        energy_boost = 0.0
        if self.enable_energy_boost and nz_filt > 1.0:
            energy_boost = self.energy_boost_gain * (nz_filt - 1.0)

        throttle = throttle_filt + throttle_correction + energy_boost

        # Stall recovery: add extra throttle when speed is below recovery band
        if speed < self.recovery_speed_mps:
            throttle += 0.1 * (self.recovery_speed_mps - speed) / self.target_speed_mps

        # Clamp
        throttle = float(np.clip(throttle, self.throttle_min, self.throttle_max))

        # --- Safety limits ---
        # Altitude protection: limit elevator near ground/ceiling
        if altitude < 1000.0:
            elevator = np.clip(elevator, -0.3, 0.5)  # Prevent diving
        if altitude > 14000.0:
            elevator = np.clip(elevator, -0.5, 0.3)  # Prevent climbing

        # Speed protection: limit elevator when too slow
        if speed < 120.0:
            elevator = np.clip(elevator, -0.2, 0.2)
            aileron = np.clip(aileron, -0.3, 0.3)

        # --- Build output ---
        jsbsim_props = {
            "fcs/elevator-cmd-norm": float(elevator),
            "fcs/aileron-cmd-norm": float(aileron),
            "fcs/rudder-cmd-norm": float(rudder),
            "fcs/throttle-cmd-norm": float(throttle),
            "saturation_flag": False,  # We already clipped everything
            "elevator_raw": float(elevator),
            "aileron_raw": float(aileron),
            "rudder_raw": float(rudder),
            "filtered_command": filtered,
            "protection_flags": protection_flags,
        }

        # Record history
        record = {
            "guidance": guidance_command,
            "filtered": filtered,
            "jsbsim": {k: v for k, v in jsbsim_props.items() if k.startswith("fcs/")},
            "pid_errors": {
                "nz_error": float(nz_error),
                "roll_rate_error": float(roll_rate_error),
                "nz_integral": float(self._nz_integral),
                "roll_integral": float(self._roll_integral),
            },
            "aircraft_state": {
                "nz_g": actual_nz,
                "p_rps": actual_roll_rate,
                "alpha_rad": alpha,
                "beta_rad": beta,
                "speed_mps": speed,
            },
            "protection_flags": protection_flags,
            "saturation_flag": False,
        }
        self.command_history.append(record)

        return jsbsim_props

    @staticmethod
    def _safe_float(value):
        """Protect against NaN and inf."""
        if value is None:
            return 0.0
        v = float(value)
        if not np.isfinite(v):
            return 0.0
        return v


class GainScheduledEnhancedController(EnhancedLowLevelController):
    """
    Gain-scheduled enhanced low-level controller.

    Extends EnhancedLowLevelController with:
    - Dynamic PID gain scheduling based on dynamic pressure, altitude, and AoA
    - Descent rate protection (h/vd < 25 emergency pull-up)
    - Conditional integration (anti-windup)
    - Large roll angle protection (limit nz when |roll| > 45°)

    Architecture: VPP → LOS → GainScheduled PID → JSBSim
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Inherits all EnhancedLowLevelController keys plus:
                - use_gain_scheduling (bool): default True
                - qbar_ref (float): reference dynamic pressure, default 0.5*1.225*250^2 = 38281.25
                - speed_threshold_mps (float): high-speed threshold, default 300
                - altitude_threshold_m (float): high-altitude threshold, default 11000
                - alpha_schedule_threshold (float): AoA scheduling threshold, default 0.3
                - roll_angle_limit_rad (float): large roll protection threshold, default π/4
                - nz_limit_at_high_roll (float): nz limit when |roll| > threshold, default 3.0
                - use_descent_protection (bool): enable descent rate protection, default True
                - descent_time_to_impact (float): h/vd threshold, default 25.0
                - use_conditional_integration (bool): anti-windup, default True
                - conditional_integration_threshold (float): saturation threshold fraction, default 0.95
        """
        super().__init__(config)

        # Gain scheduling config
        self.use_gain_scheduling = self.config.get("use_gain_scheduling", True)
        self.qbar_ref = self.config.get("qbar_ref", 0.5 * 1.225 * 250.0 ** 2)
        self.speed_threshold_mps = self.config.get("speed_threshold_mps", 300.0)
        self.altitude_threshold_m = self.config.get("altitude_threshold_m", 11000.0)
        self.alpha_schedule_threshold = self.config.get("alpha_schedule_threshold", 0.3)

        # Large roll protection
        self.roll_angle_limit_rad = self.config.get("roll_angle_limit_rad", math.pi / 4)
        self.nz_limit_at_high_roll = self.config.get("nz_limit_at_high_roll", 3.0)

        # Descent protection
        self.use_descent_protection = self.config.get("use_descent_protection", True)
        self.descent_time_to_impact = self.config.get("descent_time_to_impact", 25.0)
        self.descent_altitude_threshold = self.config.get("descent_altitude_threshold", 4000.0)
        self.descent_min_altitude = self.config.get("descent_min_altitude", 400.0)
        self.descent_pitch_threshold = self.config.get("descent_pitch_threshold", -math.pi / 4)
        self.descent_altitude_for_pitch = self.config.get("descent_altitude_for_pitch", 3500.0)

        # Conditional integration (anti-windup)
        self.use_conditional_integration = self.config.get("use_conditional_integration", True)
        self.conditional_integration_threshold = self.config.get(
            "conditional_integration_threshold", 0.95
        )

        # Store base gains for scheduling
        self._base_Kp_nz = self.Kp_nz
        self._base_Ki_nz = self.Ki_nz
        self._base_Kd_nz = self.Kd_nz
        self._base_Kp_roll = self.Kp_roll
        self._base_Ki_roll = self.Ki_roll
        self._base_Kd_roll = self.Kd_roll

        # Safety flags for diagnostics
        self._safety_flags = {}

    def reset(self):
        """Reset controller and safety flags."""
        super().reset()
        self._safety_flags = {}

    def _update_pid_gains(self, state):
        """
        Dynamic gain scheduling based on flight conditions.

        Inspired by bvr_sim StdFlightController._adaptive_pid_scaling().
        """
        if not self.use_gain_scheduling:
            return

        speed = self._safe_float(state.get("speed_mps", 250.0))
        altitude = self._safe_float(state.get("altitude_m", 5000.0))
        alpha = self._safe_float(state.get("alpha_rad", 0.0))

        # Start from base gains
        self.Kp_nz = self._base_Kp_nz
        self.Ki_nz = self._base_Ki_nz
        self.Kd_nz = self._base_Kd_nz
        self.Kp_roll = self._base_Kp_roll
        self.Ki_roll = self._base_Ki_roll
        self.Kd_roll = self._base_Kd_roll

        # 1. Dynamic pressure scheduling (core compensation)
        qbar = 0.5 * 1.225 * speed ** 2
        if qbar > 1e-6:
            qbar_scale = min(2.0, self.qbar_ref / qbar)
        else:
            qbar_scale = 2.0

        # High speed: reduce proportional gain to prevent overshoot
        # Low speed: increase proportional gain to compensate for reduced control authority
        self.Kp_nz *= qbar_scale
        self.Kp_roll *= qbar_scale
        self.Kd_nz *= qbar_scale  # More damping at high speed
        self.Kd_roll *= qbar_scale

        # 2. Altitude scheduling (thin air at high altitude)
        if altitude > self.altitude_threshold_m:
            height_scale = max(0.3, (20000.0 - altitude) / 9000.0)
            self.Kp_nz *= height_scale
            self.Ki_nz *= height_scale
            self.Kp_roll *= max(0.3, (16000.0 - altitude) / 5000.0)
            self.Ki_roll *= max(0.3, (16000.0 - altitude) / 5000.0)

        # 3. Angle-of-attack scheduling (high AoA = reduced control effectiveness)
        if alpha > self.alpha_schedule_threshold:
            alpha_scale = 1.0 + 0.5 * (alpha / self.alpha_limit)
            self.Kp_nz *= alpha_scale
            self.Kp_roll *= alpha_scale

        # 4. Speed-specific adjustments (similar to bvr_sim mach scheduling)
        if speed > self.speed_threshold_mps:
            # High speed: further reduce roll gain to prevent excessive roll rate
            self.Kp_roll *= 0.85 + 0.15 * (abs(alpha) / (1.5 * math.pi / 12.0))
            self.Ki_roll *= 0.85 + 0.15 * (abs(alpha) / (1.5 * math.pi / 12.0))
        else:
            # Low speed: increase pitch gain slightly (conservative to avoid overshoot)
            self.Kp_nz *= 1.1
            self.Ki_nz *= 1.05
            self.Kp_roll *= 1.1
            self.Ki_roll *= 1.05

    def _safety_protection(self, state, elevator_cmd, aileron_cmd, throttle_cmd, nz_filt):
        """
        Enhanced safety protection including descent rate and large roll angle.

        Returns: (override_elevator, override_aileron, override_throttle, override_nz, flags)
        """
        flags = {}
        altitude = self._safe_float(state.get("altitude_m", 5000.0))
        pitch = self._safe_float(state.get("pitch_rad", 0.0))
        roll = self._safe_float(state.get("roll_rad", 0.0))
        speed = self._safe_float(state.get("speed_mps", 250.0))

        # --- Descent rate protection (inspired by bvr_sim) ---
        if self.use_descent_protection:
            # Get vertical descent rate (positive = descending)
            vd = self._safe_float(state.get("vd_mps", 0.0))
            # If vd is not available, estimate from pitch and speed
            if vd < 0.01 and abs(pitch) > 0.01:
                vd = speed * math.sin(-pitch)  # rough estimate

            # Crashing protection: descending too fast at low altitude
            if altitude < self.descent_altitude_threshold and vd > 0.1:
                time_to_impact = altitude / vd
                if time_to_impact < self.descent_time_to_impact:
                    flags["emergency_pull_up"] = True
                    flags["time_to_impact"] = time_to_impact
                    # Override: force positive elevator (pull-up)
                    elevator_cmd = min(elevator_cmd, -0.5)  # More negative = pull up
                    throttle_cmd = max(throttle_cmd, 0.8)  # Full throttle

            # Low altitude + steep dive
            if altitude < self.descent_altitude_for_pitch and pitch < self.descent_pitch_threshold:
                flags["steep_dive"] = True
                elevator_cmd = min(elevator_cmd, -0.3)
                throttle_cmd = max(throttle_cmd, 0.8)

            # Absolute minimum altitude
            if altitude < self.descent_min_altitude:
                flags["minimum_altitude"] = True
                elevator_cmd = -0.8  # Maximum pull-up
                throttle_cmd = 0.9

        # --- Over-height protection ---
        if altitude > 12000.0 and pitch > math.pi / 4:
            flags["over_height"] = True
            elevator_cmd = max(elevator_cmd, 0.3)  # Push down

        # --- Large roll angle protection ---
        # When banked > 45°, limit nz to prevent excessive energy loss in climb
        if abs(roll) > self.roll_angle_limit_rad:
            flags["large_roll"] = True
            # Limit nz command
            nz_filt = np.clip(nz_filt, -self.nz_limit_at_high_roll, self.nz_limit_at_high_roll)
            # Also limit elevator to prevent excessive pitch while rolled
            elevator_cmd = np.clip(elevator_cmd, -0.5, 0.3)

        # --- Low speed protection ---
        if speed < 120.0:
            flags["low_speed"] = True
            elevator_cmd = np.clip(elevator_cmd, -0.2, 0.2)
            aileron_cmd = np.clip(aileron_cmd, -0.3, 0.3)
            throttle_cmd = max(throttle_cmd, 0.8)

        return elevator_cmd, aileron_cmd, throttle_cmd, nz_filt, flags

    def _conditional_integration(self, error, integral, control_signal, limit):
        """
        Conditional integration: only integrate when not saturated.
        Prevents integral windup during sustained high-demand maneuvers.
        """
        if not self.use_conditional_integration:
            return integral + error

        threshold = limit * self.conditional_integration_threshold
        if abs(control_signal) < threshold:
            return integral + error
        return integral  # Hold current value when saturated

    def compute_actuator(
        self,
        guidance_command: dict,
        aircraft_state: dict = None,
        aggressiveness: float = None,
        pid_gain_deltas=None,
    ) -> dict:
        """
        Compute actuator commands with gain scheduling and enhanced safety.

        Args:
            guidance_command (dict): Keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
            aircraft_state (dict, optional): Current aircraft state.
            aggressiveness (float, optional): PPO-learned aggressiveness in [-1, 1].
                +1 = aggressive (higher gains), -1 = conservative (lower gains).
            pid_gain_deltas (dict or sequence, optional): APIC-style per-gain
                adjustments. If provided, overrides aggressiveness scaling.

        Returns:
            dict: Actuator command dictionary with safety flags.
        """
        # --- Sanitize inputs ---
        nz_cmd = self._safe_float(guidance_command.get("nz_cmd", 1.0))
        roll_rate_cmd = self._safe_float(guidance_command.get("roll_rate_cmd", 0.0))
        throttle_cmd = self._safe_float(guidance_command.get("throttle_cmd", 0.7))

        # --- First-order filter ---
        nz_filt = self.alpha * nz_cmd + (1.0 - self.alpha) * self._prev_nz
        roll_rate_filt = self.alpha * roll_rate_cmd + (1.0 - self.alpha) * self._prev_roll_rate
        throttle_filt = self.alpha * throttle_cmd + (1.0 - self.alpha) * self._prev_throttle
        self._prev_nz = nz_filt
        self._prev_roll_rate = roll_rate_filt
        self._prev_throttle = throttle_filt

        filtered = {
            "nz_cmd": float(nz_filt),
            "roll_rate_cmd": float(roll_rate_filt),
            "throttle_cmd": float(throttle_filt),
        }

        # --- Extract aircraft state ---
        if aircraft_state is None:
            aircraft_state = {}
        actual_nz = self._safe_float(aircraft_state.get("nz_g", 1.0))
        actual_roll_rate = self._safe_float(aircraft_state.get("p_rps", 0.0))
        roll = self._safe_float(aircraft_state.get("roll_rad", 0.0))
        pitch = self._safe_float(aircraft_state.get("pitch_rad", 0.0))
        alpha = self._safe_float(aircraft_state.get("alpha_rad", 0.0))
        beta = self._safe_float(aircraft_state.get("beta_rad", 0.0))
        speed = self._safe_float(aircraft_state.get("speed_mps", 250.0))
        altitude = self._safe_float(aircraft_state.get("altitude_m", 5000.0))

        # --- Update PID gains dynamically ---
        self._update_pid_gains(aircraft_state)

        # --- APIC-style per-gain adaptation overrides aggressiveness ---
        if pid_gain_deltas is not None:
            (
                Kp_nz_eff,
                Ki_nz_eff,
                Kd_nz_eff,
                Kp_roll_eff,
                Ki_roll_eff,
                Kd_roll_eff,
            ) = self._effective_pid_gains(pid_gain_deltas)
        elif aggressiveness is not None:
            # PPO-learned aggressiveness: scale scheduled PID gains
            # aggressiveness ∈ [-1, 1] → gain_scale ∈ [0.5, 1.5]
            agg = float(np.clip(aggressiveness, -1.0, 1.0))
            gain_scale = 1.0 + agg * 0.5
            Kp_nz_eff = self.Kp_nz * gain_scale
            Ki_nz_eff = self.Ki_nz * gain_scale
            Kd_nz_eff = self.Kd_nz * gain_scale
            Kp_roll_eff = self.Kp_roll * gain_scale
            Ki_roll_eff = self.Ki_roll * gain_scale
            Kd_roll_eff = self.Kd_roll * gain_scale
        else:
            Kp_nz_eff = self.Kp_nz
            Ki_nz_eff = self.Ki_nz
            Kd_nz_eff = self.Kd_nz
            Kp_roll_eff = self.Kp_roll
            Ki_roll_eff = self.Ki_roll
            Kd_roll_eff = self.Kd_roll

        # --- Elevator: PID + feedforward ---
        nz_error = nz_filt - actual_nz
        nz_derivative = nz_error - self._nz_prev_error

        # Conditional integration for nz channel
        raw_elevator = -nz_filt * self.nz_gain
        if self.use_pid:
            elevator_pid = Kp_nz_eff * nz_error + Kd_nz_eff * nz_derivative
            self._nz_integral = self._conditional_integration(
                nz_error, self._nz_integral, raw_elevator + elevator_pid, self.elevator_max
            )
            self._nz_integral = np.clip(
                self._nz_integral, -self.integral_windup_limit, self.integral_windup_limit
            )
            elevator_pid += Ki_nz_eff * self._nz_integral
        else:
            elevator_pid = 0.0

        elevator = -nz_filt * self.nz_gain - elevator_pid
        elevator = np.clip(elevator, -self.elevator_max, self.elevator_max)
        self._nz_prev_error = nz_error

        # --- Aileron: PID + feedforward ---
        roll_rate_error = roll_rate_filt - actual_roll_rate
        roll_derivative = roll_rate_error - self._roll_prev_error

        raw_aileron = roll_rate_filt * self.roll_rate_gain
        if self.use_pid:
            aileron_pid = Kp_roll_eff * roll_rate_error + Kd_roll_eff * roll_derivative
            self._roll_integral = self._conditional_integration(
                roll_rate_error, self._roll_integral, raw_aileron + aileron_pid, self.aileron_max
            )
            self._roll_integral = np.clip(
                self._roll_integral, -self.integral_windup_limit, self.integral_windup_limit
            )
            aileron_pid += Ki_roll_eff * self._roll_integral
        else:
            aileron_pid = 0.0

        aileron = roll_rate_filt * self.roll_rate_gain + aileron_pid
        aileron = np.clip(aileron, -self.aileron_max, self.aileron_max)
        self._roll_prev_error = roll_rate_error

        # --- Rudder ---
        if self.use_rudder:
            beta_suppression = 0.0
            if self.use_beta_suppression:
                beta_Kp = self.config.get("beta_Kp", 0.5)
                beta_suppression = -beta_Kp * beta
            turn_coordination = -self.rudder_from_roll * actual_roll_rate
            roll_assist = 0.1 * roll_rate_filt
            rudder = beta_suppression + turn_coordination + roll_assist
            rudder = np.clip(rudder, -self.rudder_max, self.rudder_max)
        else:
            rudder = 0.0

        # --- Throttle ---
        throttle = throttle_filt
        if self.config.get("enable_energy_boost", False) and nz_filt > 3.0:
            energy_boost = self.config.get("energy_boost_gain", 0.02) * (nz_filt - 3.0)
            throttle += energy_boost
        if speed < 150.0 and throttle > 0.5:
            throttle *= 0.9
        throttle = np.clip(throttle, self.throttle_min, self.throttle_max)

        # --- Safety protection (enhanced) ---
        elevator, aileron, throttle, nz_filt, safety_flags = self._safety_protection(
            aircraft_state, elevator, aileron, throttle, nz_filt
        )
        self._safety_flags = safety_flags

        # --- Angle-of-attack protection ---
        if self.use_aoa_protection and abs(alpha) > self.alpha_limit:
            aoa_excess = abs(alpha) - self.alpha_limit
            aoa_scale = max(0.0, 1.0 - aoa_excess * 2.0)
            elevator = np.clip(elevator, -self.elevator_max * aoa_scale, self.elevator_max * aoa_scale)

        # --- Build output ---
        jsbsim_props = {
            "fcs/elevator-cmd-norm": float(elevator),
            "fcs/aileron-cmd-norm": float(aileron),
            "fcs/rudder-cmd-norm": float(rudder),
            "fcs/throttle-cmd-norm": float(throttle),
            "saturation_flag": False,
            "elevator_raw": float(elevator),
            "aileron_raw": float(aileron),
            "rudder_raw": float(rudder),
            "filtered_command": filtered,
            "safety_flags": safety_flags,
            "scheduled_gains": {
                "Kp_nz": float(self.Kp_nz),
                "Ki_nz": float(self.Ki_nz),
                "Kd_nz": float(self.Kd_nz),
                "Kp_roll": float(self.Kp_roll),
                "Ki_roll": float(self.Ki_roll),
                "Kd_roll": float(self.Kd_roll),
            } if self.use_gain_scheduling else {},
        }

        # Record history
        record = {
            "guidance": guidance_command,
            "filtered": filtered,
            "jsbsim": {k: v for k, v in jsbsim_props.items() if k.startswith("fcs/")},
            "pid_errors": {
                "nz_error": float(nz_error),
                "roll_rate_error": float(roll_rate_error),
                "nz_integral": float(self._nz_integral),
                "roll_integral": float(self._roll_integral),
            },
            "aircraft_state": {
                "nz_g": actual_nz,
                "p_rps": actual_roll_rate,
                "alpha_rad": alpha,
                "beta_rad": beta,
                "speed_mps": speed,
                "altitude_m": altitude,
            },
            "safety_flags": safety_flags,
            "saturation_flag": False,
        }
        self.command_history.append(record)

        return jsbsim_props
