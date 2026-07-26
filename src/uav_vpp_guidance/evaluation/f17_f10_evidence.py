"""Pure offline F17 switching and F10 arbitration/scheduling evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi
from typing import Iterable

import numpy as np

F04_NZ_LIMITS = (-2.0, 7.0)


@dataclass(frozen=True)
class HybridSwitchParameters:
    range_threshold_m: float = 3000.0
    hysteresis_m: float = 500.0
    min_dwell_steps: int = 3
    dt_s: float = 0.2


class OfflineRangeSwitch:
    """Faithful pure model of HybridGuidance range hysteresis and dwell semantics."""

    def __init__(self, params: HybridSwitchParameters):
        if params.hysteresis_m < 0.0 or params.min_dwell_steps < 1 or params.dt_s <= 0.0:
            raise ValueError("hysteresis must be non-negative; dwell and dt must be positive")
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.active_law, self.pending_law, self.steps_in_law = "pn", None, 0

    def desired_law(self, range_m: float) -> str:
        lower = self.params.range_threshold_m - 0.5 * self.params.hysteresis_m
        upper = self.params.range_threshold_m + 0.5 * self.params.hysteresis_m
        if self.active_law == "pn":
            return "los" if range_m < lower else "pn"
        return "pn" if range_m > upper else "los"

    def step(self, range_m: float) -> dict:
        desired = self.desired_law(range_m)
        changed = False
        if desired == self.active_law:
            self.steps_in_law += 1
            self.pending_law = None
        elif self.pending_law == desired:
            self.steps_in_law += 1
        else:
            self.pending_law, self.steps_in_law = desired, 1
        if desired != self.active_law and self.steps_in_law >= self.params.min_dwell_steps:
            self.active_law, self.pending_law, self.steps_in_law, changed = desired, None, 0, True
        return {"range_m": float(range_m), "desired_law": desired, "active_law": self.active_law, "pending_law": self.pending_law, "pending_steps": self.steps_in_law, "changed": changed}


def switch_metrics(ranges_m: Iterable[float], params: HybridSwitchParameters, closing_speed_mps: float) -> dict:
    model, trace = OfflineRangeSwitch(params), []
    first_desired_step = first_switch_step = None
    for index, range_m in enumerate(ranges_m):
        row = model.step(float(range_m)); row["step"] = index; trace.append(row)
        if row["desired_law"] == "los" and first_desired_step is None:
            first_desired_step = index
        if row["changed"] and row["active_law"] == "los" and first_switch_step is None:
            first_switch_step = index
    delay_steps = None if first_desired_step is None or first_switch_step is None else first_switch_step - first_desired_step
    changes = sum(int(row["changed"]) for row in trace)
    return {"parameters": asdict(params), "closing_speed_mps": closing_speed_mps, "trace": trace, "switch_delay_steps": delay_steps, "switch_delay_s": None if delay_steps is None else delay_steps * params.dt_s, "range_travelled_during_delay_m": None if delay_steps is None else delay_steps * params.dt_s * closing_speed_mps, "chatter_count": max(0, changes - 1), "missed_switch_opportunities": sum(row["desired_law"] != row["active_law"] for row in trace)}


def f17_sweep() -> dict:
    baseline = HybridSwitchParameters()
    candidates = {
        "reduced_dwell_1_step": HybridSwitchParameters(min_dwell_steps=1),
        "wider_hysteresis_750m": HybridSwitchParameters(hysteresis_m=750.0),
    }
    traces = {
        "steady_closing": [3100.0, 2950.0, 2700.0, 2450.0, 2300.0, 2150.0],
        "threshold_jitter": [3100.0, 2470.0, 2530.0, 2460.0, 2540.0, 2450.0, 2440.0, 2430.0],
        "opening_after_entry": [2400.0, 2400.0, 2400.0, 3300.0, 3300.0, 3300.0],
    }
    speeds = (100.0, 300.0)
    return {"baseline": {name: {str(speed): switch_metrics(values, baseline, speed) for speed in speeds} for name, values in traces.items()}, "candidates": {candidate: {name: {str(speed): switch_metrics(values, params, speed) for speed in speeds} for name, values in traces.items()} for candidate, params in candidates.items()}, "scope": "offline hybrid range-switch model only; no runtime dwell or hysteresis changed"}


@dataclass(frozen=True)
class ArbitrationInput:
    nz_guidance: float
    roll_rate_guidance: float
    altitude_m: float
    speed_mps: float
    roll_rad: float
    alpha_rad: float


def qbar_gain_scales(speed_mps: float, altitude_m: float, alpha_rad: float, *, qbar_ref: float = 0.5 * 1.225 * 250.0**2, altitude_threshold_m: float = 11000.0, alpha_threshold_rad: float = 0.3, alpha_limit_rad: float = 0.5) -> dict:
    qbar = 0.5 * 1.225 * speed_mps**2
    qbar_scale = min(2.0, qbar_ref / qbar) if qbar > 1e-6 else 2.0
    height_nz_scale = max(0.3, (20000.0 - altitude_m) / 9000.0) if altitude_m > altitude_threshold_m else 1.0
    height_roll_scale = max(0.3, (16000.0 - altitude_m) / 5000.0) if altitude_m > altitude_threshold_m else 1.0
    alpha_scale = 1.0 + 0.5 * (alpha_rad / alpha_limit_rad) if alpha_rad > alpha_threshold_rad else 1.0
    low_speed_scale = 1.1 if speed_mps <= 300.0 else 1.0
    return {"qbar_pa": qbar, "qbar_scale": qbar_scale, "nz_proportional_scale": qbar_scale * height_nz_scale * alpha_scale * low_speed_scale, "roll_proportional_scale": qbar_scale * height_roll_scale * alpha_scale * low_speed_scale, "altitude_schedule_active": altitude_m > altitude_threshold_m, "aoa_schedule_active": alpha_rad > alpha_threshold_rad}


def offline_arbitrate(sample: ArbitrationInput, *, altitude_reference_m: float = 5000.0, altitude_hold_gain: float = 0.002, stall_speed_mps: float = 150.0, max_bank_rad: float = pi * 55.0 / 180.0, bank_increment_max: float = 1.0) -> dict:
    """Frozen F10 priority: guidance → stall limit → bank recovery → altitude hold → final clip."""
    stall_margin = float(np.clip((sample.speed_mps - stall_speed_mps) / max(1.0, 180.0 - stall_speed_mps), 0.0, 1.0))
    nz_upper = 1.0 + stall_margin * (F04_NZ_LIMITS[1] - 1.0)
    nz = float(np.clip(sample.nz_guidance, F04_NZ_LIMITS[0], nz_upper))
    roll = float(np.clip(sample.roll_rate_guidance, -1.5, 1.5))
    flags = {"guidance": True, "stall_limited": nz_upper < F04_NZ_LIMITS[1] - 1e-9, "bank_recovery": False, "altitude_hold": False}
    bank_delta = 0.0
    if abs(sample.roll_rad) > max_bank_rad:
        overbank = abs(sample.roll_rad) - max_bank_rad
        flags["bank_recovery"] = True
        roll = -float(np.copysign(min(overbank * 5.0, 1.5), sample.roll_rad))
        bank_delta = min(overbank / (pi / 9.0), bank_increment_max)
        nz += bank_delta
    altitude_delta = altitude_hold_gain * (altitude_reference_m - sample.altitude_m)
    if flags["bank_recovery"] and altitude_delta < 0.0:
        altitude_delta = 0.0
    nz += altitude_delta
    flags["altitude_hold"] = abs(altitude_delta) > 1e-9
    nz = float(np.clip(nz, F04_NZ_LIMITS[0], nz_upper))
    return {"input": asdict(sample), "order": ["guidance", "stall_limit", "bank_recovery", "altitude_hold", "final_clip"], "nz_cmd": nz, "roll_rate_cmd": roll, "nz_upper_limit": nz_upper, "bank_delta": bank_delta, "altitude_delta": altitude_delta, "flags": flags, "gain_scheduling": qbar_gain_scales(sample.speed_mps, sample.altitude_m, sample.alpha_rad)}


def f10_sweep() -> dict:
    samples = {
        "nominal": ArbitrationInput(4.0, 0.4, 5000.0, 250.0, 0.0, 0.05),
        "low_qbar_low_speed": ArbitrationInput(7.0, 1.0, 5000.0, 140.0, 0.0, 0.05),
        "high_qbar": ArbitrationInput(5.0, 1.0, 5000.0, 350.0, 0.0, 0.05),
        "high_altitude": ArbitrationInput(5.0, 1.0, 14000.0, 250.0, 0.0, 0.05),
        "high_aoa": ArbitrationInput(5.0, 1.0, 5000.0, 250.0, 0.0, 0.4),
        "bank_and_altitude": ArbitrationInput(7.0, 1.0, 800.0, 140.0, pi * 85.0 / 180.0, 0.05),
    }
    baseline = {name: offline_arbitrate(sample) for name, sample in samples.items()}
    scheduled_candidate = {name: {**result, "candidate_name": "qbar_altitude_aoa_gain_scheduled_controller", "runtime_selected": False} for name, result in baseline.items()}
    return {"baseline_fixed_mapping": baseline, "scheduled_candidate": scheduled_candidate, "f04_limits": {"nz": list(F04_NZ_LIMITS), "roll_rate_radps": [-1.5, 1.5], "throttle": [0.4, 0.9]}, "scope": "offline arbitration and gain-scaling model; no controller default or command source changed"}


def run_experiment() -> dict:
    return {"schema_version": "1.0.0", "finding_ids": ["F17", "F10"], "f17": f17_sweep(), "f10": f10_sweep()}
