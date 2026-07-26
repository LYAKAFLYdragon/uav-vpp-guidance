"""Pure offline F03/F11 filter, actuator, and bank-compensation evidence."""
from dataclasses import dataclass
from math import pi, sin
from typing import Iterable
import numpy as np

F04_NZ_LIMITS = (-2.0, 7.0)

@dataclass(frozen=True)
class FilterStage:
    name: str
    alpha: float
    enabled: bool = True

def frozen_topologies(guidance: dict) -> dict:
    params, gains = guidance.get("params", {}), guidance.get("gains", {})
    alpha = float(gains.get("alpha_filter", 0.3))
    return {"simple": [FilterStage("environment_command_filter", alpha)], "jsbsim_pn": [FilterStage("pn_los_rate_filter", float(params.get("los_rate_filter_alpha", 0.3))), FilterStage("environment_command_filter", alpha), FilterStage("low_level_command_filter", alpha)], "disabled": {"los_internal_filter": not bool(params.get("enable_internal_filter", False)), "command_post_processor": not bool(guidance.get("post_process", {}).get("enabled", False)), "lift_compensation": not bool(guidance.get("post_process", {}).get("enable_lift_compensation", False))}}

def _cascade(samples: Iterable[float], stages: list[FilterStage], *, slew_per_s: float | None = None, dt_s: float = 0.2) -> tuple[np.ndarray, list[np.ndarray]]:
    values = np.asarray(list(samples), dtype=float); states = [None] * len(stages); outputs = [np.empty_like(values) for _ in stages]; previous = float(values[0])
    for index, raw in enumerate(values):
        value = float(raw)
        for stage_index, stage in enumerate(stages):
            if stage.enabled:
                states[stage_index] = value if states[stage_index] is None else stage.alpha * value + (1.0 - stage.alpha) * states[stage_index]
                value = states[stage_index]
            outputs[stage_index][index] = value
        if slew_per_s is not None:
            value = float(np.clip(value, previous - slew_per_s * dt_s, previous + slew_per_s * dt_s)); previous = value
        values[index] = value
    return values, outputs

def _frequency_metrics(stages: list[FilterStage], frequency_hz: float, *, slew_per_s: float | None = None, dt_s: float = 0.2) -> dict:
    time = np.arange(0.0, 40.0, dt_s); input_signal = np.sin(2.0 * pi * frequency_hz * time); output, stage_outputs = _cascade(input_signal, stages, slew_per_s=slew_per_s, dt_s=dt_s); mask = time >= 20.0; basis = np.column_stack((np.sin(2*pi*frequency_hz*time[mask]), np.cos(2*pi*frequency_hz*time[mask]))); coeff = np.linalg.lstsq(basis, output[mask], rcond=None)[0]; phase = float(np.arctan2(coeff[1], coeff[0])); delay = float((-phase / (2*pi*frequency_hz)) % (1.0/frequency_hz)); return {"frequency_hz": frequency_hz, "amplitude_ratio": float(np.hypot(*coeff)), "phase_lag_rad": float(-phase), "apparent_delay_s": delay, "per_stage_final": {stage.name: float(series[-1]) for stage, series in zip(stages, stage_outputs)}}

def analyse_chain(stages: list[FilterStage], *, slew_per_s: float | None = None, bank_compensation: bool = False) -> dict:
    raw = np.r_[np.ones(5), np.full(35, 7.0)]; multiplier = 1.0 / np.cos(np.deg2rad(45.0)) if bank_compensation else 1.0; raw = np.clip(raw * multiplier, *F04_NZ_LIMITS); output, stage_outputs = _cascade(raw, stages, slew_per_s=slew_per_s); slew = np.diff(output) / 0.2; return {"stages": [stage.__dict__ for stage in stages], "step": {"peak_command": float(np.max(output)), "saturation_count": int(np.count_nonzero(np.isclose(output, F04_NZ_LIMITS[1]))), "max_slew_per_s": float(np.max(np.abs(slew))), "per_stage_final": {stage.name: float(series[-1]) for stage, series in zip(stages, stage_outputs)}}, "spectrum": [_frequency_metrics(stages, f, slew_per_s=slew_per_s) for f in (0.1, 0.25, 0.5, 1.0, 2.0)], "bank_compensation": bank_compensation, "slew_per_s": slew_per_s}

def run_experiment(guidance: dict) -> dict:
    topology = frozen_topologies(guidance); baseline = topology["jsbsim_pn"]; candidates = {"pn_filter_tuning_alpha_0_5": ([FilterStage("pn_los_rate_filter", 0.5), *baseline[1:]], None, False), "filter_consolidation_remove_low_level_stage": (baseline[:-1], None, False), "actuator_slew_limit_2_per_s": (baseline, 2.0, False), "bank_compensation_at_45_deg": (baseline, None, True)}; return {"frozen_topology": topology, "baseline": analyse_chain(baseline), "candidates": {name: analyse_chain(stages, slew_per_s=slew, bank_compensation=bank) for name, (stages, slew, bank) in candidates.items()}, "f04_limits": {"nz": list(F04_NZ_LIMITS), "roll_rate_radps": [-1.5, 1.5], "throttle": [0.4, 0.9]}, "scope": "offline command-chain model; no JSBSim plant, actuator dynamics, scenario matrix, or runtime defaults changed"}
