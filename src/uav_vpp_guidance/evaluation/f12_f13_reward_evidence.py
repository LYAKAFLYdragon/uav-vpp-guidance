"""Pure offline F12/F13 reward decomposition evidence; no reward runtime import."""
from __future__ import annotations
from copy import deepcopy

DEFAULT_REWARD = {"w_range": 0.6, "w_angle": 0.9, "w_energy": 0.0, "w_safety": 3.0, "w_saturation": 0.5, "w_smooth": 0.2, "w_turn_rate": 0.5, "w_closing": 0.1, "w_alive": 0.02, "terminal_success": 400.0, "terminal_failure": -300.0, "terminal_crash": -600.0}


def _step(sample: dict, weights: dict, previous: dict | None) -> dict:
    range_m, altitude, rate = sample["range_m"], sample["altitude_m"], sample["range_rate_mps"]
    command = sample["command"]
    range_reward = weights["w_range"] * 0.5 if 700 <= range_m <= 1100 else -weights["w_range"] * min(1.0, abs(range_m - 900) / 3100)
    angle = -weights["w_angle"] * sample["angle_error"]
    safety = -weights["w_safety"] * max(0.0, 1.0 - (altitude - 500.0) / 5000.0) if altitude < 1500 else 0.0
    saturation = -weights["w_saturation"] * (max(0.0, abs(command["nz_cmd"]) - 6.5) / 7.0 + max(0.0, abs(command["roll_rate_cmd"]) - 1.4) / 1.5)
    smooth = 0.0 if previous is None else -weights["w_smooth"] * sum(abs(command[k] - previous[k]) for k in command)
    closing = weights["w_closing"] * (-rate / 200.0)
    terms = {"reward_range": range_reward, "reward_angle": angle, "reward_safety": safety, "reward_saturation": saturation, "reward_smooth": smooth, "reward_closing": closing, "reward_alive": weights["w_alive"], "terminal_reward": sample.get("terminal_reward", 0.0)}
    terms["reward_total"] = sum(terms.values())
    return terms


def decompose_episode(samples: list[dict], weights: dict) -> dict:
    prior, rows = None, []
    for sample in samples:
        rows.append(_step(sample, weights, prior)); prior = sample["command"]
    aggregate = {name: sum(row[name] for row in rows) for name in rows[0]}
    safety_occupancy = sum(sample["altitude_m"] < 1500 for sample in samples) / len(samples)
    saturation_occupancy = sum(abs(s["command"]["nz_cmd"]) > 6.5 or abs(s["command"]["roll_rate_cmd"]) > 1.4 for s in samples) / len(samples)
    return {"per_step": rows, "aggregate": aggregate, "safety_trigger_occupancy": safety_occupancy, "saturation_occupancy": saturation_occupancy}


def run_experiment() -> dict:
    samples = [{"range_m": 2200.0, "altitude_m": 5000.0, "range_rate_mps": -180.0, "angle_error": 0.4, "command": {"nz_cmd": 2.0, "roll_rate_cmd": 0.3, "throttle_cmd": 0.7}}, {"range_m": 1000.0, "altitude_m": 1200.0, "range_rate_mps": -220.0, "angle_error": 0.1, "command": {"nz_cmd": 7.0, "roll_rate_cmd": 1.5, "throttle_cmd": 0.9}}, {"range_m": 850.0, "altitude_m": 800.0, "range_rate_mps": -120.0, "angle_error": 0.05, "command": {"nz_cmd": 4.0, "roll_rate_cmd": 0.4, "throttle_cmd": 0.7}, "terminal_reward": 400.0}]
    counterfactual = deepcopy(DEFAULT_REWARD); counterfactual.update({"w_alive": 0.0, "w_closing": 0.0})
    baseline, ablated = decompose_episode(samples, DEFAULT_REWARD), decompose_episode(samples, counterfactual)
    return {"samples": "deterministic synthetic trace; not policy or JSBSim evaluation", "default": baseline, "counterfactual_disable_alive_and_closing": ablated, "anti_exploitation_checks": {"terminal_reward_separate_from_step_terms": True, "low_altitude_safety_occupancy_reported": True, "command_saturation_occupancy_reported": True, "no_default_reward_weight_changed": True}}
