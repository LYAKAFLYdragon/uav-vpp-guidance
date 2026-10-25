#!/usr/bin/env python3
"""Collect paired local impulse responses at real handoff states.

The source trajectory is replayed from a legal reset. At its recorded first
mode change, one baseline and six symmetric one-axis action pulses are run.
This is a non-learning JSBSim collector; it never changes the controller.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.common.hash import config_sha256, file_sha256
from uav_vpp_guidance.common.provenance import record_config_override_if_changed
from uav_vpp_guidance.envs.end_to_end_opponent import EndToEndOpponent
from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V1-20260903"
DEFAULT_CONFIG = ROOT / "config/experiment/mild_disadvantage_handoff_local_response_collector_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/mild_disadvantage_handoff_local_response_v1"
ACTIVE_CONFIG_PATH = DEFAULT_CONFIG
MATCHED_TELEMETRY_RUNNER = ROOT / "scripts/run_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py"
MATCHED_PROBE_CONFIG = ROOT / "config/experiment/mild_disadvantage_envelope_recovery_matched_telemetry_v1.yaml"
CONDITIONS = ("baseline", "axis0_plus", "axis0_minus", "axis1_plus", "axis1_minus", "axis2_plus", "axis2_minus")
OFFSETS = (4, 12, 48)


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": file_sha256(path) if path.is_file() else None}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_config(config: dict[str, Any], key: str, value: Any, source: str) -> None:
    current: Any = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    old = current.get(parts[-1])
    current[parts[-1]] = value
    record_config_override_if_changed(config, key, value, old, source)


def normalise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [normalise(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalise(value[key]) for key in sorted(value)}
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def state_hash(obs: dict[str, Any]) -> str:
    payload = {key: obs.get(key, {}) for key in ("own_state", "target_state", "relative_state", "observation_vector")}
    encoded = json.dumps(normalise(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fvalue(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class H0Adapter:
    def __init__(self, agent: PPOAgent) -> None:
        self.agent = agent

    def action(self, observation: Any) -> np.ndarray:
        return np.asarray(self.agent.get_deterministic_action(np.asarray(observation, dtype=np.float64)[:16]), dtype=np.float64).reshape(3)

    def get_deterministic_action(self, observation: Any) -> np.ndarray:
        return self.action(observation)


class HandoffPulsePolicy:
    """Apply one symmetric pulse to the first nominal-policy action after handoff."""

    def __init__(self, base: Any, label: str, amplitude: float) -> None:
        self.base = base
        self.label = label
        self.amplitude = float(amplitude)
        self.calls = 0
        self.base_action: np.ndarray | None = None
        self.pulsed_action: np.ndarray | None = None
        self.pulse_axis: int | None = None
        self.pulse_sign = 0
        self.pulse_clipped = False

    def get_deterministic_action(self, observation: Any) -> np.ndarray:
        base = np.asarray(self.base.get_deterministic_action(observation), dtype=np.float64).reshape(3)
        if self.calls == 0 and self.label != "baseline":
            axis = int(self.label[4])
            sign = 1 if self.label.endswith("plus") else -1
            candidate = base + np.eye(3, dtype=np.float64)[axis] * sign * self.amplitude
            self.base_action = base.copy()
            self.pulsed_action = candidate.copy()
            self.pulse_axis = axis
            self.pulse_sign = sign
            self.pulse_clipped = bool(np.any(candidate < -1.0) or np.any(candidate > 1.0))
            if self.pulse_clipped:
                candidate = np.clip(candidate, -1.0, 1.0)
            action = candidate
        else:
            action = base
        self.calls += 1
        return action


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scenario_from_source(source: dict[str, Any]) -> dict[str, Any]:
    first = source["rows"][0]
    x = fvalue(first.get("target_point_x_m"))
    y = fvalue(first.get("target_point_y_abs_m"))
    if not finite(x):
        x = 625.0
    if not finite(y):
        y = 200.0
    return {
        "name": source["candidate"],
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
        "target_init": {"position_m": [x, y, 5000.0], "velocity_mps": 200.0, "heading_deg": 175.0},
    }


def build_opponent(name: str, config: dict[str, Any]):
    entry = config["opponent_registry"][name]
    if name == "expert":
        return ExpertOpponent(entry.get("config", {})), {"stage": "expert"}
    return EndToEndOpponent.from_registry(entry), {"stage": "end_to_end", **entry}


def prepare_config(base: dict[str, Any], spec: dict[str, Any], opponent: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    source = "run_mild_disadvantage_handoff_local_response_collector_v1.py"
    set_config(cfg, "experiment.name", spec["experiment"]["name"], f"{source}:name")
    set_config(cfg, "experiment.source_id", SOURCE_ID, f"{source}:source_id")
    set_config(cfg, "experiment.mode", "evaluation", f"{source}:mode")
    set_config(cfg, "protocol.source_id", SOURCE_ID, f"{source}:protocol_source_id")
    set_config(cfg, "protocol.opponent", opponent, f"{source}:opponent")
    set_config(cfg, "backend", "jsbsim", f"{source}:backend")
    set_config(cfg, "env.backend", "jsbsim", f"{source}:env_backend")
    set_config(cfg, "env.use_jsbsim", True, f"{source}:env_use_jsbsim")
    set_config(cfg, "env.strict_backend", True, f"{source}:strict_backend")
    set_config(cfg, "env.legacy_project_root", str(resolve("E:/uav-vpp-guidance-artifacts/CAN-20260705/assets")), f"{source}:jsbsim_root")
    set_config(cfg, "env.max_high_level_steps", 512, f"{source}:max_steps")
    set_config(cfg, "trajectory_prediction.checkpoint_path", str(resolve(spec["assets"]["crossing_predictor"])), f"{source}:predictor")
    set_config(cfg, "trajectory_prediction.allow_device_fallback", False, f"{source}:predictor_fallback")
    set_config(cfg, "observation.include_opponent_stage", True, f"{source}:opponent_stage")
    set_config(cfg, "opponent_registry.end_to_end.checkpoint", str(resolve(spec["assets"]["end_to_end_checkpoint"])), f"{source}:end_to_end_asset")
    set_config(cfg, "opponent_registry.end_to_end.device", "cpu", f"{source}:end_to_end_device")
    set_config(cfg, "opponent_registry.expert.stage", "expert", f"{source}:expert_stage")
    set_config(cfg, "opponent_registry.end_to_end.stage", "end_to_end", f"{source}:end_to_end_stage")
    return cfg


def source_traces(root: Path, opponent: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted((root / opponent / "trajectories").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        modes = [row.get("mode", "") for row in rows]
        handoff = next((i for i in range(1, len(modes)) if modes[i] != modes[i - 1]), None)
        if handoff is None:
            continue
        candidate = rows[0].get("candidate", path.stem.rsplit("_", 1)[0])
        seed = None
        result_path = root / opponent / "episode_results.json"
        if result_path.is_file():
            data = json.loads(result_path.read_text(encoding="utf-8")).get("episodes", [])
            same = [item for item in data if item.get("candidate") == candidate]
            index = int(path.stem.rsplit("_", 1)[-1]) - 1
            if 0 <= index < len(same):
                seed = same[index].get("seed")
        if seed is None:
            continue
        output.append({
            "source_trace_id": f"{opponent}/{path.name}", "opponent": opponent, "candidate": candidate,
            "path": path, "path_sha256": file_sha256(path), "rows": rows, "handoff_index": handoff,
            "handoff_hash": rows[handoff].get("state_hash_before_action", ""), "seed": int(seed),
        })
    return output


def policy_action(mode: str, obs: dict[str, Any], h0: H0Adapter, crossing: FrozenSpecialistPolicy) -> np.ndarray:
    action = crossing.get_deterministic_action(obs["observation_vector"]) if mode == "crossing" else h0.action(obs["observation_vector"])
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError(f"invalid action {action}")
    return action


def pulse_action(label: str, base: np.ndarray, amplitude: float) -> tuple[np.ndarray, int | None, int]:
    if label == "baseline":
        return base.copy(), None, 0
    axis = int(label[4])
    sign = 1 if label.endswith("plus") else -1
    return base + np.eye(3, dtype=np.float64)[axis] * sign * amplitude, axis, sign


def run_condition(cfg: dict[str, Any], source: dict[str, Any], label: str, h0: H0Adapter, crossing: FrozenSpecialistPolicy, opponent_policy: Any, opponent_config: dict[str, Any], amplitude: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
    trace: list[dict[str, Any]] = []
    reset_calls = 0
    handoff_hash = ""
    pulse_axis: int | None = None
    pulse_sign = 0
    pulse_clipped = False
    reason = "timeout"
    try:
        obs = env.reset(scenario=scenario_from_source(source), seed=source["seed"])
        reset_calls = 1
        stop = min(int(env.max_steps), source["handoff_index"] + max(OFFSETS) + 1)
        for step in range(stop):
            source_row = source["rows"][step] if step < len(source["rows"]) else source["rows"][-1]
            mode = source_row.get("mode", "h0_current_geometry")
            before = state_hash(obs)
            base = policy_action(mode, obs, h0, crossing)
            action = base.copy()
            delta = np.zeros(3, dtype=np.float64)
            if step == source["handoff_index"]:
                action, pulse_axis, pulse_sign = pulse_action(label, base, amplitude)
                delta = action - base
                pulse_clipped = bool(np.any(action < -1.0) or np.any(action > 1.0))
                handoff_hash = before
                if pulse_clipped:
                    action = np.clip(action, -1.0, 1.0)
            pre_own = obs.get("own_state", {})
            pre_target = obs.get("target_state", {})
            pre_rel = obs.get("relative_state", {})
            env.set_runtime_specialist_context(specialist_key=mode, specialist_profile="local_response", specialist_mode_name=mode)
            obs, _, terminated, truncated, info = env.step(action)
            own = info.get("own_state", {})
            rel = info.get("relative_state", {})
            velocity = np.asarray(own.get("velocity_vector_mps", [np.nan] * 3), dtype=float)
            speed = float(np.linalg.norm(velocity)) if velocity.shape == (3,) else float("nan")
            altitude = fvalue(own.get("altitude_m"))
            trace.append({
                "source_id": SOURCE_ID, "source_trace_id": source["source_trace_id"], "opponent": source["opponent"], "candidate": source["candidate"], "seed": source["seed"], "condition": label,
                "step": step, "handoff_step": source["handoff_index"], "response_offset": step - source["handoff_index"], "mode": mode,
                "state_hash_before_action": before, "source_handoff_hash": source["handoff_hash"], "handoff_replay_hash": handoff_hash,
                "action_0": float(action[0]), "action_1": float(action[1]), "action_2": float(action[2]), "baseline_action_0": float(base[0]), "baseline_action_1": float(base[1]), "baseline_action_2": float(base[2]),
                "pulse_delta_0": float(delta[0]), "pulse_delta_1": float(delta[1]), "pulse_delta_2": float(delta[2]), "pulse_axis": "" if pulse_axis is None or step != source["handoff_index"] else pulse_axis, "pulse_sign": pulse_sign if step == source["handoff_index"] else 0, "pulse_amplitude": amplitude if label != "baseline" and step == source["handoff_index"] else 0.0, "pulse_clipped": pulse_clipped if step == source["handoff_index"] else False,
                "range_m": fvalue(info.get("range_m", rel.get("range_m"))), "range_rate_mps": fvalue(info.get("range_rate_mps", rel.get("range_rate_mps"))), "ata_deg": fvalue(info.get("ata_deg")), "aa_deg": fvalue(info.get("aa_deg")), "specific_energy_m2ps2": 0.5 * speed * speed + 9.80665 * altitude, "altitude_m": altitude,
                "roll_rad": fvalue(own.get("roll_rad")), "p_rps": fvalue(own.get("p_rps")), "nz_g": fvalue(own.get("nz_g")), "alpha_rad": fvalue(own.get("alpha_rad")), "backend": info.get("backend"), "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)), "prediction_fallback": bool(info.get("prediction_fallback", False)), "terminal": bool(terminated or truncated), "terminal_reason": info.get("reason", "") or "",
                "pre_own_state": json.dumps(normalise(pre_own), ensure_ascii=False, separators=(",", ":")), "pre_target_state": json.dumps(normalise(pre_target), ensure_ascii=False, separators=(",", ":")), "pre_relative_state": json.dumps(normalise(pre_rel), ensure_ascii=False, separators=(",", ":")),
            })
            if terminated or truncated:
                reason = str(info.get("reason", "unknown"))
                break
    finally:
        backend = getattr(env, "_backend", None)
        backend_fallback = bool(getattr(env, "_backend_fallback_occurred", False))
        env.close()
    result = {
        "source_id": SOURCE_ID, "source_trace_id": source["source_trace_id"], "opponent": source["opponent"], "candidate": source["candidate"], "seed": source["seed"], "condition": label,
        "pulse_axis": pulse_axis, "pulse_sign": pulse_sign, "pulse_amplitude": 0.0 if label == "baseline" else amplitude, "pulse_clipped": pulse_clipped, "handoff_step": source["handoff_index"], "source_handoff_hash": source["handoff_hash"], "handoff_replay_hash": handoff_hash, "handoff_hash_match": handoff_hash == source["handoff_hash"], "reset_calls": reset_calls, "trace_rows": len(trace), "backend_effective": backend, "backend_fallback_occurred": backend_fallback, "terminal_reason": reason,
        "response_4_available": any(row["response_offset"] == 4 for row in trace), "response_12_available": any(row["response_offset"] == 12 for row in trace), "response_48_available": any(row["response_offset"] == 48 for row in trace), "pre_handoff_prediction_fallback": any(row["prediction_fallback"] for row in trace if row["response_offset"] < 0), "response_window_prediction_fallback": any(row["prediction_fallback"] for row in trace if row["response_offset"] >= 0),
    }
    return result, trace


def run_condition_matched(
    env: CloseRangeTrackingEnv,
    source: dict[str, Any],
    label: str,
    h0: H0Adapter,
    crossing: FrozenSpecialistPolicy,
    opponent_policy: Any,
    opponent_config: dict[str, Any],
    amplitude: float,
    matched: Any,
    legacy: Any,
    matched_spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the verified recovery chain and pulse its first H0 action."""
    stable_points = [
        legacy.EnvelopePoint(float(point[0]), float(point[1]))
        for point in matched_spec["recovery"]["stable_points_m"]
    ]
    recovery_cfg = matched_spec["recovery"]
    recovery = legacy.CapabilityEnvelopeRecoveryPolicy(
        stable_points,
        **{
            key: recovery_cfg[key]
            for key in (
                "nominal_mode",
                "recovery_mode",
                "x_tolerance_m",
                "y_tolerance_m",
                "max_recovery_steps",
                "required_dwell_steps",
                "max_clipping_rate",
                "min_altitude_m",
                "max_range_m",
            )
        },
    )
    pulse_h0 = HandoffPulsePolicy(h0, label, amplitude)
    candidate = matched_spec["probe"]["candidates"][source["candidate"]]
    scenario = legacy._make_scenario(
        source["candidate"], candidate, matched_spec["probe"]["fixed_target_conditions"]
    )
    trajectory_result, raw_trace = matched._run_episode(
        env,
        pulse_h0,
        crossing,
        recovery,
        scenario,
        int(source["seed"]),
        source["opponent"],
        source["candidate"],
        matched_spec["probe"]["calibration"],
        legacy,
    )
    handoff_rows = [row for row in raw_trace if int(row.get("step", -1)) == int(source["handoff_index"])]
    handoff_row = handoff_rows[0] if handoff_rows else {}
    pulse_step = int(source["handoff_index"])
    for row in raw_trace:
        step = int(row.get("step", -1))
        row["source_id"] = SOURCE_ID
        row["source_trace_id"] = source["source_trace_id"]
        row["seed"] = source["seed"]
        row["condition"] = label
        row["handoff_step"] = pulse_step
        row["response_offset"] = step - pulse_step
        row["source_handoff_hash"] = source["handoff_hash"]
        row["handoff_replay_hash"] = handoff_row.get("state_hash_before_action", "")
        row["handoff_hash_match"] = row["handoff_replay_hash"] == source["handoff_hash"]
        row["baseline_action_0"] = float(row.get("action_0", float("nan")))
        row["baseline_action_1"] = float(row.get("action_1", float("nan")))
        row["baseline_action_2"] = float(row.get("action_2", float("nan")))
        row["pulse_delta_0"] = 0.0
        row["pulse_delta_1"] = 0.0
        row["pulse_delta_2"] = 0.0
        row["pulse_axis"] = ""
        row["pulse_sign"] = 0
        row["pulse_amplitude"] = 0.0
        row["pulse_clipped"] = False
        if step == pulse_step and label != "baseline":
            if pulse_h0.base_action is None or pulse_h0.pulsed_action is None:
                raise RuntimeError(f"{source['source_trace_id']}/{label}: handoff did not invoke H0 pulse")
            row["baseline_action_0"] = float(pulse_h0.base_action[0])
            row["baseline_action_1"] = float(pulse_h0.base_action[1])
            row["baseline_action_2"] = float(pulse_h0.base_action[2])
            row["pulse_delta_0"] = float(row["action_0"]) - row["baseline_action_0"]
            row["pulse_delta_1"] = float(row["action_1"]) - row["baseline_action_1"]
            row["pulse_delta_2"] = float(row["action_2"]) - row["baseline_action_2"]
            row["pulse_axis"] = pulse_h0.pulse_axis
            row["pulse_sign"] = pulse_h0.pulse_sign
            row["pulse_amplitude"] = amplitude
            row["pulse_clipped"] = pulse_h0.pulse_clipped
        own_speed = fvalue(row.get("own_speed_mps"))
        altitude = fvalue(row.get("altitude_m"))
        row["specific_energy_m2ps2"] = 0.5 * own_speed * own_speed + 9.80665 * altitude
        row["range_m"] = fvalue(row.get("range_m"))
        row["range_rate_mps"] = fvalue(row.get("range_rate_mps"))
        row["ata_deg"] = fvalue(row.get("ata_deg"))
        row["aa_deg"] = fvalue(row.get("aa_deg"))
        row["roll_rad"] = fvalue(row.get("roll_rad"))
        row["p_rps"] = fvalue(row.get("p_rps"))
        row["nz_g"] = fvalue(row.get("nz_g"))
        row["alpha_rad"] = fvalue(row.get("alpha_rad"))
        row["backend"] = row.get("backend")
        row["backend_fallback_occurred"] = bool(row.get("backend_fallback_occurred", False))
        row["prediction_fallback"] = bool(row.get("prediction_fallback", False))
        row["terminal"] = bool(row.get("terminal", False))
        row["terminal_reason"] = row.get("terminal_reason", "") or ""
    result = dict(trajectory_result)
    result.update({
        "source_id": SOURCE_ID,
        "source_trace_id": source["source_trace_id"],
        "candidate": source["candidate"],
        "opponent": source["opponent"],
        "seed": source["seed"],
        "condition": label,
        "pulse_axis": pulse_h0.pulse_axis,
        "pulse_sign": pulse_h0.pulse_sign,
        "pulse_amplitude": 0.0 if label == "baseline" else amplitude,
        "pulse_clipped": pulse_h0.pulse_clipped,
        "handoff_step": pulse_step,
        "source_handoff_hash": source["handoff_hash"],
        "handoff_replay_hash": handoff_row.get("state_hash_before_action", ""),
        "handoff_hash_match": handoff_row.get("state_hash_before_action", "") == source["handoff_hash"],
        "response_4_available": any(int(row.get("response_offset", -1)) == 4 for row in raw_trace),
        "response_12_available": any(int(row.get("response_offset", -1)) == 12 for row in raw_trace),
        "response_48_available": any(int(row.get("response_offset", -1)) == 48 for row in raw_trace),
        "pre_handoff_prediction_fallback": any(bool(row.get("prediction_fallback", False)) for row in raw_trace if int(row.get("response_offset", -1)) < 0),
        "response_window_prediction_fallback": any(bool(row.get("prediction_fallback", False)) for row in raw_trace if 0 <= int(row.get("response_offset", -1)) <= 48),
    })
    return result, raw_trace


def response_records(results: list[dict[str, Any]], traces: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    metrics = ("range_m", "range_rate_mps", "ata_deg", "aa_deg", "specific_energy_m2ps2", "altitude_m")
    output: list[dict[str, Any]] = []
    for result in results:
        trace = {int(row["response_offset"]): row for row in traces[(result["source_trace_id"], result["condition"])]}
        for offset in OFFSETS:
            sample = trace.get(offset)
            row = {key: result.get(key) for key in ("source_id", "source_trace_id", "opponent", "candidate", "seed", "condition", "pulse_axis", "pulse_sign", "pulse_amplitude", "handoff_hash_match", "backend_effective", "backend_fallback_occurred", "response_window_prediction_fallback")}
            row["response_offset"] = offset
            row["response_available"] = sample is not None
            for metric in metrics:
                row[metric] = sample.get(metric) if sample else float("nan")
            output.append(row)
    return output


def run(
    spec: dict[str, Any],
    output_root: Path,
    limit_sources: int | None = None,
    selected_conditions: tuple[str, ...] = CONDITIONS,
    source_trace_id: str | None = None,
) -> dict[str, Any]:
    source_root = resolve(spec["source_traces_root"])
    base = load(resolve(spec["base_config"]))
    matched = load_module(MATCHED_TELEMETRY_RUNNER, "mild_disadvantage_matched_telemetry_helpers")
    matched_probe = matched._load_module(matched.MATCHED_RUNNER, "mild_disadvantage_matched_probe_helpers")
    legacy = matched._load_module(matched_probe.LEGACY_RUNNER, "mild_disadvantage_recovery_probe_helpers")
    matched_spec = load(MATCHED_PROBE_CONFIG)
    sources = [item for opponent in spec["protocol"]["opponents"] for item in source_traces(source_root, opponent)]
    if len(sources) != int(spec["protocol"]["required_source_traces"]):
        raise RuntimeError(f"expected {spec['protocol']['required_source_traces']} real handoff traces, found {len(sources)}")
    if source_trace_id is not None:
        sources = [item for item in sources if item["source_trace_id"] == source_trace_id]
        if len(sources) != 1:
            raise RuntimeError(f"expected exactly one source trace for {source_trace_id}, found {len(sources)}")
    if limit_sources is not None:
        sources = sources[:limit_sources]
    print(f"Loaded {len(sources)} real handoff traces", flush=True)
    results: list[dict[str, Any]] = []
    traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifests: list[dict[str, Any]] = []
    for opponent in spec["protocol"]["opponents"]:
        print(f"Preparing opponent {opponent}", flush=True)
        cfg = matched._prepare_config(base, matched_spec, opponent, matched_probe)
        set_config(cfg, "experiment.name", spec["experiment"]["name"], "run_mild_disadvantage_handoff_local_response_collector_v1.py:name")
        set_config(cfg, "experiment.source_id", SOURCE_ID, "run_mild_disadvantage_handoff_local_response_collector_v1.py:source_id")
        set_config(cfg, "protocol.source_id", SOURCE_ID, "run_mild_disadvantage_handoff_local_response_collector_v1.py:protocol_source_id")
        set_config(cfg, "protocol.opponent", opponent, "run_mild_disadvantage_handoff_local_response_collector_v1.py:protocol_opponent")
        set_config(cfg, "env.legacy_project_root", str(resolve("E:/uav-vpp-guidance-artifacts/CAN-20260705/assets")), "run_mild_disadvantage_handoff_local_response_collector_v1.py:jsbsim_root")
        set_config(cfg, "trajectory_prediction.allow_device_fallback", False, "run_mild_disadvantage_handoff_local_response_collector_v1.py:predictor_fallback")
        print(f"Preparing isolated conditions for {opponent}", flush=True)
        resolved = output_root / opponent / "resolved_config.yaml"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        own_sources = [item for item in sources if item["opponent"] == opponent]
        if source_trace_id is not None and own_sources:
            set_config(
                cfg,
                "env.max_high_level_steps",
                int(own_sources[0]["handoff_index"]) + max(OFFSETS) + 1,
                "run_mild_disadvantage_handoff_local_response_collector_v1.py:isolated_response_window",
            )
        for source in own_sources:
            print(f"Replaying {source['source_trace_id']} at handoff {source['handoff_index']}", flush=True)
            for label in selected_conditions:
                # Recreate all stateful runtime objects for every paired condition.
                # This prevents controller, predictor, FCS, or opponent state from
                # leaking from one pulse condition into the next.
                opponent_policy, opponent_config = build_opponent(opponent, cfg)
                h0_agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
                h0_agent.load(str(resolve(spec["assets"]["h0_checkpoint"])))
                h0 = H0Adapter(h0_agent)
                crossing = FrozenSpecialistPolicy(
                    str(resolve(spec["assets"]["crossing_checkpoint"])),
                    str(resolve(spec["assets"]["crossing_config"])),
                    device="cpu",
                )
                env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
                try:
                    result, trace = run_condition_matched(
                        env, source, label, h0, crossing, opponent_policy, opponent_config,
                        float(spec["protocol"]["pulse_amplitude"]), matched, legacy, matched_spec,
                    )
                    results.append(result)
                    traces[(source["source_trace_id"], label)] = trace
                    filename = source["source_trace_id"].replace("/", "__").replace(".csv", "") + "__" + label + ".csv"
                    write_csv(output_root / opponent / "traces" / filename, trace)
                finally:
                    env.close()
        manifests.append({"source_id": SOURCE_ID, "opponent": opponent, "source_trace_count": len(own_sources), "condition_count": len(own_sources) * len(selected_conditions), "resolved_config": receipt(resolved), "backend_requested": "jsbsim", "strict_backend": True})
    write_csv(output_root / "condition_results.csv", results)
    write_csv(output_root / "local_response_records.csv", response_records(results, traces))
    manifest = {"source_id": SOURCE_ID, "status": "completed", "training_started": False, "training_permitted": False, "jsbsim_started": True, "compiler_runtime_probe_started": False, "source_trace_count": len(sources), "condition_count": len(results), "expected_condition_count": len(sources) * len(selected_conditions), "pulse_conditions": list(selected_conditions), "pulse_amplitude": float(spec["protocol"]["pulse_amplitude"]), "response_offsets": list(OFFSETS), "opponents": list(spec["protocol"]["opponents"]), "source_traces_root": str(source_root), "base_config": receipt(resolve(spec["base_config"])), "collector_config": receipt(ACTIVE_CONFIG_PATH), "run_manifests": manifests, "source_trace_manifest": [{key: value for key, value in item.items() if key not in {"rows", "path"}} | {"path": str(item["path"])} for item in sources], "git_info": get_git_info(str(ROOT)), "config_canonical_sha256": config_sha256(spec, truncate=0), "contract": {"real_handoff_source": True, "replay_from_legal_reset": True, "same_state_hash_required": True, "symmetric_three_axis_pulses": True, "response_offsets": list(OFFSETS), "pre_handoff_prediction_warmup_allowed": True, "response_window_prediction_fallback_forbidden": True}}
    write_json(output_root / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    global SOURCE_ID, ACTIVE_CONFIG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-traces-root", type=Path, default=None)
    parser.add_argument("--source-trace-id", type=str, default=None)
    parser.add_argument("--execute", action="store_true", help="Required to run JSBSim")
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    spec = load(args.config.resolve())
    configured_source_id = str(spec["experiment"]["source_id"])
    if not configured_source_id:
        raise SystemExit("Source ID is empty")
    SOURCE_ID = configured_source_id
    ACTIVE_CONFIG_PATH = args.config.resolve()
    if args.source_traces_root is not None:
        spec["source_traces_root"] = str(args.source_traces_root.resolve())
    if not args.execute:
        print(f"LOCKED: no JSBSim execution. Planned Source ID: {SOURCE_ID}")
        return 0
    root = args.output_root.resolve()
    if root.exists() and any(path.is_file() for path in root.rglob("*")):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {root}")
    manifest = run(
        spec,
        root,
        limit_sources=args.limit_sources,
        selected_conditions=tuple(args.conditions),
        source_trace_id=args.source_trace_id,
    )
    print(json.dumps({"source_id": SOURCE_ID, "condition_count": manifest["condition_count"], "output_root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
