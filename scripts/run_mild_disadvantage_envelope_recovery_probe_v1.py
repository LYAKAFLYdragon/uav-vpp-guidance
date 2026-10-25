#!/usr/bin/env python
"""Run the one-shot non-learning JSBSim envelope-recovery probe."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.common.hash import config_sha256, file_sha256
from uav_vpp_guidance.common.provenance import record_config_override_if_changed
from uav_vpp_guidance.envs.end_to_end_opponent import EndToEndOpponent
from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.capability_envelope_recovery import (
    CapabilityEnvelopeRecoveryPolicy,
    EnvelopePoint,
)


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-PHYSICAL-PROBE-V1-20260903"
DEFAULT_CONFIG = REPO_ROOT / "config/experiment/mild_disadvantage_envelope_recovery_probe_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/mild_disadvantage_envelope_recovery_v1/physical_probe_v1"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _receipt(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": file_sha256(path) if exists else None,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _set_nested(config: dict[str, Any], key: str, value: Any, source: str) -> None:
    current: Any = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    old = current.get(parts[-1])
    current[parts[-1]] = value
    record_config_override_if_changed(config, key, value, old, source)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _heading_from_velocity(state: dict[str, Any]) -> float:
    velocity = _vector(state.get("velocity_vector_mps", [np.nan, np.nan, 0.0]))
    if velocity.size < 2 or not np.isfinite(velocity[:2]).all():
        raise ValueError("own velocity is not finite enough to define body frame")
    return float(math.atan2(velocity[1], velocity[0]))


def _body_relative_position(own_state: dict[str, Any], target_state: dict[str, Any]) -> tuple[float, float]:
    own = _vector(own_state["position_m"])
    target = _vector(target_state["position_m"])
    if own.size < 2 or target.size < 2 or not np.isfinite(np.r_[own[:2], target[:2]]).all():
        raise ValueError("position is not finite enough to define body frame")
    delta = target[:2] - own[:2]
    heading = _heading_from_velocity(own_state)
    c, s = math.cos(heading), math.sin(heading)
    return float(c * delta[0] + s * delta[1]), float(-s * delta[0] + c * delta[1])


def _speed(state: dict[str, Any]) -> float:
    return float(np.linalg.norm(_vector(state["velocity_vector_mps"])))


def _make_scenario(name: str, point: dict[str, float], fixed: dict[str, float]) -> dict[str, Any]:
    return {
        "name": name,
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [point["target_x_m"], point["target_y_m"], fixed["target_z_m"]],
            "velocity_mps": fixed["target_speed_mps"],
            "heading_deg": fixed["target_heading_deg"],
        },
    }


def _prepare_config(base: dict[str, Any], spec: dict[str, Any], opponent: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    source = "run_mild_disadvantage_envelope_recovery_probe_v1.py"
    _set_nested(cfg, "experiment.name", spec["experiment"]["name"], f"{source}:name")
    _set_nested(cfg, "experiment.source_id", SOURCE_ID, f"{source}:source_id")
    _set_nested(cfg, "experiment.mode", "evaluation", f"{source}:mode")
    _set_nested(cfg, "backend", "jsbsim", f"{source}:backend")
    _set_nested(cfg, "env.backend", "jsbsim", f"{source}:env_backend")
    _set_nested(cfg, "env.use_jsbsim", True, f"{source}:env_use_jsbsim")
    _set_nested(cfg, "env.strict_backend", True, f"{source}:strict_backend")
    _set_nested(cfg, "env.max_high_level_steps", int(spec["probe"]["max_high_level_steps"]), f"{source}:max_steps")
    _set_nested(cfg, "observation.include_opponent_stage", True, f"{source}:observation_contract")
    end_to_end = _resolve(spec["assets"]["end_to_end_checkpoint"])
    _set_nested(cfg, "opponent_registry.end_to_end.checkpoint", str(end_to_end), f"{source}:end_to_end_asset")
    _set_nested(cfg, "opponent_registry.end_to_end.device", "cpu", f"{source}:end_to_end_device")
    _set_nested(cfg, "opponent_registry.expert.stage", "expert", f"{source}:expert_stage")
    _set_nested(cfg, "opponent_registry.end_to_end.stage", "end_to_end", f"{source}:end_to_end_stage")
    scenarios = {
        name: _make_scenario(name, point, spec["probe"]["fixed_target_conditions"])
        for name, point in spec["probe"]["candidates"].items()
    }
    _set_nested(cfg, "scenarios", scenarios, f"{source}:candidate_scenarios")
    _set_nested(cfg, "protocol.source_id", SOURCE_ID, f"{source}:protocol_source_id")
    _set_nested(cfg, "protocol.opponent", opponent, f"{source}:protocol_opponent")
    return cfg


class _TruncateTaskFlagAdapter:
    """Expose the first 16 physical features to the 16-D H0 checkpoint."""

    adapter_name = "truncate_task_flag_suffix_only"

    def __init__(self, agent: PPOAgent):
        self.agent = agent

    def get_deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.size < 16:
            raise ValueError(f"H0 observation unexpectedly has {vector.size} dimensions")
        action = self.agent.get_deterministic_action(vector[:16])
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("H0 produced an invalid 3-D action")
        return action


def _state_from_snapshot(
    own_state: dict[str, Any],
    target_state: dict[str, Any],
    info: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    x_body, y_body = _body_relative_position(own_state, target_state)
    rel = info.get("relative_state", {})
    own_alt = float(_vector(own_state["position_m"])[2])
    own_speed = _speed(own_state)
    target_speed = _speed(target_state)
    own_ok = calibration["own_speed_mps"][0] <= own_speed <= calibration["own_speed_mps"][1]
    target_ok = calibration["target_speed_mps"][0] <= target_speed <= calibration["target_speed_mps"][1]
    clipping_rate = float(np.mean([
        bool(info.get("nz_saturated", False)),
        bool(info.get("roll_rate_saturated", False)),
        bool(info.get("throttle_saturated", False)),
    ]))
    return {
        "target_x_m": x_body,
        "target_y_m": y_body,
        "altitude_m": own_alt,
        "range_m": float(rel.get("range_m", np.linalg.norm(_vector(target_state["position_m"]) - _vector(own_state["position_m"])))),
        "clipping_rate": clipping_rate,
        "envelope_condition_valid": bool(own_ok and target_ok),
        "envelope_condition_detail": "body_frame_from_own_velocity_and_speed_window",
        "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
        "terminal": bool(info.get("terminated", False) or info.get("truncated", False)),
    }


def _initial_info(env: CloseRangeTrackingEnv) -> dict[str, Any]:
    own, target = env._get_current_states(noisy=False)
    return {
        "own_state": own,
        "target_state": target,
        "relative_state": {},
        "backend": getattr(env, "_backend", None),
        "backend_fallback_occurred": bool(getattr(env, "_backend_fallback_occurred", False)),
        "nz_saturated": False,
        "roll_rate_saturated": False,
        "throttle_saturated": False,
    }


def _run_episode(
    env: CloseRangeTrackingEnv,
    h0_policy: _TruncateTaskFlagAdapter,
    crossing_policy: Any,
    recovery: CapabilityEnvelopeRecoveryPolicy,
    scenario: dict[str, Any],
    seed: int,
    opponent: str,
    candidate: str,
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs = env.reset(scenario=scenario, seed=seed)
    reset_calls = 1
    recovery.reset()
    initial_info = _initial_info(env)
    state = _state_from_snapshot(initial_info["own_state"], initial_info["target_state"], initial_info, calibration)
    trajectory: list[dict[str, Any]] = []
    entering_step = None
    first_recovery_step = None
    max_dwell = 0
    recover_steps = 0
    hold_steps = 0
    crossing_steps = 0
    h0_steps = 0
    min_range = float("inf")
    min_ata = float("inf")
    reason = "timeout"
    info: dict[str, Any] = initial_info

    for step in range(int(env.max_steps)):
        decision = recovery.decide(state)
        if decision.phase in {"recover", "hold", "complete"} and first_recovery_step is None:
            first_recovery_step = step
        if decision.phase in {"hold", "complete"} and entering_step is None:
            entering_step = step
        recover_steps += int(decision.phase == "recover")
        hold_steps += int(decision.phase == "hold")
        max_dwell = max(max_dwell, int(decision.dwell_steps))
        mode = decision.mode
        if mode == "crossing":
            action = crossing_policy.get_deterministic_action(obs["observation_vector"])
            crossing_steps += 1
            adapter_name = "exact_18d"
        else:
            action = h0_policy.get_deterministic_action(obs["observation_vector"])
            h0_steps += 1
            adapter_name = "truncate_task_flag_suffix_only"
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError(f"invalid action at step {step}: {action}")
        env.set_runtime_specialist_context(
            specialist_key=mode,
            specialist_profile="envelope_recovery" if mode == "crossing" else "h0_nominal",
            specialist_mode_name=mode,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        own = info.get("own_state", {})
        target = info.get("target_state", {})
        rel = info.get("relative_state", {})
        state = _state_from_snapshot(own, target, {**info, "terminated": terminated, "truncated": truncated}, calibration)
        range_m = float(rel.get("range_m", np.nan))
        ata_deg = float(np.rad2deg(rel.get("ata_rad", np.nan)))
        min_range = min(min_range, range_m)
        min_ata = min(min_ata, ata_deg)
        target_point = decision.target
        trajectory.append({
            "step": step,
            "time_s": step * float(env.env_config.get("high_level_dt", 0.2)),
            "candidate": candidate,
            "opponent": opponent,
            "mode": mode,
            "adapter": adapter_name,
            "recovery_phase": decision.phase,
            "recovery_reason": decision.reason,
            "target_point_x_m": target_point.target_x_m if target_point else np.nan,
            "target_point_y_abs_m": target_point.target_y_abs_m if target_point else np.nan,
            "body_target_x_m": state["target_x_m"],
            "body_target_y_m": state["target_y_m"],
            "range_m": range_m,
            "range_rate_mps": float(rel.get("range_rate_mps", np.nan)),
            "ata_deg": ata_deg,
            "aa_deg": float(np.rad2deg(rel.get("aa_rad", np.nan))),
            "altitude_m": state["altitude_m"],
            "own_speed_mps": _speed(own),
            "target_speed_mps": _speed(target),
            "clipping_rate": state["clipping_rate"],
            "action_0": float(action[0]),
            "action_1": float(action[1]),
            "action_2": float(action[2]),
            "backend": info.get("backend"),
            "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
            "mode_switch_effective": bool(info.get("mode_switch_effective", False)),
            "effective_guidance_mode": info.get("effective_guidance_mode", "unknown"),
            "terminal": bool(terminated or truncated),
            "terminal_reason": info.get("reason", ""),
        })
        if terminated or truncated:
            reason = str(info.get("reason", "unknown"))
            break

    backend_values = {str(row["backend"]) for row in trajectory}
    fallback = any(bool(row["backend_fallback_occurred"]) for row in trajectory)
    finite_telemetry = bool(trajectory) and all(
        _finite(row[key])
        for row in trajectory
        for key in ("range_m", "range_rate_mps", "ata_deg", "aa_deg", "altitude_m", "action_0", "action_1", "action_2")
    )
    result = {
        "candidate": candidate,
        "target_x_m": scenario["target_init"]["position_m"][0],
        "target_y_m": scenario["target_init"]["position_m"][1],
        "opponent": opponent,
        "seed": seed,
        "method": "h0_with_bounded_crossing_recovery",
        "reset_calls": reset_calls,
        "length": len(trajectory),
        "reason": reason,
        "is_success": reason == "success",
        "is_crash": reason == "crash",
        "is_out_of_bounds": reason == "out_of_bounds",
        "is_timeout": reason == "timeout",
        "entered_stable_envelope": entering_step is not None,
        "first_recovery_step": first_recovery_step,
        "entered_envelope_step": entering_step,
        "recovery_steps": recover_steps,
        "hold_steps": hold_steps,
        "max_dwell_steps": max_dwell,
        "crossing_steps": crossing_steps,
        "h0_steps": h0_steps,
        "recovery_budget_ok": recover_steps <= recovery.max_recovery_steps,
        "dwell_gate_ok": max_dwell >= recovery.required_dwell_steps if entering_step is not None else False,
        "min_range_m": min_range,
        "min_ata_deg": min_ata,
        "backend_values": sorted(backend_values),
        "backend_effective": getattr(env, "_backend", None),
        "backend_fallback_occurred": fallback,
        "finite_telemetry": finite_telemetry,
        "training_started": False,
        "fallback_or_reset_or_padding": bool(fallback or reset_calls != 1),
    }
    return result, trajectory


def _build_opponent(name: str, cfg: dict[str, Any]):
    entry = cfg["opponent_registry"][name]
    if name == "expert":
        return ExpertOpponent(entry.get("config", {})), {"stage": "expert"}
    return EndToEndOpponent.from_registry(entry), {"stage": "end_to_end", **entry}


def _run_opponent(spec: dict[str, Any], base: dict[str, Any], opponent: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = _prepare_config(base, spec, opponent)
    root.mkdir(parents=True, exist_ok=True)
    resolved = root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    h0_path = _resolve(spec["assets"]["h0_checkpoint"])
    crossing_path = _resolve(spec["assets"]["crossing_checkpoint"])
    crossing_config = _resolve(spec["assets"]["crossing_config"])
    for path in (h0_path, crossing_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    opponent_policy, opponent_config = _build_opponent(opponent, cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
    try:
        sample = env.reset(seed=1)
        h0_agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
        h0_load = h0_agent.load(str(h0_path))
        from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy
        crossing_policy = FrozenSpecialistPolicy(str(crossing_path), str(crossing_config), device="cpu")
        h0_policy = _TruncateTaskFlagAdapter(h0_agent)
        stable_points = [EnvelopePoint(float(p[0]), float(p[1])) for p in spec["recovery"]["stable_points_m"]]
        recovery_cfg = spec["recovery"]
        rows: list[dict[str, Any]] = []
        all_trajectories: list[tuple[str, int, list[dict[str, Any]]]] = []
        candidates = spec["probe"]["candidates"]
        repeats = int(spec["probe"]["repetitions_per_candidate_and_opponent"])
        seed_base = int(spec["probe"]["evaluation_seed_base"])
        calibration = spec["probe"]["calibration"]
        for candidate_index, (candidate, point) in enumerate(candidates.items()):
            scenario = _make_scenario(candidate, point, spec["probe"]["fixed_target_conditions"])
            for repetition in range(repeats):
                seed = seed_base + candidate_index * 100 + (0 if opponent == "expert" else 50) + repetition
                recovery = CapabilityEnvelopeRecoveryPolicy(stable_points, **{
                    key: recovery_cfg[key]
                    for key in ("nominal_mode", "recovery_mode", "x_tolerance_m", "y_tolerance_m", "max_recovery_steps", "required_dwell_steps", "max_clipping_rate", "min_altitude_m", "max_range_m")
                })
                result, trajectory = _run_episode(env, h0_policy, crossing_policy, recovery, scenario, seed, opponent, candidate, calibration)
                rows.append(result)
                all_trajectories.append((candidate, repetition, trajectory))
                _write_csv(root / "trajectories" / f"{candidate}_{repetition + 1:02d}.csv", trajectory)
        run_manifest = {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "status": "completed",
            "training_started": False,
            "training_permitted": False,
            "backend_requested": "jsbsim",
            "backend_effective": getattr(env, "_backend", None),
            "backend_fallback_occurred": bool(getattr(env, "_backend_fallback_occurred", False)),
            "observation_runtime_dim": int(sample["observation_vector"].shape[0]),
            "h0_checkpoint": _receipt(h0_path),
            "h0_checkpoint_obs_dim": int(h0_load.get("obs_dim", 0)),
            "h0_checkpoint_action_dim": int(h0_load.get("action_dim", 0)),
            "crossing_checkpoint": _receipt(crossing_path),
            "crossing_config": _receipt(crossing_config),
            "crossing_checkpoint_obs_dim": crossing_policy.obs_dim,
            "crossing_checkpoint_action_dim": crossing_policy.action_dim,
            "end_to_end_checkpoint": _receipt(_resolve(spec["assets"]["end_to_end_checkpoint"])),
            "resolved_config": _receipt(resolved),
            "config_sha256": file_sha256(resolved),
            "config_canonical_sha256": config_sha256(cfg, truncate=0),
            "git_info": get_git_info(str(REPO_ROOT)),
            "episode_count": len(rows),
            "reset_calls_total": sum(int(row["reset_calls"]) for row in rows),
            "observation_adapters": {"h0": "truncate_task_flag_suffix_only", "crossing": "exact_18d"},
        }
    finally:
        env.close()
    _write_json(root / "run_manifest.json", run_manifest)
    _write_json(root / "episode_results.json", {"source_id": SOURCE_ID, "opponent": opponent, "episodes": rows})
    return rows, run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Required to execute the physical probe")
    args = parser.parse_args(argv)
    spec = _load(args.config.resolve())
    if not args.execute:
        print(f"LOCKED: no JSBSim execution. Planned Source ID: {SOURCE_ID}")
        return 0
    if spec["experiment"]["source_id"] != SOURCE_ID:
        raise SystemExit("Source ID mismatch")
    root = args.output_dir.resolve()
    if root.exists() and any(path.is_file() for path in root.rglob("*")):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {root}")
    base = _load(_resolve(spec["base_config"]))
    all_rows: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    for opponent in spec["probe"]["opponents"]:
        rows, manifest = _run_opponent(spec, base, opponent, root / opponent)
        all_rows.extend(rows)
        run_manifests.append(manifest)
    _write_csv(root / "episode_results.csv", all_rows)
    _write_json(root / "pilot_manifest.json", {
        "source_id": SOURCE_ID,
        "status": "completed",
        "training_started": False,
        "training_permitted": False,
        "backend_requested": "jsbsim",
        "opponents": spec["probe"]["opponents"],
        "candidates": spec["probe"]["candidates"],
        "repetitions_per_candidate_and_opponent": spec["probe"]["repetitions_per_candidate_and_opponent"],
        "raw_episode_count": len(all_rows),
        "config": _receipt(args.config.resolve()),
        "base_config": _receipt(_resolve(spec["base_config"])),
        "config_canonical_sha256": config_sha256(spec, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "runs": run_manifests,
        "decision_gate": {
            "per_opponent_min_entered_and_dwelled": 2,
            "max_recovery_steps": 48,
            "required_dwell_steps": 10,
            "no_new_crash_oob_fallback": True,
        },
    })
    print(f"Completed 12-episode strict JSBSim recovery probe under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
