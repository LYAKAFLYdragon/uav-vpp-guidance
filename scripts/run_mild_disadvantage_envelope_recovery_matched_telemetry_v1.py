#!/usr/bin/env python
"""Collect missing causal-chain telemetry for the matched-contract probe."""

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
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-TELEMETRY-COLLECTOR-V1-20260903"
DEFAULT_CONFIG = REPO_ROOT / "config/experiment/mild_disadvantage_envelope_recovery_matched_telemetry_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/mild_disadvantage_envelope_recovery_v1/matched_telemetry_v1"
MATCHED_RUNNER = REPO_ROOT / "scripts/run_mild_disadvantage_envelope_recovery_matched_probe_v1.py"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _normalise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalise(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise(value[key]) for key in sorted(value)}
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _state_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _normalise(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> str:
    return json.dumps(_normalise(value), ensure_ascii=False, separators=(",", ":"))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prepare_config(base: dict[str, Any], spec: dict[str, Any], opponent: str, matched: Any) -> dict[str, Any]:
    cfg = matched._prepare_config(base, spec, opponent)
    source = "run_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py"
    _set_nested(cfg, "experiment.name", spec["experiment"]["name"], f"{source}:name")
    _set_nested(cfg, "experiment.source_id", SOURCE_ID, f"{source}:source_id")
    _set_nested(cfg, "protocol.source_id", SOURCE_ID, f"{source}:protocol_source_id")
    _set_nested(cfg, "telemetry_contract", spec["telemetry_contract"], f"{source}:telemetry_contract")
    return cfg


def _build_opponent(name: str, cfg: dict[str, Any]):
    entry = cfg["opponent_registry"][name]
    if name == "expert":
        return ExpertOpponent(entry.get("config", {})), {"stage": "expert"}
    return EndToEndOpponent.from_registry(entry), {"stage": "end_to_end", **entry}


def _pre_state_payload(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "own_state": obs.get("own_state", {}),
        "target_state": obs.get("target_state", {}),
        "relative_state": obs.get("relative_state", {}),
        "observation_vector": obs.get("observation_vector", []),
    }


def _run_episode(
    env: CloseRangeTrackingEnv,
    h0_policy: Any,
    crossing_policy: Any,
    recovery: Any,
    scenario: dict[str, Any],
    seed: int,
    opponent: str,
    candidate: str,
    calibration: dict[str, Any],
    legacy: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs = env.reset(scenario=scenario, seed=seed)
    recovery.reset()
    reset_calls = 1
    initial = legacy._initial_info(env)
    state = legacy._state_from_snapshot(
        initial["own_state"], initial["target_state"], initial, calibration
    )
    trajectory: list[dict[str, Any]] = []
    first_recovery_step = None
    entering_step = None
    max_dwell = 0
    recovery_steps = 0
    hold_steps = 0
    crossing_steps = 0
    h0_steps = 0
    min_range = float("inf")
    min_ata = float("inf")
    reason = "timeout"

    for step in range(int(env.max_steps)):
        decision = recovery.decide(state)
        if decision.phase in {"recover", "hold", "complete"} and first_recovery_step is None:
            first_recovery_step = step
        if decision.phase in {"hold", "complete"} and entering_step is None:
            entering_step = step
        recovery_steps += int(decision.phase == "recover")
        hold_steps += int(decision.phase == "hold")
        max_dwell = max(max_dwell, int(decision.dwell_steps))
        mode = decision.mode
        if mode == "crossing":
            action = crossing_policy.get_deterministic_action(obs["observation_vector"])
            crossing_steps += 1
            adapter = "exact_18d"
        else:
            action = h0_policy.get_deterministic_action(obs["observation_vector"])
            h0_steps += 1
            adapter = "truncate_task_flag_suffix_only"
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError(f"invalid action at step {step}: {action}")
        before_hash = _state_hash(_pre_state_payload(obs))
        pre_own = obs.get("own_state", {})
        pre_target = obs.get("target_state", {})
        pre_rel = obs.get("relative_state", {})
        env.set_runtime_specialist_context(
            specialist_key=mode,
            specialist_profile="envelope_recovery" if mode == "crossing" else "h0_nominal",
            specialist_mode_name=mode,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        own = info.get("own_state", {})
        target = info.get("target_state", {})
        rel = info.get("relative_state", {})
        state = legacy._state_from_snapshot(
            own,
            target,
            {**info, "terminated": terminated, "truncated": truncated},
            calibration,
        )
        range_m = _float(info.get("range_m", rel.get("range_m")))
        ata_deg = _float(info.get("ata_deg"))
        min_range = min(min_range, range_m)
        min_ata = min(min_ata, ata_deg)
        raw_command = info.get("raw_command") or {}
        guidance_command = info.get("guidance_command") or {}
        body_rates = own.get("body_rates_rps", [np.nan, np.nan, np.nan])
        trajectory.append({
            "step": step,
            "time_s": step * float(env.env_config.get("high_level_dt", 0.2)),
            "candidate": candidate,
            "opponent": opponent,
            "mode": mode,
            "adapter": adapter,
            "recovery_phase": decision.phase,
            "recovery_reason": decision.reason,
            "state_hash_before_action": before_hash,
            "state_hash_schema": "canonical_json_v1",
            "pre_own_state": _json_value(pre_own),
            "pre_target_state": _json_value(pre_target),
            "pre_relative_state": _json_value(pre_rel),
            "action_0": float(action[0]),
            "action_1": float(action[1]),
            "action_2": float(action[2]),
            "target_point_x_m": _float(getattr(decision.target, "target_x_m", np.nan)),
            "target_point_y_abs_m": _float(getattr(decision.target, "target_y_abs_m", np.nan)),
            "anchor_mode": info.get("anchor_mode"),
            "virtual_point_source": info.get("virtual_point_source"),
            "action_semantics": info.get("action_semantics", "tactical_basis_v1"),
            "vp_position_neu": _json_value(info.get("vp_position_neu")),
            "anchor_pos": _json_value(info.get("anchor_pos")),
            "vp_offset": _json_value(info.get("vp_offset")),
            "vp_world_offset": _json_value(info.get("vp_world_offset")),
            "offset_frame": info.get("offset_frame"),
            "vp_forward_bias_m": _float(info.get("vp_forward_bias_m")),
            "vp_lateral_bias_m": _float(info.get("vp_lateral_bias_m")),
            "raw_nz_cmd": _float(raw_command.get("nz_cmd")),
            "raw_roll_rate_cmd": _float(raw_command.get("roll_rate_cmd")),
            "raw_throttle_cmd": _float(raw_command.get("throttle_cmd")),
            "guidance_nz_cmd": _float(guidance_command.get("nz_cmd")),
            "guidance_roll_rate_cmd": _float(guidance_command.get("roll_rate_cmd")),
            "guidance_throttle_cmd": _float(guidance_command.get("throttle_cmd")),
            "elevator_cmd": _float(info.get("elevator_cmd")),
            "aileron_cmd": _float(info.get("aileron_cmd")),
            "rudder_cmd": _float(info.get("rudder_cmd")),
            "throttle_actual": _float(info.get("throttle_actual")),
            "roll_rad": _float(own.get("roll_rad")),
            "pitch_rad": _float(own.get("pitch_rad")),
            "yaw_rad": _float(own.get("yaw_rad")),
            "p_rps": _float(own.get("p_rps", body_rates[0] if len(body_rates) > 0 else np.nan)),
            "q_rps": _float(own.get("q_rps", body_rates[1] if len(body_rates) > 1 else np.nan)),
            "r_rps": _float(own.get("r_rps", body_rates[2] if len(body_rates) > 2 else np.nan)),
            "nz_g": _float(own.get("nz_g")),
            "alpha_rad": _float(own.get("alpha_rad")),
            "beta_rad": _float(own.get("beta_rad")),
            "range_m": range_m,
            "range_rate_mps": _float(info.get("range_rate_mps", rel.get("range_rate_mps"))),
            "ata_deg": ata_deg,
            "aa_deg": _float(info.get("aa_deg")),
            "altitude_m": _float(own.get("altitude_m")),
            "own_speed_mps": _float(np.linalg.norm(np.asarray(own.get("velocity_vector_mps", [np.nan] * 3), dtype=float))),
            "target_speed_mps": _float(np.linalg.norm(np.asarray(target.get("velocity_vector_mps", [np.nan] * 3), dtype=float))),
            "prediction_enabled": bool(info.get("prediction_enabled", False)),
            "prediction_valid": bool(info.get("prediction_valid", False)),
            "prediction_fallback": bool(info.get("prediction_fallback", False)),
            "clipping_rate": float(np.mean([
                bool(info.get("nz_saturated", False)),
                bool(info.get("roll_rate_saturated", False)),
                bool(info.get("throttle_saturated", False)),
            ])),
            "backend": info.get("backend"),
            "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
            "mode_switch_effective": bool(info.get("mode_switch_effective", False)),
            "effective_guidance_mode": info.get("effective_guidance_mode", "unknown"),
            "terminal": bool(terminated or truncated),
            "terminal_reason": info.get("reason", "") or "",
        })
        if terminated or truncated:
            reason = str(info.get("reason", "unknown"))
            break

    finite_fields = (
        "state_hash_before_action", "action_0", "action_1", "action_2",
        "vp_forward_bias_m", "vp_lateral_bias_m", "raw_nz_cmd",
        "raw_roll_rate_cmd", "raw_throttle_cmd", "guidance_nz_cmd",
        "guidance_roll_rate_cmd", "guidance_throttle_cmd", "roll_rad",
        "p_rps", "nz_g", "range_m", "range_rate_mps", "ata_deg", "aa_deg",
    )
    finite_telemetry = bool(trajectory) and all(
        (bool(row[key]) if key == "state_hash_before_action" else _finite(row[key]))
        for row in trajectory for key in finite_fields
    )
    result = {
        "candidate": candidate,
        "target_x_m": scenario["target_init"]["position_m"][0],
        "target_y_m": scenario["target_init"]["position_m"][1],
        "opponent": opponent,
        "seed": seed,
        "method": "h0_with_bounded_crossing_recovery_telemetry",
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
        "recovery_steps": recovery_steps,
        "hold_steps": hold_steps,
        "max_dwell_steps": max_dwell,
        "crossing_steps": crossing_steps,
        "h0_steps": h0_steps,
        "recovery_budget_ok": recovery_steps <= recovery.max_recovery_steps,
        "dwell_gate_ok": max_dwell >= recovery.required_dwell_steps if entering_step is not None else False,
        "min_range_m": min_range,
        "min_ata_deg": min_ata,
        "backend_effective": getattr(env, "_backend", None),
        "backend_fallback_occurred": any(bool(row["backend_fallback_occurred"]) for row in trajectory),
        "finite_telemetry": finite_telemetry,
        "prediction_fallback_occurred": any(bool(row["prediction_fallback"]) for row in trajectory),
        "training_started": False,
        "fallback_or_reset_or_padding": bool(reset_calls != 1),
    }
    return result, trajectory


def _run_opponent(spec: dict[str, Any], base: dict[str, Any], opponent: str, root: Path, matched: Any, legacy: Any):
    cfg = _prepare_config(base, spec, opponent, matched)
    root.mkdir(parents=True, exist_ok=True)
    resolved = root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    paths = {
        "h0": _resolve(spec["assets"]["h0_checkpoint"]),
        "crossing": _resolve(spec["assets"]["crossing_checkpoint"]),
        "crossing_config": _resolve(spec["assets"]["crossing_config"]),
        "predictor": _resolve(spec["assets"]["crossing_predictor"]),
        "end_to_end": _resolve(spec["assets"]["end_to_end_checkpoint"]),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    opponent_policy, opponent_config = _build_opponent(opponent, cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
    rows: list[dict[str, Any]] = []
    try:
        h0_agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
        h0_load = h0_agent.load(str(paths["h0"]))
        crossing_policy = FrozenSpecialistPolicy(str(paths["crossing"]), str(paths["crossing_config"]), device="cpu")
        h0_policy = legacy._TruncateTaskFlagAdapter(h0_agent)
        stable_points = [legacy.EnvelopePoint(float(p[0]), float(p[1])) for p in spec["recovery"]["stable_points_m"]]
        recovery_cfg = spec["recovery"]
        repeats = int(spec["probe"]["repetitions_per_candidate_and_opponent"])
        base_seed = int(spec["probe"]["evaluation_seed_base"])
        calibration = spec["probe"]["calibration"]
        for candidate_index, (candidate, point) in enumerate(spec["probe"]["candidates"].items()):
            scenario = legacy._make_scenario(candidate, point, spec["probe"]["fixed_target_conditions"])
            for repetition in range(repeats):
                seed = base_seed + candidate_index * 100 + (0 if opponent == "expert" else 50) + repetition
                recovery = legacy.CapabilityEnvelopeRecoveryPolicy(stable_points, **{
                    key: recovery_cfg[key]
                    for key in ("nominal_mode", "recovery_mode", "x_tolerance_m", "y_tolerance_m", "max_recovery_steps", "required_dwell_steps", "max_clipping_rate", "min_altitude_m", "max_range_m")
                })
                result, trajectory = _run_episode(
                    env, h0_policy, crossing_policy, recovery, scenario, seed,
                    opponent, candidate, calibration, legacy,
                )
                result["source_id"] = SOURCE_ID
                result["matched_contract"] = True
                rows.append(result)
                _write_csv(root / "trajectories" / f"{candidate}_{repetition + 1:02d}.csv", trajectory)
    finally:
        backend_effective = getattr(env, "_backend", None)
        backend_fallback = bool(getattr(env, "_backend_fallback_occurred", False))
        env.close()
    manifest = {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "status": "completed",
        "training_started": False,
        "training_permitted": False,
        "backend_requested": "jsbsim",
        "backend_effective": backend_effective,
        "backend_fallback_occurred": backend_fallback,
        "task_name": spec["matched_contract"]["task_name"],
        "matched_contract": spec["matched_contract"],
        "telemetry_contract": spec["telemetry_contract"],
        "h0_checkpoint": _receipt(paths["h0"]),
        "h0_checkpoint_obs_dim": int(h0_load.get("obs_dim", 0)),
        "h0_checkpoint_action_dim": int(h0_load.get("action_dim", 0)),
        "crossing_checkpoint": _receipt(paths["crossing"]),
        "crossing_config": _receipt(paths["crossing_config"]),
        "crossing_predictor": _receipt(paths["predictor"]),
        "end_to_end_checkpoint": _receipt(paths["end_to_end"]),
        "resolved_config": _receipt(resolved),
        "config_sha256": file_sha256(resolved),
        "config_canonical_sha256": config_sha256(cfg, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "episode_count": len(rows),
        "reset_calls_total": sum(int(row["reset_calls"]) for row in rows),
        "prediction_fallback_occurred": any(bool(row["prediction_fallback_occurred"]) for row in rows),
    }
    _write_json(root / "run_manifest.json", manifest)
    _write_json(root / "episode_results.json", {"source_id": SOURCE_ID, "opponent": opponent, "episodes": rows})
    return rows, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Required to execute JSBSim")
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
    matched = _load_module(MATCHED_RUNNER, "matched_recovery_probe_runner_for_telemetry")
    legacy = matched._legacy_module()
    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for opponent in spec["probe"]["opponents"]:
        rows, manifest = _run_opponent(spec, base, opponent, root / opponent, matched, legacy)
        all_rows.extend(rows)
        manifests.append(manifest)
    _write_csv(root / "episode_results.csv", all_rows)
    _write_json(root / "pilot_manifest.json", {
        "source_id": SOURCE_ID,
        "status": "completed",
        "training_started": False,
        "training_permitted": False,
        "backend_requested": "jsbsim",
        "task_name": spec["matched_contract"]["task_name"],
        "matched_contract": spec["matched_contract"],
        "telemetry_contract": spec["telemetry_contract"],
        "opponents": spec["probe"]["opponents"],
        "candidates": spec["probe"]["candidates"],
        "repetitions_per_candidate_and_opponent": spec["probe"]["repetitions_per_candidate_and_opponent"],
        "raw_episode_count": len(all_rows),
        "config": _receipt(args.config.resolve()),
        "base_config": _receipt(_resolve(spec["base_config"])),
        "config_canonical_sha256": config_sha256(spec, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "runs": manifests,
        "decision_gate": {
            "strict_jsbsim": True,
            "one_reset_per_episode": True,
            "required_intermediate_telemetry": True,
            "state_hash_before_action": True,
            "prediction_fallback_forbidden": True,
        },
    })
    print(f"Completed 12-episode telemetry collector under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
