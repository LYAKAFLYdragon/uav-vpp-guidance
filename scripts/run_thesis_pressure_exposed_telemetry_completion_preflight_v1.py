#!/usr/bin/env python3
"""Non-learning physical-telemetry preflight for the corrected pressure cohort."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uav_vpp_guidance.envs.observation import compute_relative_geometry  # noqa: E402
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import classify_relative_geometry  # noqa: E402


RUNNER_PATH = ROOT / "scripts" / "run_thesis_disadvantage_pressure_exposed_postmerge_collector_v1.py"
PREFLIGHT_ID = "ENGINEERING-PRESSURE-EXPOSED-TELEMETRY-COMPLETION-PREFLIGHT-V1-NOT-A-SOURCE-ID"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


class TelemetryPreflightError(RuntimeError):
    """Raised when physical telemetry cannot support a future ablation source."""


def _runner():
    spec = importlib.util.spec_from_file_location("pressure_collector", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise TelemetryPreflightError("cannot import frozen pressure collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _state_speed(state: Mapping[str, Any]) -> float:
    for key in ("velocity_vector_mps", "velocity_ned", "velocity"):
        value = state.get(key)
        if value is not None:
            return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))
    raise TelemetryPreflightError("state lacks a velocity vector")


def _state_altitude(state: Mapping[str, Any]) -> float:
    if state.get("altitude_m") is not None:
        return float(state["altitude_m"])
    for key in ("position_m", "position_neu"):
        value = state.get(key)
        if value is not None:
            return float(np.asarray(value, dtype=np.float64)[2])
    raise TelemetryPreflightError("state lacks altitude")


def _reset_receipt(env: Any, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
    env.reset(scenario=copy.deepcopy(dict(scenario)), seed=seed)
    own_state, target_state = env._get_current_states(noisy=False)
    rel = compute_relative_geometry(own_state, target_state)
    aa_deg = abs(math.degrees(float(rel["aa_rad"])))
    ata_deg = abs(math.degrees(float(rel["ata_rad"])))
    metadata = scenario["metadata"]
    actual = {
        "range_m": float(rel["range_m"]),
        "own_speed_mps": _state_speed(own_state),
        "target_speed_mps": _state_speed(target_state),
        "altitude_diff_m": _state_altitude(target_state) - _state_altitude(own_state),
        "aa_deg": aa_deg,
        "ata_deg": ata_deg,
        "dynamic_taxonomy": classify_relative_geometry(aa_deg, ata_deg),
    }
    expected = {
        "range_m": float(metadata["initial_range_m"]),
        "own_speed_mps": float(metadata["own_speed_mps"]),
        "target_speed_mps": float(metadata["target_speed_mps"]),
        "altitude_diff_m": float(metadata["altitude_diff_m"]),
        "dynamic_taxonomy": str(metadata["initial_class"]),
    }
    tolerance = {"range_m": 5.0, "speed_mps": 2.0, "altitude_diff_m": 5.0}
    passes = bool(
        abs(actual["range_m"] - expected["range_m"]) <= tolerance["range_m"]
        and abs(actual["own_speed_mps"] - expected["own_speed_mps"]) <= tolerance["speed_mps"]
        and abs(actual["target_speed_mps"] - expected["target_speed_mps"]) <= tolerance["speed_mps"]
        and abs(actual["altitude_diff_m"] - expected["altitude_diff_m"]) <= tolerance["altitude_diff_m"]
        and actual["dynamic_taxonomy"] == expected["dynamic_taxonomy"]
    )
    return {"expected": expected, "actual": actual, "tolerance": tolerance, "passes": passes}


def _finite_vector(value: Any, dimension: int) -> bool:
    return bool(
        isinstance(value, (list, tuple, np.ndarray))
        and len(value) == dimension
        and all(_finite(item) is not None for item in value)
    )


def _physical_trace(ledger: Mapping[str, Any], vpp_info: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(ledger["steps"]) != len(vpp_info):
        raise TelemetryPreflightError("VPP info capture is not one-to-one with ledger steps")
    trace = []
    for step, captured in zip(ledger["steps"], vpp_info):
        own = step.get("own_state")
        target = step.get("target_state")
        if not isinstance(own, Mapping) or not isinstance(target, Mapping):
            raise TelemetryPreflightError("collector step lacks raw JSBSim own/target state")
        rel = compute_relative_geometry(own, target)
        aa_deg = abs(math.degrees(float(rel["aa_rad"])))
        ata_deg = abs(math.degrees(float(rel["ata_rad"])))
        own_speed = _state_speed(own)
        target_speed = _state_speed(target)
        own_altitude = _state_altitude(own)
        target_altitude = _state_altitude(target)
        action = step.get("normalized_vpp_action")
        observation_66d = step.get("observation_66d")
        row = {
            "step": int(step["step"]),
            "phase": str(step["phase"]),
            "dynamic_taxonomy": classify_relative_geometry(aa_deg, ata_deg),
            "range_m": float(rel["range_m"]),
            "range_rate_mps": float(rel["range_rate_mps"]),
            "aa_deg": aa_deg,
            "ata_deg": ata_deg,
            "own_speed_mps": own_speed,
            "target_speed_mps": target_speed,
            "own_altitude_m": own_altitude,
            "target_altitude_m": target_altitude,
            "own_specific_energy_m": own_altitude + own_speed * own_speed / (2.0 * 9.80665),
            "target_in_attack_zone": bool(step["target_in_attack_zone"]),
            "ego_in_attack_zone": bool(step["ego_in_attack_zone"]),
            "observation_66d_valid": isinstance(observation_66d, list) and len(observation_66d) == 66 and all(_finite(value) is not None for value in observation_66d),
            "vpp_action_valid": isinstance(action, list) and len(action) == 3 and all(_finite(value) is not None for value in action),
            "tactical_basis_enabled": captured.get("tactical_basis_enabled") is True,
            "tactical_basis_world_offset": captured.get("tactical_basis_world_offset"),
            "tactical_basis_world_offset_valid": _finite_vector(captured.get("tactical_basis_world_offset"), 3),
            "tactical_basis_ll_world_valid": _finite_vector(captured.get("tactical_basis_ll_world"), 3),
            "tactical_basis_io_world_valid": _finite_vector(captured.get("tactical_basis_io_world"), 3),
            "tactical_basis_cd_world_valid": _finite_vector(captured.get("tactical_basis_cd_world"), 3),
            "guidance_command_present": step.get("guidance_command") is not None,
            "aoa_present": _finite(own.get("alpha_rad", own.get("aoa_rad"))) is not None,
            "nz_command_present": step.get("nz_cmd") is not None,
            "nz_actual_g": _finite(step.get("actual_nz_g")),
            "prediction_fallback": bool(step["prediction_fallback"]),
            "backend": str(step["backend"]),
            "backend_fallback_occurred": bool(step["backend_fallback_occurred"]),
        }
        trace.append(row)
    return trace


def _trace_passes(trace: list[Mapping[str, Any]]) -> bool:
    if not trace:
        return False
    for row in trace:
        if row["backend"] != "jsbsim" or row["backend_fallback_occurred"]:
            return False
        numeric = ("range_m", "range_rate_mps", "aa_deg", "ata_deg", "own_speed_mps", "target_speed_mps", "own_altitude_m", "target_altitude_m", "own_specific_energy_m")
        if any(_finite(row[key]) is None for key in numeric):
            return False
    # 66-D vectors are unavailable only during the fixed ten-frame warm-up.
    post_warmup = trace[9:]
    return bool(post_warmup) and all(
        row["observation_66d_valid"] and row["vpp_action_valid"] and row["tactical_basis_enabled"]
        and row["tactical_basis_world_offset_valid"] and row["tactical_basis_ll_world_valid"]
        and row["tactical_basis_io_world_valid"] and row["tactical_basis_cd_world_valid"]
        and row["guidance_command_present"] and row["aoa_present"]
        and row["nz_command_present"] and row["nz_actual_g"] is not None
        for row in post_warmup
    )


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    runner = _runner()
    design = runner.validate_design(config_path, check_external_inputs=True)
    config = runner.load_config(config_path)
    if output_root.exists():
        raise TelemetryPreflightError("preflight output root must be absent")
    manifest = runner._load_yaml(runner._resolve(str(config["inputs"]["scenario_manifest"])))
    registry = runner._load_yaml(runner._resolve(str(config["inputs"]["runtime_registry"])))
    output_root.mkdir(parents=True)
    records = []
    for opponent in OPPONENTS:
        for scenario in manifest["scenarios"]:
            env = runner._build_env(config, registry, opponent)
            try:
                receipt = _reset_receipt(env, scenario, int(scenario["metadata"]["scenario_seed"]))
                encoder = runner.FrozenP3Encoder(runner._resolve(str(config["inputs"]["frozen_p3_encoder"])), str(config["inputs"]["frozen_p3_encoder_sha256"]))
                captured_vpp_info: list[Mapping[str, Any]] = []
                original_step = env.step

                def _capturing_step(action: Any):
                    outcome = original_step(action)
                    info = outcome[4]
                    captured_vpp_info.append({
                        key: info.get(key)
                        for key in (
                            "tactical_basis_enabled", "tactical_basis_world_offset",
                            "tactical_basis_ll_world", "tactical_basis_io_world", "tactical_basis_cd_world",
                        )
                    })
                    return outcome

                env.step = _capturing_step
                episode = runner.raw_v5.FiveStateRawSIV5Episode(env, runner._specialist(registry), encoder, config)
                result, ledger = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                trace = _physical_trace(ledger, captured_vpp_info)
                record = {
                    "preflight_id": PREFLIGHT_ID,
                    "opponent": opponent,
                    "scenario_name": scenario["name"],
                    "scenario_signature": scenario["metadata"]["mirror_pair_id"],
                    "mirror_sign": scenario["metadata"]["mirror_sign"],
                    "reset_receipt": receipt,
                    "trace": trace,
                    "trace_passes": _trace_passes(trace),
                    "terminal_reason": result["terminal_reason"],
                }
            finally:
                env.close()
            records.append(record)
            path = output_root / "episodes" / opponent / f"{scenario['name']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    per_opponent = {}
    for opponent in OPPONENTS:
        group = [row for row in records if row["opponent"] == opponent]
        per_opponent[opponent] = {
            "records": len(group),
            "reset_receipts_passed": sum(bool(row["reset_receipt"]["passes"]) for row in group),
            "physical_trace_passed": sum(bool(row["trace_passes"]) for row in group),
            "signatures": sorted({row["scenario_signature"] for row in group}),
            "mirrors": sorted({row["mirror_sign"] for row in group}),
        }
    passes = all(
        item["records"] == 16 and item["reset_receipts_passed"] == 16 and item["physical_trace_passed"] == 16
        and len(item["signatures"]) == 8 and item["mirrors"] == ["negative", "positive"]
        for item in per_opponent.values()
    )
    index = []
    for record in records:
        path = output_root / "episodes" / record["opponent"] / f"{record['scenario_name']}.json"
        index.append({"path": str(path.relative_to(output_root)), "sha256": _hash(path)})
    (output_root / "records_index.json").write_text(json.dumps({"records": index}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "preflight_id": PREFLIGHT_ID,
        "source_config": str(config_path.resolve()),
        "source_config_sha256": _hash(config_path),
        "source_manifest_sha256": _hash(runner._resolve(str(config["inputs"]["scenario_manifest"]))),
        "records": len(records),
        "per_opponent": per_opponent,
        "passes": passes,
        "next_action": "compact_ablation_source_design_permitted" if passes else "stop_no_ablation_or_training",
        "boundary": "non-learning physical-telemetry contract only; no method comparison or performance claim",
    }
    (output_root / "telemetry_completion_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.collector_config.resolve(), args.output_root.resolve())
    except TelemetryPreflightError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
