#!/usr/bin/env python3
"""Replay artifact-qualified handoffs and measure symmetric local VPP authority."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor as v15
import run_thesis_temporal_corridor_transfer_v2 as v2
from uav_vpp_guidance.evaluation.thesis_local_action_authority_map import METHODS, SOURCE_ID, summarize
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer_v2 import is_ego_failure
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return v15._load(path)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _config(path: Path) -> dict[str, Any]:
    overlay = _yaml(path)
    if "base_config" not in overlay:
        return overlay
    base = _resolve(overlay["base_config"])
    if _hash(base) != str(overlay.get("base_config_sha256") or ""):
        raise RuntimeError("local-authority authorization base hash mismatch")
    value = copy.deepcopy(_yaml(base))
    value["authorization"] = copy.deepcopy(overlay["authorization"])
    value["_execution"] = copy.deepcopy(overlay.get("execution", {}))
    return value


def _cohort(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(config["inputs"]["cohort_manifest"])
    if _hash(path) != str(config["inputs"]["cohort_manifest_sha256"]):
        raise RuntimeError("local-authority cohort hash mismatch")
    cohort = _yaml(path)
    expected_entries = int(config["gates"]["expected_records"]) // len(METHODS)
    if cohort.get("source_id") != SOURCE_ID + f"-COHORT{expected_entries}" or len(cohort.get("entries", [])) != expected_entries:
        raise RuntimeError("local-authority cohort contract drifted")
    for entry in cohort["entries"]:
        for key, hash_key in (("historical_record", "historical_record_sha256"), ("historical_authorized_config", "historical_authorized_config_sha256")):
            artifact = _resolve(entry[key])
            if not artifact.is_file() or _hash(artifact) != str(entry[hash_key]):
                raise RuntimeError(f"local-authority historical artifact mismatch: {key}")
    return cohort


def validate_design(path: Path) -> dict[str, Any]:
    config = _config(path)
    if config.get("source_id") != SOURCE_ID or tuple(config.get("methods", ())) != METHODS:
        raise RuntimeError("local-authority design identity drifted")
    for key, value in config["authorization"].items():
        if key != "execution_permitted" and value is not False:
            raise RuntimeError(f"local-authority must prohibit {key}")
    cohort = _cohort(config)
    output = _resolve(config["outputs"]["root"])
    if output.exists() or not output.is_absolute():
        raise RuntimeError("local-authority output root must be absolute and absent")
    return {"source_id": SOURCE_ID, "cohort_entries": len(cohort["entries"]), "planned_records": len(cohort["entries"]) * len(METHODS), "output_root_absent": True, "execution_permitted": bool(config["authorization"]["execution_permitted"])}


def _source(entry: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    archived = _json(_resolve(entry["historical_authorized_config"]))["config"]
    manifest = _yaml(_resolve(archived["inputs"]["manifest"]))
    scenario = next(item for item in manifest["scenarios"] if item["metadata"]["pair_key"] == entry["pair_key"])
    registry = _yaml(_resolve(archived["inputs"]["runtime_registry"]))
    return archived, scenario, registry


def _episode(entry: Mapping[str, Any], config: Mapping[str, Any], registry: Mapping[str, Any]):
    policy = engine._specialist(registry, "frozen_fixed_head_on")
    encoder = FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"])
    if entry["historical_source_kind"] == "v15":
        env = v15.v14.build_record_only_env(config, registry, entry["opponent"])
        return v15.FixedEncoderTemporalCorridorEpisode(env, policy, encoder, config, v15._compiler(config))
    env = v2.build_record_only_env(config, registry, entry["opponent"])
    return v2.Episode(env, policy, encoder, config, v2.v16._compiler(config))


def _prepare(ep: Any, entry: Mapping[str, Any], scenario: Mapping[str, Any]) -> tuple[bool, str | None]:
    ep.history.clear(); ep.history_features.reset(); ep.observation = None; ep.base = None; ep.last_info = {}; ep.total_steps = 0; ep.phase_trace = []
    raw = v15.v14.v11
    event = raw.RawSIFirstPassConfig(**dict(ep.config["observability"]["raw_si_first_pass_event"]))
    ep.phase_tracker = raw.RawSIReentryPhaseTracker(raw.RawSIReentryPhaseConfig(first_pass=event))
    observation = ep.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(scenario["metadata"]["scenario_seed"]))
    if ep.env._backend != "jsbsim":
        raise RuntimeError("local-authority requires strict JSBSim")
    if entry["historical_source_kind"] == "v15":
        ep._assert_raw_interface(); mode = "local_authority_v15_run_in"
    else:
        ep._assert_path_interface(); mode = "local_authority_v2_run_in"
    ep.env.set_runtime_specialist_context(specialist_key="head_on", specialist_profile=None, specialist_mode_name=mode, specialist_reason="artifact_matched_replay")
    ep.observation_schema = copy.deepcopy(observation["observation_schema"]); ep._observe(observation)
    done = False
    while not done and ep.total_steps < int(ep.config["observability"]["max_high_level_steps"]):
        if ep._trigger_ready():
            break
        _info, done = ep._reference_step()
    if not ep._trigger_ready():
        return False, None
    if entry["historical_source_kind"] == "v15":
        replay_hash = v15.v14._stable_hash({"base": ep.base, "history": tuple(ep.history), "observation": ep.observation["observation_vector"], "phase": ep._phase()})
    else:
        replay_hash = v2.v16.v15.v14._stable_hash({"base": ep.base, "history": tuple(ep.history), "phase": ep._phase()})
    return replay_hash == entry["historical_handoff_hash"], replay_hash


def _metrics(base: np.ndarray) -> dict[str, float]:
    angle = lambda sin, cos: abs(float(np.degrees(np.arctan2(float(sin), float(cos)))))
    return {"range_m": float(base[0] * 12000.0), "range_rate_mps": float(base[1] * 400.0), "ata_deg": angle(base[8], base[9]), "aa_deg": angle(base[10], base[11]), "specific_energy_height_m": float(base[14] + base[12] ** 2 / (2.0 * 9.80665)), "altitude_m": float(base[14])}


def _branch(entry: Mapping[str, Any], method: str, pulse: Mapping[str, Any]) -> dict[str, Any]:
    source_config, scenario, registry = _source(entry); ep = _episode(entry, source_config, registry)
    try:
        matched, replay_hash = _prepare(ep, entry, scenario)
        result: dict[str, Any] = {"source_id": SOURCE_ID, "cohort_key": entry["cohort_key"], "historical_source_kind": entry["historical_source_kind"], "opponent": entry["opponent"], "method": method, "historic_handoff_hash": entry["historical_handoff_hash"], "replay_handoff_hash": replay_hash, "replay_matched": matched, "window_complete": False, "pulse_clipping": False, "ego_failure": False}
        if not matched:
            return result
        initial = _metrics(ep.base); terminal = "window_horizon"; trace = []
        axis = None if method == "reference" else (0 if method.startswith("longitudinal") else 1 if method.startswith("lateral") else 2)
        sign = 0 if method == "reference" else (1 if method.endswith("plus") else -1)
        for step in range(int(pulse["local_window_steps"])):
            reference = v15.v14.v11.v9.finite_action(ep.policy.get_deterministic_action(ep.observation["observation_vector"]))
            action = reference.copy()
            if axis is not None and step < int(pulse["pulse_steps"]):
                raw_action = action.copy(); raw_action[axis] += sign * float(pulse["normalized_delta"]); action = np.clip(raw_action, *pulse["action_clip"]); result["pulse_clipping"] |= not np.allclose(raw_action, action, rtol=0.0, atol=1e-12)
            observation, _reward, terminated, truncated, info = ep.env.step(np.asarray(action, dtype=np.float32)); ep.total_steps += 1; ep.last_info = info; ep._observe(observation)
            own = info.get("own_state", {}); telemetry = {"vpp_world_offset_m": ep._finite_vector(info.get("vp_world_offset"), "vp_world_offset"), "guidance_command_normalized": ep._guidance_vector(info), "actual_nz_g": float(own.get("nz_g", info.get("nz_g"))), "phase": ep._phase()}
            trace.append(telemetry)
            if bool(info.get("backend_fallback_occurred", False)) or info.get("prediction_fallback") is True:
                raise RuntimeError("local-authority runtime fallback")
            terminal = str(info.get("reason") or "window_horizon")
            if terminated or truncated:
                break
        result.update({"initial_metrics": initial, "final_metrics": _metrics(ep.base), "window_complete": len(trace) == int(pulse["local_window_steps"]), "terminal_reason": terminal, "ego_failure": is_ego_failure(terminal), "telemetry": trace})
        return result
    finally:
        ep.close()


def _authorize(config: Mapping[str, Any], expected: int) -> None:
    execution = config.get("_execution", {})
    if config["authorization"]["execution_permitted"] is not True or int(execution.get("expected_records", -1)) != expected:
        raise RuntimeError("local-authority execution is not authorized")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != execution.get("required_implementation_git_sha"):
        raise RuntimeError("local-authority implementation SHA mismatch")
    smoke = _resolve(execution.get("required_smoke_gate", ""))
    if not smoke.is_file() or _hash(smoke) != execution.get("required_smoke_gate_sha256"):
        raise RuntimeError("local-authority required smoke gate mismatch")
    smoke_payload = _json(smoke)
    if smoke_payload.get("source_id_under_test") != SOURCE_ID or smoke_payload.get("passes") is not True:
        raise RuntimeError("local-authority required smoke gate did not pass")
    for item in execution.get("authorized_code_files", []):
        path = _resolve(item["path"])
        if not path.is_file() or _hash(path) != item["sha256"]:
            raise RuntimeError("local-authority authorized code mismatch")


def _run(config: Mapping[str, Any], cohort: Mapping[str, Any], output: Path, entries: list[Mapping[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    records = []
    try:
        for entry in entries:
            for method in methods:
                row = _branch(entry, method, config["pulse"]); records.append(row)
                safe = entry["cohort_key"].replace(":", "__").replace("/", "_")
                _write(output / "records" / method / entry["opponent"] / f"{safe}.json", {"record": row})
        gate = summarize(records, config["gates"]); _write(output / "gate.json", {"source_id": SOURCE_ID, **gate}); return gate
    except Exception as error:
        _write(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)}); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--execute", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--output", type=Path); args = parser.parse_args()
    config = _config(args.config); preflight = validate_design(args.config); cohort = _cohort(config)
    if not args.execute and not args.smoke:
        print(json.dumps(preflight, ensure_ascii=False, indent=2)); return 0
    if args.smoke:
        if config["authorization"]["execution_permitted"] is not False: raise RuntimeError("local-authority smoke requires design config")
        source_kinds = tuple(sorted({str(entry["historical_source_kind"]) for entry in cohort["entries"]}))
        selected = [
            next(entry for entry in cohort["entries"] if entry["opponent"] == opponent and entry["historical_source_kind"] == source_kind)
            for source_kind in source_kinds
            for opponent in config["opponents"]
        ]
        output = args.output or ROOT / "outputs/diagnostics/local_action_authority_map_v1_smoke"
        if not output.is_absolute(): output = ROOT / output
        if output.exists(): raise RuntimeError("local-authority smoke output exists")
        output.mkdir(parents=True); gate = _run(config, cohort, output, selected, ("reference",)); passed = gate["records"] == len(selected) and all(row["replay_matched"] and row["window_complete"] for row in [json.loads(path.read_text(encoding="utf-8"))["record"] for path in (output / "records" / "reference").rglob("*.json")]); _write(output / "smoke_gate.json", {"source_id_under_test": SOURCE_ID, "records": len(selected), "passes": passed}); print(json.dumps({"output": str(output), "passes": passed}, ensure_ascii=False)); return 0
    _authorize(config, preflight["planned_records"])
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(): raise RuntimeError("local-authority requires clean worktree")
    output = _resolve(config["outputs"]["root"]); output.mkdir(parents=True); _write(output / "authorized_config.json", {"config": config, "config_sha256": _hash(args.config)}); gate = _run(config, cohort, output, cohort["entries"], METHODS); _write(output / "run_manifest.json", {"source_id": SOURCE_ID, "records": preflight["planned_records"], "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "training": "prohibited"}); print(json.dumps({"source_id": SOURCE_ID, **gate}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
