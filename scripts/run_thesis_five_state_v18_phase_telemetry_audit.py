#!/usr/bin/env python3
"""Collect V18 telemetry without changing V17's frozen controller or scenarios."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_v17_path_context_preflight as v17
from uav_vpp_guidance.common.provenance import record_config_override
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import classify_relative_geometry
from uav_vpp_guidance.evaluation.thesis_v18_phase_telemetry_audit import OPPONENTS, SOURCE_ID, summarize_records
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder


class V18RuntimeError(RuntimeError):
    """Raised when V18 would leave its preregistered telemetry-only contract."""


def _load_design(path: Path) -> dict[str, Any]:
    overlay = v17._load(path)
    if "base_config" not in overlay:
        return overlay
    base = v17._resolve(str(overlay["base_config"]))
    if v17._hash(base) != str(overlay.get("base_config_sha256") or ""):
        raise V18RuntimeError("V18 authorization overlay base hash mismatch")
    result = copy.deepcopy(v17._load(base))
    result["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    result["execution"] = copy.deepcopy(overlay.get("execution", {}))
    return result


def _runtime_config(design: Mapping[str, Any]) -> dict[str, Any]:
    runtime_path = v17._resolve(str(design["inputs"]["runtime_template"]))
    runtime = copy.deepcopy(v17._load(runtime_path))
    old_output = runtime.get("outputs", {}).get("root")
    runtime.setdefault("outputs", {})["root"] = str(design["outputs"]["root"])
    record_config_override(runtime, "outputs.root", runtime["outputs"]["root"], old_output, "run_thesis_five_state_v18_phase_telemetry_audit.py:telemetry_output")
    return runtime


def validate_design(config_path: Path) -> dict[str, Any]:
    design = _load_design(config_path)
    if design.get("source_id") != SOURCE_ID or design.get("status") not in {"design_only_not_authorized", "authorized_one_time_execution"}:
        raise V18RuntimeError("V18 design identity drifted")
    for key in ("training_permitted", "tuning_permitted", "reward_change_permitted", "candidate_policy_loading_permitted", "combat_finetune_permitted"):
        if design.get("authorization", {}).get(key) is not False:
            raise V18RuntimeError(f"V18 must prohibit {key}")
    if tuple(design.get("opponents", {}).get("order", ())) != OPPONENTS or design["opponents"].get("pooled_gate") != "prohibited":
        raise V18RuntimeError("V18 opponent contract drifted")
    inputs = design["inputs"]
    for key in ("manifest", "runtime_template", "runtime_registry"):
        path = v17._resolve(str(inputs[key]))
        if not path.is_file() or v17._hash(path) != str(inputs[key + "_sha256"]):
            raise V18RuntimeError(f"V18 frozen input mismatch: {key}")
    for key in ("gate", "run_manifest"):
        path = Path(str(inputs["v17_reference"][key]))
        if not path.is_file() or v17._hash(path) != str(inputs["v17_reference"][key + "_sha256"]):
            raise V18RuntimeError(f"V18 V17 reference mismatch: {key}")
    manifest = v17._load(v17._resolve(str(inputs["manifest"])))
    if len(manifest.get("scenarios", [])) != 24:
        raise V18RuntimeError("V18 requires the exact V17 24-scenario universe")
    output = v17._resolve(str(design["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise V18RuntimeError("V18 output root must be absolute and absent")
    free_gb = shutil.disk_usage(output.parent).free / 1024**3
    return {"source_id": SOURCE_ID, "mode": "nonlearning_same_envelope_phase_telemetry_audit", "planned_records": len(OPPONENTS) * len(manifest["scenarios"]), "output_root_absent": True, "free_disk_gb": round(free_gb, 3), "execution_permitted": bool(design.get("authorization", {}).get("execution_permitted"))}


class PhaseTelemetryEpisode(v17.PathContextEpisode):
    """V17-equivalent episode that records current-state telemetry after each step."""

    def run(self, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self.telemetry: list[dict[str, Any]] = []
        result = super().run(scenario, seed)
        result["steps"] = self.telemetry
        return result

    def _observe(self, observation: Mapping[str, Any]) -> None:
        super()._observe(observation)
        if self.total_steps > 0:
            self.telemetry.append(self._telemetry_step())

    def _telemetry_step(self) -> dict[str, Any]:
        if self.base is None:
            raise V18RuntimeError("V18 telemetry requested without base observation")
        base = np.asarray(self.base, dtype=np.float64)
        info = self.last_info
        if base.shape != (16,) or not np.isfinite(base).all():
            raise V18RuntimeError("V18 base observation is invalid")
        if bool(info.get("backend_fallback_occurred", False)) or bool(info.get("prediction_fallback", False)) or info.get("prediction_valid") is not True:
            raise V18RuntimeError("V18 fallback occurred")
        ata = math.atan2(float(base[8]), float(base[9]))
        aa = math.atan2(float(base[10]), float(base[11]))
        own_speed, target_speed, own_altitude, target_altitude = (float(base[index]) for index in (12, 13, 14, 15))
        own_state = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
        return {
            "step": int(self.total_steps), "base_observation_16d": base.tolist(),
            "raw_range_m": float(base[0] * 5000.0), "raw_range_rate_mps": float(base[1] * 200.0),
            "ata_rad": ata, "aa_rad": aa, "los_azimuth_rad": math.atan2(float(base[4]), float(base[5])), "los_elevation_rad": math.atan2(float(base[6]), float(base[7])),
            "own_speed_mps": own_speed, "target_speed_mps": target_speed, "own_altitude_m": own_altitude, "target_altitude_m": target_altitude,
            "own_specific_energy_j_per_kg": 0.5 * own_speed**2 + 9.80665 * own_altitude,
            "target_specific_energy_j_per_kg": 0.5 * target_speed**2 + 9.80665 * target_altitude,
            "dynamic_taxonomy": classify_relative_geometry(math.degrees(ata), math.degrees(aa)), "phase": self._phase(), "first_pass_complete": bool(self.phase_tracker.first_pass_complete),
            "vpp_world_offset_m": self._finite_vector(info.get("vp_world_offset"), "vp_world_offset"), "guidance_command_normalized": self._guidance_vector(info),
            "actual_nz_g": float(own_state.get("nz_g", info.get("nz_g"))),
            "saturation_flags": {"nz": bool(info.get("nz_saturated", False)), "roll_rate": bool(info.get("roll_rate_saturated", False)), "throttle": bool(info.get("throttle_saturated", False))},
        }


def _record(scenario: Mapping[str, Any], opponent: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"source_id": SOURCE_ID, "opponent": opponent, "pair_key": scenario["metadata"]["pair_key"], **dict(result)}


def _require_execution_authorization(design: Mapping[str, Any], scenario_count: int) -> None:
    execution = design.get("execution")
    if design.get("authorization", {}).get("execution_permitted") is not True or not isinstance(execution, Mapping):
        raise V18RuntimeError("V18 execution is not authorized")
    if int(execution.get("expected_records", -1)) != len(OPPONENTS) * scenario_count:
        raise V18RuntimeError("V18 authorization record universe drifted")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if not required_sha or subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False).returncode:
        raise V18RuntimeError("V18 implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise V18RuntimeError("V18 authorization file whitelist is missing")
    for item in files:
        path = v17._resolve(str(item.get("path") or ""))
        if not path.is_file() or v17._hash(path) != str(item.get("sha256") or ""):
            raise V18RuntimeError(f"V18 authorized code hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--execute", action="store_true"); args = parser.parse_args()
    design = _load_design(args.config); preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2)); return 0
    manifest = v17._load(v17._resolve(str(design["inputs"]["manifest"])))
    _require_execution_authorization(design, len(manifest["scenarios"]))
    if not preflight["output_root_absent"]:
        raise V18RuntimeError("V18 output root is not empty")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise V18RuntimeError("V18 requires a clean worktree")
    runtime = _runtime_config(design); registry = v17._load(v17._resolve(str(design["inputs"]["runtime_registry"])))
    output = v17._resolve(str(design["outputs"]["root"])); output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                env = v17.v16.v15.v14.build_record_only_env(runtime, registry, opponent); policy = engine._specialist(registry, "frozen_fixed_head_on")
                encoder = FrozenP3Encoder(Path(runtime["inputs"]["frozen_p3_encoder"]), runtime["inputs"]["frozen_p3_encoder_sha256"])
                episode = PhaseTelemetryEpisode(env, policy, encoder, runtime, v17.v16._compiler(runtime))
                try: result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                finally: episode.close()
                row = _record(scenario, opponent, result); records.append(row); v17._write_json(output / "episodes" / opponent / f"{scenario['name']}.json", {"record": row})
        summary = summarize_records(records, expected=design["expected_v17_phase_replay"], records_per_opponent=int(design["gates"]["records_per_opponent"]))
        v17._write_json(output / "phase_telemetry_gate.json", {"source_id": SOURCE_ID, **summary}); v17._write_json(output / "run_manifest.json", {"source_id": SOURCE_ID, "records": len(records), "config_sha256": v17._hash(args.config), "training": "prohibited", "reward_recording": "prohibited", "runtime_provenance": runtime.get("provenance", {})})
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    except Exception as error:
        v17._write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)}); raise


if __name__ == "__main__": raise SystemExit(main())
