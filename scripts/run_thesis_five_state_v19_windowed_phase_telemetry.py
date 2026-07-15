#!/usr/bin/env python3
"""Run V19's non-learning, windowed telemetry audit."""

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

import run_thesis_five_state_v18_phase_telemetry_audit as v18
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import classify_relative_geometry
from uav_vpp_guidance.evaluation.thesis_v19_windowed_phase_telemetry import OPPONENTS, SOURCE_ID, summarize_records
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder


class V19RuntimeError(RuntimeError):
    """Raised when V19 would leave its windowed, non-learning contract."""


def _load_design(path: Path) -> dict[str, Any]:
    return v18._load_design(path)


def _runtime_config(design: Mapping[str, Any]) -> dict[str, Any]:
    return v18._runtime_config(design)


def validate_design(config_path: Path) -> dict[str, Any]:
    design = _load_design(config_path)
    if design.get("source_id") != SOURCE_ID or design.get("status") not in {"design_only_not_authorized", "authorized_one_time_execution"}:
        raise V19RuntimeError("V19 design identity drifted")
    for key in ("training_permitted", "tuning_permitted", "reward_change_permitted", "candidate_policy_loading_permitted", "combat_finetune_permitted"):
        if design.get("authorization", {}).get(key) is not False:
            raise V19RuntimeError(f"V19 must prohibit {key}")
    inputs = design["inputs"]
    for key in ("manifest", "runtime_template", "runtime_registry"):
        path = v18.v17._resolve(str(inputs[key]))
        if not path.is_file() or v18.v17._hash(path) != str(inputs[key + "_sha256"]):
            raise V19RuntimeError(f"V19 frozen input mismatch: {key}")
    reference = Path(str(inputs["v17_reference_gate"]))
    if not reference.is_file() or v18.v17._hash(reference) != str(inputs["v17_reference_gate_sha256"]):
        raise V19RuntimeError("V19 V17 reference mismatch")
    manifest = v18.v17._load(v18.v17._resolve(str(inputs["manifest"])))
    output = v18.v17._resolve(str(design["outputs"]["root"]))
    if len(manifest.get("scenarios", [])) != 24 or output.exists() or not output.is_absolute():
        raise V19RuntimeError("V19 requires an absent output and the exact 24-scenario V17 universe")
    return {"source_id": SOURCE_ID, "mode": "nonlearning_windowed_phase_telemetry_audit", "planned_records": 72, "output_root_absent": True, "free_disk_gb": round(shutil.disk_usage(output.parent).free / 1024**3, 3), "execution_permitted": bool(design["authorization"].get("execution_permitted"))}


class WindowedTelemetryEpisode(v18.PhaseTelemetryEpisode):
    """V17-equivalent control with V19's warm-up-aware telemetry classification."""

    def _telemetry_step(self) -> dict[str, Any]:
        if self.base is None:
            raise V19RuntimeError("V19 telemetry requested without a base observation")
        base = np.asarray(self.base, dtype=np.float64)
        info = self.last_info
        if base.shape != (16,) or not np.isfinite(base).all() or bool(info.get("backend_fallback_occurred", False)) or bool(info.get("prediction_fallback", False)):
            raise V19RuntimeError("V19 telemetry contract failure")
        ata, aa = math.atan2(float(base[8]), float(base[9])), math.atan2(float(base[10]), float(base[11]))
        own_speed, target_speed, own_altitude, target_altitude = (float(base[index]) for index in (12, 13, 14, 15))
        own_state = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
        history_length = len(self.history)
        return {"step": int(self.total_steps), "base_observation_16d": base.tolist(), "phase": self._phase(), "first_pass_complete": bool(self.phase_tracker.first_pass_complete), "history_length": history_length, "prediction_valid": bool(info.get("prediction_valid", False)), "prediction_fallback": bool(info.get("prediction_fallback", False)), "raw_range_m": float(base[0] * 5000.0), "raw_range_rate_mps": float(base[1] * 200.0), "ata_rad": ata, "aa_rad": aa, "own_speed_mps": own_speed, "target_speed_mps": target_speed, "own_altitude_m": own_altitude, "target_altitude_m": target_altitude, "own_specific_energy_j_per_kg": 0.5 * own_speed**2 + 9.80665 * own_altitude, "target_specific_energy_j_per_kg": 0.5 * target_speed**2 + 9.80665 * target_altitude, "dynamic_taxonomy": classify_relative_geometry(math.degrees(ata), math.degrees(aa)), "vpp_world_offset_m": self._finite_vector(info.get("vp_world_offset"), "vp_world_offset"), "guidance_command_normalized": self._guidance_vector(info), "actual_nz_g": float(own_state.get("nz_g", info.get("nz_g"))), "saturation_flags": {"nz": bool(info.get("nz_saturated", False)), "roll_rate": bool(info.get("roll_rate_saturated", False)), "throttle": bool(info.get("throttle_saturated", False))}}


def _authorize(design: Mapping[str, Any], scenario_count: int) -> None:
    execution = design.get("execution")
    if design["authorization"].get("execution_permitted") is not True or not isinstance(execution, Mapping) or int(execution.get("expected_records", -1)) != scenario_count * len(OPPONENTS):
        raise V19RuntimeError("V19 execution is not authorized")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if not required_sha or subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False).returncode:
        raise V19RuntimeError("V19 implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise V19RuntimeError("V19 authorization file whitelist is missing")
    for item in files:
        path = v18.v17._resolve(str(item.get("path") or ""))
        if not path.is_file() or v18.v17._hash(path) != str(item.get("sha256") or ""):
            raise V19RuntimeError(f"V19 authorized code hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--execute", action="store_true"); args = parser.parse_args()
    design = _load_design(args.config); preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2)); return 0
    manifest = v18.v17._load(v18.v17._resolve(str(design["inputs"]["manifest"]))); _authorize(design, len(manifest["scenarios"]))
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(): raise V19RuntimeError("V19 requires a clean worktree")
    runtime, registry, output, records = _runtime_config(design), v18.v17._load(v18.v17._resolve(str(design["inputs"]["runtime_registry"]))), v18.v17._resolve(str(design["outputs"]["root"])), []
    output.mkdir(parents=True)
    try:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                env = v18.v17.v16.v15.v14.build_record_only_env(runtime, registry, opponent); policy = engine._specialist(registry, "frozen_fixed_head_on"); encoder = FrozenP3Encoder(Path(runtime["inputs"]["frozen_p3_encoder"]), runtime["inputs"]["frozen_p3_encoder_sha256"]); episode = WindowedTelemetryEpisode(env, policy, encoder, runtime, v18.v17.v16._compiler(runtime))
                try: result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                finally: episode.close()
                row = {"source_id": SOURCE_ID, "opponent": opponent, "pair_key": scenario["metadata"]["pair_key"], **result}; records.append(row); v18.v17._write_json(output / "episodes" / opponent / f"{scenario['name']}.json", {"record": row})
        summary = summarize_records(records, expected=design["expected_v17_phase_replay"], records_per_opponent=24, minimum_ready_steps=int(design["gates"]["per_opponent_analysis_ready_steps_min"])); v18.v17._write_json(output / "windowed_phase_telemetry_gate.json", {"source_id": SOURCE_ID, **summary}); v18.v17._write_json(output / "run_manifest.json", {"source_id": SOURCE_ID, "records": len(records), "config_sha256": v18.v17._hash(args.config), "training": "prohibited", "reward_recording": "prohibited", "runtime_provenance": runtime.get("provenance", {})}); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    except Exception as error:
        v18.v17._write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)}); raise


if __name__ == "__main__": raise SystemExit(main())
