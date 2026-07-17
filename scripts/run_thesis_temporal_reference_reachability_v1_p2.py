#!/usr/bin/env python3
"""P2: replay frozen handoffs and screen temporal references without executing them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_local_action_authority_map_v1 as historical
from uav_vpp_guidance.evaluation.temporal_reference_compiler import (
    TemporalReferenceConfig,
    TemporalReferenceTemplate,
    compile_temporal_reference_program,
)
from uav_vpp_guidance.evaluation.temporal_reference_p2 import summarize_rows


SOURCE_ID = "THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1"
P2_STAGE = "P2_offline_envelope_screen"


class TemporalReferenceP2RunnerError(RuntimeError):
    """Raised when a P2 baseline replay would violate its non-learning contract."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TemporalReferenceP2RunnerError(f"YAML mapping required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _design(path: Path) -> dict[str, Any]:
    config = _load(path)
    if config.get("source_id") != SOURCE_ID or config.get("stage") != P2_STAGE:
        raise TemporalReferenceP2RunnerError("P2 source ID or stage drifted")
    authorization = config.get("authorization", {})
    required = {
        "execution_permitted": True,
        "baseline_replay_permitted": True,
        "candidate_action_execution_permitted": False,
        "training_permitted": False,
        "tuning_permitted": False,
        "reward_change_permitted": False,
        "guidance_pid_change_permitted": False,
        "snapshot_restore_permitted": False,
        "future_state_injection_permitted": False,
        "history_padding_permitted": False,
    }
    if any(authorization.get(key) is not value for key, value in required.items()):
        raise TemporalReferenceP2RunnerError("P2 authorization contract drifted")
    return config


def _validated_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = config["inputs"]
    for key in ("p0_source_manifest", "p1_design", "historical_cohort"):
        path = _resolve(inputs[key])
        expected = str(inputs[key + "_sha256"])
        if not path.is_file() or _hash(path) != expected:
            raise TemporalReferenceP2RunnerError(f"P2 frozen input hash mismatch: {key}")
    source = json.loads(_resolve(inputs["p0_source_manifest"]).read_text(encoding="utf-8"))
    if source.get("source_id") != SOURCE_ID or source.get("status") != "p0_p1_frozen_design_execution_not_authorized":
        raise TemporalReferenceP2RunnerError("P0 source snapshot is not valid")
    design = _load(_resolve(inputs["p1_design"]))
    reference = design.get("temporal_reference")
    if not isinstance(reference, dict) or design.get("source_id") != SOURCE_ID:
        raise TemporalReferenceP2RunnerError("P1 reference design is invalid")
    cohort = _load(_resolve(inputs["historical_cohort"]))
    if cohort.get("source_id") != inputs["historical_cohort_source_id"] or len(cohort.get("entries", [])) != 12:
        raise TemporalReferenceP2RunnerError("historical handoff cohort drifted")
    return reference, cohort


def _compiler_config(reference: Mapping[str, Any]) -> TemporalReferenceConfig:
    return TemporalReferenceConfig(
        horizon_steps=int(reference["horizon_steps"]),
        high_level_dt_s=float(reference["high_level_dt_s"]),
        max_action_delta=float(reference["max_action_delta"]),
        max_action_slew=float(reference["max_action_slew"]),
        max_basis_condition_number=float(reference["max_basis_condition_number"]),
    )


def _active_basis(ep: Any) -> tuple[dict[str, Any], dict[str, Any], list[list[float]]]:
    own, target = ep.env._get_current_states(noisy=True)
    target = ep.env._task_get_target_state(target, own)
    own, target = ep.env._task_pre_step(own, target)
    generator = getattr(ep.env, "virtual_point_generator", None)
    if generator is None or str(getattr(generator, "action_semantics", "")) != "tactical_basis_v1":
        raise TemporalReferenceP2RunnerError("P2 replay did not expose tactical_basis_v1")
    vectors = generator._build_tactical_basis_vectors(own, target)[:3]
    basis = [np.asarray(vector, dtype=np.float64).reshape(3).tolist() for vector in vectors]
    return dict(own), dict(target), basis


def _baseline_continuation(entry: Mapping[str, Any], steps: int) -> dict[str, Any]:
    """Replay one legal handoff and record baseline-only post-handoff inputs."""

    source_config, scenario, registry = historical._source(entry)
    episode = historical._episode(entry, source_config, registry)
    try:
        matched, replay_hash = historical._prepare(episode, entry, scenario)
        record: dict[str, Any] = {
            "cohort_key": str(entry["cohort_key"]),
            "opponent": str(entry["opponent"]),
            "scenario_signature": str(entry["scenario_signature"]),
            "mirror_sign": str(scenario["metadata"].get("mirror_sign", "unknown")),
            "historical_handoff_hash": str(entry["historical_handoff_hash"]),
            "replay_handoff_hash": replay_hash,
            "replay_matched": bool(matched),
            "strict_jsbsim": bool(getattr(episode.env, "_backend", None) == "jsbsim"),
            "baseline_complete": False,
            "backend_fallback": False,
            "prediction_fallback": False,
            "baseline_steps": 0,
            "baseline": [],
        }
        if not matched:
            return record
        done = False
        for step in range(steps):
            own, target, basis = _active_basis(episode)
            reference = historical.v15.v14.v11.v9.finite_action(
                episode.policy.get_deterministic_action(episode.observation["observation_vector"])
            )
            observation, _reward, terminated, truncated, info = episode.env.step(
                np.asarray(reference, dtype=np.float32)
            )
            episode.total_steps += 1
            episode.last_info = info
            episode._observe(observation)
            backend_fallback = bool(info.get("backend_fallback_occurred", False))
            prediction_fallback = bool(info.get("prediction_fallback", False))
            record["backend_fallback"] = bool(record["backend_fallback"] or backend_fallback)
            record["prediction_fallback"] = bool(record["prediction_fallback"] or prediction_fallback)
            if backend_fallback or prediction_fallback:
                break
            own_after = info.get("own_state", {})
            record["baseline"].append(
                {
                    "step": step,
                    "reference_action": np.asarray(reference, dtype=np.float64).tolist(),
                    "basis_vectors_world_m": basis,
                    "own_position_neu": own.get("position_neu"),
                    "target_position_neu": target.get("position_neu"),
                    "vp_world_offset_m": np.asarray(info.get("vp_world_offset"), dtype=np.float64).reshape(3).tolist(),
                    "guidance_command": info.get("guidance_command"),
                    "aoa_rad": float(own_after.get("alpha_rad", own_after.get("aoa_rad", 0.0))),
                    "actual_nz_g": float(own_after.get("nz_g", info.get("nz_g", 0.0))),
                    "phase": str(episode._phase()),
                }
            )
            if terminated or truncated:
                done = True
                break
        record["baseline_steps"] = len(record["baseline"])
        record["baseline_complete"] = bool(
            not done
            and record["baseline_steps"] == steps
            and not record["backend_fallback"]
            and not record["prediction_fallback"]
        )
        return record
    finally:
        episode.close()


def _variants(reference: Mapping[str, Any]) -> list[tuple[str, TemporalReferenceTemplate, int, int]]:
    values: list[tuple[str, TemporalReferenceTemplate, int, int]] = []
    for name, value in reference["templates"].items():
        template = TemporalReferenceTemplate.from_mapping(str(name), value)
        for lateral_sign in value["lateral_sign_candidates"]:
            for vertical_sign in value["vertical_sign_candidates"]:
                values.append((f"{name}:lat{int(lateral_sign):+d}:vert{int(vertical_sign):+d}", template, int(lateral_sign), int(vertical_sign)))
    return values


def _candidate_row(
    baseline: Mapping[str, Any],
    *,
    candidate_variant: str,
    template: TemporalReferenceTemplate,
    lateral_sign: int,
    vertical_sign: int,
    compiler_config: TemporalReferenceConfig,
) -> dict[str, Any]:
    steps = baseline.get("baseline", [])
    row: dict[str, Any] = {
        key: baseline.get(key)
        for key in (
            "cohort_key",
            "opponent",
            "scenario_signature",
            "mirror_sign",
            "historical_handoff_hash",
            "replay_handoff_hash",
            "replay_matched",
            "strict_jsbsim",
            "baseline_complete",
            "backend_fallback",
            "prediction_fallback",
            "baseline_steps",
        )
    }
    row["candidate_variant"] = candidate_variant
    if len(steps) != compiler_config.horizon_steps:
        row.update(
            {
                "action_box_limited_fraction": 1.0,
                "mean_projection_residual_m": float("inf"),
                "peak_actual_reference_delta_m": 0.0,
                "max_basis_condition_number": float("inf"),
            }
        )
        return row
    receipts = compile_temporal_reference_program(
        template=template,
        reference_actions=[step["reference_action"] for step in steps],
        basis_vectors_by_step=[step["basis_vectors_world_m"] for step in steps],
        config=compiler_config,
        lateral_sign=lateral_sign,
        vertical_sign=vertical_sign,
    )
    row.update(
        {
            "action_box_limited_fraction": float(
                np.mean([bool(receipt["action_box_limited"]) for receipt in receipts])
            ),
            "slew_limited_fraction": float(
                np.mean([bool(receipt["slew_limited"]) for receipt in receipts])
            ),
            "mean_projection_residual_m": float(
                np.mean([float(receipt["projection_residual_norm_m"]) for receipt in receipts])
            ),
            "peak_actual_reference_delta_m": float(
                max(
                    np.linalg.norm(np.asarray(receipt["actual_reference_world_delta_m"], dtype=np.float64))
                    for receipt in receipts
                )
            ),
            "max_basis_condition_number": float(
                max(float(receipt["basis_condition_number"]) for receipt in receipts)
            ),
            "receipts": receipts,
        }
    )
    return row


def validate_design(path: Path) -> dict[str, Any]:
    config = _design(path)
    reference, cohort = _validated_inputs(config)
    compiler = _compiler_config(reference)
    output = _resolve(config["outputs"]["root"])
    if not output.is_absolute() or output.exists():
        raise TemporalReferenceP2RunnerError("P2 output root must be absolute and absent")
    return {
        "source_id": SOURCE_ID,
        "stage": P2_STAGE,
        "historical_handoff_states": len(cohort["entries"]),
        "candidate_variants": len(_variants(reference)),
        "baseline_continuation_steps": compiler.horizon_steps,
        "candidate_actions_enter_env": False,
        "output_root_absent": True,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _design(config_path)
    reference, cohort = _validated_inputs(config)
    compiler = _compiler_config(reference)
    output = _resolve(config["outputs"]["root"])
    if output.exists():
        raise TemporalReferenceP2RunnerError("P2 refuses to overwrite an existing output root")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise TemporalReferenceP2RunnerError("P2 requires a clean worktree before the one authorized replay")
    output.mkdir(parents=True)
    baseline_records: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    try:
        for entry in cohort["entries"]:
            if str(entry.get("historical_source_kind")) != config["collection_contract"]["historical_source_kind"]:
                raise TemporalReferenceP2RunnerError("P2 cohort contains an unauthorized historical source")
            baseline = _baseline_continuation(entry, compiler.horizon_steps)
            baseline_records.append(baseline)
            safe = str(entry["cohort_key"]).replace(":", "__").replace("/", "_")
            _write_json(output / "baseline_records" / str(entry["opponent"]) / f"{safe}.json", {"record": baseline})
            for variant, template, lateral_sign, vertical_sign in _variants(reference):
                row = _candidate_row(
                    baseline,
                    candidate_variant=variant,
                    template=template,
                    lateral_sign=lateral_sign,
                    vertical_sign=vertical_sign,
                    compiler_config=compiler,
                )
                candidate_rows.append(row)
        summary = summarize_rows(candidate_rows, config["gates"])
        _write_json(
            output / "run_manifest.json",
            {
                "source_id": SOURCE_ID,
                "stage": P2_STAGE,
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "config_sha256": _hash(config_path),
                "p0_source_manifest_sha256": str(config["inputs"]["p0_source_manifest_sha256"]),
                "historical_cohort_sha256": str(config["inputs"]["historical_cohort_sha256"]),
                "candidate_actions_enter_env": False,
                "baseline_replays": len(baseline_records),
            },
        )
        _write_json(output / "offline_envelope_summary.json", {"source_id": SOURCE_ID, "stage": P2_STAGE, **summary})
        _write_json(output / "baseline_replay_index.json", {"source_id": SOURCE_ID, "records": baseline_records})
        _write_json(output / "p2_verification.json", {"source_id": SOURCE_ID, "stage": P2_STAGE, **summary})
        fields = [key for key in candidate_rows[0] if key != "receipts"]
        with (output / "offline_envelope_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in candidate_rows:
                writer.writerow({key: row.get(key) for key in fields})
        return {"output": str(output), "baseline_records": len(baseline_records), **summary}
    except Exception as error:
        _write_json(
            output / "failure.json",
            {"source_id": SOURCE_ID, "stage": P2_STAGE, "error": repr(error), "baseline_records": len(baseline_records), "candidate_rows": len(candidate_rows)},
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(validate_design(args.config), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
