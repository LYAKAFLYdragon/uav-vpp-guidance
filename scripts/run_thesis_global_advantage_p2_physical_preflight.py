"""Strict, non-learning JSBSim reachability preflight for P2-B2."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.global_advantage_runin_contract import base_observation
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as pilot
from uav_vpp_guidance.training.thesis_shared_skill_geometry import PhaseTracker


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT-V1"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_physical_preflight.yaml"


class PreflightError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise PreflightError(f"Expected YAML mapping: {path}")
    return payload


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _finite_vector(value: Any, name: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise PreflightError(f"{name} must be finite 3-D")
    return [float(item) for item in vector]


def _finite_state(state: Mapping[str, Any], name: str) -> dict[str, float]:
    result = {key: float(state[key]) for key in ("speed_mps", "altitude_m", "nz_g")}
    if not all(math.isfinite(value) for value in result.values()):
        raise PreflightError(f"{name} contains non-finite values")
    return result


def build_preflight_scenarios(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("source_id") != "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1":
        raise PreflightError("unexpected preflight manifest source")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != 60 or len({item["metadata"]["geometry_cell_id"] for item in scenarios}) != 60:
        raise PreflightError("preflight manifest must contain 60 unique geometry cells")
    return scenarios


def _episode(
    runtime: Mapping[str, Any], registry: Mapping[str, Any], scenario: Mapping[str, Any], opponent: str, maximum_steps: int
) -> dict[str, Any]:
    env = pilot._build_env(runtime, registry, opponent)
    reference = pilot._specialist(registry, "run_in_head_on")
    try:
        observation = env.reset(scenario=dict(scenario), seed=int(scenario["metadata"]["scenario_seed"]))
        if getattr(env, "_backend", None) != "jsbsim":
            raise PreflightError("strict JSBSim backend was not selected")
        tracker = PhaseTracker()
        steps: list[dict[str, Any]] = []
        terminal_reason = "horizon"
        for index in range(1, maximum_steps + 1):
            action = _finite_vector(reference.get_deterministic_action(observation["observation_vector"]), "reference action")
            next_observation, _reward, terminated, truncated, info = env.step(action)
            own = _finite_state(info.get("own_state", {}), "own state")
            target = _finite_state(info.get("target_state", {}), "target state")
            base = base_observation(next_observation)
            phase = tracker.update(base["range_m"], base["range_rate_mps"])
            fallback = bool(info.get("prediction_fallback", False)) and info.get("prediction_fallback_phase") != "warmup"
            record = {
                "step": index,
                "phase": phase,
                "first_pass_complete": bool(info.get("first_pass_complete", False)),
                "backend": info.get("backend"),
                "backend_fallback": bool(info.get("backend_fallback_occurred", False)),
                "prediction_fallback": fallback,
                "action": action,
                "own": own,
                "target": target,
                "terminal_reason": info.get("reason"),
            }
            steps.append(record)
            observation = next_observation
            if terminated or truncated:
                terminal_reason = str(info.get("reason") or "terminated")
                break
        return {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "scenario_name": scenario["name"],
            "geometry_cell_id": scenario["metadata"]["geometry_cell_id"],
            "step_count": len(steps),
            "terminal_reason": terminal_reason,
            "first_pass_observed": any(item["first_pass_complete"] for item in steps),
            "phase_counts": dict(Counter(item["phase"] for item in steps)),
            "valid": bool(steps)
            and all(item["backend"] == "jsbsim" and not item["backend_fallback"] and not item["prediction_fallback"] for item in steps),
            "steps": steps,
        }
    finally:
        env.close()


def run(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID or config["authorization"].get("execution_permitted") is not True:
        raise PreflightError("P2 physical preflight is not authorised")
    if any(config["authorization"].get(key) is not False for key in ("training_permitted", "tuning_permitted", "candidate_policy_loading_permitted", "heldout_claims_permitted")):
        raise PreflightError("P2 physical preflight authorization drifted")
    contract = config["contract"]
    if contract["backend"] != "jsbsim" or contract["strict_backend"] is not True or contract["fresh_environment_per_episode"] is not True:
        raise PreflightError("P2 physical contract drifted")
    manifest = _load_yaml(_path(config["inputs"]["manifest"]))
    runtime = _load_yaml(_path(config["inputs"]["runtime_config"]))
    registry = _load_yaml(_path(config["inputs"]["runtime_registry"]))
    scenarios = build_preflight_scenarios(manifest)
    opponents = tuple(contract["opponents"])
    if opponents != ("expert", "end_to_end", "independent_ppo_vpp"):
        raise PreflightError("opponent contract drifted")
    output = Path(config["outputs"]["root"])
    if output.exists():
        raise PreflightError("output root must be fresh")
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for opponent in opponents:
            for scenario in scenarios:
                record = _episode(runtime, registry, scenario, opponent, int(contract["max_high_level_steps"]))
                _write_json(output / "episodes" / opponent / f"{scenario['name']}.json", record)
                records.append(record)
        minimum = int(contract["min_valid_steps"])
        expected = len(scenarios) * len(opponents)
        phase_matrix = {
            opponent: {
                phase: sum(record["phase_counts"].get(phase, 0) for record in records if record["opponent"] == opponent)
                for phase in ("pre_merge", "post_merge", "re_entry")
            }
            for opponent in opponents
        }
        passed = len(records) == expected and all(record["valid"] and record["step_count"] >= minimum for record in records)
        gate = {"source_id": SOURCE_ID, "record_count": len(records), "expected_record_count": expected, "passed": passed, "phase_matrix": phase_matrix, "phase_gaps": {opponent: [phase for phase, count in values.items() if count == 0] for opponent, values in phase_matrix.items()}}
        _write_json(output / "physical_preflight_gate.json", gate)
        _write_json(output / "run_manifest.json", {"source_id": SOURCE_ID, "records": records, "config_sha256": _sha256(config_path), "manifest_sha256": _sha256(_path(config["inputs"]["manifest"])), "gate": gate})
        return gate
    except Exception:
        _write_json(output / "runner_failure_manifest.json", {"source_id": SOURCE_ID, "completed_record_count": len(records)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    if not args.execute:
        print(json.dumps({"source_id": config.get("source_id"), "mode": "validated_not_executed", "execution_permitted": config["authorization"]["execution_permitted"], "planned_records": 180}, indent=2))
        return 0
    gate = run(args.config)
    print(json.dumps(gate, indent=2))
    return 0 if gate["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
