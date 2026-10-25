#!/usr/bin/env python
"""Run a crossing-training-contract-matched, non-learning recovery probe."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-CONTRACT-PROBE-V1-20260903"
DEFAULT_CONFIG = REPO_ROOT / "config/experiment/mild_disadvantage_envelope_recovery_matched_probe_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1"
LEGACY_RUNNER = REPO_ROOT / "scripts/run_mild_disadvantage_envelope_recovery_probe_v1.py"
_LEGACY_MODULE = None


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("recovery_probe_helpers", LEGACY_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_module():
    global _LEGACY_MODULE
    if _LEGACY_MODULE is None:
        _LEGACY_MODULE = _load_legacy_module()
    return _LEGACY_MODULE


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


def _prepare_config(base: dict[str, Any], spec: dict[str, Any], opponent: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    source = "run_mild_disadvantage_envelope_recovery_matched_probe_v1.py"
    _set_nested(cfg, "experiment.name", spec["experiment"]["name"], f"{source}:name")
    _set_nested(cfg, "experiment.source_id", SOURCE_ID, f"{source}:source_id")
    _set_nested(cfg, "experiment.mode", "evaluation", f"{source}:mode")
    _set_nested(cfg, "task.name", spec["matched_contract"]["task_name"], f"{source}:task_context")
    _set_nested(cfg, "backend", "jsbsim", f"{source}:backend")
    _set_nested(cfg, "env.backend", "jsbsim", f"{source}:env_backend")
    _set_nested(cfg, "env.use_jsbsim", True, f"{source}:env_use_jsbsim")
    _set_nested(cfg, "env.strict_backend", True, f"{source}:strict_backend")
    _set_nested(cfg, "env.max_high_level_steps", int(spec["probe"]["max_high_level_steps"]), f"{source}:max_steps")
    _set_nested(cfg, "trajectory_prediction.checkpoint_path", str(_resolve(spec["assets"]["crossing_predictor"])), f"{source}:predictor_asset")
    _set_nested(cfg, "trajectory_prediction.allow_device_fallback", False, f"{source}:predictor_device")
    _set_nested(cfg, "observation.include_opponent_stage", True, f"{source}:observation_contract")
    end_to_end = _resolve(spec["assets"]["end_to_end_checkpoint"])
    _set_nested(cfg, "opponent_registry.end_to_end.checkpoint", str(end_to_end), f"{source}:end_to_end_asset")
    _set_nested(cfg, "opponent_registry.end_to_end.device", "cpu", f"{source}:end_to_end_device")
    _set_nested(cfg, "opponent_registry.expert.stage", "expert", f"{source}:expert_stage")
    _set_nested(cfg, "opponent_registry.end_to_end.stage", "end_to_end", f"{source}:end_to_end_stage")
    legacy = _legacy_module()
    scenarios = {
        name: legacy._make_scenario(name, point, spec["probe"]["fixed_target_conditions"])
        for name, point in spec["probe"]["candidates"].items()
    }
    _set_nested(cfg, "scenarios", scenarios, f"{source}:candidate_scenarios")
    _set_nested(cfg, "protocol.source_id", SOURCE_ID, f"{source}:protocol_source_id")
    _set_nested(cfg, "protocol.opponent", opponent, f"{source}:protocol_opponent")
    return cfg


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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
    predictor_path = _resolve(spec["assets"]["crossing_predictor"])
    end_to_end_path = _resolve(spec["assets"]["end_to_end_checkpoint"])
    for path in (h0_path, crossing_path, crossing_config, predictor_path, end_to_end_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    opponent_policy, opponent_config = _build_opponent(opponent, cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
    try:
        h0_agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
        h0_load = h0_agent.load(str(h0_path))
        crossing_policy = FrozenSpecialistPolicy(str(crossing_path), str(crossing_config), device="cpu")
        legacy = _legacy_module()
        h0_policy = legacy._TruncateTaskFlagAdapter(h0_agent)
        stable_points = [legacy.EnvelopePoint(float(p[0]), float(p[1])) for p in spec["recovery"]["stable_points_m"]]
        recovery_cfg = spec["recovery"]
        rows: list[dict[str, Any]] = []
        repeats = int(spec["probe"]["repetitions_per_candidate_and_opponent"])
        seed_base = int(spec["probe"]["evaluation_seed_base"])
        calibration = spec["probe"]["calibration"]
        for candidate_index, (candidate, point) in enumerate(spec["probe"]["candidates"].items()):
            scenario = legacy._make_scenario(candidate, point, spec["probe"]["fixed_target_conditions"])
            for repetition in range(repeats):
                seed = seed_base + candidate_index * 100 + (0 if opponent == "expert" else 50) + repetition
                recovery = legacy.CapabilityEnvelopeRecoveryPolicy(stable_points, **{
                    key: recovery_cfg[key]
                    for key in ("nominal_mode", "recovery_mode", "x_tolerance_m", "y_tolerance_m", "max_recovery_steps", "required_dwell_steps", "max_clipping_rate", "min_altitude_m", "max_range_m")
                })
                result, trajectory = legacy._run_episode(
                    env, h0_policy, crossing_policy, recovery, scenario, seed,
                    opponent, candidate, calibration,
                )
                result["matched_contract"] = True
                result["task_name"] = spec["matched_contract"]["task_name"]
                rows.append(result)
                _write_csv(root / "trajectories" / f"{candidate}_{repetition + 1:02d}.csv", trajectory)
    finally:
        env.close()
    run_manifest = {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "status": "completed",
        "training_started": False,
        "training_permitted": False,
        "backend_requested": "jsbsim",
        "backend_effective": getattr(env, "_backend", None),
        "backend_fallback_occurred": bool(getattr(env, "_backend_fallback_occurred", False)),
        "task_name": spec["matched_contract"]["task_name"],
        "h0_checkpoint": _receipt(h0_path),
        "h0_checkpoint_obs_dim": int(h0_load.get("obs_dim", 0)),
        "h0_checkpoint_action_dim": int(h0_load.get("action_dim", 0)),
        "crossing_checkpoint": _receipt(crossing_path),
        "crossing_config": _receipt(crossing_config),
        "crossing_predictor": _receipt(predictor_path),
        "crossing_checkpoint_obs_dim": crossing_policy.obs_dim,
        "crossing_checkpoint_action_dim": crossing_policy.action_dim,
        "end_to_end_checkpoint": _receipt(end_to_end_path),
        "resolved_config": _receipt(resolved),
        "config_sha256": file_sha256(resolved),
        "config_canonical_sha256": config_sha256(cfg, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "episode_count": len(rows),
        "reset_calls_total": sum(int(row["reset_calls"]) for row in rows),
        "observation_adapters": {"h0": "truncate_task_flag_suffix_only", "crossing": "exact_18d"},
        "matched_contract": spec["matched_contract"],
    }
    _write_json(root / "run_manifest.json", run_manifest)
    _write_json(root / "episode_results.json", {"source_id": SOURCE_ID, "opponent": opponent, "episodes": rows})
    return rows, run_manifest


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
        "task_name": spec["matched_contract"]["task_name"],
        "matched_contract": spec["matched_contract"],
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
    print(f"Completed 12-episode matched-contract probe under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
