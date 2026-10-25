#!/usr/bin/env python
"""Run a non-learning local difficulty scan around disadvantage."""

from __future__ import annotations

import argparse
import copy
import csv
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
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_method
from uav_vpp_guidance.utils.config import merge_config


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V1-20260903"
DEFAULT_CONFIG = REPO_ROOT / "config/experiment/mild_disadvantage_boundary_scan_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/mild_disadvantage_boundary_scan_v1/scan_v1"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_sha256(path) if path.is_file() else None,
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


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _build_opponent(name: str, config: dict[str, Any]):
    entry = config["opponent_registry"][name]
    if name == "expert":
        return ExpertOpponent(entry.get("config", {})), {"stage": name}
    if name == "end_to_end":
        return EndToEndOpponent.from_registry(entry), {"stage": name, **entry}
    raise ValueError(f"Unsupported opponent: {name}")


def _prepare_config(base: dict[str, Any], scan: dict[str, Any], seed: int, source_id: str) -> dict[str, Any]:
    cfg = merge_config(copy.deepcopy(base), copy.deepcopy(base["methods"]["h0_current_geometry"]))
    source = "run_mild_disadvantage_boundary_scan_v1.py"
    _set_nested(cfg, "experiment.source_id", source_id, f"{source}:source_id")
    _set_nested(cfg, "experiment.name", "mild_disadvantage_boundary_scan_v1", f"{source}:name")
    _set_nested(cfg, "experiment.mode", "evaluation", f"{source}:mode")
    _set_nested(cfg, "experiment.seed", int(seed), f"{source}:seed")
    _set_nested(cfg, "ppo.seed", int(seed), f"{source}:ppo_seed")
    _set_nested(cfg, "backend", "jsbsim", f"{source}:backend")
    _set_nested(cfg, "env.backend", "jsbsim", f"{source}:env_backend")
    _set_nested(cfg, "env.use_jsbsim", True, f"{source}:env_use_jsbsim")
    _set_nested(cfg, "env.strict_backend", True, f"{source}:strict_backend")
    scenario_defs = copy.deepcopy(scan["scan"]["scenarios"])
    for scenario_name, scenario in scenario_defs.items():
        # The evaluator uses scenario.name for the raw record label. Keep it
        # explicit so a boundary point cannot be misclassified as random.
        scenario.setdefault("name", scenario_name)
    _set_nested(cfg, "scenarios", scenario_defs, f"{source}:scan_scenarios")
    _set_nested(cfg, "protocol", scan["protocol"], f"{source}:scan_protocol")
    cfg["scan"] = scan["scan"]
    return cfg


def _run_one(base: dict[str, Any], scan: dict[str, Any], opponent: str, root: Path, source_id: str) -> dict[str, Any]:
    seed = int(scan["scan"]["evaluation_seed"])
    cfg = _prepare_config(base, scan, seed, source_id)
    root.mkdir(parents=True, exist_ok=True)
    resolved = root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    checkpoint = _resolve(cfg["methods"]["h0_current_geometry"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing H0 checkpoint: {checkpoint}")
    opponent_policy, opponent_config = _build_opponent(opponent, cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
    try:
        sample_obs = env.reset(seed=seed)
        agent = PPOAgent(
            obs_dim=int(sample_obs["observation_vector"].shape[0]),
            action_dim=int(cfg["policy"]["action_dim"]),
            config=cfg,
            device="cpu",
        )
        load_info = agent.load(str(checkpoint))
        scenarios = list(scan["scan"]["scenarios"].keys())
        metrics = evaluate_method(
            env,
            agent,
            cfg,
            "h0_current_geometry",
            num_episodes=len(scenarios),
            seeds=[seed],
            scenarios=scenarios,
            save_trajectories=True,
            output_dir=str(root),
            training_seed=int(scan["scan"]["training_seed"]),
        )
        backend = getattr(env, "_backend", None)
        fallback = bool(getattr(env, "_backend_fallback_occurred", False))
        fallback_reason = getattr(env, "_backend_fallback_reason", None)
    finally:
        env.close()
    manifest = {
        "source_id": source_id,
        "phase": "pilot",
        "method": "h0_current_geometry",
        "opponent": opponent,
        "training_started": False,
        "training_permitted": False,
        "training_seed": int(scan["scan"]["training_seed"]),
        "evaluation_seeds": [seed],
        "scenarios": scenarios,
        "episodes_per_scenario": int(scan["scan"]["episodes_per_scenario"]),
        "checkpoint": _receipt(checkpoint),
        "resolved_config": _receipt(resolved),
        "config_sha256": file_sha256(resolved),
        "config_canonical_sha256": config_sha256(cfg, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "backend_requested": "jsbsim",
        "backend_effective": backend,
        "backend_fallback_occurred": fallback,
        "backend_fallback_reason": fallback_reason,
        "obs_dim": int(sample_obs["observation_vector"].shape[0]),
        "action_dim": int(cfg["policy"]["action_dim"]),
        "checkpoint_load": load_info,
        "raw_episode_count": len(metrics.get("raw_episodes", [])),
    }
    _write_json(root / "run_manifest.json", manifest)
    _write_json(root / "metrics.json", metrics)
    return {"manifest": manifest, "metrics": metrics}


def _summaries(scan: dict[str, Any], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = scan["scan"]["scenarios"]
    by_opp: list[dict[str, Any]] = []
    by_scenario: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for result in results:
        manifest = result["manifest"]
        metrics = result["metrics"]
        episodes = list(metrics.get("raw_episodes", []))
        by_opp.append({
            "opponent": manifest["opponent"],
            "method": manifest["method"],
            "episodes": len(episodes),
            "successes": sum(bool(e.get("is_success", False)) for e in episodes),
            "success_rate": metrics.get("success_rate", 0.0),
            "crash_rate": metrics.get("crash_rate", 0.0),
            "out_of_bounds_rate": metrics.get("out_of_bounds_rate", 0.0),
            "mean_return": metrics.get("mean_return", 0.0),
            "backend_effective": manifest["backend_effective"],
            "backend_fallback_occurred": manifest["backend_fallback_occurred"],
        })
        for name, definition in scenarios.items():
            eps = [e for e in episodes if e.get("scenario") == name]
            item = metrics.get("per_scenario", {}).get(name, {})
            terminals: dict[str, int] = {}
            for episode in eps:
                reason = str(episode.get("reason", "unknown"))
                terminals[reason] = terminals.get(reason, 0) + 1
            by_scenario.append({
                "opponent": manifest["opponent"],
                "method": manifest["method"],
                "scenario": name,
                "axis": definition["axis"],
                "parameter_value": definition["parameter_value"],
                "episodes": len(eps),
                "successes": sum(bool(e.get("is_success", False)) for e in eps),
                "success_rate": item.get("success_rate", 0.0),
                "crash_rate": item.get("crash_rate", 0.0),
                "out_of_bounds_rate": item.get("out_of_bounds_rate", 0.0),
                "mean_return": item.get("mean_return", 0.0),
                "mean_final_range_m": item.get("mean_final_range_m", 0.0),
                "mean_final_ata_deg": item.get("mean_final_ata_deg", 0.0),
                "terminal_counts": json.dumps(terminals, sort_keys=True),
            })
        raw.extend({"opponent": manifest["opponent"], **e} for e in episodes)
    return by_opp, by_scenario, raw


def _boundary_analysis(scan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for opponent in scan["protocol"]["opponents"]:
        result[opponent] = {}
        for axis in scan["scan"]["axes"]:
            axis_rows = [r for r in rows if r["opponent"] == opponent and r["axis"] == axis]
            axis_rows.sort(key=lambda r: float(r["parameter_value"]))
            successes = [float(r["parameter_value"]) for r in axis_rows if int(r["successes"]) > 0]
            failures = [float(r["parameter_value"]) for r in axis_rows if int(r["successes"]) == 0]
            transitions = []
            for left, right in zip(axis_rows, axis_rows[1:]):
                left_success = int(left["successes"]) > 0
                right_success = int(right["successes"]) > 0
                if left_success == right_success:
                    continue
                left_level = float(left["parameter_value"])
                right_level = float(right["parameter_value"])
                transitions.append({
                    "lower_level": min(left_level, right_level),
                    "upper_level": max(left_level, right_level),
                    "lower_level_success": left_success if left_level < right_level else right_success,
                    "upper_level_success": right_success if right_level > left_level else left_success,
                    "success_level": left_level if left_success else right_level,
                    "failure_level": left_level if not left_success else right_level,
                })
            result[opponent][axis] = {
                "levels": [float(r["parameter_value"]) for r in axis_rows],
                "success_levels": successes,
                "failure_levels": failures,
                "transition_brackets": transitions,
                "success_to_failure_bracket": transitions[0] if transitions else None,
            }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-id", type=str, default=None)
    parser.add_argument("--execute", action="store_true", help="Required to run JSBSim")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        planned = _load(args.config.resolve())
        print(
            "LOCKED: no simulation started. Planned Source ID: "
            f"{planned.get('experiment', {}).get('source_id', SOURCE_ID)}"
        )
        return 0
    scan = _load(args.config.resolve())
    source_id = args.source_id or scan["experiment"]["source_id"]
    if scan["experiment"]["source_id"] != source_id or scan["protocol"]["source_id"] != source_id:
        raise SystemExit("Source ID mismatch")
    base_path = _resolve(scan["base_config"])
    base = _load(base_path)
    root = args.output_dir.resolve()
    if root.exists() and any(p.is_file() for p in root.rglob("*")):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for opponent in scan["protocol"]["opponents"]:
        results.append(_run_one(base, scan, opponent, root / "h0_current_geometry" / opponent, source_id))
    by_opp, by_scenario, raw = _summaries(scan, results)
    _write_csv(root / "metrics_by_opponent.csv", by_opp)
    _write_csv(root / "metrics_by_scenario.csv", by_scenario)
    _write_csv(root / "episode_level_results.csv", raw)
    boundary = _boundary_analysis(scan, by_scenario)
    _write_json(root / "boundary_analysis.json", boundary)
    _write_json(root / "pilot_manifest.json", {
        "source_id": source_id,
        "phase": "pilot",
        "config": _receipt(args.config.resolve()),
        "base_config": _receipt(base_path),
        "config_canonical_sha256": config_sha256(scan, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "methods": scan["protocol"]["methods"],
        "opponents": scan["protocol"]["opponents"],
        "scenarios": scan["scan"]["scenarios"],
        "axes": scan["scan"]["axes"],
        "training_started": False,
        "training_permitted": False,
        "backend": "jsbsim",
        "runs": [r["manifest"] for r in results],
        "raw_episode_count": len(raw),
        "status": "completed",
    })
    (root / "analysis.md").write_text(
        "# Mild Disadvantage Boundary Scan\n\n"
        f"Source ID: `{source_id}`\n\n"
        "This non-learning scan varies one initial-condition axis at a time. "
        "See `boundary_analysis.json` for the observed success-to-failure brackets.\n",
        encoding="utf-8",
    )
    print(f"Completed disadvantage boundary scan under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
