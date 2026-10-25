#!/usr/bin/env python
"""Run a non-learning pilot on a simplified scenario ladder.

Only the initial scenario definitions differ from the formal H0 protocol.
The frozen H0 policy, opponent implementations, reward, action interface and
JSBSim execution chain are kept unchanged.
"""

from __future__ import annotations

import argparse
import csv
import copy
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


SOURCE_ID = "THESIS-SIMPLIFIED-SCENARIO-FEASIBILITY-PILOT-V1-20260903"
DEFAULT_CONFIG = REPO_ROOT / "config/experiment/simplified_scenario_feasibility_pilot_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/simplified_scenario_feasibility_pilot_v1/pilot_v1"


def _receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def _set_nested(config: dict[str, Any], key: str, value: Any, source: str) -> None:
    current: Any = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    old = current.get(parts[-1])
    current[parts[-1]] = value
    record_config_override_if_changed(config, key, value, old, source)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _build_opponent(name: str, config: dict[str, Any]):
    registry = config["opponent_registry"]
    if name == "expert":
        return ExpertOpponent(registry["expert"].get("config", {})), {"stage": name}
    if name == "end_to_end":
        entry = registry["end_to_end"]
        return EndToEndOpponent.from_registry(entry), {"stage": name, **entry}
    raise ValueError(f"Unsupported opponent: {name}")


def _assert_new_output(path: Path) -> None:
    if path.exists() and any(item.is_file() for item in path.rglob("*")):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {path}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_config(base: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = merge_config(copy.deepcopy(base), copy.deepcopy(base["methods"]["h0_current_geometry"]))
    source = "run_simplified_scenario_feasibility_pilot_v1.py"
    _set_nested(cfg, "experiment.source_id", SOURCE_ID, f"{source}:source_id")
    _set_nested(cfg, "experiment.mode", "evaluation", f"{source}:mode")
    _set_nested(cfg, "experiment.seed", int(seed), f"{source}:seed")
    _set_nested(cfg, "ppo.seed", int(seed), f"{source}:ppo_seed")
    _set_nested(cfg, "backend", "jsbsim", f"{source}:backend")
    _set_nested(cfg, "env.backend", "jsbsim", f"{source}:env_backend")
    _set_nested(cfg, "env.use_jsbsim", True, f"{source}:env_use_jsbsim")
    _set_nested(cfg, "env.strict_backend", True, f"{source}:strict_backend")
    return cfg


def _run_one(
    base: dict[str, Any],
    opponent_name: str,
    output_dir: Path,
    evaluation_seed: int,
    scenarios: list[str],
    episodes_per_scenario: int,
) -> dict[str, Any]:
    cfg = _prepare_config(base, evaluation_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    opponent, opponent_config = _build_opponent(opponent_name, cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent, opponent_config=opponent_config)
    checkpoint = _resolve(cfg["methods"]["h0_current_geometry"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing frozen H0 checkpoint: {checkpoint}")
    try:
        sample_obs = env.reset(seed=evaluation_seed)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(cfg["policy"]["action_dim"])
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=cfg, device="cpu")
        load_info = agent.load(str(checkpoint))
        metrics = evaluate_method(
            env,
            agent,
            cfg,
            "h0_current_geometry",
            num_episodes=episodes_per_scenario * len(scenarios),
            seeds=[evaluation_seed],
            scenarios=scenarios,
            save_trajectories=True,
            output_dir=str(output_dir),
            training_seed=9201,
        )
    finally:
        backend = getattr(env, "_backend", None)
        fallback = bool(getattr(env, "_backend_fallback_occurred", False))
        fallback_reason = getattr(env, "_backend_fallback_reason", None)
        env.close()

    manifest = {
        "source_id": SOURCE_ID,
        "phase": "pilot",
        "method": "h0_current_geometry",
        "opponent": opponent_name,
        "training_started": False,
        "training_permitted": False,
        "training_seed": 9201,
        "evaluation_seeds": [evaluation_seed],
        "scenarios": scenarios,
        "episodes_per_scenario": episodes_per_scenario,
        "checkpoint": _receipt(checkpoint),
        "resolved_config": _receipt(resolved_config_path),
        "config_sha256": file_sha256(resolved_config_path),
        "config_canonical_sha256": config_sha256(cfg, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "backend_requested": "jsbsim",
        "backend_effective": backend,
        "backend_fallback_occurred": fallback,
        "backend_fallback_reason": fallback_reason,
        "observation_schema": sample_obs.get("observation_schema", {}),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "checkpoint_load": load_info,
        "raw_episode_count": len(metrics.get("raw_episodes", [])),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "metrics.json", metrics)
    return {"manifest": manifest, "metrics": metrics}


def _summary_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    opponent_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for result in results:
        manifest = result["manifest"]
        metrics = result["metrics"]
        episodes = list(metrics.get("raw_episodes", []))
        overall = metrics.get("per_scenario", {})
        opponent_rows.append({
            "opponent": manifest["opponent"],
            "method": manifest["method"],
            "episodes": len(episodes),
            "successes": sum(bool(item.get("is_success", False)) for item in episodes),
            "success_rate": metrics.get("success_rate", 0.0),
            "mean_return": metrics.get("mean_return", 0.0),
            "backend_effective": manifest["backend_effective"],
            "backend_fallback_occurred": manifest["backend_fallback_occurred"],
        })
        for scenario, item in overall.items():
            scenario_episodes = [episode for episode in episodes if episode.get("scenario") == scenario]
            terminals: dict[str, int] = {}
            for episode in scenario_episodes:
                reason = str(episode.get("reason", "unknown"))
                terminals[reason] = terminals.get(reason, 0) + 1
            scenario_rows.append({
                "opponent": manifest["opponent"],
                "method": manifest["method"],
                "scenario": scenario,
                "episodes": len(scenario_episodes),
                "successes": sum(bool(episode.get("is_success", False)) for episode in scenario_episodes),
                "success_rate": item.get("success_rate", 0.0),
                "mean_return": item.get("mean_return", 0.0),
                "mean_final_range_m": item.get("mean_final_range_m", 0.0),
                "mean_final_ata_deg": item.get("mean_final_ata_deg", 0.0),
                "terminal_counts": json.dumps(terminals, sort_keys=True),
            })
    return opponent_rows, scenario_rows


def _write_aggregate_outputs(root: Path, results: list[dict[str, Any]]) -> None:
    opponent_rows, scenario_rows = _summary_rows(results)
    _write_csv(root / "metrics_by_opponent.csv", opponent_rows)
    _write_csv(root / "metrics_by_scenario.csv", scenario_rows)
    raw_rows = []
    for result in results:
        for episode in result["metrics"].get("raw_episodes", []):
            raw_rows.append({"opponent": result["manifest"]["opponent"], **episode})
    _write_csv(root / "episode_level_results.csv", raw_rows)
    (root / "analysis.md").write_text(
        "# Simplified Scenario Feasibility Pilot\n\n"
        f"Source ID: `{SOURCE_ID}`\n\n"
        "This is a non-learning diagnostic. Only initial scenario geometry was changed; "
        "the frozen H0 policy, opponents, reward and JSBSim chain were held constant.\n\n"
        "The scenario-level table is derived from the raw episode records.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Required to run JSBSim")
    parser.add_argument("--rebuild-summary", action="store_true", help="Rebuild summaries from completed raw metrics without running JSBSim")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute and not args.rebuild_summary:
        print(f"LOCKED: no simulation started. Planned Source ID: {SOURCE_ID}")
        return 0
    base = _load(args.config.resolve())
    protocol = base["protocol"]
    scenarios = list(protocol["scenarios"])
    opponents = list(protocol["opponents"])
    if protocol["source_id"] != SOURCE_ID:
        raise SystemExit("Config source_id does not match the runner source_id")
    if protocol.get("training_started") is not False or protocol.get("training_permitted") is not False:
        raise SystemExit("This pilot is evaluation-only")
    if args.rebuild_summary:
        root = args.output_dir.resolve()
        results = []
        for opponent in protocol["opponents"]:
            run_dir = root / "h0_current_geometry" / str(opponent)
            manifest_path = run_dir / "run_manifest.json"
            metrics_path = run_dir / "metrics.json"
            if not manifest_path.is_file() or not metrics_path.is_file():
                raise SystemExit(f"Missing completed run artifacts for {opponent}")
            results.append({
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            })
        _write_aggregate_outputs(root, results)
        print(f"Rebuilt summaries from existing raw episodes under {root}")
        return 0
    _assert_new_output(args.output_dir.resolve())
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for opponent in opponents:
        run_dir = args.output_dir.resolve() / "h0_current_geometry" / opponent
        results.append(_run_one(
            base,
            opponent,
            run_dir,
            evaluation_seed=int(protocol["evaluation_seeds"][0]),
            scenarios=scenarios,
            episodes_per_scenario=int(protocol["episodes_per_scenario"]),
        ))

    _write_aggregate_outputs(args.output_dir.resolve(), results)
    raw_rows = []
    for result in results:
        for episode in result["metrics"].get("raw_episodes", []):
            raw_rows.append({"opponent": result["manifest"]["opponent"], **episode})

    root_manifest = {
        "source_id": SOURCE_ID,
        "phase": "pilot",
        "config": _receipt(args.config.resolve()),
        "config_canonical_sha256": config_sha256(base, truncate=0),
        "git_info": get_git_info(str(REPO_ROOT)),
        "methods": ["h0_current_geometry"],
        "opponents": opponents,
        "scenarios": scenarios,
        "training_started": False,
        "training_permitted": False,
        "backend": "jsbsim",
        "runs": [item["manifest"] for item in results],
        "raw_episode_count": len(raw_rows),
        "status": "completed",
    }
    _write_json(args.output_dir.resolve() / "pilot_manifest.json", root_manifest)
    _write_json(args.output_dir.resolve() / "metrics.json", {"runs": [item["metrics"] for item in results]})
    print(f"Completed simplified-scenario pilot under {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
