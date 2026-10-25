#!/usr/bin/env python
"""Run a non-learning two-dimensional empirical capability-envelope scan."""

from __future__ import annotations

import argparse
import copy
import csv
import json
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
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_method
from uav_vpp_guidance.utils.config import merge_config


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-CAPABILITY-ENVELOPE-V1-20260903"
DEFAULT_CONFIG = REPO_ROOT / "config/experiment/mild_disadvantage_capability_envelope_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/mild_disadvantage_capability_envelope_v1/envelope_v1"


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
    source = "run_mild_disadvantage_capability_envelope_v1.py"
    _set_nested(cfg, "experiment.source_id", source_id, f"{source}:source_id")
    _set_nested(cfg, "experiment.name", "mild_disadvantage_capability_envelope_v1", f"{source}:name")
    _set_nested(cfg, "experiment.mode", "evaluation", f"{source}:mode")
    _set_nested(cfg, "experiment.seed", int(seed), f"{source}:seed")
    _set_nested(cfg, "ppo.seed", int(seed), f"{source}:ppo_seed")
    _set_nested(cfg, "backend", "jsbsim", f"{source}:backend")
    _set_nested(cfg, "env.backend", "jsbsim", f"{source}:env_backend")
    _set_nested(cfg, "env.use_jsbsim", True, f"{source}:env_use_jsbsim")
    _set_nested(cfg, "env.strict_backend", True, f"{source}:strict_backend")
    scenario_defs = copy.deepcopy(scan["scan"]["scenarios"])
    fixed = scan["scan"]["fixed_target_conditions"]
    for scenario_name, scenario in scenario_defs.items():
        scenario["name"] = scenario_name
        scenario["own_init"] = {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 0.0,
        }
        scenario["target_init"] = {
            "position_m": [scenario["target_x_m"], scenario["target_y_m"], fixed["target_z_m"]],
            "velocity_mps": fixed["target_speed_mps"],
            "heading_deg": fixed["target_heading_deg"],
        }
    _set_nested(cfg, "scenarios", scenario_defs, f"{source}:scan_scenarios")
    _set_nested(cfg, "protocol", scan["protocol"], f"{source}:scan_protocol")
    cfg["scan"] = scan["scan"]
    return cfg


def _run_one(base: dict[str, Any], scan: dict[str, Any], opponent: str, root: Path, source_id: str) -> dict[str, Any]:
    seed0 = int(scan["scan"]["evaluation_seed"])
    repetitions = int(scan["scan"]["repetitions_per_cell"])
    seeds = [seed0 + offset for offset in range(repetitions)]
    cfg = _prepare_config(base, scan, seed0, source_id)
    root.mkdir(parents=True, exist_ok=True)
    resolved = root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    checkpoint = _resolve(cfg["methods"]["h0_current_geometry"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing H0 checkpoint: {checkpoint}")
    opponent_policy, opponent_config = _build_opponent(opponent, cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
    try:
        sample_obs = env.reset(seed=seed0)
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
            seeds=seeds,
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
        "evaluation_seeds": seeds,
        "scenarios": scenarios,
        "episodes_per_scenario": repetitions,
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


def _finite_mean(values: list[Any]) -> float:
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            clean.append(number)
    return float(np.mean(clean)) if clean else float("nan")


def _cell_rows(scan: dict[str, Any], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = scan["scan"]["scenarios"]
    by_opponent: list[dict[str, Any]] = []
    cells_by_opp: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        manifest = result["manifest"]
        metrics = result["metrics"]
        for scenario_name, definition in scenarios.items():
            episodes = [e for e in metrics.get("raw_episodes", []) if e.get("scenario") == scenario_name]
            item = metrics.get("per_scenario", {}).get(scenario_name, {})
            terminal_counts: dict[str, int] = {}
            for episode in episodes:
                reason = str(episode.get("reason", "unknown"))
                terminal_counts[reason] = terminal_counts.get(reason, 0) + 1
            clipping_rates = []
            fallback_rates = []
            for episode in episodes:
                clipping_rates.append(_finite_mean([
                    episode.get("nz_cmd_saturation_rate"),
                    episode.get("roll_rate_cmd_saturation_rate"),
                    episode.get("throttle_cmd_saturation_rate"),
                ]))
                fallback_rates.append(_finite_mean([
                    episode.get("prediction_fallback_rate"),
                    episode.get("runtime_fallback_rate"),
                ]))
            row = {
                "opponent": manifest["opponent"],
                "method": manifest["method"],
                "scenario": scenario_name,
                "target_x_m": definition["target_x_m"],
                "target_y_m": definition["target_y_m"],
                "episodes": len(episodes),
                "successes": sum(bool(e.get("is_success", False)) for e in episodes),
                "success_rate": item.get("success_rate", 0.0),
                "crash_rate": item.get("crash_rate", 0.0),
                "out_of_bounds_rate": item.get("out_of_bounds_rate", 0.0),
                "timeout_rate": item.get("timeout_rate", 0.0),
                "mean_return": item.get("mean_return", 0.0),
                "mean_min_range_m": item.get("mean_min_range_m", 0.0),
                "mean_min_ata_deg": item.get("mean_min_ata_deg", 0.0),
                "mean_final_range_m": item.get("mean_final_range_m", 0.0),
                "mean_final_ata_deg": item.get("mean_final_ata_deg", 0.0),
                "mean_clipping_rate": _finite_mean(clipping_rates),
                "mean_fallback_rate": _finite_mean(fallback_rates),
                "terminal_counts": json.dumps(terminal_counts, sort_keys=True),
                "backend_effective": manifest["backend_effective"],
                "backend_fallback_occurred": manifest["backend_fallback_occurred"],
            }
            by_opponent.append(row)
            cells_by_opp[(manifest["opponent"], scenario_name)] = row

    envelope_rows: list[dict[str, Any]] = []
    opponents = list(scan["protocol"]["opponents"])
    for scenario_name, definition in scenarios.items():
        rows = [cells_by_opp[(opponent, scenario_name)] for opponent in opponents]
        stable = all(
            row["successes"] == int(scan["scan"]["repetitions_per_cell"])
            and row["crash_rate"] == 0.0
            and row["out_of_bounds_rate"] == 0.0
            and row["mean_fallback_rate"] == 0.0
            for row in rows
        )
        non_feasible = all(row["successes"] == 0 for row in rows)
        classification = "stable_feasible" if stable else "non_feasible" if non_feasible else "transition"
        envelope_rows.append({
            "scenario": scenario_name,
            "target_x_m": definition["target_x_m"],
            "target_y_m": definition["target_y_m"],
            "classification": classification,
            "opponent_success_rates": json.dumps(
                {row["opponent"]: row["success_rate"] for row in rows}, sort_keys=True
            ),
            "opponent_terminal_counts": json.dumps(
                {row["opponent"]: json.loads(row["terminal_counts"]) for row in rows}, sort_keys=True
            ),
        })
    return by_opponent, envelope_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-id", type=str, default=None)
    parser.add_argument("--execute", action="store_true", help="Required to run JSBSim")
    args = parser.parse_args(argv)
    scan = _load(args.config.resolve())
    source_id = args.source_id or scan["experiment"]["source_id"]
    if not args.execute:
        print(f"LOCKED: no simulation started. Planned Source ID: {source_id}")
        return 0
    if scan["experiment"]["source_id"] != source_id or scan["protocol"]["source_id"] != source_id:
        raise SystemExit("Source ID mismatch")
    base_path = _resolve(scan["base_config"])
    base = _load(base_path)
    root = args.output_dir.resolve()
    if root.exists() and any(path.is_file() for path in root.rglob("*")):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for opponent in scan["protocol"]["opponents"]:
        results.append(_run_one(base, scan, opponent, root / "h0_current_geometry" / opponent, source_id))
    by_opponent, envelope_rows = _cell_rows(scan, results)
    _write_csv(root / "metrics_by_cell_and_opponent.csv", by_opponent)
    _write_csv(root / "envelope_classification.csv", envelope_rows)
    _write_json(root / "envelope_analysis.json", {
        "source_id": source_id,
        "classification_rules": scan["protocol"]["classification"],
        "axes": scan["scan"]["axes"],
        "fixed_target_conditions": scan["scan"]["fixed_target_conditions"],
        "cells": envelope_rows,
    })
    raw = []
    for result in results:
        raw.extend({"opponent": result["manifest"]["opponent"], **episode} for episode in result["metrics"].get("raw_episodes", []))
    _write_csv(root / "episode_level_results.csv", raw)
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
        "fixed_target_conditions": scan["scan"]["fixed_target_conditions"],
        "repetitions_per_cell": scan["scan"]["repetitions_per_cell"],
        "training_started": False,
        "training_permitted": False,
        "backend": "jsbsim",
        "runs": [result["manifest"] for result in results],
        "raw_episode_count": len(raw),
        "status": "completed",
    })
    counts: dict[str, int] = {}
    for row in envelope_rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    (root / "analysis.md").write_text(
        "# Mild Disadvantage Capability Envelope V1\n\n"
        f"Source ID: `{source_id}`\n\n"
        "本轮在固定 H0 控制器下扫描目标纵向位置和侧向位置，未训练、未调奖励或控制接口。\n\n"
        f"网格为 {len(scan['scan']['axes']['target_x_m']['levels'])} × {len(scan['scan']['axes']['target_y_m']['levels'])}，"
        f"每个网格点每个对手重复 {scan['scan']['repetitions_per_cell']} 次，共 {len(raw)} 条回合。\n\n"
        f"联合分类计数：{json.dumps(counts, ensure_ascii=False, sort_keys=True)}。\n\n"
        "详细指标见 `metrics_by_cell_and_opponent.csv`，联合分类见 `envelope_classification.csv` 和 `envelope_analysis.json`。\n",
        encoding="utf-8",
    )
    print(f"Completed capability envelope scan under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
