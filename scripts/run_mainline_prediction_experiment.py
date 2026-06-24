#!/usr/bin/env python
"""Run a compact mainline prediction/VPP experiment.

This runner is intentionally narrow:
- train every compared policy from scratch with the same PPO capacity/budget;
- evaluate on the same fixed scenarios and seeds;
- report task metrics and prediction telemetry in one place;
- include a paired observation-ablation suite for
  add_prediction_to_observation=false/true.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    aggregate_metrics,
    evaluate_method,
)
from uav_vpp_guidance.common.provenance import record_config_override_if_changed
from uav_vpp_guidance.training.train_prediction_vpp_ppo import (
    load_experiment_config,
    train_ppo,
)
from uav_vpp_guidance.utils.config import merge_config
from uav_vpp_guidance.utils.seed import set_seed


MAINLINE_METHODS = [
    "no_prediction",
    "oracle_future",
    "cv_prediction",
    "ca_prediction",
    "lstm_frozen",
    "gru_frozen",
]

DEFAULT_OBS_PREDICTORS = ["cv_prediction"]

METRIC_FIELDS = [
    "num_episodes",
    "success_rate",
    "mean_return",
    "std_return",
    "mean_prediction_error_m",
    "mean_prediction_fallback_rate",
    "mean_post_warmup_fallback_rate",
    "mean_virtual_point_shift_m",
    "mean_anchor_shift_m",
    "mean_prediction_enabled_rate",
    "mean_prediction_valid_rate",
    "mean_length",
    "timeout_rate",
    "crash_rate",
    "out_of_bounds_rate",
    "score_win_rate",
]

RAW_FIELDS = [
    "suite",
    "method",
    "source_predictor",
    "training_seed",
    "evaluation_seed",
    "episode_seed",
    "scenario",
    "return",
    "length",
    "reason",
    "is_success",
    "is_crash",
    "is_timeout",
    "is_out_of_bounds",
    "score_win",
    "prediction_enabled_rate",
    "prediction_valid_rate",
    "prediction_fallback_rate",
    "post_warmup_fallback_rate",
    "warmup_fallback_rate",
    "runtime_fallback_rate",
    "mean_prediction_error_m",
    "median_prediction_error_m",
    "mean_env_prediction_error_m",
    "mean_offline_aligned_error_m",
    "prediction_error_count",
    "mean_virtual_point_shift_m",
    "mean_anchor_shift_m",
    "final_range_m",
    "final_ata_deg",
    "min_range_m",
    "min_ata_deg",
    "time_to_first_advantage_s",
    "advantage_hold_time_s",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _safe_float(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f


def _format_metric(value: Any, digits: int = 3) -> str:
    value_f = _safe_float(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.{digits}f}"


def _get_nested(config: dict[str, Any], key: str) -> Any:
    cur: Any = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_with_override(
    config: dict[str, Any],
    key: str,
    value: Any,
    source: str,
) -> None:
    old_value = _get_nested(config, key)
    cur = config
    parts = key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
    record_config_override_if_changed(
        config,
        key=key,
        new_value=value,
        old_value=old_value,
        source=source,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _base_vpp_override(anchor_mode: str) -> dict[str, Any]:
    return {
        "anchor_mode": anchor_mode,
        "return_info": True,
        "action_dim": 3,
        "d_long_range": [-1500.0, 1500.0],
        "d_lat_range": [-800.0, 800.0],
        "d_vert_range": [-500.0, 500.0],
        "smoothing_alpha": 0.3,
    }


def _oracle_method_override() -> dict[str, Any]:
    return {
        "name": "oracle_future",
        "trajectory_prediction": {
            "enabled": False,
            "prediction": {
                "lookahead_time_s": 1.0,
                "output_mode": "absolute_position",
                "fallback_mode": "constant_velocity",
            },
            "integration": {
                "add_prediction_to_observation": False,
                "add_uncertainty_to_observation": False,
            },
        },
        "virtual_point": _base_vpp_override("oracle_future_position"),
    }


def _with_prediction_observation(
    method_name: str,
    source_method: str,
    method_override: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    override = copy.deepcopy(method_override)
    override["name"] = method_name
    tp_cfg = override.setdefault("trajectory_prediction", {})
    int_cfg = tp_cfg.setdefault("integration", {})
    int_cfg["add_prediction_to_observation"] = enabled
    int_cfg["add_uncertainty_to_observation"] = enabled
    override["source_predictor"] = source_method
    return override


def build_method_registry(base_config: dict[str, Any], obs_predictors: list[str]) -> dict[str, dict[str, Any]]:
    configured = copy.deepcopy(base_config.get("methods", {}))
    registry: dict[str, dict[str, Any]] = {}

    for method in MAINLINE_METHODS:
        if method == "oracle_future":
            registry[method] = {
                "suite": "mainline_anchor",
                "source_predictor": "",
                "override": _oracle_method_override(),
            }
        else:
            if method not in configured:
                raise KeyError(f"Method {method!r} is missing from config.methods")
            registry[method] = {
                "suite": "mainline_anchor",
                "source_predictor": method,
                "override": configured[method],
            }

    for source in obs_predictors:
        if source not in configured:
            raise KeyError(f"Observation predictor {source!r} is missing from config.methods")
        short = source.replace("_prediction", "").replace("_frozen", "")
        false_name = f"{short}_anchor_only"
        true_name = f"{short}_obs_aug"
        registry[false_name] = {
            "suite": "prediction_observation",
            "source_predictor": source,
            "override": _with_prediction_observation(
                false_name, source, configured[source], enabled=False
            ),
        }
        registry[true_name] = {
            "suite": "prediction_observation",
            "source_predictor": source,
            "override": _with_prediction_observation(
                true_name, source, configured[source], enabled=True
            ),
        }
    return registry


def build_training_config(
    base_config: dict[str, Any],
    method_override: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    clean_override = copy.deepcopy(method_override)
    method_label = clean_override.get("name", "method")
    for metadata_key in ("checkpoint", "source_predictor"):
        clean_override.pop(metadata_key, None)

    cfg = merge_config(copy.deepcopy(base_config), clean_override)
    cfg.pop("methods", None)

    source = f"run_mainline_prediction_experiment.py:{method_label}"
    _set_with_override(cfg, "backend", args.backend, source)
    _set_with_override(cfg, "env.backend", args.backend, source)
    _set_with_override(cfg, "env.use_jsbsim", args.backend == "jsbsim", source)
    _set_with_override(cfg, "env.strict_backend", args.backend == "jsbsim", source)

    _set_with_override(cfg, "experiment.seed", int(seed), source)
    _set_with_override(cfg, "experiment.mode", "train", source)
    _set_with_override(cfg, "experiment.name", method_label, source)

    _set_with_override(cfg, "ppo.total_timesteps", int(args.train_timesteps), source)
    _set_with_override(cfg, "ppo.rollout_steps", int(args.rollout_steps), source)
    _set_with_override(cfg, "ppo.device", args.device, source)
    _set_with_override(cfg, "ppo.seed", int(seed), source)

    _set_with_override(cfg, "evaluation.eval_interval", 0, source)
    _set_with_override(cfg, "evaluation.eval_episodes", 0, source)
    _set_with_override(cfg, "evaluation.seeds", [int(seed)], source)
    _set_with_override(cfg, "evaluation.save_trajectories", False, source)

    if not isinstance(cfg.get("checkpoint"), dict):
        old_checkpoint = cfg.get("checkpoint")
        cfg["checkpoint"] = {}
        record_config_override_if_changed(
            cfg,
            key="checkpoint",
            new_value={},
            old_value=old_checkpoint,
            source=source,
        )
    _set_with_override(cfg, "checkpoint.save_interval", 0, source)
    _set_with_override(cfg, "checkpoint.save_best", False, source)
    _set_with_override(cfg, "checkpoint.save_last", True, source)

    if not args.allow_mode_switch:
        _set_with_override(cfg, "guidance.direct_track_mode", False, source)
        _set_with_override(cfg, "guidance.mode_switch.enabled", False, source)

    return cfg


def train_and_evaluate(
    *,
    method_name: str,
    method_meta: dict[str, Any],
    base_config: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    scenario_names: list[str],
    eval_seeds: list[int],
) -> dict[str, Any]:
    suite = method_meta["suite"]
    source_predictor = method_meta.get("source_predictor", "")
    method_override = method_meta["override"]
    run_dir = output_dir / "runs" / suite / method_name / f"seed_{seed}"
    cfg = build_training_config(base_config, method_override, seed, args)

    set_seed(seed)
    train_ppo(cfg, str(run_dir), smoke=False)

    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")

    env = CloseRangeTrackingEnv(cfg)
    sample_obs = env.reset(seed=seed)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(cfg.get("policy", {}).get("action_dim", 3))
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=cfg, device=args.device)
    agent.load(str(checkpoint_path))

    episodes_per_eval_seed = int(args.eval_episodes_per_scenario) * len(scenario_names)
    metrics = evaluate_method(
        env,
        agent,
        cfg,
        method_name,
        num_episodes=episodes_per_eval_seed,
        seeds=eval_seeds,
        scenarios=scenario_names,
        save_trajectories=args.save_trajectories,
        output_dir=str(run_dir / "evaluation"),
        training_seed=seed,
    )
    env.close()

    for ep in metrics["raw_episodes"]:
        ep["suite"] = suite
        ep["method"] = method_name
        ep["source_predictor"] = source_predictor
        ep["training_seed"] = seed
    metrics["suite"] = suite
    metrics["method"] = method_name
    metrics["source_predictor"] = source_predictor
    metrics["training_seed"] = seed
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["obs_dim"] = obs_dim
    metrics["action_dim"] = action_dim
    metrics["run_dir"] = str(run_dir)
    return metrics


def _group_episodes(
    raw_episodes: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for ep in raw_episodes:
        group_key = tuple(ep.get(key, "") for key in keys)
        groups.setdefault(group_key, []).append(ep)
    return groups


def build_summary_rows(raw_episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for (suite, method), episodes in sorted(_group_episodes(raw_episodes, ("suite", "method")).items()):
        agg = aggregate_metrics(episodes)
        row = {"suite": suite, "method": method, "scenario": "all"}
        for field in METRIC_FIELDS:
            row[field] = agg.get(field)
        row["prediction_rmse_m"] = agg.get("mean_prediction_error_m")
        row["fallback_rate"] = agg.get("mean_prediction_fallback_rate")
        row["vpp_shift_m"] = agg.get("mean_virtual_point_shift_m")
        row["anchor_shift_m"] = agg.get("mean_anchor_shift_m")
        method_rows.append(row)

    for (suite, method, scenario), episodes in sorted(
        _group_episodes(raw_episodes, ("suite", "method", "scenario")).items()
    ):
        agg = aggregate_metrics(episodes)
        row = {"suite": suite, "method": method, "scenario": scenario}
        for field in METRIC_FIELDS:
            row[field] = agg.get(field)
        row["prediction_rmse_m"] = agg.get("mean_prediction_error_m")
        row["fallback_rate"] = agg.get("mean_prediction_fallback_rate")
        row["vpp_shift_m"] = agg.get("mean_virtual_point_shift_m")
        row["anchor_shift_m"] = agg.get("mean_anchor_shift_m")
        scenario_rows.append(row)

    return method_rows, scenario_rows


def build_observation_comparison_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in method_rows}
    rows: list[dict[str, Any]] = []
    for method, row in by_method.items():
        if not method.endswith("_obs_aug"):
            continue
        prefix = method[: -len("_obs_aug")]
        anchor_name = f"{prefix}_anchor_only"
        anchor = by_method.get(anchor_name)
        if anchor is None:
            continue
        rows.append(
            {
                "predictor": prefix,
                "anchor_only_method": anchor_name,
                "obs_aug_method": method,
                "anchor_success_rate": anchor.get("success_rate"),
                "obs_aug_success_rate": row.get("success_rate"),
                "success_rate_delta": _safe_float(row.get("success_rate"))
                - _safe_float(anchor.get("success_rate")),
                "anchor_mean_return": anchor.get("mean_return"),
                "obs_aug_mean_return": row.get("mean_return"),
                "mean_return_delta": _safe_float(row.get("mean_return"))
                - _safe_float(anchor.get("mean_return")),
                "anchor_prediction_rmse_m": anchor.get("prediction_rmse_m"),
                "obs_aug_prediction_rmse_m": row.get("prediction_rmse_m"),
                "anchor_fallback_rate": anchor.get("fallback_rate"),
                "obs_aug_fallback_rate": row.get("fallback_rate"),
                "anchor_vpp_shift_m": anchor.get("vpp_shift_m"),
                "obs_aug_vpp_shift_m": row.get("vpp_shift_m"),
            }
        )
    return rows


def write_summary_md(
    path: Path,
    *,
    args: argparse.Namespace,
    scenario_names: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    method_rows: list[dict[str, Any]],
    obs_rows: list[dict[str, Any]],
    elapsed_s: float,
) -> None:
    lines = [
        "# Mainline Prediction/VPP Minimal Experiment",
        "",
        "## Design",
        "",
        f"- Backend: `{args.backend}`",
        f"- Train seeds: `{train_seeds}`",
        f"- Eval seeds: `{eval_seeds}`",
        f"- Scenarios: `{scenario_names}`",
        f"- PPO budget per method/seed: `{args.train_timesteps}` steps, rollout `{args.rollout_steps}`",
        f"- Eval episodes per scenario per eval seed: `{args.eval_episodes_per_scenario}`",
        f"- Mode switch/direct-track allowed: `{args.allow_mode_switch}`",
        f"- Runtime: `{elapsed_s:.1f}s`",
        "",
        "Note: `oracle_future` uses `oracle_future_position`, which is implemented as true-current-velocity extrapolation. With `target_mode=constant_velocity`, it is expected to be close to CV rather than a separate maneuver oracle.",
        "",
        "## Mainline Results",
        "",
        "| method | success | return | RMSE m | fallback | VPP shift m | scenario suite |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in method_rows:
        if row["suite"] != "mainline_anchor":
            continue
        lines.append(
            "| {method} | {success} | {ret} | {rmse} | {fallback} | {shift} | {suite} |".format(
                method=row["method"],
                success=_format_metric(row.get("success_rate")),
                ret=_format_metric(row.get("mean_return"), 2),
                rmse=_format_metric(row.get("prediction_rmse_m"), 2),
                fallback=_format_metric(row.get("fallback_rate")),
                shift=_format_metric(row.get("vpp_shift_m"), 2),
                suite=row["suite"],
            )
        )

    lines.extend(
        [
            "",
            "## Prediction Observation Ablation",
            "",
            "| predictor | anchor success | obs success | success delta | anchor return | obs return | return delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in obs_rows:
        lines.append(
            "| {pred} | {asr} | {osr} | {dsr} | {ar} | {or_} | {dr} |".format(
                pred=row["predictor"],
                asr=_format_metric(row.get("anchor_success_rate")),
                osr=_format_metric(row.get("obs_aug_success_rate")),
                dsr=_format_metric(row.get("success_rate_delta")),
                ar=_format_metric(row.get("anchor_mean_return"), 2),
                or_=_format_metric(row.get("obs_aug_mean_return"), 2),
                dr=_format_metric(row.get("mean_return_delta"), 2),
            )
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `raw_episodes.csv`: per-episode returns, success, prediction RMSE, fallback, VPP shift.",
            "- `summary_by_method.csv`: aggregate metrics by method.",
            "- `summary_by_method_scenario.csv`: per-scenario aggregate metrics.",
            "- `prediction_observation_comparison.csv`: paired false/true observation ablation.",
            "- `run_manifest.json`: exact CLI and method metadata.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
        help="Base experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/mainline_prediction_minimal/<timestamp>.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="simple",
        choices=["simple", "jsbsim"],
        help="Simulation backend.",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=1024,
        help="PPO training budget per method and training seed.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=256,
        help="PPO rollout length per update.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Training seeds.",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Evaluation seeds. Defaults to --seeds.",
    )
    parser.add_argument(
        "--eval-episodes-per-scenario",
        type=int,
        default=1,
        help="Evaluation episodes per scenario per eval seed.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Scenario names. Defaults to all scenarios in the config.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to run. Defaults to all mainline methods plus observation ablation.",
    )
    parser.add_argument(
        "--obs-predictors",
        type=str,
        nargs="+",
        default=DEFAULT_OBS_PREDICTORS,
        help="Predictor methods to use for add_prediction_to_observation false/true pairs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for PPO and neural predictors.",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save per-step trajectory CSVs under each run directory.",
    )
    parser.add_argument(
        "--allow-mode-switch",
        action="store_true",
        help="Keep inherited guidance.mode_switch/direct-track behavior. Default is off to isolate VPP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    config_path = REPO_ROOT / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    base_config = load_experiment_config(str(config_path))

    scenario_names = args.scenarios or list(base_config.get("scenarios", {}).keys())
    if not scenario_names:
        raise ValueError("No scenarios found in config and none were provided.")

    eval_seeds = args.eval_seeds if args.eval_seeds is not None else args.seeds
    registry = build_method_registry(base_config, args.obs_predictors)
    methods_to_run = args.methods or list(registry.keys())
    unknown = [method for method in methods_to_run if method not in registry]
    if unknown:
        raise KeyError(f"Unknown methods requested: {unknown}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "outputs" / "mainline_prediction_minimal" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at": timestamp,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "backend": args.backend,
        "train_timesteps": args.train_timesteps,
        "rollout_steps": args.rollout_steps,
        "train_seeds": args.seeds,
        "eval_seeds": eval_seeds,
        "eval_episodes_per_scenario": args.eval_episodes_per_scenario,
        "allow_mode_switch": args.allow_mode_switch,
        "scenarios": scenario_names,
        "methods": {
            method: {
                "suite": registry[method]["suite"],
                "source_predictor": registry[method].get("source_predictor", ""),
                "override": registry[method]["override"],
            }
            for method in methods_to_run
        },
        "oracle_note": (
            "oracle_future_position is implemented as current true velocity extrapolation; "
            "under constant_velocity target dynamics it is expected to behave like CV."
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    all_metrics: list[dict[str, Any]] = []
    raw_episodes: list[dict[str, Any]] = []

    for seed in args.seeds:
        for method in methods_to_run:
            print(f"\n=== {registry[method]['suite']} | {method} | seed {seed} ===")
            metrics = train_and_evaluate(
                method_name=method,
                method_meta=registry[method],
                base_config=base_config,
                seed=seed,
                args=args,
                output_dir=output_dir,
                scenario_names=scenario_names,
                eval_seeds=eval_seeds,
            )
            all_metrics.append(metrics)
            raw_episodes.extend(metrics["raw_episodes"])

    method_rows, scenario_rows = build_summary_rows(raw_episodes)
    obs_rows = build_observation_comparison_rows(method_rows)

    summary_fields = [
        "suite",
        "method",
        "scenario",
        *METRIC_FIELDS,
        "prediction_rmse_m",
        "fallback_rate",
        "vpp_shift_m",
        "anchor_shift_m",
    ]
    _write_csv(output_dir / "raw_episodes.csv", raw_episodes, RAW_FIELDS)
    _write_csv(output_dir / "summary_by_method.csv", method_rows, summary_fields)
    _write_csv(output_dir / "summary_by_method_scenario.csv", scenario_rows, summary_fields)
    _write_csv(
        output_dir / "prediction_observation_comparison.csv",
        obs_rows,
        [
            "predictor",
            "anchor_only_method",
            "obs_aug_method",
            "anchor_success_rate",
            "obs_aug_success_rate",
            "success_rate_delta",
            "anchor_mean_return",
            "obs_aug_mean_return",
            "mean_return_delta",
            "anchor_prediction_rmse_m",
            "obs_aug_prediction_rmse_m",
            "anchor_fallback_rate",
            "obs_aug_fallback_rate",
            "anchor_vpp_shift_m",
            "obs_aug_vpp_shift_m",
        ],
    )

    elapsed_s = time.time() - start_time
    write_summary_md(
        output_dir / "summary.md",
        args=args,
        scenario_names=scenario_names,
        train_seeds=args.seeds,
        eval_seeds=eval_seeds,
        method_rows=method_rows,
        obs_rows=obs_rows,
        elapsed_s=elapsed_s,
    )

    results_json = {
        "method_rows": method_rows,
        "scenario_rows": scenario_rows,
        "observation_comparison": obs_rows,
        "run_metrics": [
            {
                key: value
                for key, value in metrics.items()
                if key not in {"raw_episodes", "per_seed", "per_scenario"}
            }
            for metrics in all_metrics
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(results_json, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    print(f"\nDone. Outputs written to: {output_dir}")
    print("Main summary:")
    for row in method_rows:
        print(
            f"  {row['suite']}::{row['method']}: "
            f"success={_format_metric(row.get('success_rate'))}, "
            f"return={_format_metric(row.get('mean_return'), 2)}, "
            f"rmse={_format_metric(row.get('prediction_rmse_m'), 2)}, "
            f"fallback={_format_metric(row.get('fallback_rate'))}, "
            f"vpp_shift={_format_metric(row.get('vpp_shift_m'), 2)}"
        )


if __name__ == "__main__":
    main()
