#!/usr/bin/env python
"""Run the independent H0/H1/H2 predicted-geometry protocol.

This runner has a hard execution lock. Without ``--execute`` it performs no
model loading, environment construction, reset, training, or JSBSim work.
H0/H1 always use a frozen policy. H2 may train only the high-level PPO.
"""

from __future__ import annotations

import argparse
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
from uav_vpp_guidance.evaluation.prediction_highlevel_observation_v1 import (
    EXPECTED_METHODS,
    EXPECTED_OPPONENTS,
    EXPECTED_SCENARIOS,
    SOURCE_ID,
    file_receipt,
    load_resolved_config,
    method_config,
)
from uav_vpp_guidance.training.train_prediction_vpp_ppo import train_ppo


DEFAULT_CONFIG = REPO_ROOT / "config/experiment/prediction_highlevel_observation_v1.yaml"
DEFAULT_ROOT = REPO_ROOT / "outputs/prediction_highlevel_observation_v1"


def _assert_new_output_root(output_root: Path) -> None:
    """Refuse to overwrite an existing execution or analysis artifact tree."""
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output path is not a directory: {output_root}")
    if not output_root.exists():
        return
    existing = [path for path in output_root.rglob("*") if path.is_file()]
    if existing:
        sample = ", ".join(str(path.relative_to(output_root)) for path in existing[:5])
        raise ValueError(
            "Refusing to overwrite an existing output tree; choose a new "
            f"--output-dir. Existing files: {sample}"
        )


def _require_pilot_verification(pilot_path: Path, source_id: str) -> None:
    """Require an independently written, passing pilot-verifier receipt."""
    if not pilot_path.is_file():
        raise ValueError(
            "Formal execution requires an independent pilot verification receipt: "
            f"{pilot_path}"
        )
    try:
        receipt = json.loads(pilot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid pilot verification receipt: {pilot_path}") from exc
    if receipt.get("source_id") != source_id:
        raise ValueError("Pilot verification receipt has the wrong source_id")
    if receipt.get("phase") not in (None, "pilot"):
        raise ValueError("Pilot verification receipt has the wrong phase")
    if receipt.get("ok") is not True:
        raise ValueError("Pilot verification has not passed")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _nested(config: dict[str, Any], key: str) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_nested(config: dict[str, Any], key: str, value: Any, source: str) -> None:
    current: Any = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    old = current.get(parts[-1])
    current[parts[-1]] = value
    record_config_override_if_changed(config, key, value, old, source)


def _build_opponent(stage: str, config: dict[str, Any]):
    registry = config.get("opponent_registry", {})
    if stage == "expert":
        return ExpertOpponent(registry.get("expert", {}).get("config", {})), {"stage": stage}
    if stage == "end_to_end":
        entry = registry.get("end_to_end", {})
        return EndToEndOpponent.from_registry(entry), {"stage": stage, **entry}
    raise ValueError(f"Unsupported opponent stage: {stage}")


def _prepare_config(
    base: dict[str, Any],
    method_name: str,
    args: argparse.Namespace,
    seed: int,
    training: bool,
    source_id: str,
) -> dict[str, Any]:
    cfg = method_config(base, method_name)
    source = f"run_prediction_highlevel_observation_v1.py:{method_name}"
    _set_nested(cfg, "experiment.source_id", source_id, source)
    _set_nested(cfg, "experiment.mode", "train" if training else "evaluation", source)
    _set_nested(cfg, "experiment.seed", int(seed), source)
    _set_nested(cfg, "ppo.seed", int(seed), source)
    _set_nested(cfg, "backend", args.backend, source)
    _set_nested(cfg, "env.backend", args.backend, source)
    _set_nested(cfg, "env.use_jsbsim", args.backend == "jsbsim", source)
    _set_nested(cfg, "env.strict_backend", args.backend == "jsbsim", source)
    if training:
        _set_nested(cfg, "ppo.total_timesteps", int(args.train_timesteps), source)
        _set_nested(cfg, "evaluation.eval_interval", 0, source)
        _set_nested(cfg, "evaluation.eval_episodes", 0, source)
        _set_nested(cfg, "checkpoint.save_best", False, source)
        _set_nested(cfg, "checkpoint.save_last", True, source)
    return cfg


def _resolve_checkpoint(
    method_meta: dict[str, Any], override: str | None, seed: int | None = None
) -> Path | None:
    value = override
    if value is None and seed is not None:
        mapping = method_meta.get("checkpoint_by_training_seed", {})
        value = mapping.get(seed, mapping.get(str(seed)))
    value = value or method_meta.get("checkpoint")
    if not value:
        return None
    return Path(value).resolve() if Path(value).is_absolute() else (REPO_ROOT / value).resolve()


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run_evaluation(
    *,
    base: dict[str, Any],
    method_name: str,
    opponent_stage: str,
    checkpoint: Path,
    seed_label: int,
    args: argparse.Namespace,
    phase_root: Path,
    source_id: str,
) -> dict[str, Any]:
    cfg = _prepare_config(base, method_name, args, seed_label, training=False, source_id=source_id)
    run_dir = phase_root / method_name / f"seed{seed_label}" / opponent_stage
    if "prediction_decision_bridge_v1" in str(run_dir):
        raise ValueError("New protocol cannot write prediction_decision_bridge_v1 outputs")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config(run_dir / "resolved_config.yaml", cfg)
    opponent, opponent_config = _build_opponent(opponent_stage, cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent, opponent_config=opponent_config)
    try:
        sample_obs = env.reset(seed=seed_label)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(cfg.get("policy", {}).get("action_dim", 3))
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=cfg, device="cpu")
        loaded = agent.load(str(checkpoint))
        metrics = evaluate_method(
            env,
            agent,
            cfg,
            method_name,
            num_episodes=args.episodes_per_scenario * len(args.scenarios),
            seeds=args.eval_seeds,
            scenarios=args.scenarios,
            save_trajectories=args.save_trajectories,
            output_dir=str(run_dir),
            training_seed=seed_label,
        )
    finally:
        env.close()

    for episode in metrics.get("raw_episodes", []):
        episode["method"] = method_name
        episode["opponent"] = opponent_stage
        episode["training_seed"] = seed_label
        episode["observation_contract_version"] = (
            "prediction_highlevel_observation_v1"
            if cfg.get("trajectory_prediction", {}).get("integration", {}).get("replace_base_geometry", False)
            else "legacy"
        )
    manifest = {
        "source_id": source_id,
        "phase": args.phase,
        "method": method_name,
        "opponent": opponent_stage,
        "training_seed": seed_label,
        "evaluation_seeds": args.eval_seeds,
        "scenarios": args.scenarios,
        "episodes_per_scenario": args.episodes_per_scenario,
        "training_started": False,
        "ppo_update_permitted": False,
        "training_scope": (
            "high_level_ppo_only"
            if method_name == "h2_retrained_predicted_geometry"
            else "none"
        ),
        "predictor_frozen": method_name != "h0_current_geometry",
        "low_level_frozen": True,
        "low_level_execution_chain_frozen": True,
        "checkpoint": file_receipt(checkpoint),
        "predictor_checkpoint": (
            file_receipt(REPO_ROOT / cfg["trajectory_prediction"]["checkpoint_path"])
            if cfg.get("trajectory_prediction", {}).get("enabled", False)
            else None
        ),
        "config_sha256": file_sha256(run_dir / "resolved_config.yaml"),
        "config_path": str((run_dir / "resolved_config.yaml").resolve()),
        "git_info": get_git_info(str(REPO_ROOT)),
        "backend_requested": args.backend,
        "backend_effective": getattr(env, "_backend", None),
        "observation_schema": sample_obs.get("observation_schema", {}),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "checkpoint_load": {
            "loaded": True,
            "obs_dim": loaded.get("obs_dim"),
            "action_dim": loaded.get("action_dim"),
            "optimizer_loaded": bool(loaded.get("optimizer_loaded", False)),
            "total_updates": loaded.get("total_updates"),
            "total_timesteps": loaded.get("total_timesteps"),
        },
        "backend_fallback_occurred": bool(
            getattr(env, "_backend_fallback_occurred", False)
        ),
        "backend_fallback_reason": getattr(
            env, "_backend_fallback_reason", None
        ),
        "raw_episode_count": len(metrics.get("raw_episodes", [])),
    }
    _json_dump(run_dir / "run_manifest.json", manifest)
    _json_dump(run_dir / "metrics.json", metrics)
    return manifest


def _train_h2(
    *,
    base: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    phase_root: Path,
    source_id: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    method_name = "h2_retrained_predicted_geometry"
    cfg = _prepare_config(base, method_name, args, seed, training=True, source_id=source_id)
    h0_checkpoint = _resolve_checkpoint(
        base["methods"]["h0_current_geometry"], args.h0_h1_checkpoint, seed=seed
    )
    if h0_checkpoint is None:
        raise ValueError("H2 warm start requires the H0 checkpoint")
    _set_nested(
        cfg,
        "warm_start.enabled",
        True,
        f"run_prediction_highlevel_observation_v1.py:{method_name}:h0_warm_start",
    )
    _set_nested(
        cfg,
        "warm_start.checkpoint",
        str(h0_checkpoint),
        f"run_prediction_highlevel_observation_v1.py:{method_name}:h0_warm_start",
    )
    _set_nested(
        cfg,
        "warm_start.strict_dims",
        True,
        f"run_prediction_highlevel_observation_v1.py:{method_name}:h0_warm_start",
    )
    _set_nested(
        cfg,
        "warm_start.load_optimizer",
        False,
        f"run_prediction_highlevel_observation_v1.py:{method_name}:h0_warm_start",
    )
    _set_nested(
        cfg,
        "training_scope",
        "high_level_ppo_only",
        f"run_prediction_highlevel_observation_v1.py:{method_name}:scope",
    )
    train_dir = phase_root / method_name / f"seed{seed}" / "training"
    _write_config(train_dir / "resolved_config.yaml", cfg)
    train_ppo(cfg, str(train_dir), smoke=False)
    checkpoint = train_dir / "checkpoints" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"H2 training did not produce {checkpoint}")
    manifest = {
        "source_id": source_id,
        "phase": args.phase,
        "method": method_name,
        "training_seed": seed,
        "training_started": True,
        "ppo_update_permitted": True,
        "training_scope": "high_level_ppo_only",
        "predictor_frozen": True,
        "low_level_frozen": True,
        "low_level_execution_chain_frozen": True,
        "warm_start_checkpoint": file_receipt(h0_checkpoint),
        "checkpoint": file_receipt(checkpoint),
        "config_sha256": file_sha256(train_dir / "resolved_config.yaml"),
        "git_info": get_git_info(str(REPO_ROOT)),
        "config_payload_sha256": config_sha256(cfg, truncate=0),
    }
    _json_dump(train_dir / "run_manifest.json", manifest)
    return cfg, checkpoint, manifest


def _write_pilot_outputs(phase_root: Path, root_manifest: dict[str, Any], error: str | None) -> None:
    source_id = str(root_manifest.get("source_id") or SOURCE_ID)
    pilot_status = {
        "source_id": source_id,
        "phase": "pilot",
        "status": "runner_failed" if error else "pending_independent_verification",
        "error": error,
        "formal_authorized": False,
        "verification_required": True,
        "verification_path": str((phase_root / "pilot_verification.json").resolve()),
        "run_count": len(root_manifest.get("runs", [])),
    }
    _json_dump(phase_root / "pilot_status.json", pilot_status)
    _json_dump(phase_root / "pilot_config_snapshot.json", root_manifest)
    (phase_root / "pilot_logs").mkdir(parents=True, exist_ok=True)
    _json_dump(phase_root / "causality_runtime_check.json", {
        "status": "not_run" if error else "requires_episode_telemetry_review",
        "future_state_access_allowed": False,
        "source_id": source_id,
    })
    summary = "# H0/H1/H2 Pilot Summary\n\n"
    summary += "Pilot failed before completion: " + error if error else "Pilot execution completed; run verifier before formal use."
    (phase_root / "pilot_summary.md").write_text(summary + "\n", encoding="utf-8")
    (phase_root / "pilot_failure_report.md").write_text(
        "# Pilot Failure Report\n\n" + (error or "No runner-level failure. Any verifier failure must be recorded separately.") + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=EXPECTED_METHODS, default=list(EXPECTED_METHODS))
    parser.add_argument("--opponents", nargs="+", choices=EXPECTED_OPPONENTS, default=list(EXPECTED_OPPONENTS))
    parser.add_argument("--scenarios", nargs="+", choices=EXPECTED_SCENARIOS, default=list(EXPECTED_SCENARIOS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--episodes-per-scenario", type=int, default=None)
    parser.add_argument("--train-timesteps", type=int, default=None)
    parser.add_argument("--backend", choices=("simple", "jsbsim"), default="jsbsim")
    parser.add_argument("--h0-h1-checkpoint", type=str, default=None)
    parser.add_argument("--h2-checkpoint", type=str, default=None)
    parser.add_argument(
        "--pilot-verification",
        type=Path,
        default=None,
        help="Independent pilot verification receipt required for formal execution",
    )
    parser.add_argument("--train-h2", action="store_true", help="Train H2 once per train seed")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Required to start model loading or simulation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print("LOCKED: no experiment started. Re-run with --execute to load models or run simulation.")
        print(f"Planned Source ID: {SOURCE_ID}; phase={args.phase}")
        return 0
    args.train_seeds = args.train_seeds or ([42, 43] if args.phase == "pilot" else [42, 43, 44])
    args.episodes_per_scenario = args.episodes_per_scenario or (1 if args.phase == "pilot" else 12)
    args.train_timesteps = args.train_timesteps or (4096 if args.phase == "pilot" else 200000)
    if args.phase == "formal":
        if tuple(args.methods) != EXPECTED_METHODS:
            raise SystemExit("Formal execution requires the complete H0/H1/H2 matrix")
        if tuple(args.opponents) != EXPECTED_OPPONENTS:
            raise SystemExit("Formal execution requires both locked opponents")
        if tuple(args.scenarios) != EXPECTED_SCENARIOS:
            raise SystemExit("Formal execution requires all five locked scenarios")
        if len(args.train_seeds) < 3:
            raise SystemExit("Formal execution requires at least three training seeds")
    if args.phase == "formal" and "h2_retrained_predicted_geometry" in args.methods and not args.train_h2 and not args.h2_checkpoint:
        raise SystemExit("Formal H2 requires --train-h2 or --h2-checkpoint")
    if args.train_h2 and "h2_retrained_predicted_geometry" not in args.methods:
        raise SystemExit("--train-h2 requires h2_retrained_predicted_geometry in --methods")

    config = load_resolved_config(args.config.resolve())
    source_id = str(config.get("protocol", {}).get("source_id") or SOURCE_ID)
    output_root = (args.output_dir or DEFAULT_ROOT / args.phase).resolve()
    if "prediction_decision_bridge_v1" in str(output_root):
        raise SystemExit("Refusing forbidden output directory")
    try:
        _assert_new_output_root(output_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    pilot_verification_path = (
        args.pilot_verification.resolve()
        if args.pilot_verification
        else (output_root.parent / "pilot" / "pilot_verification.json").resolve()
    )
    if args.phase == "formal":
        try:
            _require_pilot_verification(pilot_verification_path, source_id)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    output_root.mkdir(parents=True, exist_ok=True)
    root_manifest = {
        "source_id": source_id,
        "phase": args.phase,
        "config_path": str(args.config.resolve()),
        "config_receipt": file_receipt(args.config.resolve()),
        "git_info": get_git_info(str(REPO_ROOT)),
        "methods": args.methods,
        "opponents": args.opponents,
        "scenarios": args.scenarios,
        "train_seeds": args.train_seeds,
        "evaluation_seeds": args.eval_seeds,
        "episodes_per_scenario": args.episodes_per_scenario,
        "train_timesteps": args.train_timesteps,
        "training_requested": bool(args.train_h2),
        "execution_started": True,
        "pilot_verification_path": str(pilot_verification_path) if args.phase == "formal" else None,
        "runs": [],
    }
    error = None
    try:
        for method_name in args.methods:
            if method_name == "h2_retrained_predicted_geometry":
                if args.train_h2:
                    checkpoints = {}
                    for seed in args.train_seeds:
                        _cfg, checkpoint, train_manifest = _train_h2(
                            base=config, seed=seed, args=args, phase_root=output_root,
                            source_id=source_id,
                        )
                        root_manifest["runs"].append(train_manifest)
                        checkpoints[seed] = checkpoint
                else:
                    checkpoints = {}
                    for seed in args.train_seeds:
                        checkpoint = _resolve_checkpoint(
                            config["methods"][method_name], args.h2_checkpoint, seed=seed
                        )
                        if checkpoint is None:
                            raise ValueError(
                                "H2 requires --h2-checkpoint or a checkpoint_by_training_seed entry"
                            )
                        checkpoints[seed] = checkpoint
            else:
                checkpoints = {}
                for seed in args.train_seeds:
                    checkpoint = _resolve_checkpoint(
                        config["methods"][method_name], args.h0_h1_checkpoint, seed=seed
                    )
                    if checkpoint is None:
                        raise ValueError(f"{method_name} requires a checkpoint for seed {seed}")
                    checkpoints[seed] = checkpoint

            for seed, checkpoint in checkpoints.items():
                for opponent in args.opponents:
                    root_manifest["runs"].append(
                        _run_evaluation(
                            base=config,
                            method_name=method_name,
                            opponent_stage=opponent,
                            checkpoint=checkpoint,
                            seed_label=seed,
                            args=args,
                            phase_root=output_root,
                            source_id=source_id,
                        )
                    )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        root_manifest["error"] = error
    root_manifest["status"] = "failed" if error else "completed"
    _json_dump(output_root / ("formal_manifest.json" if args.phase == "formal" else "run_manifest.json"), root_manifest)
    if args.phase == "pilot":
        _write_pilot_outputs(output_root, root_manifest, error)
    if error:
        raise SystemExit(error)
    print(f"Completed {args.phase} protocol under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
