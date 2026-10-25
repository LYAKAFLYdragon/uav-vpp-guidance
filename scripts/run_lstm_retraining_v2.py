#!/usr/bin/env python
"""Run the independent V2 corpus, predictor, policy and heldout protocol."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.baselines.rule_based_pursuit import RuleBasedPursuitPolicy
from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.common.provenance import record_config_override_if_changed
from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_method
from uav_vpp_guidance.evaluation.lstm_retraining_v2 import (
    SOURCE_ID,
    canonical_config_sha256,
    file_receipt,
    inspect_policy_checkpoint,
    source_manifest,
    strict_load_predictor,
    verify_corpus,
    verify_formal_metrics,
    write_json,
)
from uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios import evaluate_scenario
from uav_vpp_guidance.utils.config import load_yaml_config


DEFAULT_ROOT = REPO_ROOT / "outputs/predicted_geometry_lstm_vpp_retraining_v2"
COLLECT_CONFIG = REPO_ROOT / "config/experiment/collect_lstm_retraining_v2.yaml"
PREDICTOR_CONFIG = REPO_ROOT / "config/trajectory_prediction_lstm_retraining_v2.yaml"
POLICY_CONFIG = REPO_ROOT / "config/experiment/train_vpp_ppo_lstm_retraining_v2.yaml"
JSBSIM_SOURCE = Path(r"E:\academic_paper\safe_fly\JSBSim_FlightControl\envs\JSBSim\data")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def yaml_dump(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def prepare_jsbsim_data(root: Path) -> Path:
    """Copy the verified F-16 data into the new source tree without overwriting."""
    if not JSBSIM_SOURCE.is_dir():
        raise FileNotFoundError(f"JSBSim source data directory is missing: {JSBSIM_SOURCE}")
    runtime_root = root / "jsbsim_runtime_root"
    destination = runtime_root / "envs" / "JSBSim" / "data"
    if destination.exists() and any(destination.rglob("*")):
        raise RuntimeError(f"refusing to overwrite existing JSBSim runtime data: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    receipts = []
    for source in sorted(JSBSIM_SOURCE.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(JSBSIM_SOURCE)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        receipts.append({"source": str(source.resolve()), "destination": str(target.resolve()), "source_receipt": file_receipt(source), "destination_receipt": file_receipt(target)})
    if not receipts:
        raise RuntimeError("JSBSim source data directory contains no files")
    dump(root / "jsbsim_data_manifest.json", {"source_id": SOURCE_ID, "source_root": str(JSBSIM_SOURCE.resolve()), "destination_root": str(destination.resolve()), "file_count": len(receipts), "files": receipts})
    return runtime_root


def run_command(command: list[str], cwd: Path, log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.write("START_UTC: " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        result = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "log": str(log_path.resolve()),
    }


def collect_corpus(config: dict, root: Path) -> dict:
    corpus_root = root / "corpus"
    existing_manifest = root / "corpus_manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if any(corpus_root.rglob("*.csv")):
            refreshed = verify_corpus(corpus_root / "trajectories", min_files=6, min_rows=15)
            refreshed["source_id"] = SOURCE_ID
            refreshed["config_sha256"] = canonical_config_sha256(config)
            refreshed["collector_backend"] = "jsbsim"
            refreshed["collector_reset_policy"] = "one initial reset per episode; no mid-episode reset"
            if refreshed.get("ok"):
                dump(existing_manifest, refreshed)
                return refreshed
    if any(corpus_root.rglob("*.csv")):
        raise RuntimeError(f"refusing to overwrite existing corpus: {corpus_root}")
    cfg = json.loads(json.dumps(config))
    runtime_root = prepare_jsbsim_data(root)
    old_backend = cfg.get("backend")
    cfg["backend"] = "jsbsim"
    record_config_override_if_changed(cfg, "backend", "jsbsim", old_backend, "run_lstm_retraining_v2.py:corpus")
    env_cfg = cfg.setdefault("env", {})
    old_env_backend = env_cfg.get("backend")
    env_cfg["backend"] = "jsbsim"
    record_config_override_if_changed(cfg, "env.backend", "jsbsim", old_env_backend, "run_lstm_retraining_v2.py:corpus")
    old_strict = env_cfg.get("strict_backend")
    env_cfg["strict_backend"] = True
    record_config_override_if_changed(cfg, "env.strict_backend", True, old_strict, "run_lstm_retraining_v2.py:corpus")
    old_legacy_root = env_cfg.get("legacy_project_root")
    env_cfg["legacy_project_root"] = str(runtime_root)
    record_config_override_if_changed(cfg, "env.legacy_project_root", str(runtime_root), old_legacy_root, "run_lstm_retraining_v2.py:corpus")
    env = CloseRangeTrackingEnv(cfg)
    if getattr(env, "_backend", None) != "jsbsim" or getattr(env, "_backend_fallback_occurred", False):
        raise RuntimeError("corpus collector did not initialize strict JSBSim")
    collector = cfg["collector"]
    policy = RuleBasedPursuitPolicy(mode=collector["rule_mode"])
    metrics = []
    try:
        for scenario_name, scenario_cfg in cfg["scenarios"].items():
            result = evaluate_scenario(
                env,
                scenario_name,
                scenario_cfg,
                num_episodes=int(collector["episodes_per_scenario"]),
                seeds=[int(seed) for seed in collector["seeds"]],
                policy=policy,
                save_trajectories=True,
                output_root=str(corpus_root),
                backend="jsbsim",
            )
            metrics.append(result)
    finally:
        env.close()
    dump(corpus_root / "scenario_metrics.json", metrics)
    report = verify_corpus(corpus_root / "trajectories", min_files=6, min_rows=15)
    report["source_id"] = SOURCE_ID
    report["config_sha256"] = canonical_config_sha256(cfg)
    report["collector_backend"] = "jsbsim"
    report["collector_reset_policy"] = "one initial reset per episode; no mid-episode reset"
    dump(root / "corpus_manifest.json", report)
    return report


def train_predictor(root: Path, args: argparse.Namespace) -> dict:
    corpus_dir = root / "corpus" / "trajectories" / "no_prediction_vpp" / "jsbsim"
    output_dir = root / "predictor_training"
    command = [
        sys.executable, "-m", "uav_vpp_guidance.trajectory_prediction.train_pipeline",
        "--config", str(PREDICTOR_CONFIG), "--data-dir", str(corpus_dir), "--model-type", "lstm",
        "--output-dir", str(output_dir), "--seed", str(args.predictor_seed),
        "--epochs", str(args.predictor_epochs), "--batch-size", str(args.predictor_batch_size),
        "--val-ratio", "0.2", "--device", "cpu",
    ]
    result = run_command(command, REPO_ROOT, root / "logs" / "predictor_training.log")
    if result["returncode"] != 0:
        raise RuntimeError(f"predictor training failed; see {result['log']}")
    checkpoint = output_dir / "best_model.pt"
    strict = strict_load_predictor(checkpoint)
    manifest = {"source_id": SOURCE_ID, "stage": "predictor_training", "config": file_receipt(PREDICTOR_CONFIG), "config_sha256": canonical_config_sha256(load_yaml_config(PREDICTOR_CONFIG)), "seed": args.predictor_seed, "epochs": args.predictor_epochs, "command_result": result, "checkpoint": strict}
    dump(output_dir / "run_manifest.json", manifest)
    return manifest


def train_policy(root: Path, args: argparse.Namespace, predictor_path: Path) -> list[dict]:
    results = []
    for seed in args.policy_seeds:
        output_dir = root / "policy_training" / f"seed_{seed}"
        cfg = load_yaml_config(POLICY_CONFIG)
        env_cfg = cfg.setdefault("env", {})
        legacy_root = str((root / "jsbsim_runtime_root").resolve())
        old_root = env_cfg.get("legacy_project_root")
        env_cfg["legacy_project_root"] = legacy_root
        record_config_override_if_changed(
            cfg, "env.legacy_project_root", legacy_root, old_root,
            "run_lstm_retraining_v2.py:policy",
        )
        prediction_cfg = cfg.setdefault("trajectory_prediction", {})
        old_checkpoint = prediction_cfg.get("checkpoint_path")
        prediction_cfg["checkpoint_path"] = str(predictor_path.resolve())
        record_config_override_if_changed(
            cfg, "trajectory_prediction.checkpoint_path",
            str(predictor_path.resolve()), old_checkpoint,
            "run_lstm_retraining_v2.py:policy",
        )
        history_cfg = prediction_cfg.setdefault("history", {})
        old_padding = history_cfg.get("allow_neural_padding", False)
        history_cfg["allow_neural_padding"] = True
        record_config_override_if_changed(
            cfg, "trajectory_prediction.history.allow_neural_padding", True,
            old_padding, "run_lstm_retraining_v2.py:policy",
        )
        resolved_config = output_dir / "resolved_config.yaml"
        yaml_dump(resolved_config, cfg)
        command = [
            sys.executable, "-m", "uav_vpp_guidance.training.train_prediction_vpp_ppo",
            "--config", str(resolved_config), "--predictor-type", "lstm", "--checkpoint", str(predictor_path),
            "--seed", str(seed), "--output-dir", str(output_dir), "--device", "cpu",
        ]
        result = run_command(command, REPO_ROOT, root / "logs" / f"policy_training_seed_{seed}.log")
        if result["returncode"] != 0:
            raise RuntimeError(f"policy training seed {seed} failed; see {result['log']}")
        checkpoint = output_dir / "checkpoints" / "last.pt"
        info = inspect_policy_checkpoint(checkpoint, min_timesteps=100_000)
        manifest = {
            "source_id": SOURCE_ID,
            "stage": "policy_training",
            "seed": seed,
            "config": file_receipt(resolved_config),
            "config_sha256": canonical_config_sha256(cfg),
            "predictor": file_receipt(predictor_path),
            "training_backend": cfg.get("backend"),
            "formal_evaluation_backend": "jsbsim",
            "command_result": result,
            "checkpoint": info,
            "git_info": get_git_info(str(REPO_ROOT)),
        }
        dump(output_dir / "run_manifest.json", manifest)
        results.append(manifest)
    return results


def evaluate_policy(root: Path, args: argparse.Namespace, predictor_path: Path, policy_path: Path, training_seed: int) -> dict:
    cfg = load_yaml_config(POLICY_CONFIG)
    heldout = {
        "heldout_head_on_diagonal": {
            "name": "heldout_head_on_diagonal", "own_init": {"position_m": [0.0, 0.0, 5300.0], "velocity_mps": 212.0, "heading_deg": 2.0}, "target_init": {"position_m": [2800.0, 300.0, 5250.0], "velocity_mps": 188.0, "heading_deg": 188.0}
        },
        "heldout_cross_left_far": {
            "name": "heldout_cross_left_far", "own_init": {"position_m": [0.0, 0.0, 5150.0], "velocity_mps": 218.0, "heading_deg": 6.0}, "target_init": {"position_m": [1900.0, 1250.0, 5450.0], "velocity_mps": 202.0, "heading_deg": 232.0}
        },
        "heldout_cross_right_far": {
            "name": "heldout_cross_right_far", "own_init": {"position_m": [0.0, 0.0, 4850.0], "velocity_mps": 218.0, "heading_deg": -6.0}, "target_init": {"position_m": [1900.0, -1250.0, 4550.0], "velocity_mps": 202.0, "heading_deg": 128.0}
        },
        "heldout_neutral_high": {
            "name": "heldout_neutral_high", "own_init": {"position_m": [0.0, 0.0, 5600.0], "velocity_mps": 208.0, "heading_deg": 0.0}, "target_init": {"position_m": [2100.0, 650.0, 5750.0], "velocity_mps": 200.0, "heading_deg": 168.0}
        },
        "heldout_disadvantage_low": {
            "name": "heldout_disadvantage_low", "own_init": {"position_m": [0.0, 0.0, 4300.0], "velocity_mps": 208.0, "heading_deg": 0.0}, "target_init": {"position_m": [-900.0, -500.0, 4250.0], "velocity_mps": 228.0, "heading_deg": 25.0}
        },
        "heldout_disadvantage_high": {
            "name": "heldout_disadvantage_high", "own_init": {"position_m": [0.0, 0.0, 5750.0], "velocity_mps": 208.0, "heading_deg": 0.0}, "target_init": {"position_m": [-750.0, 500.0, 5800.0], "velocity_mps": 228.0, "heading_deg": 335.0}
        },
    }
    old_backend = cfg.get("backend")
    cfg["backend"] = "jsbsim"
    record_config_override_if_changed(
        cfg, "backend", "jsbsim", old_backend,
        "run_lstm_retraining_v2.py:evaluation",
    )
    env_cfg = cfg.setdefault("env", {})
    legacy_root = str((root / "jsbsim_runtime_root").resolve())
    old_root = env_cfg.get("legacy_project_root")
    env_cfg["legacy_project_root"] = legacy_root
    record_config_override_if_changed(
        cfg, "env.legacy_project_root", legacy_root, old_root,
        "run_lstm_retraining_v2.py:evaluation",
    )
    for key, value in (("backend", "jsbsim"), ("use_jsbsim", True), ("strict_backend", True)):
        old_value = env_cfg.get(key)
        env_cfg[key] = value
        record_config_override_if_changed(
            cfg, f"env.{key}", value, old_value,
            "run_lstm_retraining_v2.py:evaluation",
        )
    prediction_cfg = cfg.setdefault("trajectory_prediction", {})
    old_checkpoint = prediction_cfg.get("checkpoint_path")
    prediction_cfg["checkpoint_path"] = str(predictor_path)
    record_config_override_if_changed(
        cfg, "trajectory_prediction.checkpoint_path", str(predictor_path), old_checkpoint,
        "run_lstm_retraining_v2.py:evaluation",
    )
    history_cfg = prediction_cfg.setdefault("history", {})
    old_padding = history_cfg.get("allow_neural_padding", False)
    history_cfg["allow_neural_padding"] = True
    record_config_override_if_changed(
        cfg, "trajectory_prediction.history.allow_neural_padding", True,
        old_padding, "run_lstm_retraining_v2.py:evaluation",
    )
    cfg["scenarios"] = heldout
    eval_dir = root / "heldout" / f"seed_{training_seed}" / "expert"
    yaml_dump(eval_dir / "resolved_config.yaml", cfg)
    opponent = ExpertOpponent({})
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent, opponent_config={"stage": "expert"})
    if getattr(env, "_backend", None) != "jsbsim" or getattr(env, "_backend_fallback_occurred", False):
        raise RuntimeError("heldout evaluation did not initialize strict JSBSim")
    sample_obs = env.reset(seed=args.eval_seeds[0])
    agent = PPOAgent(obs_dim=int(sample_obs["observation_vector"].shape[0]), action_dim=3, config=cfg, device="cpu")
    agent.load(str(policy_path))
    reset_count = 0
    original_reset = env.reset
    def tracked_reset(*reset_args, **reset_kwargs):
        nonlocal reset_count
        reset_count += 1
        return original_reset(*reset_args, **reset_kwargs)
    env.reset = tracked_reset
    try:
        metrics = evaluate_method(
            env, agent, cfg, "lstm_vpp_retraining_v2",
            num_episodes=args.eval_episodes_per_scenario * len(heldout),
            seeds=args.eval_seeds, scenarios=list(heldout), save_trajectories=True,
            output_dir=str(eval_dir), training_seed=training_seed,
        )
    finally:
        env.close()
    expected = args.eval_episodes_per_scenario * len(heldout) * len(args.eval_seeds)
    verification = verify_formal_metrics(metrics, expected)
    verification["reset_count"] = reset_count
    verification["expected_reset_count"] = expected
    if reset_count != expected:
        verification["ok"] = False
        verification["errors"].append(f"reset_count={reset_count}, expected {expected}")
    manifest = {
        "source_id": SOURCE_ID,
        "stage": "heldout_strict_jsbsim_evaluation",
        "training_seed": training_seed,
        "evaluation_seeds": args.eval_seeds,
        "scenarios": list(heldout),
        "episodes_per_scenario": args.eval_episodes_per_scenario,
        "backend_requested": "jsbsim",
        "backend_effective": getattr(env, "_backend", None),
        "backend_fallback_occurred": bool(getattr(env, "_backend_fallback_occurred", False)),
        "predictor": strict_load_predictor(predictor_path),
        "policy": inspect_policy_checkpoint(policy_path),
        "policy_checkpoint": file_receipt(policy_path),
        "resolved_config": file_receipt(eval_dir / "resolved_config.yaml"),
        "config_sha256": canonical_config_sha256(cfg),
        "git_info": get_git_info(str(REPO_ROOT)),
        "reset_count": reset_count,
        "expected_reset_count": expected,
        "verification": verification,
        "metrics_path": str((eval_dir / "metrics.json").resolve()),
    }
    dump(eval_dir / "metrics.json", metrics)
    dump(eval_dir / "run_manifest.json", manifest)
    return manifest


def build_asset_manifest(root: Path, predictor_manifest: dict, policy_manifests: list[dict], eval_manifest: dict) -> dict:
    predictor = root / "predictor_training" / "best_model.pt"
    assets = [{"role": "trajectory_predictor", "contract": predictor_manifest["checkpoint"], "source_id": SOURCE_ID}]
    for item in policy_manifests:
        assets.append({"role": "lstm_vpp_policy", "seed": item["seed"], "contract": item["checkpoint"], "source_id": SOURCE_ID})
    manifest = {
        "asset_manifest_id": SOURCE_ID + "-ASSETS",
        "source_id": SOURCE_ID,
        "historical_assets_reused": False,
        "predictor_contract": predictor_manifest["checkpoint"]["architecture"],
        "assets": assets,
        "heldout_evaluation": eval_manifest,
        "git_info": get_git_info(str(REPO_ROOT)),
    }
    dump(root / "asset_manifest.json", manifest)
    return manifest


def make_bundle(root: Path, asset_manifest: dict, predictor_manifest: dict, policy_manifests: list[dict], eval_manifest: dict) -> Path:
    bundle = root / "portable_bundle"
    if bundle.exists():
        raise RuntimeError(f"refusing to overwrite bundle: {bundle}")
    bundle.mkdir(parents=True)
    files: list[Path] = [root / "asset_manifest.json", root / "corpus_manifest.json", root / "jsbsim_data_manifest.json", root / "predictor_training" / "run_manifest.json", root / "heldout" / f"seed_{eval_manifest['training_seed']}" / "expert" / "run_manifest.json", root / "heldout" / f"seed_{eval_manifest['training_seed']}" / "expert" / "metrics.json", root / "heldout" / f"seed_{eval_manifest['training_seed']}" / "expert" / "resolved_config.yaml"]
    files.append(root / "predictor_training" / "best_model.pt")
    for item in policy_manifests:
        policy_dir = root / "policy_training" / f"seed_{item['seed']}"
        files.extend([policy_dir / "run_manifest.json", policy_dir / "config_snapshot.yaml", policy_dir / "checkpoints" / "last.pt"])
    files.extend(sorted((root / "jsbsim_runtime_root" / "envs" / "JSBSim" / "data").rglob("*")))
    copied = []
    for source in files:
        if source.is_file():
            target = bundle / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({"path": str(target.relative_to(bundle)), "sha256": file_receipt(target)["sha256"], "size_bytes": target.stat().st_size})
    dump(bundle / "bundle_info.json", {"source_id": SOURCE_ID, "asset_manifest": asset_manifest["asset_manifest_id"], "files": copied, "git_info": get_git_info(str(REPO_ROOT))})
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required before collection or training")
    parser.add_argument("--phase", choices=["all", "collect", "predictor", "policy", "evaluate", "package"], default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--predictor-seed", type=int, default=7101)
    parser.add_argument("--predictor-epochs", type=int, default=100)
    parser.add_argument("--predictor-batch-size", type=int, default=64)
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[7101, 7102, 7103])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[8101, 8102, 8103])
    parser.add_argument("--eval-episodes-per-scenario", type=int, default=2)
    args = parser.parse_args()
    if not args.execute:
        print(f"LOCKED: no collection, training, model loading, or JSBSim execution. Source ID: {SOURCE_ID}")
        return 0
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_config(COLLECT_CONFIG)
    if args.phase == "package":
        corpus_report = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
        predictor_manifest = json.loads(
            (root / "predictor_training" / "run_manifest.json").read_text(encoding="utf-8")
        )
        policy_manifests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "policy_training").glob("seed_*/run_manifest.json"))
        ]
        eval_manifest_path = root / "heldout" / "seed_7101" / "expert" / "run_manifest.json"
        eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
        if not eval_manifest.get("verification", {}).get("ok", False):
            raise RuntimeError("refusing to package an evaluation that failed its strict gate")
        asset_manifest = build_asset_manifest(
            root, predictor_manifest, policy_manifests, eval_manifest
        )
        bundle = make_bundle(
            root, asset_manifest, predictor_manifest, policy_manifests, eval_manifest
        )
        final = {
            "source_id": SOURCE_ID,
            "status": "completed",
            "corpus": corpus_report,
            "predictor": predictor_manifest,
            "policies": policy_manifests,
            "evaluation": eval_manifest,
            "asset_manifest": str((root / "asset_manifest.json").resolve()),
            "portable_bundle": str(bundle.resolve()),
        }
        dump(root / "source_run_manifest.json", final)
        print(json.dumps(final, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.phase in ("all", "collect"):
        corpus_report = collect_corpus(cfg, root)
    else:
        corpus_report = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    if args.phase == "collect":
        print(json.dumps({"source_id": SOURCE_ID, "status": "corpus_completed", "corpus": corpus_report}, indent=2, ensure_ascii=False))
        return 0
    if args.phase in ("all", "predictor"):
        predictor_manifest = train_predictor(root, args)
    else:
        predictor_manifest = json.loads((root / "predictor_training" / "run_manifest.json").read_text(encoding="utf-8"))
    if args.phase == "predictor":
        print(json.dumps({"source_id": SOURCE_ID, "status": "predictor_completed", "predictor": predictor_manifest}, indent=2, ensure_ascii=False))
        return 0
    predictor_path = root / "predictor_training" / "best_model.pt"
    if args.phase in ("all", "policy"):
        policy_manifests = train_policy(root, args, predictor_path)
    else:
        policy_manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "policy_training").glob("seed_*/run_manifest.json"))]
    if args.phase == "policy":
        print(json.dumps({"source_id": SOURCE_ID, "status": "policy_completed", "policies": policy_manifests}, indent=2, ensure_ascii=False))
        return 0
    policy_choice = policy_manifests[0]
    policy_path = Path(policy_choice["checkpoint"]["receipt"]["path"])
    if args.phase in ("all", "evaluate"):
        eval_manifest = evaluate_policy(root, args, predictor_path, policy_path, int(policy_choice["seed"]))
    else:
        eval_manifest = json.loads(next((root / "heldout").rglob("run_manifest.json")).read_text(encoding="utf-8"))
    if args.phase == "evaluate":
        print(json.dumps({"source_id": SOURCE_ID, "status": "evaluation_completed", "evaluation": eval_manifest}, indent=2, ensure_ascii=False))
        return 0
    asset_manifest = build_asset_manifest(root, predictor_manifest, policy_manifests, eval_manifest)
    bundle = make_bundle(root, asset_manifest, predictor_manifest, policy_manifests, eval_manifest)
    final = {"source_id": SOURCE_ID, "status": "completed", "corpus": corpus_report, "predictor": predictor_manifest, "policies": policy_manifests, "evaluation": eval_manifest, "asset_manifest": str((root / "asset_manifest.json").resolve()), "portable_bundle": str(bundle.resolve())}
    dump(root / "source_run_manifest.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
