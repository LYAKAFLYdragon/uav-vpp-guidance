#!/usr/bin/env python3
"""Train the preregistered single re-entry PPO candidate after authorization."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import build_record_only_env
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as geometry
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_reentry_terminal_reward_pilot import TerminalRewardEpisode
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.training.thesis_shared_skill_policy import ThesisSharedSkillPPOAgent


SOURCE_ID = "THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = ("candidate", "frozen_fixed_head_on", "frozen_fixed_crossing")


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_runtime_config(path: Path) -> tuple[dict[str, Any], Path]:
    overlay = _load(path)
    base_value = overlay.pop("base_config")
    base_path = _resolve(base_value)
    expected = str(overlay.pop("base_config_sha256")).lower()
    if _hash(base_path) != expected:
        raise ValueError("runtime design base-config hash mismatch")
    base = _load(base_path)
    if "base_config" in base:
        base, _ = load_runtime_config(base_path)
    return _merge(base, overlay), base_path


def _agent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["proposed_training"]
    return {"training": {"ppo": copy.deepcopy(training["ppo"]), "policy": copy.deepcopy(training["policy"])}}


def _checkpoint_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "profile": config["training"]["profile"],
        "observation_dim": 66,
        "action_dim": 3,
        "p3_encoder_sha256": config["inputs"]["frozen_p3_encoder_sha256"],
        "training_seed": int(config["proposed_training"]["training_seed"]),
        "routing_enabled": False,
    }


def _require_verified_preflight(config: Mapping[str, Any]) -> None:
    authorization = config["authorization"]
    expected_source = authorization["required_preflight_source_id"]
    gate_path = Path(authorization["required_preflight_gate"])
    verification_path = Path(authorization["required_preflight_verification"])
    if not gate_path.is_file() or not verification_path.is_file():
        raise ValueError("required preflight artifacts are missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    gate_passes = gate.get("passes")
    gate_passes = gate_passes is True or (
        isinstance(gate_passes, int) and not isinstance(gate_passes, bool) and gate_passes == 1
    )
    if gate.get("source_id") != expected_source or not gate_passes:
        raise ValueError("preflight gate did not pass")
    if verification.get("source_id") != expected_source or verification.get("verification_status") != "pass" or verification.get("passes") is not True:
        raise ValueError("preflight independent verification did not pass")


def _require_frozen_inputs(config: Mapping[str, Any]) -> None:
    for label in ("train_distribution", "dev_manifest", "heldout_manifest", "runtime_template", "runtime_registry"):
        path = _resolve(config["inputs"][label])
        expected = str(config["inputs"][f"{label}_sha256"]).lower()
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"{label} hash mismatch")
    encoder = Path(config["inputs"]["frozen_p3_encoder"])
    expected_encoder = str(config["inputs"]["frozen_p3_encoder_sha256"]).lower()
    if not encoder.is_file() or _hash(encoder) != expected_encoder:
        raise ValueError("frozen P3 encoder hash mismatch")


def _sample_train_scenario(distribution: Mapping[str, Any], episode_index: int) -> dict[str, Any]:
    """Sample a deterministic continuous geometry without touching dev/heldout."""

    sampling = distribution["sampling"]
    rng = np.random.default_rng(int(distribution["sampling"]["scenario_seed_range"][0]) + int(episode_index))
    range_m = float(rng.uniform(sampling["initial_range_m"]["minimum"], sampling["initial_range_m"]["maximum"]))
    own_speed = float(rng.uniform(sampling["own_speed_mps"]["minimum"], sampling["own_speed_mps"]["maximum"]))
    target_speed = float(rng.uniform(sampling["target_speed_mps"]["minimum"], sampling["target_speed_mps"]["maximum"]))
    altitude_diff = float(rng.choice(sampling["target_minus_own_altitude_m"]["values"]))
    state = list(distribution["initial_states"])[episode_index % len(distribution["initial_states"])]
    mirror = list(sampling["mirror_sign"]["values"])[episode_index % len(sampling["mirror_sign"]["values"])]
    sign = -1.0 if mirror == "negative" else 1.0
    los_azimuth = sign * 18.0
    target_los_angle = 20.0 if state == "head_on" else 90.0
    horizontal = math.sqrt(range_m ** 2 - altitude_diff ** 2)
    fraction = horizontal / range_m
    own_heading = geometry._horizontal_heading(los_azimuth, fraction, 20.0, -int(sign))
    target_heading = geometry._horizontal_heading((los_azimuth + 180.0) % 360.0, fraction, target_los_angle, int(sign))
    azimuth = math.radians(los_azimuth)
    seed = int(sampling["scenario_seed_range"][0]) + int(episode_index)
    return {
        "name": f"reward_pilot_train_{episode_index:06d}",
        "own_init": {
            "position_m": [0.0, 0.0, 5200.0],
            "velocity_mps": own_speed,
            "heading_deg": own_heading,
        },
        "target_init": {
            "position_m": [horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), 5200.0 + altitude_diff],
            "velocity_mps": target_speed,
            "heading_deg": target_heading,
        },
        "metadata": {
            "split": "train",
            "initial_class": state,
            "height_condition": "continuous",
            "mirror_sign": mirror,
            "scenario_signature": f"reward_pilot_train::{episode_index}::{seed}",
            "scenario_seed": seed,
            "continuous_run_in": True,
            "phase_contract": "raw_physical_first_pass_aware_v1",
        },
    }


def _episode_result(
    *,
    opponent: str,
    method: str,
    scenario: Mapping[str, Any],
    handoff: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid = [metric for metric in metrics if metric["valid_target_step"]]
    window = valid[:20]
    final = metrics[-1] if metrics else {}
    return {
        "opponent": opponent,
        "method": method,
        "scenario_signature": scenario["metadata"]["scenario_signature"],
        "mirror_sign": scenario["metadata"]["mirror_sign"],
        "handoff_reached": bool(handoff.get("handoff_reached")),
        "handoff_step": handoff.get("handoff_step"),
        "valid_target_steps": len(valid),
        "qualifying": len(window) == 20,
        "intent_loss_auc20": float(np.mean([metric["after_intent_loss"] for metric in window])) if len(window) == 20 else None,
        "range_opening_fraction_auc20": float(np.mean([float(metric["range_rate_mps"] > 0.0) for metric in window])) if window else None,
        "specific_energy_height_delta_m_at_20": float(window[-1]["specific_energy_height_delta_m"]) if len(window) == 20 else None,
        "target_attack_zone_exposure_fraction_auc20": float(np.mean([float(metric["target_in_attack_zone"]) for metric in window])) if window else None,
        "ego_attack_zone_exposure_fraction_auc20": float(np.mean([float(metric["ego_in_attack_zone"]) for metric in window])) if window else None,
        "terminal_reason": final.get("terminal_reason", handoff.get("terminal_reason", "no_handoff")),
        "ego_failure": bool(final.get("ego_failure", False)),
        "win": final.get("terminal_reward", 0.0) > 0.0,
    }


def _run_method(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    opponent: str,
    method: str,
    *,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    env = build_record_only_env(config, registry, opponent)
    run_in = engine._specialist(registry, "frozen_fixed_head_on")
    encoder = FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"])
    episode = TerminalRewardEpisode(env, run_in, encoder, config)
    candidate = None
    baseline = None
    if method == "candidate":
        if checkpoint is None:
            raise ValueError("candidate evaluation requires checkpoint")
        candidate = ThesisSharedSkillPPOAgent(_agent_config(config), device="cpu")
        candidate.load_strict(checkpoint, _checkpoint_metadata(config))
    elif method == "frozen_fixed_head_on":
        baseline = engine._specialist(registry, "frozen_fixed_head_on")
    elif method == "frozen_fixed_crossing":
        baseline = engine._specialist(registry, "frozen_fixed_crossing")
    else:
        raise ValueError(f"unknown method: {method}")
    results = []
    try:
        for scenario in scenarios:
            observation, handoff = episode.reset_to_handoff(scenario, int(scenario["metadata"]["scenario_seed"]))
            metrics = []
            while observation is not None:
                if candidate is not None:
                    action = candidate.select_action(observation, deterministic=True)[0]
                else:
                    assert baseline is not None and episode.observation is not None
                    action = baseline.get_deterministic_action(episode.observation["observation_vector"])
                observation, _reward, done, metric = episode.step(action)
                metrics.append(metric)
                if done:
                    observation = None
            results.append(_episode_result(opponent=opponent, method=method, scenario=scenario, handoff=handoff, metrics=metrics))
    finally:
        episode.close()
    return results


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualifying = [record for record in records if record["qualifying"]]
    return {
        "episodes": len(records),
        "handoff_episodes": sum(bool(record["handoff_reached"]) for record in records),
        "qualifying_episodes": len(qualifying),
        "mean_intent_loss_auc20": float(np.mean([record["intent_loss_auc20"] for record in qualifying])) if qualifying else None,
        "ego_crash_oob_rate": sum(bool(record["ego_failure"]) for record in records) / max(len(records), 1),
    }


def _paired_delta(candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> list[float]:
    by_signature = {record["scenario_signature"]: record for record in baseline if record["qualifying"]}
    return [
        float(record["intent_loss_auc20"] - by_signature[record["scenario_signature"]]["intent_loss_auc20"])
        for record in candidate
        if record["qualifying"] and record["scenario_signature"] in by_signature
    ]


def _evaluate_dev(
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    per_opponent = {}
    worst_delta = -float("inf")
    safety_pass = True
    for opponent in OPPONENTS:
        candidate_summary = _summary(candidate[opponent])
        head_summary = _summary(baseline[opponent]["frozen_fixed_head_on"])
        deltas = _paired_delta(candidate[opponent], baseline[opponent]["frozen_fixed_head_on"])
        mean_delta = float(np.mean(deltas)) if deltas else None
        crash_delta = candidate_summary["ego_crash_oob_rate"] - head_summary["ego_crash_oob_rate"]
        safety = crash_delta <= 0.05
        safety_pass = safety_pass and safety
        worst_delta = max(worst_delta, mean_delta if mean_delta is not None else float("inf"))
        per_opponent[opponent] = {
            "paired_count": len(deltas),
            "mean_paired_intent_loss_delta_vs_head_on": mean_delta,
            "ego_crash_oob_delta_vs_head_on": crash_delta,
            "safety_pass": safety,
            "candidate": candidate_summary,
            "head_on": head_summary,
        }
    return {
        "selection_score_worst_opponent_mean_delta": worst_delta,
        "safety_pass": safety_pass,
        "per_opponent": per_opponent,
    }


def run_pilot(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Run the authorized dev-only pilot; heldout execution is prohibited here."""

    registry = _load(_resolve(config["inputs"]["runtime_registry"]))
    dev_scenarios = _load(_resolve(config["inputs"]["dev_manifest"]))["scenarios"]
    distribution = _load(_resolve(config["inputs"]["train_distribution"]))
    training = config["proposed_training"]
    random.seed(int(training["training_seed"]))
    np.random.seed(int(training["training_seed"]))
    torch.manual_seed(int(training["training_seed"]))
    baseline = {
        opponent: {
            "frozen_fixed_head_on": _run_method(config, registry, dev_scenarios, opponent, "frozen_fixed_head_on"),
            "frozen_fixed_crossing": _run_method(config, registry, dev_scenarios, opponent, "frozen_fixed_crossing"),
        }
        for opponent in OPPONENTS
    }
    engine.write_json(output_root / "dev_baseline.json", baseline)
    agent = ThesisSharedSkillPPOAgent(_agent_config(config), device="cpu")
    encoder = FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"])
    episodes = {
        opponent: TerminalRewardEpisode(
            build_record_only_env(config, registry, opponent),
            engine._specialist(registry, "frozen_fixed_head_on"),
            encoder,
            config,
        )
        for opponent in OPPONENTS
    }
    checkpoints = [int(value) for value in training["checkpoint_evaluation_steps"]]
    checkpoint_index = 0
    summaries = []
    evaluations = []
    stopped = False
    stop_reason = None
    attempts = 0
    try:
        while agent.total_timesteps < int(training["total_timesteps"]):
            if attempts >= int(training["max_episode_attempts"]):
                stopped, stop_reason = True, "max_episode_attempts_before_budget"
                break
            opponent = OPPONENTS[attempts % len(OPPONENTS)]
            scenario = _sample_train_scenario(distribution, attempts)
            attempts += 1
            episode = episodes[opponent]
            observation, handoff = episode.reset_to_handoff(scenario, int(scenario["metadata"]["scenario_seed"]))
            metrics = []
            while observation is not None and agent.total_timesteps < int(training["total_timesteps"]):
                action, log_prob, value = agent.select_action(observation, deterministic=False)
                next_observation, reward, done, metric = episode.step(action)
                agent.store_transition(observation, action, log_prob, reward, done, value)
                metrics.append(metric)
                if agent.buffer.full:
                    agent.update(None if done else next_observation)
                observation = None if done else next_observation
                while checkpoint_index < len(checkpoints) and agent.total_timesteps >= checkpoints[checkpoint_index]:
                    if len(agent.buffer):
                        agent.update(None if observation is None else observation)
                    step = checkpoints[checkpoint_index]
                    checkpoint = output_root / "checkpoints" / f"step_{step}.pt"
                    agent.save(checkpoint, _checkpoint_metadata(config))
                    candidate = {opponent_id: _run_method(config, registry, dev_scenarios, opponent_id, "candidate", checkpoint=checkpoint) for opponent_id in OPPONENTS}
                    evaluation = _evaluate_dev(candidate, baseline)
                    evaluation.update({"step": step, "checkpoint": str(checkpoint)})
                    evaluations.append(evaluation)
                    engine.write_json(output_root / "dev_candidate" / f"step_{step}.json", {"records": candidate, "summary": evaluation})
                    checkpoint_index += 1
                    if not evaluation["safety_pass"]:
                        stopped, stop_reason = True, f"dev_safety_stop_at_{step}"
                        observation = None
                        break
            summaries.append({
                "attempt": attempts,
                "opponent": opponent,
                "handoff_reached": bool(handoff.get("handoff_reached")),
                "candidate_steps": len(metrics),
                "terminal_reason": metrics[-1]["terminal_reason"] if metrics else handoff.get("terminal_reason"),
            })
            if stopped:
                break
    finally:
        for episode in episodes.values():
            episode.close()
    if len(agent.buffer):
        agent.update(None)
    last = output_root / "checkpoints" / "last.pt"
    agent.save(last, _checkpoint_metadata(config))
    eligible = [item for item in evaluations if item["safety_pass"] and math.isfinite(item["selection_score_worst_opponent_mean_delta"])]
    selected = min(eligible, key=lambda item: (item["selection_score_worst_opponent_mean_delta"], item["step"])) if eligible else None
    best = None
    if selected:
        best = output_root / "checkpoints" / "best.pt"
        shutil.copy2(Path(selected["checkpoint"]), best)
    result = {
        "source_id": SOURCE_ID,
        "training_steps": agent.total_timesteps,
        "attempts": attempts,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "summaries": summaries,
        "dev_evaluations": evaluations,
        "selected_checkpoint": None if best is None else str(best),
        "heldout_executed": False,
    }
    engine.write_json(output_root / "training_summary.json", result)
    return result


def _emit_completion_result(result: Mapping[str, Any]) -> None:
    """Best-effort stdout reporting after the authoritative files are written."""
    try:
        print(json.dumps(result, indent=2))
    except OSError:
        # A supervising shell may have closed stdout after timing out.  The
        # completed manifest and summary must remain the authoritative result.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, _base = load_runtime_config(args.config)
    if not args.execute:
        print(json.dumps({"source_id": SOURCE_ID, "mode": "runtime_preflight"}))
        return 0
    if config["authorization"]["execution_permitted"] is not True or config["authorization"]["training_permitted"] is not True:
        raise ValueError("pilot execution is not authorized")
    if config["authorization"].get("heldout_execution_permitted") is not False:
        raise ValueError("dev-only pilot must prohibit heldout execution")
    _require_verified_preflight(config)
    _require_frozen_inputs(config)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("pilot requires clean worktree")
    output_root = Path(config["outputs"]["runtime_root"])
    if output_root.exists():
        raise ValueError("pilot output root already exists")
    if shutil.disk_usage(output_root.parent).free / 1024 ** 3 < float(config["observability"]["min_free_disk_gb"]):
        raise ValueError("insufficient free disk")
    output_root.mkdir(parents=True)
    try:
        result = run_pilot(config, output_root)
        engine.write_json(output_root / "run_manifest.json", {
            "source_id": SOURCE_ID,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": _hash(args.config),
            "training_permitted": True,
            "heldout_executed": False,
        })
    except Exception as error:
        engine.write_json(output_root / "failure.json", {"source_id": SOURCE_ID, "error": repr(error)})
        raise
    # The completed run is already persisted above.  A closed parent stdout pipe
    # must not manufacture a failure artifact after a valid scientific result.
    _emit_completion_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
