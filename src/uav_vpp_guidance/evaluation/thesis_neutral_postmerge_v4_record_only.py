"""Reward-free post-handoff execution for V4 physical-support records."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as legacy
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as neutral_engine


class NeutralPostMergeV4RecordError(RuntimeError):
    """Raised when the reward-unobserved runtime record contract is broken."""


def _finite_action(action: Any) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(np.abs(result) > 1.0 + 1e-6):
        raise NeutralPostMergeV4RecordError("V4 action is not a finite normalized 3-D VPP action")
    return result


def build_record_only_env(
    config: Mapping[str, Any], registry: Mapping[str, Any], opponent_id: str
) -> CloseRangeTrackingEnv:
    """Build the frozen JSBSim chain without reading a training configuration."""

    runtime = neutral_engine._runtime(config)
    base = copy.deepcopy(neutral_engine.load_yaml(legacy.BASE_ENV_CONFIG))
    base["backend"] = "jsbsim"
    base["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
    base.setdefault("env", {}).update(
        {
            "use_jsbsim": True,
            "strict_backend": True,
            "legacy_project_root": str(runtime["legacy_project_root"]),
            "aircraft_model": str(runtime["aircraft_model"]),
            "high_level_dt": float(runtime["high_level_dt_s"]),
            "max_high_level_steps": int(config["runtime_feasibility"]["max_high_level_steps"]),
            "success_range_m": 0.0,
            "success_ata_deg": 0.0,
            "success_hold_time_s": 1.0,
            "max_range_m": 12000.0,
        }
    )
    base["observation"] = {
        "include_gains": True,
        "include_guidance_state": False,
        "include_saturation": False,
        "include_opponent_stage": False,
        "include_task_type": True,
    }
    base["trajectory_prediction"] = copy.deepcopy(runtime["trajectory_prediction"])
    base["virtual_point"] = copy.deepcopy(runtime["virtual_point"])
    base["opponent_stage"] = opponent_id
    base["attack_zone"] = copy.deepcopy(config["fixed_contract"]["attack_zone"])
    entry = registry["opponents"][opponent_id]
    return CloseRangeTrackingEnv(
        base,
        opponent_policy=neutral_engine._instantiate_opponent(entry),
        opponent_config={"stage": opponent_id, **copy.deepcopy(entry)},
    )


class RewardFreeNeutralPostMergeEpisode(neutral_engine.NeutralPostMergeEpisode):
    """Use the frozen run-in, but record post-handoff dynamics without reward code."""

    def record_only_step(self, action: Any) -> tuple[np.ndarray | None, bool, dict[str, Any]]:
        if self.observation is None or self.base is None or self.handoff_step is None:
            raise NeutralPostMergeV4RecordError("V4 record requested before handoff")
        pre_phase, pre_state = self.phase, self.dynamic_state
        pre_first_pass = self.first_pass_complete
        pre_pair_allowed = self._target_pair_allowed()
        pre_prediction_valid = self.last_info.get("prediction_valid") is True
        pre_prediction_fallback = self.last_info.get("prediction_fallback") is True
        policy_observation = self.policy_observation()
        pre_base = self.base.copy()
        pre_history = tuple(item.copy() for item in self.history)
        normalized = _finite_action(action)
        observation, _environment_reward, terminated, truncated, info = self.env.step(normalized)
        self.total_steps += 1
        self.last_info = info
        self.first_pass_complete = bool(info.get("first_pass_complete", False))
        self._observe(observation)
        valid_target_step = bool(
            pre_first_pass
            and pre_state == neutral_engine.TARGET_STATE
            and pre_phase == neutral_engine.TARGET_PHASE
            and pre_pair_allowed
            and pre_prediction_valid
            and not pre_prediction_fallback
        )
        done = bool(terminated or truncated or self.total_steps >= self.env.max_steps)
        next_observation = None if done else self.policy_observation()
        own_state = info.get("own_state", {}) if isinstance(info.get("own_state", {}), Mapping) else {}
        metric = {
            "step": self.total_steps,
            "phase": pre_phase,
            "dynamic_state": pre_state,
            "valid_target_step": valid_target_step,
            "range_m": float(legacy._base_map(pre_base)["range_m"]),
            "range_rate_mps": float(legacy._base_map(pre_base)["range_rate_mps"]),
            "target_in_attack_zone": bool(info.get("target_in_attack_zone", False)),
            "ego_in_attack_zone": bool(info.get("ego_in_attack_zone", False)),
            "ego_hp": float(info.get("ego_hp", 100.0)),
            "target_hp": float(info.get("target_hp", 100.0)),
            "vp_forward_bias_m": float(info.get("vp_forward_bias_m", np.nan)),
            "vp_lateral_bias_m": float(info.get("vp_lateral_bias_m", np.nan)),
            "vp_vertical_bias_m": float(info.get("vp_vertical_bias_m", np.nan)),
            "nz_cmd": float(info.get("nz_cmd", np.nan)),
            "actual_nz_g": float(own_state.get("nz_g", info.get("nz_g", np.nan))),
            "ego_attack_aoa_deg": float(info.get("ego_attack_aoa_deg", np.nan)),
            "terminal_reason": str(info.get("reason") or "timeout"),
            "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
        }
        self.post_handoff_steps.append(
            self._step_record(
                stage="post_handoff",
                action=normalized,
                phase=pre_phase,
                dynamic_state=pre_state,
                observation_66d=policy_observation,
                info=self.last_info,
                base_vector=pre_base,
                history=pre_history,
            )
        )
        if metric["backend_fallback_occurred"]:
            raise NeutralPostMergeV4RecordError("V4 post-handoff reported a backend fallback")
        if "reward" in metric:
            raise NeutralPostMergeV4RecordError("V4 record-only metric must not contain reward")
        return next_observation, done, metric


def record_only_result(
    *,
    opponent: str,
    scenario: Mapping[str, Any],
    handoff: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a physical-support result without a reward or performance score."""

    if any("reward" in metric for metric in metrics):
        raise NeutralPostMergeV4RecordError("V4 results cannot serialize reward-bearing metrics")
    valid = [metric for metric in metrics if bool(metric.get("valid_target_step"))][:20]
    final = metrics[-1] if metrics else {}
    return {
        "opponent": opponent,
        "scenario": scenario["name"],
        "handoff_reached": bool(handoff.get("handoff_reached")),
        "handoff_step": handoff.get("handoff_step"),
        "handoff_state_sha256": handoff.get("handoff_state_sha256"),
        "valid_target_steps": sum(bool(metric.get("valid_target_step")) for metric in metrics),
        "qualifying": len(valid) == 20,
        "telemetry_complete": bool(ledger["summary"]["full_post_handoff_telemetry"]),
        "terminal_reason": str(final.get("terminal_reason", handoff.get("terminal_reason", "no_handoff"))),
        "training_reward": "prohibited",
        "environment_reward_recording": "prohibited",
    }


def runtime_record(
    *,
    source_id: str,
    opponent: str,
    scenario: Mapping[str, Any],
    result: Mapping[str, Any],
    ledger: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize one V4 record while excluding rewards and paired scores."""

    if source_id != "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V4":
        raise NeutralPostMergeV4RecordError("V4 runtime record source ID drifted")
    if opponent not in ("expert", "end_to_end", "independent_ppo_vpp"):
        raise NeutralPostMergeV4RecordError("V4 runtime record opponent drifted")
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        raise NeutralPostMergeV4RecordError("V4 scenario metadata is missing")
    required_result = (
        "handoff_reached",
        "valid_target_steps",
        "qualifying",
        "telemetry_complete",
        "terminal_reason",
        "training_reward",
        "environment_reward_recording",
    )
    if (
        any(key not in result for key in required_result)
        or result.get("training_reward") != "prohibited"
        or result.get("environment_reward_recording") != "prohibited"
    ):
        raise NeutralPostMergeV4RecordError("V4 runtime result lacks the reward-unobserved contract")
    if "reward" in result:
        raise NeutralPostMergeV4RecordError("V4 runtime result must not contain reward")
    summary = ledger.get("summary") if isinstance(ledger.get("summary"), Mapping) else None
    if summary is None or summary.get("no_backend_fallback") is not True:
        raise NeutralPostMergeV4RecordError("V4 ledger reports a backend fallback")
    if summary.get("full_post_handoff_telemetry") is not True or result.get("telemetry_complete") is not True:
        raise NeutralPostMergeV4RecordError("V4 runtime telemetry is incomplete")
    for key in (
        "scenario_application_passed",
        "raw_si_phase_replay_passed",
        "initial_pre_merge_semantics_passed",
        "record_only_no_reward_terms",
    ):
        if runtime_evidence.get(key) is not True:
            raise NeutralPostMergeV4RecordError(f"V4 runtime evidence failed: {key}")
    return {
        "source_id": source_id,
        "opponent": opponent,
        "family_id": metadata.get("feasibility_family_id"),
        "pair_key": metadata.get("pair_key"),
        "metadata": dict(metadata),
        "result": {
            "handoff_reached": bool(result["handoff_reached"]),
            "valid_target_steps": int(result["valid_target_steps"]),
            "qualifying": bool(result["qualifying"]),
            "telemetry_complete": True,
            "terminal_reason": str(result["terminal_reason"]),
            "handoff_state_sha256": result.get("handoff_state_sha256"),
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
        },
        "runtime_evidence": {
            "scenario_application_passed": True,
            "raw_si_phase_replay_passed": True,
            "initial_pre_merge_semantics_passed": True,
            "record_only_no_reward_terms": True,
        },
        "paired_delta_eligible": False,
        "role": "nonlearning_reward_unobserved_runtime_feasibility",
    }
