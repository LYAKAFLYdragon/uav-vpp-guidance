"""Phase-aware online episode for the preregistered terminal-reward pilot."""
from __future__ import annotations

from collections import deque
import copy
from typing import Any, Mapping

import numpy as np

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import classify_dynamic_state
from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.raw_si_reentry_phase import (
    RawSIReentryPhaseConfig,
    RawSIReentryPhaseTracker,
)
from uav_vpp_guidance.evaluation.terminal_outcome_reward import (
    TerminalOutcomeRewardConfig,
    terminal_outcome_reward,
)
from uav_vpp_guidance.evaluation.single_motif_continuous_66d import project_base_observation
from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as legacy
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder, intent_loss, physical_intent_vector


RANGE_REFERENCE_M = 5000.0
RANGE_RATE_REFERENCE_MPS = 200.0
PROFILE = "range_extension"


class TerminalRewardPilotError(RuntimeError):
    """Raised when the online pilot would violate a frozen contract."""


def _terminal_reward_config(config: Mapping[str, Any]) -> TerminalOutcomeRewardConfig:
    values = config["reward"]["terminal_outcome_reward"]
    return TerminalOutcomeRewardConfig(
        win_bonus=float(values["win_bonus"]),
        loss_penalty=float(values["loss_penalty"]),
        draw_penalty=float(values["draw_penalty"]),
        ego_crash_oob_supplemental_penalty=float(values["ego_crash_oob_supplemental_penalty"]),
    )


class TerminalRewardEpisode:
    """One legal reset, frozen run-in, then a raw-phase-aware candidate handoff."""

    def __init__(
        self,
        env: Any,
        run_in_policy: FrozenSpecialistPolicy,
        encoder: FrozenP3Encoder,
        config: Mapping[str, Any],
    ) -> None:
        self.env = env
        self.run_in_policy = run_in_policy
        self.encoder = encoder
        self.config = config
        self.contract = FiveStateObservationContract()
        self.history: deque[np.ndarray] = deque(maxlen=10)
        self.history_features = TemporalGeometryFeatureExtractor(10)
        self.observation_schema: Mapping[str, Any] = {}
        self.observation: Mapping[str, Any] | None = None
        self.base: np.ndarray | None = None
        self.phase = "pre_merge"
        self.dynamic_state = "unknown"
        self.last_info: Mapping[str, Any] = {}
        self.total_steps = 0
        self.handoff_step: int | None = None
        self._reset_phase_tracker()

    def _reset_phase_tracker(self) -> None:
        event = self.config["observability"]["raw_si_first_pass_event"]
        self.phase_tracker = RawSIReentryPhaseTracker(
            RawSIReentryPhaseConfig(first_pass=RawSIFirstPassConfig(**dict(event)))
        )

    def _observe(self, observation: Mapping[str, Any]) -> None:
        base = project_base_observation(observation["observation_vector"], self.observation_schema)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise TerminalRewardPilotError("pilot requires a finite 16-D base observation")
        self.history.append(base.copy())
        self.history_features.update(legacy._history_frame(base))
        values = legacy._base_map(base)
        self.phase = self.phase_tracker.update(
            values["range_m"] * RANGE_REFERENCE_M,
            values["range_rate_mps"] * RANGE_RATE_REFERENCE_MPS,
            step_index=self.total_steps,
        )
        self.dynamic_state = classify_dynamic_state(
            legacy._angle_deg(values["ata_sin"], values["ata_cos"]),
            legacy._angle_deg(values["aa_sin"], values["aa_cos"]),
        )
        if self.dynamic_state == "unknown":
            raise TerminalRewardPilotError("dynamic taxonomy became unknown")
        self.observation = observation
        self.base = base

    def _policy_observation(self) -> np.ndarray:
        if self.base is None or len(self.history) != 10:
            raise TerminalRewardPilotError("66-D policy observation requires exact 10-frame history")
        profile = compile_profile(PROFILE)
        result = self.contract.compose(
            base_geometry=self.base,
            explicit_history=self.history_features.features(),
            intent_targets=profile["target_vector"],
            intent_weights=profile["weight_vector"],
            temporal_embedding=self.encoder.encode(np.stack(tuple(self.history), axis=0)),
        )
        if result.shape != (66,) or not np.isfinite(result).all():
            raise TerminalRewardPilotError("invalid 66-D policy observation")
        return result

    def _trigger_ready(self) -> bool:
        return bool(
            len(self.history) == 10
            and self.phase == "re_entry"
            and self.phase_tracker.first_pass_complete
            and self.last_info.get("prediction_valid") is True
            and self.last_info.get("prediction_fallback") is not True
        )

    def reset_to_handoff(self, scenario: Mapping[str, Any], seed: int) -> tuple[np.ndarray | None, dict[str, Any]]:
        self.history.clear()
        self.history_features.reset()
        self._reset_phase_tracker()
        self.observation = None
        self.base = None
        self.phase = "pre_merge"
        self.dynamic_state = "unknown"
        self.last_info = {}
        self.total_steps = 0
        self.handoff_step = None
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise TerminalRewardPilotError("strict JSBSim backend was not selected")
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="frozen_run_in",
            specialist_reason="terminal_reward_pilot_run_in",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        while self.total_steps < self.env.max_steps:
            if self._trigger_ready():
                self.handoff_step = self.total_steps
                self.env.set_runtime_specialist_context(
                    specialist_key="head_on",
                    specialist_profile=None,
                    specialist_mode_name="terminal_reward_candidate",
                    specialist_reason="raw_physical_reentry_handoff",
                )
                return self._policy_observation(), {
                    "handoff_reached": True,
                    "handoff_step": self.handoff_step,
                    "phase": self.phase,
                    "dynamic_state": self.dynamic_state,
                }
            if self.observation is None:
                raise TerminalRewardPilotError("run-in lost current observation")
            action = self.run_in_policy.get_deterministic_action(self.observation["observation_vector"])
            observation, _ignored_reward, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1
            self.last_info = info
            self._observe(observation)
            if bool(info.get("backend_fallback_occurred", False)):
                raise TerminalRewardPilotError("run-in reported backend fallback")
            if terminated or truncated:
                return None, {
                    "handoff_reached": False,
                    "terminal_reason": str(info.get("reason") or "terminated_before_handoff"),
                    "steps": self.total_steps,
                }
        return None, {"handoff_reached": False, "terminal_reason": "horizon_before_handoff", "steps": self.total_steps}

    def step(self, action: np.ndarray) -> tuple[np.ndarray | None, float, bool, dict[str, Any]]:
        if self.observation is None or self.handoff_step is None:
            raise TerminalRewardPilotError("candidate step requested before handoff")
        normalized_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if normalized_action.shape != (3,) or not np.isfinite(normalized_action).all() or np.any(np.abs(normalized_action) > 1.0 + 1e-6):
            raise TerminalRewardPilotError("candidate action is not finite normalized 3-D VPP")
        pre_phase = self.phase
        pre_first_pass = self.phase_tracker.first_pass_complete
        before_loss = intent_loss(self.observation, PROFILE, clip=2.0)
        before_intent = physical_intent_vector(self.observation)
        observation, _ignored_reward, terminated, truncated, info = self.env.step(normalized_action)
        self.total_steps += 1
        self.last_info = info
        self._observe(observation)
        after_loss = intent_loss(self.observation, PROFILE, clip=2.0)
        after_intent = physical_intent_vector(self.observation)
        done = bool(terminated or truncated or self.total_steps >= self.env.max_steps)
        reward_cfg = self.config["reward"]["base_geometry_reward"]
        progress = float(np.clip(before_loss - after_loss, -2.0, 2.0))
        alignment = max(0.0, 1.0 - min(after_loss, 2.0) / 2.0)
        saturation_count = sum(
            bool(info.get(key, False))
            for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated")
        )
        terminal_reason = str(info.get("reason") or "")
        receipt = terminal_outcome_reward(
            terminal_reason,
            terminal=done,
            config=_terminal_reward_config(self.config),
        )
        reward = (
            float(reward_cfg["progress_weight"]) * progress
            + float(reward_cfg["alignment_weight"]) * alignment
            - float(reward_cfg["physical_command_saturation_penalty"]) * saturation_count
            + receipt.reward
        )
        if bool(info.get("backend_fallback_occurred", False)):
            raise TerminalRewardPilotError("candidate step reported backend fallback")
        next_observation = None if done else self._policy_observation()
        own_state = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
        return next_observation, float(reward), done, {
            "phase": pre_phase,
            "first_pass_complete": pre_first_pass,
            "valid_target_step": pre_phase == "re_entry" and pre_first_pass,
            "before_intent_loss": float(before_loss),
            "after_intent_loss": float(after_loss),
            "range_m": float(before_intent[2]),
            "range_rate_mps": float(before_intent[3]),
            "specific_energy_height_delta_m": float(after_intent[4]),
            "target_in_attack_zone": bool(info.get("target_in_attack_zone", False)),
            "ego_in_attack_zone": bool(info.get("ego_in_attack_zone", False)),
            "ego_failure": receipt.ego_crash_or_oob,
            "terminal_reason": terminal_reason if done else "nonterminal",
            "terminal_reward": receipt.reward,
            "reward": float(reward),
            "nz_cmd": float(info.get("nz_cmd", np.nan)),
            "actual_nz_g": float(own_state.get("nz_g", info.get("nz_g", np.nan)),
            ),
        }

    def close(self) -> None:
        self.env.close()
