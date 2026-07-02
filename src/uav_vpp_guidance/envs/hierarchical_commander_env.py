"""Macro-step wrapper environment for the hierarchical commander MVP."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np


class HierarchicalCommanderEnv:
    """Wrap one or more combat lanes and expose discrete commander actions."""

    def __init__(
        self,
        lane_envs: List[Dict[str, Any]],
        mode_registry: Dict[int, Dict[str, Any]],
        macro_action_repeat_steps: int = 12,
        switch_penalty: float = 0.0,
        mode_alignment_shaping: Optional[Dict[str, Any]] = None,
        min_mode_hold_steps: int = 0,
        rng: Optional[np.random.Generator] = None,
        close_lane_envs_on_close: bool = True,
    ):
        if not lane_envs:
            raise ValueError("lane_envs must contain at least one wrapped lane")
        if not mode_registry:
            raise ValueError("mode_registry must contain at least one commander mode")

        self.lane_envs = lane_envs
        self.mode_registry = mode_registry
        self.macro_action_repeat_steps = max(1, int(macro_action_repeat_steps))
        self.switch_penalty = float(switch_penalty)
        self.mode_alignment_shaping = mode_alignment_shaping or {}
        self.min_mode_hold_steps = max(0, int(min_mode_hold_steps))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.close_lane_envs_on_close = bool(close_lane_envs_on_close)

        self._current_lane: Optional[Dict[str, Any]] = None
        self._current_env = None
        self._current_obs: Optional[Dict[str, Any]] = None
        self._active_mode_id: Optional[int] = None
        self._switch_count = 0
        self._steps_since_switch = 0
        self._macro_steps = 0
        self._current_episode_seed: Optional[int] = None
        self._hold_steps_remaining = 0

    @property
    def max_steps(self) -> int:
        if self._current_env is not None:
            return int(self._current_env.max_steps)
        return int(max(lane["env"].max_steps for lane in self.lane_envs))

    @property
    def current_task_name(self) -> Optional[str]:
        if self._current_lane is None:
            return None
        return str(self._current_lane.get("task_name"))

    def _select_lane(self, lane_index: Optional[int] = None) -> Dict[str, Any]:
        if lane_index is not None:
            return self.lane_envs[int(lane_index)]

        weights = np.asarray([lane["weight"] for lane in self.lane_envs], dtype=np.float64)
        weights = weights / np.sum(weights)
        idx = int(self.rng.choice(len(self.lane_envs), p=weights))
        return self.lane_envs[idx]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        lane_index: Optional[int] = None,
        scenario: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self._current_episode_seed = int(seed)

        lane = self._select_lane(lane_index)
        env = lane["env"]
        scenario_pool = list(lane.get("scenario_pool", []))
        scenario_payload = scenario
        if scenario_payload is None:
            if not scenario_pool:
                raise ValueError("lane scenario_pool must contain at least one scenario")
            scenario_entry = scenario_pool[int(self.rng.integers(0, len(scenario_pool)))]
            scenario_payload = copy.deepcopy(scenario_entry["scenario"])

        obs = env.reset(scenario=scenario_payload, seed=self._current_episode_seed)
        self._current_lane = lane
        self._current_env = env
        self._current_obs = obs
        self._active_mode_id = None
        self._switch_count = 0
        self._steps_since_switch = 0
        self._macro_steps = 0
        return obs

    def _build_commander_info(self, mode_id: int, switched: bool) -> Dict[str, Any]:
        mode = self.mode_registry[int(mode_id)]
        shaping_reward, preferred_mode_id, matched = self._mode_alignment_shaping_reward(
            int(mode_id)
        )
        return {
            "commander_mode_id": int(mode_id),
            "commander_mode_name": mode["name"],
            "commander_selected_specialist": mode["specialist_key"],
            "commander_switch_count": int(self._switch_count),
            "commander_steps_since_switch": int(self._steps_since_switch),
            "commander_macro_action_repeat_steps": int(self.macro_action_repeat_steps),
            "commander_task_oracle_gate": self.current_task_name,
            "commander_mode_switched": bool(switched),
            "commander_macro_step_index": int(self._macro_steps),
            "commander_mode_alignment_shaping_reward": float(shaping_reward),
            "commander_mode_alignment_preferred_mode_id": preferred_mode_id,
            "commander_mode_alignment_matched": matched,
        }

    def _mode_alignment_shaping_reward(
        self, mode_id: int
    ) -> tuple[float, Optional[int], Optional[bool]]:
        task_name = self.current_task_name
        if task_name is None:
            return 0.0, None, None
        task_cfg = self.mode_alignment_shaping.get(str(task_name))
        if not isinstance(task_cfg, dict):
            return 0.0, None, None

        preferred_mode_id = task_cfg.get("preferred_mode_id")
        if preferred_mode_id is None:
            return 0.0, None, None
        preferred_mode_id = int(preferred_mode_id)
        preferred_reward = float(task_cfg.get("preferred_mode_reward", 0.0))
        non_preferred_penalty = float(task_cfg.get("non_preferred_mode_penalty", 0.0))
        matched = int(mode_id) == preferred_mode_id
        shaping_reward = preferred_reward if matched else -non_preferred_penalty
        return float(shaping_reward), preferred_mode_id, matched

    def step(self, mode_id: int):
        if self._current_env is None or self._current_obs is None:
            raise RuntimeError("reset() must be called before step()")
        if int(mode_id) not in self.mode_registry:
            raise KeyError(f"Unknown commander mode id: {mode_id}")

        # Enforce minimum mode hold: if hold steps remain, ignore switch requests
        requested_mode_id = int(mode_id)
        if self._active_mode_id is not None and requested_mode_id != self._active_mode_id:
            if self._hold_steps_remaining > 0:
                requested_mode_id = self._active_mode_id

        switched = self._active_mode_id is not None and requested_mode_id != self._active_mode_id
        if switched:
            self._switch_count += 1
            self._steps_since_switch = 0
            self._hold_steps_remaining = self.min_mode_hold_steps
        self._active_mode_id = requested_mode_id

        mode = self.mode_registry[requested_mode_id]
        specialist = mode["policy"]
        mode_alignment_reward, _, _ = self._mode_alignment_shaping_reward(requested_mode_id)
        total_reward = (-self.switch_penalty if switched else 0.0) + mode_alignment_reward
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        executed_steps = 0
        obs = self._current_obs

        for _ in range(self.macro_action_repeat_steps):
            action = specialist.get_deterministic_action(obs["observation_vector"])
            obs, reward, terminated, truncated, info = self._current_env.step(action)
            total_reward += float(reward)
            executed_steps += 1
            if terminated or truncated:
                break

        self._macro_steps += 1
        self._steps_since_switch += executed_steps
        self._hold_steps_remaining = max(0, self._hold_steps_remaining - 1)
        self._current_obs = obs
        info = {**info, **self._build_commander_info(requested_mode_id, switched)}
        return obs, total_reward, terminated, truncated, info

    def close(self):
        if not self.close_lane_envs_on_close:
            return
        for lane in self.lane_envs:
            lane["env"].close()
