"""ThresholdOptimizationRunner for Stage 6H.2 formal LHS20 scan.

Runs a single gate-parameter configuration against regression, candidate,
and negative suites, then returns a structured verdict.
"""

import copy
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from ..agents.ppo_agent import PPOAgent
from ..envs.scenario_registry import ScenarioRegistry, initialize_canonical_scenarios
from ..envs.tracking_env import CloseRangeTrackingEnv
from ..utils.config import merge_config
from .evaluate_prediction_comparison import evaluate_single_episode


class ThresholdOptimizationRunner:
    """Evaluate one gate-threshold configuration against canonical suites."""

    def __init__(
        self,
        base_config: Dict,
        checkpoint_path: str,
        device: str = "cpu",
        seeds: Tuple[int, ...] = tuple(range(10)),
    ):
        """
        Args:
            base_config: Full experiment config (loaded from YAML).
            checkpoint_path: Path to PPO checkpoint .pt file.
            device: 'cpu' or 'cuda'.
            seeds: Seeds to evaluate per scenario.
        """
        self.base_config = copy.deepcopy(base_config)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.seeds = seeds

        # Ensure canonical scenarios are registered
        initialize_canonical_scenarios()

        # Pre-build env + agent once; we will re-create env per config
        # because gate parameters are env-level.
        self._obs_dim = self._infer_obs_dim()

    def _infer_obs_dim(self) -> int:
        """Spin up a temporary env to get observation dimension."""
        tmp_config = copy.deepcopy(self.base_config)
        tmp_config["backend"] = "simple"
        tmp_config["env"]["backend"] = "simple"
        tmp_config["env"]["use_jsbsim"] = False
        tmp_config["guidance"]["mode_switch"] = {"enabled": False}
        env = CloseRangeTrackingEnv(tmp_config)
        obs = env.reset(seed=0)
        dim = int(obs["observation_vector"].shape[0])
        env.close()
        return dim

    def _make_env(self, gate_config: Dict) -> CloseRangeTrackingEnv:
        """Create an env with the given gate config."""
        cfg = copy.deepcopy(self.base_config)
        cfg["backend"] = "simple"
        cfg["env"]["backend"] = "simple"
        cfg["env"]["use_jsbsim"] = False
        cfg["guidance"]["mode_switch"] = copy.deepcopy(gate_config)
        return CloseRangeTrackingEnv(cfg)

    def _make_agent(self, config: Dict) -> PPOAgent:
        """Load a fresh agent."""
        agent = PPOAgent(
            obs_dim=self._obs_dim,
            action_dim=3,
            config=config,
            device=self.device,
        )
        agent.load(self.checkpoint_path)
        return agent

    def evaluate_suite(
        self,
        scenarios: List[Dict],
        gate_config: Dict,
    ) -> List[Dict]:
        """Evaluate a list of scenarios under a gate config.

        Returns a flat list of episode result dicts.
        """
        env = self._make_env(gate_config)
        agent = self._make_agent(env.config)
        episodes = []
        try:
            for scen in scenarios:
                for seed in self.seeds:
                    result, _ = evaluate_single_episode(
                        env=env,
                        agent=agent,
                        config=env.config,
                        scenario=scen,
                        seed=seed,
                        save_trajectory=False,
                        method_name="no_prediction",
                    )
                    episodes.append(result)
        finally:
            env.close()
        return episodes

    def evaluate_config(self, gate_config: Dict) -> Dict:
        """Run all canonical suites and return a verdict dict.

        Hard constraints (from Stage 6H.2 spec):
          1. regression 40/40 success
          2. candidate >= 38/40 success
          3. negative_tail_chase 10/10 mode_switch + success
          4. negative_fleeing 0/10 success
          5. negative_offset_attack 0/10 success
        """
        regression_scens = ScenarioRegistry.get_regression_suite()
        candidate_scens = ScenarioRegistry.get_candidate_suite()
        negative_scens = ScenarioRegistry.get_negative_suite()

        regression_eps = self.evaluate_suite(regression_scens, gate_config)
        candidate_eps = self.evaluate_suite(candidate_scens, gate_config)
        negative_eps = self.evaluate_suite(negative_scens, gate_config)

        def _count_success(eps: List[Dict]) -> int:
            return sum(1 for e in eps if e.get("is_success", False))

        def _count_mode_switch(eps: List[Dict]) -> int:
            return sum(1 for e in eps if e.get("mode_switch_effective", False))

        regression_success = _count_success(regression_eps)
        candidate_success = _count_success(candidate_eps)

        # Negative breakdown by scenario name
        neg_by_scen: Dict[str, List[Dict]] = {}
        for e in negative_eps:
            name = e.get("scenario", "unknown")
            neg_by_scen.setdefault(name, []).append(e)

        tail_chase_eps = neg_by_scen.get("negative_tail_chase", [])
        fleeing_eps = neg_by_scen.get("negative_fleeing", [])
        offset_eps = neg_by_scen.get("negative_offset_attack", [])

        tail_chase_success = _count_success(tail_chase_eps)
        tail_chase_switch = _count_mode_switch(tail_chase_eps)
        fleeing_success = _count_success(fleeing_eps)
        offset_success = _count_success(offset_eps)

        # Hard constraints scaled by number of seeds
        n_seeds = len(self.seeds)
        regression_req = 4 * n_seeds
        candidate_req_total = 4 * n_seeds
        candidate_req_min = int(np.ceil(0.95 * candidate_req_total))
        tail_chase_req = 1 * n_seeds
        neg_req = 1 * n_seeds

        violations = []
        if regression_success < regression_req:
            violations.append(f"regression_{regression_success}/{regression_req}")
        if candidate_success < candidate_req_min:
            violations.append(f"candidate_{candidate_success}/{candidate_req_total}")
        if tail_chase_success < tail_chase_req:
            violations.append(f"tail_chase_success_{tail_chase_success}/{tail_chase_req}")
        if tail_chase_switch < tail_chase_req:
            violations.append(f"tail_chase_switch_{tail_chase_switch}/{tail_chase_req}")
        if fleeing_success > 0:
            violations.append(f"fleeing_should_fail_got_{fleeing_success}/{neg_req}")
        if offset_success > 0:
            violations.append(f"offset_should_fail_got_{offset_success}/{neg_req}")

        verdict = "PASS" if not violations else "FAIL"

        return {
            "aspect_threshold_deg": gate_config.get("aspect_threshold_deg"),
            "range_threshold_m": gate_config.get("range_threshold_m"),
            "closing_speed_threshold_mps": gate_config.get("closing_speed_threshold_mps"),
            "regression_success": regression_success,
            "regression_total": len(regression_eps),
            "candidate_success": candidate_success,
            "candidate_total": len(candidate_eps),
            "tail_chase_success": tail_chase_success,
            "tail_chase_switch": tail_chase_switch,
            "tail_chase_total": len(tail_chase_eps),
            "fleeing_success": fleeing_success,
            "fleeing_total": len(fleeing_eps),
            "offset_success": offset_success,
            "offset_total": len(offset_eps),
            "verdict": verdict,
            "violations": "; ".join(violations) if violations else "",
        }
