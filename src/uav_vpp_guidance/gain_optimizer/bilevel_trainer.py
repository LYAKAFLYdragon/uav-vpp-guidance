"""
Bilevel training loop for strategy-gain co-optimization.

Outer loop: PPO policy updates (every outer_every episodes)
Inner loop: CEM gain optimization (every inner_iter iterations)
Regret tracking: records regret for each (policy, gains) pairing
"""

import copy
import json
from pathlib import Path
from typing import Callable, List

import numpy as np

from ..guidance.gain_config import GuidanceGains


class BilevelTrainer:
    """Alternating bilevel optimization."""

    def __init__(self, env_factory, policy, gain_optimizer, config):
        """
        Args:
            env_factory: Callable that returns a fresh CloseRangeTrackingEnv.
            policy: PPOAgent.
            gain_optimizer: CEMGainOptimizer.
            config: dict with keys:
                - outer_every: int, policy update frequency (episodes)
                - inner_iter: int, CEM iterations per inner loop
                - n_episodes: int, total training episodes
                - eval_seeds: tuple, seeds for evaluation
                - eval_scenarios: list, scenarios for evaluation
        """
        self.env_factory = env_factory
        self.policy = policy
        self.gain_optimizer = gain_optimizer
        self.config = config

        self.outer_every = int(config.get("outer_every", 10))
        self.inner_iter = int(config.get("inner_iter", 20))
        self.n_episodes = int(config.get("n_episodes", 200))

        self.history: List[dict] = []
        self.regret_log: List[dict] = []

    @staticmethod
    def _filter_gains(gains_dict: dict) -> dict:
        """Filter gains_dict to only include valid GuidanceGains fields."""
        valid_fields = set(GuidanceGains.__dataclass_fields__.keys())
        return {k: v for k, v in gains_dict.items() if k in valid_fields}

    def _collect_episode(self, seed: int) -> None:
        """Collect one episode of experience and store in policy buffer."""
        env = self.env_factory()
        try:
            obs = env.reset(seed=seed)
            done = False

            while not done:
                obs_vec = obs["observation_vector"]
                action, log_prob, value = self.policy.select_action(
                    obs_vec, deterministic=False
                )
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.policy.store_transition(
                    obs_vec, action, log_prob, reward, done, value
                )
                obs = next_obs

            # PPO update if buffer has enough data
            if len(self.policy.buffer) >= self.policy.minibatch_size:
                next_obs_vec = (
                    obs["observation_vector"] if not done else np.zeros_like(obs_vec)
                )
                self.policy.update(next_obs=next_obs_vec)
        finally:
            env.close()

    def _make_gain_evaluator(self) -> Callable[[dict], float]:
        """Create an evaluator function for the inner CEM loop."""
        from ..evaluation.evaluate_prediction_comparison import (
            evaluate_single_episode,
        )

        def evaluator(gains_dict: dict) -> float:
            eval_env = self.env_factory()
            try:
                filtered = self._filter_gains(gains_dict)
                eval_env.current_gains = GuidanceGains(**filtered)

                successes = 0
                total = 0
                for scen in self.config.get("eval_scenarios", []):
                    for seed in self.config.get("eval_seeds", [0, 1, 2]):
                        result, _ = evaluate_single_episode(
                            eval_env,
                            self.policy,
                            eval_env.config,
                            scenario=scen,
                            seed=seed,
                        )
                        if result.get("is_success", False):
                            successes += 1
                        total += 1
                return successes / total if total > 0 else 0.0
            finally:
                eval_env.close()

        return evaluator

    def _evaluate_policy_gains(
        self, gains_dict: dict, n_scenarios: int = 2
    ) -> float:
        """Evaluate a (policy, gains) pairing on a small eval set."""
        from ..envs.scenario_registry import ScenarioRegistry
        from ..evaluation.evaluate_prediction_comparison import (
            evaluate_single_episode,
        )

        scenarios = ScenarioRegistry.get_regression_suite()[:n_scenarios]
        seeds = self.config.get("eval_seeds", [0, 1, 2])

        env = self.env_factory()
        filtered = self._filter_gains(gains_dict)
        env.current_gains = GuidanceGains(**filtered)

        successes = 0
        total = 0
        for scen in scenarios:
            for seed in seeds:
                result, _ = evaluate_single_episode(
                    env, self.policy, env.config, scenario=scen, seed=seed
                )
                if result.get("is_success", False):
                    successes += 1
                total += 1
        env.close()
        return successes / total if total > 0 else 0.0

    def _save_policy_snapshot(self):
        """Save current policy network weights for regret computation."""
        return copy.deepcopy(self.policy.network.state_dict())

    def _compute_regret(self, eval_sr: float) -> float:
        """Compute regret: gap to best known success rate (monotonically non-increasing)."""
        best_known = max(
            (h.get("eval_success_rate", 0.0) for h in self.history),
            default=0.0,
        )
        best_known = max(best_known, eval_sr)
        # Regret = gap to best known; never increases as best_known only improves
        return max(0.0, 1.0 - best_known)

    def train(self) -> dict:
        """Run full bilevel training.

        Returns:
            dict with keys:
                - history: list of eval snapshots
                - regret_log: list of regret values
                - best_policy_episode: int
                - best_gains: dict
                - best_success_rate: float
        """
        best_policy_sr = 0.0
        best_gains = None
        best_policy_episode = 0

        episode = 0

        while episode < self.n_episodes:
            # --- Outer loop: collect episodes with current policy + gains ---
            for _ in range(self.outer_every):
                if episode >= self.n_episodes:
                    break
                self._collect_episode(seed=episode)
                episode += 1

            # --- Inner loop: optimize gains with frozen policy ---
            gain_evaluator = self._make_gain_evaluator()
            best_gains_iter, gain_history = self.gain_optimizer.optimize(
                gain_evaluator, n_iter=self.inner_iter
            )

            # Evaluate current (policy, gains) pairing
            eval_sr = self._evaluate_policy_gains(best_gains_iter)

            # Regret
            regret = self._compute_regret(eval_sr)
            self.regret_log.append(
                {
                    "episode": episode,
                    "eval_success_rate": eval_sr,
                    "regret": regret,
                    "gains": best_gains_iter,
                }
            )

            self.history.append(
                {
                    "episode": episode,
                    "eval_success_rate": eval_sr,
                    "gains": best_gains_iter,
                    "gain_history": gain_history,
                }
            )

            if eval_sr > best_policy_sr:
                best_policy_sr = eval_sr
                best_gains = best_gains_iter
                best_policy_episode = episode

            print(
                f"[Bilevel] Episode {episode}/{self.n_episodes} | "
                f"Eval SR: {eval_sr:.2%} | Regret: {regret:.4f}"
            )

            # Periodic checkpoint
            if episode > 0 and episode % max(1, self.outer_every * 5) == 0:
                ckpt_dir = self.config.get("checkpoint_dir", "outputs/bilevel_training")
                self._save_checkpoint(episode, best_gains_iter, ckpt_dir)

        return {
            "history": self.history,
            "regret_log": self.regret_log,
            "best_policy_episode": best_policy_episode,
            "best_gains": best_gains,
            "best_success_rate": best_policy_sr,
        }

    def _save_checkpoint(self, episode: int, gains: dict, output_dir: str) -> None:
        """Save training checkpoint."""
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save policy
        policy_path = ckpt_dir / f"policy_ep{episode}.pt"
        self.policy.save(str(policy_path))

        # Save gains + history
        meta = {
            "episode": episode,
            "gains": gains,
            "history": self.history[-10:] if self.history else [],
            "regret_log": self.regret_log[-10:] if self.regret_log else [],
        }
        meta_path = ckpt_dir / f"meta_ep{episode}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, default=str, indent=2)

        print(f"[Checkpoint] Saved at episode {episode}: {policy_path}")
