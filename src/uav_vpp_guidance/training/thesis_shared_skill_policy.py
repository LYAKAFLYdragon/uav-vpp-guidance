"""Strict, profile-conditioned 66-D to 3-D VPP policy interface for P4.

The class only defines the architecture and action semantics.  It contains no
checkpoint fallback and this P4 readiness patch does not train or select any
weights.  Keeping the interface in code lets the verifier test the exact
shape that later geometry pretraining and combat finetuning must preserve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from uav_vpp_guidance.agents.replay_buffer import PPORolloutBuffer


OBSERVATION_DIM = 66
VPP_ACTION_DIM = 3
VPP_ACTION_COMPONENTS = (
    "forward_bias_normalized",
    "lateral_bias_normalized",
    "vertical_bias_normalized",
)


class ThesisSharedSkillPolicy(nn.Module):
    """A deterministic MLP action head shared by all four future skills."""

    def __init__(self, observation_dim: int = OBSERVATION_DIM, hidden_dim: int = 128):
        super().__init__()
        if int(observation_dim) != OBSERVATION_DIM:
            raise ValueError(
                f"P4 shared skills require {OBSERVATION_DIM}-D observations, "
                f"got {observation_dim}"
            )
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be positive")
        self.observation_dim = OBSERVATION_DIM
        self.action_dim = VPP_ACTION_DIM
        self.hidden_dim = int(hidden_dim)
        self.feature_network = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(self.hidden_dim, self.action_dim)

    def _check_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(
                "P4 shared-skill observation must have shape "
                f"[batch, {self.observation_dim}], got {tuple(observation.shape)}"
            )
        if not torch.isfinite(observation).all():
            raise ValueError("P4 shared-skill observation contains non-finite values")
        return observation

    def raw_action_mean(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the unsquashed Gaussian mean used by the PPO actor."""

        observation = self._check_observation(observation)
        return self.mean_head(self.feature_network(observation))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return bounded normalized [forward, lateral, vertical] VPP biases."""

        return torch.tanh(self.raw_action_mean(observation))


class ThesisSharedSkillActorCritic(nn.Module):
    """PPO actor-critic with the strict P4 policy as its only action head."""

    def __init__(self, observation_dim: int = OBSERVATION_DIM, hidden_dim: int = 128):
        super().__init__()
        self.actor = ThesisSharedSkillPolicy(observation_dim, hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((VPP_ACTION_DIM,), -0.5))

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        checked = self.actor._check_observation(observation)
        return self.critic(checked).squeeze(-1)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        *,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        checked = self.actor._check_observation(observation)
        mean = self.actor.raw_action_mean(checked)
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = Normal(mean, std)
        value = self.critic(checked).squeeze(-1)
        action_was_none = action is None
        if action_was_none:
            raw_action = mean if deterministic else distribution.rsample()
            action = torch.tanh(raw_action)
            log_prob = distribution.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.log(1.0 - action.square() + 1e-6).sum(dim=-1)
        else:
            action = torch.as_tensor(action, dtype=checked.dtype, device=checked.device)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            if action.shape != (checked.shape[0], VPP_ACTION_DIM):
                raise ValueError(
                    "P4 shared-skill action must have shape "
                    f"[{checked.shape[0]}, {VPP_ACTION_DIM}], got {tuple(action.shape)}"
                )
            bounded = torch.clamp(action, -0.999999, 0.999999)
            raw_action = 0.5 * torch.log((1.0 + bounded) / (1.0 - bounded))
            log_prob = distribution.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.log(1.0 - bounded.square() + 1e-6).sum(dim=-1)
        if not torch.isfinite(log_prob).all() or not torch.isfinite(value).all():
            raise ValueError("P4 actor-critic emitted non-finite log probability or value")
        entropy = distribution.entropy().sum(dim=-1)
        if action_was_none:
            return action, log_prob, entropy, value
        return log_prob, entropy, value

    @torch.no_grad()
    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)


class ThesisSharedSkillPPOAgent:
    """Small strict PPO implementation used only for the new P4 skill library."""

    CHECKPOINT_SCHEMA = "thesis_five_state_shared_skill_ppo_v1"

    def __init__(self, config: Mapping[str, Any], *, device: str = "cpu"):
        ppo = dict(config["training"]["ppo"])
        policy = dict(config["training"]["policy"])
        self.obs_dim = OBSERVATION_DIM
        self.action_dim = VPP_ACTION_DIM
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.network = ThesisSharedSkillActorCritic(
            observation_dim=self.obs_dim,
            hidden_dim=int(policy["hidden_dim"]),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(ppo["learning_rate"]),
            eps=1e-5,
        )
        self.buffer = PPORolloutBuffer(
            capacity=int(ppo["rollout_steps"]),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        self.gamma = float(ppo["gamma"])
        self.gae_lambda = float(ppo["gae_lambda"])
        self.clip_coef = float(ppo["clip_coef"])
        self.value_coef = float(ppo["value_coef"])
        self.entropy_coef = float(ppo["entropy_coef"])
        self.max_grad_norm = float(ppo["max_grad_norm"])
        self.update_epochs = int(ppo["update_epochs"])
        self.minibatch_size = int(ppo["minibatch_size"])
        self.total_timesteps = 0
        self.total_updates = 0

    def select_action(self, observation: np.ndarray, *, deterministic: bool = False):
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (self.obs_dim,):
            raise ValueError(f"P4 policy expected {(self.obs_dim,)}, got {vector.shape}")
        with torch.no_grad():
            action, log_prob, _entropy, value = self.network.get_action_and_value(
                torch.as_tensor(vector, dtype=torch.float32, device=self.device),
                deterministic=deterministic,
            )
        result = action.cpu().numpy().reshape(-1)
        if result.shape != (self.action_dim,) or not np.isfinite(result).all():
            raise ValueError("P4 policy emitted an invalid normalized 3-D VPP action")
        return result, float(log_prob.item()), float(value.item())

    def store_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        normalized_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if vector.shape != (self.obs_dim,) or normalized_action.shape != (self.action_dim,):
            raise ValueError("P4 rollout buffer refuses truncated observations or actions")
        self.buffer.add(vector, normalized_action, log_prob, reward, done, value)
        self.total_timesteps += 1

    def update(self, next_observation: np.ndarray | None) -> dict[str, float]:
        if not len(self.buffer):
            return {}
        next_value = 0.0
        if next_observation is not None:
            vector = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            if vector.shape != (self.obs_dim,):
                raise ValueError("P4 bootstrap value requires an exact 66-D observation")
            with torch.no_grad():
                next_value = float(
                    self.network.get_value(
                        torch.as_tensor(vector, dtype=torch.float32, device=self.device)
                    ).item()
                )
        self.buffer.compute_gae(next_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        count = 0
        for _ in range(self.update_epochs):
            for batch in self.buffer.get_minibatches(self.minibatch_size):
                log_prob, entropy, values = self.network.get_action_and_value(
                    batch["obs"], action=batch["actions"]
                )
                ratio = torch.exp(log_prob - batch["log_probs"])
                surrogate = torch.minimum(
                    ratio * batch["advantages"],
                    torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    * batch["advantages"],
                )
                policy_loss = -surrogate.mean()
                value_loss = 0.5 * (values - batch["returns"]).square().mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy_mean.item())
                totals["approx_kl"] += float(((ratio - 1.0) - torch.log(ratio + 1e-8)).mean().item())
                count += 1
        self.buffer.clear()
        self.total_updates += 1
        return {name: value / max(count, 1) for name, value in totals.items()}

    def save(self, path: Path, metadata: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_schema": self.CHECKPOINT_SCHEMA,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "total_timesteps": self.total_timesteps,
            "total_updates": self.total_updates,
            "metadata": dict(metadata),
        }
        torch.save(payload, path)

    def load_strict(self, path: Path, expected_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"P4 checkpoint is missing: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("checkpoint_schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError("P4 checkpoint schema mismatch; fallback is prohibited")
        if payload.get("obs_dim") != self.obs_dim or payload.get("action_dim") != self.action_dim:
            raise ValueError("P4 checkpoint shape mismatch; fallback is prohibited")
        metadata = payload.get("metadata", {})
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"P4 checkpoint metadata mismatch for {key}; fallback is prohibited")
        self.network.load_state_dict(payload["network_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.total_timesteps = int(payload.get("total_timesteps", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        return payload
