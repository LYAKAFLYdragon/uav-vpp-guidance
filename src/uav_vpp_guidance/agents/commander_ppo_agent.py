"""Discrete PPO agent for the hierarchical commander."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .policy_network import CategoricalMLPActorCritic
from .replay_buffer import PPORolloutBuffer


class CommanderPPOAgent:
    """PPO agent for discrete high-level mode selection."""

    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            requested = torch.device("cpu")
        self.device = requested

        ppo_cfg = config.get("ppo", config)
        self.lr = float(ppo_cfg.get("learning_rate", 3.0e-4))
        self.gamma = float(ppo_cfg.get("gamma", 0.99))
        self.gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
        self.clip_coef = float(ppo_cfg.get("clip_coef", 0.2))
        self.value_coef = float(ppo_cfg.get("value_coef", 0.5))
        self.entropy_coef = float(ppo_cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
        self.update_epochs = int(ppo_cfg.get("update_epochs", 10))
        self.minibatch_size = int(ppo_cfg.get("minibatch_size", 256))
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))

        policy_cfg = config.get("policy", {})
        hidden_sizes = policy_cfg.get("hidden_sizes", [128, 128])
        activation = policy_cfg.get("activation", "tanh")

        self.network = CategoricalMLPActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)
        self.buffer = PPORolloutBuffer(
            capacity=self.rollout_steps,
            obs_dim=self.obs_dim,
            action_dim=1,
            device=self.device,
        )
        commander_cfg = config.get("commander", {})
        oracle_imitation_cfg = commander_cfg.get("oracle_imitation", {})
        self.oracle_imitation_enabled = bool(oracle_imitation_cfg.get("enabled", False))
        self.oracle_imitation_loss_weight = float(
            oracle_imitation_cfg.get("loss_weight", 0.0)
        )
        self.oracle_imitation_warmstart_updates = int(
            oracle_imitation_cfg.get("warmstart_updates", 0)
        )
        self.oracle_imitation_decay_schedule = str(
            oracle_imitation_cfg.get("decay_schedule", "linear_to_zero")
        )
        self._oracle_actions = np.full(self.rollout_steps, -1, dtype=np.int64)
        self._oracle_action_mask = np.zeros(self.rollout_steps, dtype=np.float32)
        self._oracle_imitation_weights = np.zeros(self.rollout_steps, dtype=np.float32)

        self.total_updates = 0
        self.total_timesteps = 0

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
        if obs_t.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected obs shape ({self.obs_dim},), got {obs_t.shape}"
            )

        with torch.no_grad():
            action_t, log_prob_t, entropy_t, value_t = self.network.get_action_and_value(
                obs_t, deterministic=deterministic
            )
        action = int(action_t.cpu().numpy().reshape(-1)[0])
        log_prob = float(log_prob_t.cpu().numpy().reshape(-1)[0])
        value = float(value_t.cpu().numpy().reshape(-1)[0])

        if not np.isfinite(log_prob):
            raise ValueError(f"Non-finite log_prob detected: {log_prob}")
        if not np.isfinite(value):
            raise ValueError(f"Non-finite value detected: {value}")
        return action, log_prob, value

    @staticmethod
    def compute_oracle_imitation_weight(
        *,
        enabled: bool,
        loss_weight: float,
        warmstart_updates: int,
        decay_schedule: str,
        total_updates: int,
    ) -> float:
        if not enabled or loss_weight <= 0.0:
            return 0.0

        base_weight = float(loss_weight)
        warmstart_updates = int(warmstart_updates)
        schedule = str(decay_schedule)

        if warmstart_updates <= 0:
            return base_weight
        if schedule == "constant":
            return base_weight
        if schedule in {"hard_stop", "step"}:
            return base_weight if int(total_updates) < warmstart_updates else 0.0
        if schedule == "linear_to_zero":
            progress = min(max(int(total_updates), 0), warmstart_updates)
            scale = max(0.0, 1.0 - (progress / float(warmstart_updates)))
            return base_weight * scale
        raise ValueError(f"Unsupported oracle imitation decay_schedule: {schedule}")

    def _current_oracle_imitation_weight(self):
        return self.compute_oracle_imitation_weight(
            enabled=self.oracle_imitation_enabled,
            loss_weight=self.oracle_imitation_loss_weight,
            warmstart_updates=self.oracle_imitation_warmstart_updates,
            decay_schedule=self.oracle_imitation_decay_schedule,
            total_updates=self.total_updates,
        )

    def store_transition(
        self,
        obs,
        action,
        log_prob,
        reward,
        done,
        value,
        oracle_action=None,
        oracle_imitation_weight=None,
    ):
        idx = len(self.buffer)
        self.buffer.add(
            obs=obs,
            action=np.asarray([action], dtype=np.float32),
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )
        if oracle_action is None:
            self._oracle_actions[idx] = -1
            self._oracle_action_mask[idx] = 0.0
            self._oracle_imitation_weights[idx] = 0.0
        else:
            self._oracle_actions[idx] = int(oracle_action)
            self._oracle_action_mask[idx] = 1.0
            if oracle_imitation_weight is None:
                oracle_imitation_weight = self._current_oracle_imitation_weight()
            self._oracle_imitation_weights[idx] = float(oracle_imitation_weight)
        self.total_timesteps += 1

    def update(self, next_obs=None):
        if len(self.buffer) == 0:
            return {}

        next_value = 0.0
        if next_obs is not None:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=self.device
                ).flatten()
                next_value = float(self.network.get_value(obs_t).cpu().numpy())

        self.buffer.compute_gae(
            next_value=next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        data = self.buffer.get_data()
        if not data:
            self.buffer.clear()
            self._oracle_action_mask.fill(0.0)
            self._oracle_actions.fill(-1)
            self._oracle_imitation_weights.fill(0.0)
            return {}

        n = data["obs"].shape[0]
        oracle_actions = torch.as_tensor(
            self._oracle_actions[:n], dtype=torch.long, device=self.device
        )
        oracle_action_mask = torch.as_tensor(
            self._oracle_action_mask[:n], dtype=torch.float32, device=self.device
        )
        oracle_imitation_weights = torch.as_tensor(
            self._oracle_imitation_weights[:n], dtype=torch.float32, device=self.device
        )
        positive_weight_mask = (oracle_action_mask > 0.5) & (oracle_imitation_weights > 0.0)
        mean_imitation_weight = float(
            oracle_imitation_weights[positive_weight_mask].mean().item()
        ) if torch.any(positive_weight_mask) else 0.0
        supervised_fraction = float(positive_weight_mask.float().mean().item())

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_imitation_loss = 0.0
        imitation_batches = 0
        num_minibatches = 0
        explained_var = torch.tensor(0.0, device=self.device)

        for _epoch in range(self.update_epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                batch_idx_np = indices[start:end]
                batch_idx = torch.as_tensor(
                    batch_idx_np, dtype=torch.long, device=self.device
                )

                obs_b = data["obs"][batch_idx]
                actions_b = data["actions"][batch_idx].long().squeeze(-1)
                old_log_probs_b = data["log_probs"][batch_idx]
                advantages_b = data["advantages"][batch_idx]
                returns_b = data["returns"][batch_idx]
                old_values_b = data["values"][batch_idx]

                logits_b, new_values_b = self.network.forward(obs_b)
                dist_b = torch.distributions.Categorical(logits=logits_b)
                new_log_probs_b = dist_b.log_prob(actions_b)
                entropy_b = dist_b.entropy()

                ratio = torch.exp(new_log_probs_b - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    * advantages_b
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_values_b + torch.clamp(
                    new_values_b - old_values_b, -self.clip_coef, self.clip_coef
                )
                value_loss1 = (new_values_b - returns_b).pow(2)
                value_loss2 = (value_pred_clipped - returns_b).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                entropy = entropy_b.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                imitation_loss_value = 0.0
                oracle_mask_b = oracle_action_mask[batch_idx] > 0.5
                if torch.any(oracle_mask_b):
                    oracle_targets_b = oracle_actions[batch_idx][oracle_mask_b]
                    oracle_weights_b = oracle_imitation_weights[batch_idx][oracle_mask_b]
                    active_oracle_mask = oracle_weights_b > 0.0
                    if torch.any(active_oracle_mask):
                        oracle_targets_b = oracle_targets_b[active_oracle_mask]
                        oracle_weights_b = oracle_weights_b[active_oracle_mask]
                        ce_losses = F.cross_entropy(
                            logits_b[oracle_mask_b][active_oracle_mask],
                            oracle_targets_b,
                            reduction="none",
                        )
                        imitation_loss = torch.mean(oracle_weights_b * ce_losses)
                        loss = loss + imitation_loss
                        imitation_loss_value = float(imitation_loss.item())
                        total_imitation_loss += imitation_loss_value
                        imitation_batches += 1

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean().item()
                    clip_fraction = (
                        (abs(ratio - 1.0) > self.clip_coef).float().mean().item()
                    )
                    explained_var = 1.0 - torch.var(returns_b - new_values_b) / (
                        torch.var(returns_b) + 1e-8
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_minibatches += 1

        self.total_updates += 1
        self.buffer.clear()
        self._oracle_action_mask.fill(0.0)
        self._oracle_actions.fill(-1)
        self._oracle_imitation_weights.fill(0.0)

        if num_minibatches == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / num_minibatches,
            "value_loss": total_value_loss / num_minibatches,
            "entropy": total_entropy / num_minibatches,
            "approx_kl": total_approx_kl / num_minibatches,
            "clip_fraction": total_clip_fraction / num_minibatches,
            "explained_variance": float(explained_var.item()),
            "learning_rate": self.lr,
            "oracle_imitation_enabled": self.oracle_imitation_enabled,
            "oracle_imitation_weight": mean_imitation_weight,
            "oracle_imitation_loss": (
                total_imitation_loss / imitation_batches if imitation_batches > 0 else 0.0
            ),
            "oracle_imitation_supervised_fraction": supervised_fraction,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "total_updates": self.total_updates,
            "total_timesteps": self.total_timesteps,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"], strict=False)
        optimizer_loaded = False
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_loaded = True
        except ValueError as exc:
            if "parameter group" in str(exc):
                print(
                    "WARNING: Commander optimizer state load skipped due to "
                    f"architecture mismatch: {exc}"
                )
            else:
                raise

        self.total_updates = checkpoint.get("total_updates", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        checkpoint["optimizer_loaded"] = optimizer_loaded
        return checkpoint

    def get_deterministic_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
        action_t = self.network.get_deterministic_action(obs_t)
        return int(action_t.cpu().numpy().reshape(-1)[0])
