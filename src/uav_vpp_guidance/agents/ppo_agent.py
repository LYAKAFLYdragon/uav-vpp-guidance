"""
PPO Agent implementation.

Implements Proximal Policy Optimization with clipped surrogate objective,
value function loss, and entropy bonus.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .policy_network import MLPActorCritic
from .replay_buffer import PPORolloutBuffer


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous action spaces.

    Supports:
    - select_action: stochastic or deterministic action sampling
    - evaluate_actions: compute log_prob, entropy, value for given actions
    - update: perform PPO update on rollout buffer data
    - save/load: checkpoint management
    """

    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        """
        Args:
            obs_dim (int): Observation dimension.
            action_dim (int): Action dimension.
            config (dict): PPO and network configuration.
            device (str): torch device.
        """
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print(f"WARNING: CUDA requested but not available. Falling back to CPU.")
            requested = torch.device("cpu")
        self.device = requested

        # Extract hyperparameters
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
        self.normalize_advantage = bool(ppo_cfg.get("normalize_advantage", True))
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))

        policy_cfg = config.get("policy", {})
        hidden_sizes = policy_cfg.get("hidden_sizes", [128, 128])
        activation = policy_cfg.get("activation", "tanh")
        action_low = policy_cfg.get("action_low", [-1.0] * action_dim)
        action_high = policy_cfg.get("action_high", [1.0] * action_dim)

        # Build network
        self.network = MLPActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)

        # Rollout buffer
        self.buffer = PPORolloutBuffer(
            capacity=self.rollout_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        # Training stats
        self.total_updates = 0
        self.total_timesteps = 0

    def select_action(self, obs, deterministic=False, store=True):
        """
        Select action from observation.

        Args:
            obs (np.ndarray): Observation vector, shape (obs_dim,).
            deterministic (bool): If True, return mean action.
            store (bool): If True, store transition in buffer (training mode).

        Returns:
            tuple: (action, log_prob, value)
                action (np.ndarray): Selected action.
                log_prob (float): Log probability of the action.
                value (float): Value estimate.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
        if obs_t.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected obs shape ({self.obs_dim},), got {obs_t.shape}"
            )

        with torch.no_grad():
            action_t, log_prob_t, entropy_t, value_t = self.network.get_action_and_value(
                obs_t, deterministic=deterministic
            )

        action = action_t.cpu().numpy().flatten()
        log_prob = float(log_prob_t.cpu().numpy())
        value = float(value_t.cpu().numpy())

        # NaN/inf check
        if not np.isfinite(action).all():
            raise ValueError(f"Non-finite action detected: {action}")
        if not np.isfinite(log_prob):
            raise ValueError(f"Non-finite log_prob detected: {log_prob}")
        if not np.isfinite(value):
            raise ValueError(f"Non-finite value detected: {value}")

        return action, log_prob, value

    def store_transition(self, obs, action, log_prob, reward, done, value):
        """
        Store a transition in the rollout buffer.

        Args:
            obs (np.ndarray): Observation.
            action (np.ndarray): Action taken.
            log_prob (float): Log probability of the action.
            reward (float): Reward received.
            done (bool): Whether episode terminated.
            value (float): Value estimate.
        """
        self.buffer.add(obs, action, log_prob, reward, done, value)
        self.total_timesteps += 1

    def update(self, next_obs=None):
        """
        Perform PPO update on collected rollout data.

        Args:
            next_obs (np.ndarray, optional): Observation after the last stored transition.
                Used to compute the bootstrap value for GAE.

        Returns:
            dict: Training statistics.
        """
        if len(self.buffer) == 0:
            return {}

        # Compute bootstrap value
        next_value = 0.0
        if next_obs is not None:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=self.device
                ).flatten()
                next_value = float(self.network.get_value(obs_t).cpu().numpy())

        # Compute GAE
        self.buffer.compute_gae(
            next_value=next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Collect update stats
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_minibatches = 0

        # PPO update epochs
        for epoch in range(self.update_epochs):
            for batch in self.buffer.get_minibatches(batch_size=self.minibatch_size):
                obs_b = batch["obs"]
                actions_b = batch["actions"]
                old_log_probs_b = batch["log_probs"]
                advantages_b = batch["advantages"]
                returns_b = batch["returns"]
                old_values_b = batch["values"]

                # Evaluate actions with current policy
                new_log_probs_b, entropy_b, new_values_b = self.network.get_action_and_value(
                    obs_b, action=actions_b
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs_b - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    * advantages_b
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_pred_clipped = old_values_b + torch.clamp(
                    new_values_b - old_values_b, -self.clip_coef, self.clip_coef
                )
                value_loss1 = (new_values_b - returns_b).pow(2)
                value_loss2 = (value_pred_clipped - returns_b).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy bonus
                entropy = entropy_b.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Stats
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

        if num_minibatches == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / num_minibatches,
            "value_loss": total_value_loss / num_minibatches,
            "entropy": total_entropy / num_minibatches,
            "approx_kl": total_approx_kl / num_minibatches,
            "clip_fraction": total_clip_fraction / num_minibatches,
            "explained_variance": explained_var.item(),
            "learning_rate": self.lr,
        }

    def save(self, path):
        """
        Save model checkpoint.

        Args:
            path (str): Save path.
        """
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
        """
        Load model checkpoint.

        Args:
            path (str): Load path.

        Returns:
            dict: Loaded checkpoint metadata.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        return checkpoint

    def get_deterministic_action(self, obs):
        """
        Get deterministic action for evaluation.

        Args:
            obs (np.ndarray): Observation vector.

        Returns:
            np.ndarray: Deterministic action.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
        action_t = self.network.get_deterministic_action(obs_t)
        return action_t.cpu().numpy().flatten()
