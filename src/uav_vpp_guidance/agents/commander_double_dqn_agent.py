"""Discrete Double DQN agent for the hierarchical commander."""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .policy_network import CategoricalMLPActorCritic


class SimpleReplayBuffer:
    """Experience replay buffer for off-policy DQN."""

    def __init__(self, capacity, obs_dim, device="cpu"):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.device = device

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.oracle_actions = np.full(self.capacity, -1, dtype=np.int64)
        self.oracle_weights = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, oracle_action=-1, oracle_weight=0.0):
        idx = self.ptr % self.capacity
        self.obs[idx] = np.asarray(obs, dtype=np.float32).flatten()[:self.obs_dim]
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32).flatten()[:self.obs_dim]
        self.dones[idx] = float(done)
        self.oracle_actions[idx] = int(oracle_action) if oracle_action is not None else -1
        self.oracle_weights[idx] = float(oracle_weight)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.long, device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
            "oracle_actions": torch.as_tensor(self.oracle_actions[indices], dtype=torch.long, device=self.device),
            "oracle_weights": torch.as_tensor(self.oracle_weights[indices], dtype=torch.float32, device=self.device),
        }

    def __len__(self):
        return self.size


class CommanderDoubleDQNAgent:
    """Double DQN agent for discrete high-level mode selection."""

    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            requested = torch.device("cpu")
        self.device = requested

        dqn_cfg = config.get("double_dqn", config.get("ppo", {}))
        self.lr = float(dqn_cfg.get("learning_rate", 3.0e-4))
        self.gamma = float(dqn_cfg.get("gamma", 0.99))
        self.epsilon_start = float(dqn_cfg.get("epsilon_start", 1.0))
        self.epsilon_end = float(dqn_cfg.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = int(dqn_cfg.get("epsilon_decay_steps", 10000))
        self.target_update_freq = int(dqn_cfg.get("target_update_freq", 1000))
        self.batch_size = int(dqn_cfg.get("batch_size", 256))
        self.buffer_capacity = int(dqn_cfg.get("buffer_capacity", 10000))
        self.update_freq = int(dqn_cfg.get("update_freq", 1))
        self.learning_starts = int(dqn_cfg.get("learning_starts", 1000))
        self.max_grad_norm = float(dqn_cfg.get("max_grad_norm", 10.0))

        policy_cfg = config.get("policy", {})
        hidden_sizes = policy_cfg.get("hidden_sizes", [128, 128])
        activation = policy_cfg.get("activation", "tanh")

        self.network = CategoricalMLPActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)
        self.replay_buffer = SimpleReplayBuffer(
            capacity=self.buffer_capacity,
            obs_dim=self.obs_dim,
            device=self.device,
        )

        # Oracle imitation warmup (learn from oracle mode-selection labels)
        commander_cfg = config.get("commander", {})
        oracle_cfg = commander_cfg.get("oracle_imitation", {})
        self.oracle_imitation_enabled = bool(oracle_cfg.get("enabled", False))
        self.oracle_imitation_loss_weight = float(oracle_cfg.get("loss_weight", 0.0))
        self.oracle_imitation_warmstart_updates = int(oracle_cfg.get("warmstart_updates", 0))
        self.oracle_imitation_decay_schedule = str(oracle_cfg.get("decay_schedule", "linear_to_zero"))

        # Pending transition for next_obs bookkeeping (bridges on-policy caller to off-policy buffer)
        self._last_obs = None
        self._last_action = None
        self._last_reward = None
        self._last_done = None
        self._last_oracle_action = None
        self._last_oracle_weight = None

        self.total_updates = 0
        self.total_timesteps = 0

    def _get_epsilon(self):
        if self.epsilon_decay_steps <= 0:
            return self.epsilon_end
        progress = min(self.total_timesteps, self.epsilon_decay_steps) / self.epsilon_decay_steps
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
        if obs_t.shape[0] != self.obs_dim:
            raise ValueError(f"Expected obs shape ({self.obs_dim},), got {obs_t.shape}")

        epsilon = 0.0 if deterministic else self._get_epsilon()

        with torch.no_grad():
            logits, value = self.network.forward(obs_t)
            q_values = logits  # (action_dim,)

        if not deterministic and np.random.random() < epsilon:
            action = int(np.random.randint(0, self.action_dim))
        else:
            action = int(torch.argmax(q_values).cpu().numpy())

        # Compatibility with PPO interface: return (action, log_prob, value)
        # log_prob = 0.0 (not meaningful for epsilon-greedy)
        # value = max Q for logging
        max_q = float(q_values.max().cpu().numpy())
        return action, 0.0, max_q

    def get_deterministic_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
        with torch.no_grad():
            logits, _ = self.network.forward(obs_t)
            action = int(torch.argmax(logits).cpu().numpy())
        return action

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
        # log_prob and value are PPO-specific; keep them in signature for compatibility
        del log_prob, value

        if self._last_obs is not None:
            self.replay_buffer.add(
                self._last_obs,
                self._last_action,
                self._last_reward,
                obs,
                self._last_done,
                oracle_action=self._last_oracle_action,
                oracle_weight=self._last_oracle_weight,
            )

        self._last_obs = obs
        self._last_action = action
        self._last_reward = reward
        self._last_done = float(done)
        self._last_oracle_action = oracle_action if oracle_action is not None else -1
        self._last_oracle_weight = oracle_imitation_weight if oracle_imitation_weight is not None else 0.0
        self.total_timesteps += 1

    def _compute_oracle_imitation_weight(self):
        if not self.oracle_imitation_enabled or self.oracle_imitation_warmstart_updates <= 0:
            return 0.0
        progress = min(max(int(self.total_updates), 0), self.oracle_imitation_warmstart_updates)
        if self.oracle_imitation_decay_schedule == "linear_to_zero":
            scale = max(0.0, 1.0 - (progress / float(self.oracle_imitation_warmstart_updates)))
        elif self.oracle_imitation_decay_schedule == "constant":
            scale = 1.0 if progress < self.oracle_imitation_warmstart_updates else 0.0
        else:
            scale = max(0.0, 1.0 - (progress / float(self.oracle_imitation_warmstart_updates)))
        return self.oracle_imitation_loss_weight * scale

    def update(self, next_obs=None):
        # Flush pending transition using the provided next_obs
        if self._last_obs is not None and next_obs is not None:
            self.replay_buffer.add(
                self._last_obs,
                self._last_action,
                self._last_reward,
                next_obs,
                self._last_done,
                oracle_action=self._last_oracle_action,
                oracle_weight=self._last_oracle_weight,
            )
            self._last_obs = None
            self._last_action = None
            self._last_reward = None
            self._last_done = None
            self._last_oracle_action = None
            self._last_oracle_weight = None

        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return {}

        if self.total_timesteps % self.update_freq != 0:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {}

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        oracle_actions = batch["oracle_actions"]
        oracle_weights = batch["oracle_weights"]

        # Current Q values
        q_logits, _ = self.network.forward(obs)
        q_values = q_logits.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Double DQN target: select best action with online network, evaluate with target network
        with torch.no_grad():
            next_q_logits, _ = self.network.forward(next_obs)
            next_actions = next_q_logits.argmax(dim=-1)
            next_q_target_logits, _ = self.target_network.forward(next_obs)
            next_q_values = next_q_target_logits.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = rewards + self.gamma * next_q_values * (1.0 - dones)

        dqn_loss = F.smooth_l1_loss(q_values, target)

        # Oracle imitation warmup: cross-entropy loss on Q-logits against oracle action labels
        oracle_weight = self._compute_oracle_imitation_weight()
        if oracle_weight > 0.0:
            # Only apply to transitions that have a valid oracle action recorded
            valid_mask = oracle_actions >= 0
            if valid_mask.any():
                valid_logits = q_logits[valid_mask]
                valid_oracle = oracle_actions[valid_mask]
                oracle_loss = F.cross_entropy(valid_logits, valid_oracle)
                loss = dqn_loss + oracle_weight * oracle_loss
            else:
                loss = dqn_loss
                oracle_loss = torch.tensor(0.0, device=self.device)
        else:
            loss = dqn_loss
            oracle_loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.total_updates += 1

        # Hard update target network
        if self.total_updates % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return {
            "loss": float(loss.item()),
            "dqn_loss": float(dqn_loss.item()),
            "oracle_loss": float(oracle_loss.item()),
            "oracle_weight": float(oracle_weight),
            "mean_q": float(q_values.mean().item()),
            "mean_target": float(target.mean().item()),
            "epsilon": self._get_epsilon(),
            "learning_rate": self.lr,
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
        self.target_network.load_state_dict(checkpoint["network_state_dict"], strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as exc:
            if "parameter group" in str(exc):
                print(f"WARNING: DQN optimizer state load skipped due to architecture mismatch: {exc}")
            else:
                raise
        self.total_updates = checkpoint.get("total_updates", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        return checkpoint
