"""
PPO Rollout Buffer with GAE computation.

Stores trajectories collected during policy rollouts and computes
Generalized Advantage Estimation (GAE) for stable PPO updates.
"""

import numpy as np
import torch


class PPORolloutBuffer:
    """
    Rollout buffer for on-policy PPO training.

    Stores observations, actions, log_probs, rewards, values, and dones
    for a fixed number of rollout steps, then computes advantages and returns.
    """

    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        """
        Args:
            capacity (int): Maximum number of timesteps to store (rollout_steps).
            obs_dim (int): Observation dimension.
            action_dim (int): Action dimension.
            device (str): torch device.
        """
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device

        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        # Pre-allocate arrays
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, log_prob, reward, done, value):
        """
        Add a single transition to the buffer.

        Args:
            obs (np.ndarray): Observation vector, shape (obs_dim,).
            action (np.ndarray): Action vector, shape (action_dim,).
            log_prob (float): Log probability of the action.
            reward (float): Reward received.
            done (bool): Whether the episode terminated.
            value (float): Value estimate for the observation.
        """
        if self.full:
            raise RuntimeError("Rollout buffer is full. Call compute_gae() and clear() before adding more.")

        idx = self.ptr
        self.obs[idx] = np.asarray(obs, dtype=np.float32).flatten()[:self.obs_dim]
        self.actions[idx] = np.asarray(action, dtype=np.float32).flatten()[:self.action_dim]
        self.log_probs[idx] = float(log_prob)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)
        self.values[idx] = float(value)

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            next_value (float): Value estimate of the state after the last stored transition.
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.

        Returns:
            tuple: (advantages, returns) as numpy arrays.
        """
        if self.ptr == 0:
            return np.array([]), np.array([])

        n = self.ptr
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        # Work backwards through the rollout
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_v = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_v = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_v * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:n]

        # NaN/inf check
        if not np.isfinite(advantages).all():
            raise ValueError("Non-finite values detected in GAE advantages")
        if not np.isfinite(returns).all():
            raise ValueError("Non-finite values detected in GAE returns")

        self.advantages[:n] = advantages
        self.returns[:n] = returns
        return advantages, returns

    def get_data(self):
        """
        Get all stored data as tensors.

        Returns:
            dict: All data tensors.
        """
        n = self.ptr
        if n == 0:
            return {}

        obs = torch.as_tensor(self.obs[:n], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions[:n], dtype=torch.float32, device=self.device)
        log_probs = torch.as_tensor(self.log_probs[:n], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.advantages[:n], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.returns[:n], dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.values[:n], dtype=torch.float32, device=self.device)

        # Normalize advantages
        if advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        return {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "values": values,
        }

    def get_minibatches(self, batch_size):
        """
        Yield minibatches of stored data.

        Args:
            batch_size (int): Minibatch size.

        Yields:
            dict: Minibatch tensors.
        """
        data = self.get_data()
        if not data:
            return

        n = data["obs"].shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield {k: v[batch_idx] for k, v in data.items()}

    def clear(self):
        """Reset buffer pointers without reallocating memory."""
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.ptr
