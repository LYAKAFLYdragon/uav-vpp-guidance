"""
Neural network architectures for PPO Actor-Critic.

Implements a shared-body MLP with separate actor and critic heads.
- Actor outputs Gaussian distribution mean and learnable log_std.
- Critic outputs scalar value estimate.
- Supports deterministic / stochastic action selection.
- Supports action squashing to configured bounds.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(sizes, activation="tanh"):
    """Build an MLP with specified layer sizes and activation."""
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    act_cls = act_map.get(activation, nn.Tanh)
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    """
    MLP Actor-Critic network for continuous action spaces.

    Supports task-conditioned multi-head actor when num_tasks > 1.
    The task type is extracted from the last observation dimension.

    Args:
        obs_dim (int): Observation dimension.
        action_dim (int): Action dimension.
        hidden_sizes (list): Hidden layer sizes.
        activation (str): Activation function name.
        action_low (np.ndarray or list): Lower bound of action space.
        action_high (np.ndarray or list): Upper bound of action space.
        init_log_std (float): Initial log standard deviation.
        num_tasks (int): Number of task-conditioned actor heads. Default 1 (single head).
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        activation="tanh",
        action_low=None,
        action_high=None,
        init_log_std=0.0,
        num_tasks=1,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_tasks = int(num_tasks)

        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if self.num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")

        # Action bounds for scaling
        if action_low is None:
            action_low = np.ones(self.action_dim, dtype=np.float32) * -1.0
        if action_high is None:
            action_high = np.ones(self.action_dim, dtype=np.float32) * 1.0
        self.register_buffer(
            "action_low", torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )
        self.register_buffer(
            "action_scale", (self.action_high - self.action_low) / 2.0
        )
        self.register_buffer(
            "action_bias", (self.action_high + self.action_low) / 2.0
        )

        # Shared body
        shared_sizes = [self.obs_dim] + list(hidden_sizes)
        self.shared_net = build_mlp(shared_sizes, activation=activation)

        if self.num_tasks == 1:
            # Single actor head (backward compatible)
            self.actor_mean = nn.Linear(shared_sizes[-1], self.action_dim)
            # Learnable log standard deviation (independent per action dim)
            self.actor_log_std = nn.Parameter(torch.ones(self.action_dim) * init_log_std)
        else:
            # Multiple task-conditioned actor heads
            self.actor_means = nn.ModuleList([
                nn.Linear(shared_sizes[-1], self.action_dim) for _ in range(self.num_tasks)
            ])
            self.actor_log_stds = nn.ParameterList([
                nn.Parameter(torch.ones(self.action_dim) * init_log_std) for _ in range(self.num_tasks)
            ])

        # Critic head: shared across all tasks
        self.critic = nn.Linear(shared_sizes[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor mean: smaller final layer gain
        if self.num_tasks == 1:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.constant_(self.actor_mean.bias, 0.0)
        else:
            for actor_mean in self.actor_means:
                nn.init.orthogonal_(actor_mean.weight, gain=0.01)
                nn.init.constant_(actor_mean.bias, 0.0)
        # Critic: standard gain
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def _extract_task_id(self, obs):
        """Extract task_id from the last observation dimension (is_crossing)."""
        if self.num_tasks == 1:
            return None
        task_id = (obs[..., -1] > 0.5).long().clamp(0, self.num_tasks - 1)
        return task_id

    def _get_actor_params(self, features, task_id):
        """Get mean and log_std for the given task_id."""
        if self.num_tasks == 1:
            mean = self.actor_mean(features)
            log_std = self.actor_log_std.expand_as(mean)
            return mean, log_std

        batch_size = features.shape[0]
        mean = torch.zeros(batch_size, self.action_dim, device=features.device, dtype=features.dtype)
        log_std = torch.zeros(batch_size, self.action_dim, device=features.device, dtype=features.dtype)
        for t in range(self.num_tasks):
            mask = (task_id == t)
            if mask.any():
                mean[mask] = self.actor_means[t](features[mask])
                log_std[mask] = self.actor_log_stds[t].expand_as(mean[mask])
        return mean, log_std

    def forward(self, obs):
        """
        Forward pass through shared network.

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch, obs_dim) or (obs_dim,).

        Returns:
            tuple: (mean, value)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected obs shape [..., {self.obs_dim}], got {obs.shape}"
            )

        features = self.shared_net(obs)
        task_id = self._extract_task_id(obs)
        mean, _ = self._get_actor_params(features, task_id)
        value = self.critic(features).squeeze(-1)
        return mean, value

    def get_value(self, obs):
        """
        Get value estimate for observations.

        Args:
            obs (torch.Tensor): Observations.

        Returns:
            torch.Tensor: Value estimates.
        """
        _, value = self.forward(obs)
        return value

    def get_action_and_value(self, obs, action=None, deterministic=False):
        """
        Sample action and compute log_prob, entropy, and value.

        Args:
            obs (torch.Tensor): Observations.
            action (torch.Tensor, optional): If provided, evaluate this action.
            deterministic (bool): If True, return mean action.

        Returns:
            tuple: (action, log_prob, entropy, value)
                If action is provided, returns (log_prob, entropy, value).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected obs shape [..., {self.obs_dim}], got {obs.shape}"
            )

        features = self.shared_net(obs)
        task_id = self._extract_task_id(obs)
        mean, log_std = self._get_actor_params(features, task_id)
        value = self.critic(features).squeeze(-1)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        action_was_none = action is None

        if action_was_none:
            if deterministic:
                raw_action = mean
            else:
                raw_action = dist.rsample()
            # Squash to action bounds using tanh
            squashed_action = torch.tanh(raw_action)
            # Scale to [action_low, action_high]
            action = squashed_action * self.action_scale + self.action_bias

            # Compute log_prob with tanh correction
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            # Correction for tanh squashing: log(1 - tanh(x)^2) + eps
            log_prob -= torch.sum(
                torch.log(1.0 - squashed_action.pow(2) + 1e-6), dim=-1
            )
        else:
            # Reverse scaling: action -> [-1, 1] -> arctanh
            normalized = (action - self.action_bias) / (self.action_scale + 1e-8)
            normalized = torch.clamp(normalized, -0.999999, 0.999999)
            raw_action = 0.5 * torch.log((1 + normalized) / (1 - normalized + 1e-8))
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.sum(
                torch.log(1.0 - normalized.pow(2) + 1e-6), dim=-1
            )

        entropy = dist.entropy().sum(dim=-1)

        # NaN/inf protection
        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            raise ValueError("NaN or inf detected in log_prob during action evaluation")
        if torch.isnan(value).any() or torch.isinf(value).any():
            raise ValueError("NaN or inf detected in value during action evaluation")

        if action_was_none:
            return action, log_prob, entropy, value
        return log_prob, entropy, value

    def get_deterministic_action(self, obs):
        """
        Get deterministic (mean) action for inference.

        Args:
            obs (torch.Tensor): Observation.

        Returns:
            torch.Tensor: Deterministic action.
        """
        with torch.no_grad():
            mean, _ = self.forward(obs)
            squashed = torch.tanh(mean)
            action = squashed * self.action_scale + self.action_bias
        return action

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
