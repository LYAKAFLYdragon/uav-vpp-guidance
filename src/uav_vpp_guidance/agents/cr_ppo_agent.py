"""
Complexity-Regularized PPO (CR-PPO) agent.

Implements the CR-PPO idea (Serfilippi, 2025) for the UAV-VPP guidance
project: replace the standard entropy bonus with a *complexity* bonus,
defined as entropy multiplied by a disequilibrium (non-uniformity) measure.
For the continuous Gaussian action distribution used by the project, the
complexity is approximated by discretizing each bounded action dimension
into bins and computing the discrete entropy and disequilibrium over those
bins.

This keeps the PPO clipped surrogate and value loss unchanged, only the
exploration regularizer is changed.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from .ppo_agent import PPOAgent


class CRPPOAgent(PPOAgent):
    """
    PPO agent with complexity regularization (CR-PPO).

    Additional configuration keys under ``ppo``:
      - ``complexity_coef``: weight for the complexity bonus (default 1.0e-3).
      - ``cr_n_bins``: number of bins used for the continuous-action
        complexity approximation (default 8, must be >= 2).
    """

    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        super().__init__(obs_dim, action_dim, config, device)
        ppo_cfg = config.get("ppo", config)
        self.complexity_coef = float(ppo_cfg.get("complexity_coef", 1.0e-3))
        self.cr_n_bins = int(ppo_cfg.get("cr_n_bins", 8))
        if self.cr_n_bins < 2:
            raise ValueError(f"cr_n_bins must be >= 2, got {self.cr_n_bins}")

    def _compute_complexity(self, mean, std):
        """
        Compute CR-PPO complexity regularization for a Gaussian policy.

        Args:
            mean (Tensor): Gaussian mean, shape (batch, action_dim).
            std (Tensor): Gaussian std, shape (batch, action_dim).

        Returns:
            Tensor: per-sample complexity scalar, shape (batch,).
        """
        eps = 1e-8
        action_low = self.network.action_low.to(mean.device)
        action_high = self.network.action_high.to(mean.device)
        n_bins = self.cr_n_bins

        # Bin edges for each action dimension: (n_bins + 1, 1, action_dim)
        grid = torch.linspace(0.0, 1.0, n_bins + 1, device=mean.device)
        edges = grid.view(-1, 1, 1) * (action_high - action_low).view(1, 1, -1) + action_low.view(1, 1, -1)

        # Standardize edges w.r.t. per-sample Gaussian: (n_bins+1, batch, action_dim)
        z = (edges - mean.unsqueeze(0)) / (std.unsqueeze(0) + eps)
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

        # Probability mass falling into each bin: (n_bins, batch, action_dim)
        probs = cdf[1:] - cdf[:-1]
        probs = torch.clamp(probs, min=eps)

        uniform = 1.0 / n_bins
        # Discrete entropy and disequilibrium per dimension
        entropy_per_dim = -torch.sum(probs * torch.log(probs + eps), dim=0)
        disequilibrium_per_dim = torch.sum((probs - uniform) ** 2, dim=0)
        complexity_per_dim = entropy_per_dim * disequilibrium_per_dim

        # Sum across action dimensions, reduce across batch later
        return complexity_per_dim.sum(dim=-1)

    def update(self, next_obs=None):
        """
        Perform a CR-PPO update.

        Identical to :meth:`PPOAgent.update` except the entropy bonus is
        replaced by the complexity regularizer.
        """
        if len(self.buffer) == 0:
            return {}

        # Bootstrap value for GAE
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

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_complexity = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_minibatches = 0

        for epoch in range(self.update_epochs):
            for batch in self.buffer.get_minibatches(batch_size=self.minibatch_size):
                obs_b = batch["obs"]
                actions_b = batch["actions"]
                old_log_probs_b = batch["log_probs"]
                advantages_b = batch["advantages"]
                returns_b = batch["returns"]
                old_values_b = batch["values"]

                # Distribution parameters needed for complexity term
                mean, _ = self.network(obs_b)
                log_std = self.network.actor_log_std.expand_as(mean)
                std = torch.exp(log_std)

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

                # Complexity regularization (CR-PPO)
                complexity = self._compute_complexity(mean, std)
                complexity_bonus = complexity.mean()

                # Total loss: entropy bonus is replaced by complexity bonus
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.complexity_coef * complexity_bonus
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
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
                total_entropy += entropy_b.mean().item()
                total_complexity += complexity_bonus.item()
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
            "complexity": total_complexity / num_minibatches,
            "approx_kl": total_approx_kl / num_minibatches,
            "clip_fraction": total_clip_fraction / num_minibatches,
            "explained_variance": explained_var.item(),
            "learning_rate": self.lr,
        }
