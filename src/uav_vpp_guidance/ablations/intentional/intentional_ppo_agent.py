"""
Intentional Updates for PPO in air-combat guidance.

Adapts Sutton et al.'s "Intentional Updates for Streaming Reinforcement Learning"
to the project's existing PPO/MAPPO-style minibatch training. Three optional
components are provided:

1. Intentional Critic Update (ICU): scales the critic update by the ratio of
   TD residual to squared value-gradient norm, controlling ``Delta V(s)``.
2. Intentional Actor Update (IAU): scales the actor update by the ratio of
   normalized advantage to squared log-probability-gradient norm, controlling
   ``Delta log pi(a|s)``.
3. Combat-Aware Intentional Schedule (CAIS): adjusts the base actor/critic
   budgets by air-combat phase.

All components are toggled via configuration flags and can be used independently
or together.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ...agents.ppo_agent import PPOAgent
from ..cais_only.combat_aware_schedule import CombatAwareSchedule


class IntentionalPPOAgent(PPOAgent):
    """
    PPO agent with optional Intentional Updates (ICU / IAU / CAIS).

    Additional configuration keys under ``ppo``:
      - ``use_intentional_critic`` (bool): enable ICU.
      - ``use_intentional_actor`` (bool): enable IAU.
      - ``use_combat_aware_eta`` (bool): enable CAIS.
      - ``eta_critic`` (float): base value-prediction update budget (default 0.1).
      - ``eta_actor`` (float): base log-prob update budget (default 0.01).
      - ``iu_eps`` (float): epsilon for gradient-norm denominators (default 1e-8).
      - ``beta_adv`` (float): EMA decay for advantage normalization (default 0.999).

    When ``use_combat_aware_eta`` is true, an optional ``combat_aware`` config
    block is passed to :class:`CombatAwareSchedule`.
    """

    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        super().__init__(obs_dim, action_dim, config, device)
        ppo_cfg = config.get("ppo", config)
        self.use_intentional_critic = bool(ppo_cfg.get("use_intentional_critic", False))
        self.use_intentional_actor = bool(ppo_cfg.get("use_intentional_actor", False))
        self.use_combat_aware_eta = bool(ppo_cfg.get("use_combat_aware_eta", False))

        self.eta_critic = float(ppo_cfg.get("eta_critic", 0.1))
        self.eta_actor = float(ppo_cfg.get("eta_actor", 0.01))
        self.iu_eps = float(ppo_cfg.get("iu_eps", 1e-8))
        self.beta_adv = float(ppo_cfg.get("beta_adv", 0.999))

        self.ema_abs_adv = 1.0

        if self.use_combat_aware_eta:
            ca_config = config.get("combat_aware", {})
            self.combat_schedule = CombatAwareSchedule(ca_config)
            self.phase_buffer = []
        else:
            self.combat_schedule = None
            self.phase_buffer = None

    def store_transition(self, obs, action, log_prob, reward, done, value, info=None):
        """
        Store a transition and optionally record combat phase features.

        Args:
            obs (np.ndarray): Observation vector.
            action (np.ndarray): Action vector.
            log_prob (float): Log probability of the action.
            reward (float): Reward received.
            done (bool): Whether episode terminated.
            value (float): Value estimate.
            info (dict, optional): If provided and contains ``relative_state``,
                phase features are extracted for combat-aware scheduling.
        """
        super().store_transition(obs, action, log_prob, reward, done, value)

        if self.use_combat_aware_eta and self.phase_buffer is not None:
            features = self._extract_phase_features(info)
            self.phase_buffer.append(features)

    def _extract_phase_features(self, info):
        """Extract geometry features used by the combat-aware schedule."""
        defaults = {
            "range_m": 5000.0,
            "ata_rad": 0.0,
            "aa_rad": 0.0,
            "altitude_diff_m": 0.0,
            "speed_diff_mps": 0.0,
            "range_rate_mps": 0.0,
            "missile_threat": 0.0,
        }
        if info is None:
            return defaults

        rel = info.get("relative_state", info)
        if rel is None:
            rel = {}

        features = {}
        for key in self.combat_schedule.FEATURE_KEYS:
            val = rel.get(key, defaults.get(key, 0.0))
            try:
                features[key] = float(val)
            except (TypeError, ValueError):
                features[key] = float(defaults.get(key, 0.0))
        return features

    @staticmethod
    def _grad_norm_sq(grads):
        """Compute the squared L2 norm of a list of gradients."""
        return sum(g.pow(2).sum().item() for g in grads)

    def update(self, next_obs=None):
        """
        Perform an Intentional PPO update.

        Runs ``self.update_epochs`` passes over the rollout data, reshuffling
        the minibatches each epoch. Intentional actor/critic scales are
        recomputed per minibatch. Falls back to standard PPO if both intentional
        flags are disabled.
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

        # Prepare phase features if combat-aware scheduling is enabled
        use_phase = self.use_combat_aware_eta and self.phase_buffer is not None
        if use_phase:
            n = len(self.buffer)
            n_features = len(self.combat_schedule.FEATURE_KEYS)
            phase_array = np.zeros((n, n_features), dtype=np.float32)
            for i, feat in enumerate(self.phase_buffer[:n]):
                for j, key in enumerate(self.combat_schedule.FEATURE_KEYS):
                    phase_array[i, j] = float(feat.get(key, 0.0))
        else:
            phase_array = None

        data = self.buffer.get_data()
        if not data:
            return {}

        n = data["obs"].shape[0]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_scale_actor = 0.0
        total_scale_critic = 0.0
        num_minibatches = 0

        for epoch in range(self.update_epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                batch_idx = indices[start:end]

                obs_b = data["obs"][batch_idx]
                actions_b = data["actions"][batch_idx]
                old_log_probs_b = data["log_probs"][batch_idx]
                advantages_b = data["advantages"][batch_idx]
                returns_b = data["returns"][batch_idx]
                old_values_b = data["values"][batch_idx]

                # Combat-aware eta scales for this batch
                if use_phase and phase_array is not None:
                    batch_phase = phase_array[batch_idx]
                    scales = self.combat_schedule.get_batch_scales(batch_phase)
                    phase_actor_scale = scales["actor"]
                    phase_critic_scale = scales["critic"]
                else:
                    phase_actor_scale = 1.0
                    phase_critic_scale = 1.0

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

                # ---- Intentional Critic Update (ICU) ----
                scale_critic = 1.0
                if self.use_intentional_critic:
                    # Gradient of value predictions w.r.t. network parameters
                    value_grad = torch.autograd.grad(
                        new_values_b.sum(),
                        self.network.parameters(),
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )
                    value_grad_norm_sq = self._grad_norm_sq([g for g in value_grad if g is not None])

                    td_residual = returns_b - new_values_b
                    eta_critic_eff = (
                        self.eta_critic
                        * phase_critic_scale
                        * td_residual.abs().mean().item()
                    )
                    alpha_v = eta_critic_eff / (value_grad_norm_sq + self.iu_eps)
                    scale_critic = alpha_v / self.lr

                # ---- Intentional Actor Update (IAU) ----
                scale_actor = 1.0
                if self.use_intentional_actor:
                    # Gradient of log probabilities w.r.t. network parameters
                    log_prob_grad = torch.autograd.grad(
                        new_log_probs_b.sum(),
                        self.network.parameters(),
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )
                    log_prob_grad_norm_sq = self._grad_norm_sq([g for g in log_prob_grad if g is not None])

                    adv_abs_mean = advantages_b.abs().mean().item()
                    self.ema_abs_adv = (
                        self.beta_adv * self.ema_abs_adv
                        + (1.0 - self.beta_adv) * adv_abs_mean
                    )
                    adv_norm = adv_abs_mean / (self.ema_abs_adv + self.iu_eps)

                    eta_actor_eff = (
                        self.eta_actor * phase_actor_scale * adv_norm
                    )
                    alpha_pi = eta_actor_eff / (log_prob_grad_norm_sq + self.iu_eps)
                    scale_actor = alpha_pi / self.lr

                # Total loss with scaled actor/critic terms
                entropy = entropy_b.mean()
                loss = (
                    scale_actor * policy_loss
                    + scale_critic * self.value_coef * value_loss
                    - scale_actor * self.entropy_coef * entropy
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
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                total_scale_actor += scale_actor
                total_scale_critic += scale_critic
                num_minibatches += 1

        self.total_updates += 1
        self.buffer.clear()
        if self.phase_buffer is not None:
            self.phase_buffer.clear()

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
            "scale_actor": total_scale_actor / num_minibatches,
            "scale_critic": total_scale_critic / num_minibatches,
            "ema_abs_adv": self.ema_abs_adv,
        }
