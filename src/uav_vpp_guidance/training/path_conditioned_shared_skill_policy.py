"""New 74-D-to-3-D policy head for path-conditioned shared-intent research.

This module deliberately does not load or adapt legacy 66-D checkpoints.  Its
only purpose is to make the V17 schema executable after a separate sampler
preflight and training authorization succeed.
"""

from __future__ import annotations

import torch
from torch import nn

from uav_vpp_guidance.hierarchy.path_conditioned_observation_contract import (
    PathConditionedObservationContract,
)


OBSERVATION_DIM = PathConditionedObservationContract().obs_dim
VPP_ACTION_DIM = 3
CHECKPOINT_SCHEMA = "thesis_path_conditioned_shared_skill_ppo_v1"


class PathConditionedSharedSkillPolicy(nn.Module):
    """Strict new policy head; no implicit 66-D compatibility path exists."""

    def __init__(self, *, hidden_dim: int = 128) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.observation_dim = OBSERVATION_DIM
        self.action_dim = VPP_ACTION_DIM
        self.network = nn.Sequential(
            nn.Linear(self.observation_dim, int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), self.action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(
                "V17 path-conditioned policy requires exact "
                f"[batch, {self.observation_dim}] observations"
            )
        if not torch.isfinite(observation).all():
            raise ValueError("V17 path-conditioned observation contains non-finite values")
        return torch.tanh(self.network(observation))
