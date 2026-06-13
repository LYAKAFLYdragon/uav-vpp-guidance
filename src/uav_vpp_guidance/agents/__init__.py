"""Core RL agent implementations.

The canonical main branch retains only the standard PPO baseline (and SAC /
end-to-end variants used by existing evaluations). Method-innovation agents
(CR-PPO, Intentional PPO) live under
:mod:`uav_vpp_guidance.ablations` and inherit from :class:`PPOAgent`.
"""

from .ppo_agent import PPOAgent
from .standard_ppo_agent import StandardPPOAgent
from .sac_agent import SACAgent

__all__ = ["PPOAgent", "StandardPPOAgent", "SACAgent"]
