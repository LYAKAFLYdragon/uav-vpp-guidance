"""
Standard PPO baseline arm of the method-innovation ablation.

The canonical baseline agent lives in
:mod:`uav_vpp_guidance.agents.standard_ppo_agent` (an alias of the core
:class:`uav_vpp_guidance.agents.ppo_agent.PPOAgent`). It is re-exported here so
the four ablation arms (baseline / cr_ppo / intentional / cais_only) share a
consistent import surface. No behavior is added; the baseline is plain PPO.
"""

from ...agents.standard_ppo_agent import StandardPPOAgent

__all__ = ["StandardPPOAgent"]
