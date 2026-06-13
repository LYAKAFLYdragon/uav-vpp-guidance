"""
Standard PPO agent (canonical baseline).

This module gives the canonical baseline agent its documented name,
``standard_ppo_agent`` / :class:`StandardPPOAgent`. It is a thin alias over the
core :class:`uav_vpp_guidance.agents.ppo_agent.PPOAgent` implementation, which
remains the single source of truth and the shared base class that the
method-innovation ablation branches (CR-PPO, Intentional PPO) inherit from.

Keeping ``PPOAgent`` as the implementation avoids a wide, risky rename across
the ~30 evaluation/training/script call sites that already import it, while
still exposing the ``standard_ppo_agent`` entry point required by the canonical
main-branch layout.
"""

from .ppo_agent import PPOAgent

# Canonical name for the standard PPO baseline.
StandardPPOAgent = PPOAgent

__all__ = ["StandardPPOAgent", "PPOAgent"]
