"""
Method-innovation ablation branches.

These modules are NOT part of the canonical (paper-safe) pipeline. The main
branch retains only the standard PPO baseline (``agents.standard_ppo_agent`` /
``agents.ppo_agent.PPOAgent``), the CEM-EMA gain optimizer, and the canonical
guidance / VPP / gain-space definitions.

Sub-packages:
  - ``baseline``    : standard PPO baseline arm (re-exports the core agent).
  - ``cr_ppo``      : Complexity-Regularized PPO (CR-PPO).
  - ``intentional`` : Intentional Updates PPO (ICU / IAU).
  - ``cais_only``   : Combat-Aware Intentional Schedule (CAIS) component.

All variants inherit from the shared :class:`~uav_vpp_guidance.agents.ppo_agent.PPOAgent`
base class to avoid code duplication.
"""
