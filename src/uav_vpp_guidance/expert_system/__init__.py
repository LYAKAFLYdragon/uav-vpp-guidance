"""
ExpertVPPPolicy — Rule-driven explainable baseline for close-range air combat.

Modules:
  - situation_evaluator: computes tactical situation scores
  - rule_engine: maps situation to maneuver intent
  - expert_vpp_policy: main policy interface
"""

from .expert_vpp_policy import ExpertVPPPolicy

__all__ = ["ExpertVPPPolicy"]
