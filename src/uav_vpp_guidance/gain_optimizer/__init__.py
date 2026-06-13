"""Guidance gain optimization algorithms.

Canonical optimizers:
- CEMGainOptimizer: standard CEM.
- CEMEMAGainOptimizer: EMA-smoothed CEM (default for paper experiments).

Deprecated:
- CEMGDGainOptimizer (two-phase CEM-GD) has moved to
  ``uav_vpp_guidance.ablations.deprecated.cem_gd`` and is no longer
  recommended for flat, noisy gain landscapes. Use CEMEMAGainOptimizer
  instead. See Theorem 3' and docs/status.md.
"""

from .cem import CEMGainOptimizer
from .gain_space import GainSpace
from .bilevel_trainer import BilevelTrainer

__all__ = ["CEMGainOptimizer", "GainSpace", "BilevelTrainer"]
