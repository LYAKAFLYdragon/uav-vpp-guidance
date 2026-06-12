# Method Innovation Track 1: CR-PPO + Curriculum Learning + CEM-GD

Branch: `CL_CRPPO_CEMGD`

This branch implements the first method-innovation track from the
[research strategy report](../Documents/kimi/workspace/uav_vpp_research_strategy_report.md):
integrate three low-risk, high-ceiling improvements into the frozen VPP+PPO
baseline with the goal of **raising the performance upper bound** (especially
on the crossing scenario and initial-condition robustness).

## 1. CR-PPO — Complexity-Regularized PPO

**Motivation:** Standard entropy regularization always rewards high entropy,
which can push the policy toward an over-diffuse action distribution. CR-PPO
(Serfilippi, 2025) instead uses `entropy * disequilibrium`, rewarding a
healthy but not pathologically uniform exploration distribution.

**Implementation:**
- `src/uav_vpp_guidance/agents/cr_ppo_agent.py`
- New class `CRPPOAgent` inherits from `PPOAgent` and overrides `update()`.
- For the continuous Gaussian policy, complexity is approximated by
  discretizing each bounded action dimension into `cr_n_bins` bins, then
  computing discrete entropy and disequilibrium on those bins.

**Configuration:**
```yaml
ppo:
  complexity_coef: 1.0e-3   # weight for the complexity bonus
  cr_n_bins: 8              # discretization granularity
```

**Risk:** Very low — pure drop-in replacement for the entropy term; easy to
revert by setting `complexity_coef: 0`.

## 2. Curriculum Learning

**Motivation:** The baseline policy is trained almost exclusively on
favorable/neutral geometries, so it generalizes poorly to crossing geometry
(0% success rate). A curriculum progressively introduces harder geometries.

**Implementation:**
- `src/uav_vpp_guidance/training/curriculum.py`
- `CurriculumScheduler` now implements stage gating, scenario weights, and
  serialization.
- `scripts/train_curriculum_ppo.py` already consumes a curriculum config; it
  can be extended to use `CurriculumScheduler.get_current_scenario_weights()`.

**Configuration:**
```yaml
curriculum:
  gate_mode: min
  stages:
    - name: static_target
      scenario_names: [favorable, neutral]
      success_threshold: 0.80
      min_episodes: 100
    - name: crossing_geometry
      scenario_names: [favorable, neutral, disadvantage, challenging]
      success_threshold: 0.30
      min_episodes: 500
```

**Acceptance criterion:** crossing scenario success rate ≥ 30% on a small
smoke run.

## 3. CEM-GD — Hybrid Coarse-to-Fine Gain Optimization

**Motivation:** The bilevel gain optimizer currently uses pure CEM. CEM-GD
(Huang, 2021) runs CEM for coarse global search, then refines the best elite
with gradient descent, potentially reducing solve time and improving gain
smoothness.

**Implementation:**
- `src/uav_vpp_guidance/gain_optimizer/cem_gd.py`
- `CEMGDGainOptimizer` subclasses `CEMGainOptimizer`.
- Gradient is estimated with two-point finite differences so the optimizer
  remains compatible with the existing black-box evaluator.

**Configuration:**
```yaml
cem_gd:
  candidates: 12
  elite_ratio: 0.25
  gd_ratio: 0.5
  gd_lr: 0.05
  gd_fd_eps: 1.0e-3
```

**Acceptance criterion:** CEM-GD solve time < 80% of pure CEM at comparable
or better score.

## 4. Integration Config

`config/method_innovation_track1.yaml` combines the three innovations and can
be loaded by the existing training scripts after minor wiring.

## 5. Tests

Three focused test modules were added:
- `tests/test_cr_ppo_agent.py`
- `tests/test_curriculum_scheduler.py`
- `tests/test_cem_gd.py`

Run them with:
```bash
python -m pytest tests/test_cr_ppo_agent.py tests/test_curriculum_scheduler.py tests/test_cem_gd.py -v
```

## 6. Next Steps / Open Wiring

- [ ] Swap `PPOAgent` for `CRPPOAgent` in `scripts/train_curriculum_ppo.py`
      (or add a `--use-cr-ppo` flag).
- [ ] Use `CurriculumScheduler.get_current_scenario_weights()` in the
      training loop and advance it based on per-scenario evaluation.
- [ ] Replace `CEMGainOptimizer` with `CEMGDGainOptimizer` in
      `scripts/run_bilevel_audit.py` / `src/uav_vpp_guidance/gain_optimizer/bilevel_trainer.py`.
- [ ] Run a 3-seed smoke comparison: baseline PPO vs. CR-PPO vs.
      CR-PPO + curriculum vs. CR-PPO + curriculum + CEM-GD.
