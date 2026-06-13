# Project Status — Single Source of Truth

> **Last updated**: 2026-06-13
>
> This file is the authoritative project-status document. Other files
> (README.md, roadmap documents, experiment notes) should reference it and
> must not contradict it. Stages marked **[FROZEN]** are considered complete
> and immutable unless a new audit explicitly reopens them.

---

## 1. Canonical Configuration (frozen)

The following configuration files define the unified, paper-safe experimental
setup. All new experiments should inherit from these files.

| Component | File | Purpose |
|-----------|------|---------|
| Guidance law | `config/canonical/guidance.yaml` | LOS-rate guidance with 3-D VPP, fixed vs. optimized gains |
| Gain space | `config/canonical/gain_space.yaml` | 5-D CEM search space: `(k_los, k_pos, k_damp, k_roll, k_speed)` |
| Reward | `config/canonical/reward.yaml` | Dense per-step mixture + terminal sparse events |
| VPP action | `config/canonical/virtual_point.yaml` | 3-D spatial offset action space |
| CEM optimizer | `config/canonical/cem.yaml` | EMA-CEM as default; GD deprecated |

Key alignment decisions (Phase 1 freeze):

1. **Guidance law**: The implemented command chain is roll-rate / normal-overload /
   throttle (LOSRateGuidance). Theory reports should reference this implementation;
   the idealized translational-acceleration model is only an analytical intuition.
2. **VPP action space**: Frozen at **3-D spatial offset** (`d_long`, `d_lat`,
   `d_vert`). The 5-D description (with `tau_pred` and `speed_bias`) in legacy
   notes is deprecated.
3. **Gain space**: CEM optimizes **5-D** `(k_los, k_pos, k_damp, k_roll, k_speed)`.
   `alpha_filter` and `k_energy` are fixed.
4. **CEM variant**: **EMA-CEM is the default**. The CEM-GD hybrid has been moved
   to `src/uav_vpp_guidance/ablations/deprecated/`.
5. **Reward design**: Dense weighted mixture + terminal sparse events. The
   "extreme sparse reward" description is obsolete.

---

## 2. Stage Status

| Stage | Status | Evidence | Runnable |
|-------|--------|----------|----------|
| 6B | [FROZEN] Complete | No-Prediction vs CV vs CA core comparison | `scripts/run_paper_benchmark.py` |
| 6E | [FROZEN] Complete | 4 geometry scenarios, 1.1M steps | `scripts/run_paper_benchmark.py` |
| 6F | [FROZEN] Complete | Paper-safe benchmark, McNemar exact, cross-seed | `scripts/run_paper_benchmark.py` |
| 6G.1 | [FROZEN] Complete | Guidance-law limitation probe (720 episodes) | `scripts/run_stage6g_guidance_limitation_probe.py` |
| 6G.2 | [FROZEN] Complete | Failure root-cause attribution | `scripts/validate_stage6g_probe_outputs.py` |
| 6G.3 | [FROZEN] Complete | Oracle & terminal-protection feasibility gate | `docs/stage6g3_oracle_terminal_feasibility_gate.md` |
| 6G.4 | [FROZEN] Complete | Oracle smoke execution & telemetry | `docs/stage6g4_oracle_smoke_and_telemetry.md` |
| 6G.5A–D | [FROZEN] Complete | PN mode-switch & VPP offset root cause | `scripts/run_stage6g5d_pn_mode_switch_probe.py` |
| 6H.0-R | [FROZEN] Complete | Regression baseline recovery & backend consistency | `docs/results/stage6h0r_config_drift_audit.md` |
| 6H.0-lite | [ACTIVE] | Mode-switch threshold optimization preflight | `docs/stage6h0_lite_mode_switch_threshold_optimization_plan.md` |
| 6H.1 | [FROZEN] Complete | Gate redesign, threshold exploration | `tests/test_stage6g5d_pn_mode_switch.py` |
| 6H.2 | [FROZEN] Complete | LHS20 threshold optimization, 8/20 PASS | `scripts/run_lhs20_threshold_optimization.py` |
| 6H.3 | [FROZEN] Complete | PN LOS-rate boundary fix, hybrid blend fix | `src/uav_vpp_guidance/guidance/los_rate_guidance.py` |
| 6I | [FROZEN] Complete | Statistical comparison framework | `tests/test_statistical_comparison.py` |
| 7A | [FROZEN] Complete | Gain-only CEM optimization | `scripts/run_gain_only_cem.py` |
| 7B | [FROZEN] Complete | Full BilevelTrainer with regret tracking | `scripts/train_bilevel.py` |
| 8B | [FROZEN] Complete | Paper-ready benchmark | `scripts/run_paper_benchmark.py` |
| 8B.1 | [FROZEN] Complete | Remote sync & integration hardening | `python -m uav_vpp_guidance.training.train_bilevel --dry-run` |
| 8C | [FROZEN] Complete | Paper-safe readiness gate | `scripts/run_paper_benchmark.py` |
| 9A | [FROZEN] Complete | Experiment freeze & artifact contract | `scripts/run_paper_benchmark.py` |
| **Phase 1 freeze (this PR)** | [ACTIVE] | Canonical configs, CEM-GD deprecation, reward audit | this file |
| **Phase 2 reward audit (A2')** | [FROZEN] Complete | Dense vs. dense+PBS vs. terminal-only, 3 seeds, simple backend, standard PPO | `scripts/run_a2_reward_ablation.py`, `outputs/ablation_reward_design/` |
| **Phase 2 ablation separation** | [FROZEN] Complete | CR-PPO / Intentional / CAIS / baseline split into `src/uav_vpp_guidance/ablations/`; main branch keeps standard PPO only | `tests/test_cr_ppo_agent.py`, `tests/test_intentional_ppo_agent.py`, `tests/test_combat_aware_schedule.py` |

---

## 3. Known Drifts and Resolutions

| # | Inconsistency | Resolution | Status |
|---|---------------|------------|--------|
| 1 | Guidance-law theory vs. code | Theory reports reference the implemented 3-loop guidance; idealized model is explicitly labeled as intuition | Resolved |
| 2 | VPP action space | Frozen at 3-D; legacy 5-D fields deprecated | Resolved |
| 3 | Gain space | Frozen at 5-D CEM search; fixed params documented | Resolved |
| 4 | CEM-GD legacy code | Moved to `ablations/deprecated/`; EMA default | Resolved |
| 5 | Reward description | Unified as dense + terminal sparse; A2' ablation confirms PBS redundant (`docs/reward_audit.md` §4.1) | Resolved |
| 6 | Method-innovation branch mixing | CR-PPO → `ablations/cr_ppo/`, Intentional PPO → `ablations/intentional/`, CAIS → `ablations/cais_only/`, baseline → `ablations/baseline/`; main branch keeps only standard PPO + CEM-EMA + canonical guidance/VPP/gain-space | [FROZEN] |
| 7 | Documentation status | This file is now the single source of truth | Resolved |

---

## 4. Honest Limitations (paper scope)

1. **Guidance Law Scope**: The theoretical analysis in reports uses an idealized
   translational-acceleration model for analytical tractability. The implemented
   guidance law (LOSRateGuidance) outputs roll-rate / normal-overload / throttle.
   The idealized model provides physical intuition; actual closed-loop behavior
   is determined by the command-chain implementation.
2. **Reward Design**: The repository uses a dense reward mixture (range, angle,
   safety, smoothness, closing rate) plus terminal sparse events. Potential-based
   shaping is an optional theoretical enhancement, not a necessity. The Phase 2
   A2' ablation (3 seeds, simple backend, standard PPO) found no statistically
   significant difference between dense-only and dense + PBS (p = 0.785),
   empirically confirming PBS is redundant; it stays disabled in canonical runs.
3. **CEM Convergence**: Theorem 2' is a heuristic scaling intuition, not a
   rigorous guarantee. The lower-level objective is a noisy episodic estimate.
4. **Bilevel Equilibrium**: Proposition 10' frames the pipeline as a conceptual
   Stackelberg lens, not a proved property. The lower level is solved
   approximately by CEM with multiple good gain settings.
5. **PPO Convergence**: Theorem 4' is conditional under idealized assumptions.
   The implementation uses fixed learning rate and deep neural networks, which
   do not satisfy all stated assumptions.
