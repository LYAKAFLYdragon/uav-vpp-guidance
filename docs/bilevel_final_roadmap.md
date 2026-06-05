# Final Bilevel Roadmap

**Version**: 6G.1 → 7B  
**Date**: 2026-06-05  
**Status**: Stage 6G.1 in progress

---

## 1. Goal

Build a **bilevel optimization system** where an outer loop optimizes guidance gains (or switching thresholds) while a neural network optimizes pursuit strategy. The system must produce **paper-safe evidence** at each gate.

---

## 2. System Architecture

```
┌─────────────────────────────────────┐
│        Bilevel Optimizer            │
│  ┌─────────────────────────────┐    │
│  │  Outer: Gain Optimizer (CEM) │   │
│  │  - Switches: range, energy     │   │
│  │  - Thresholds: range, speed    │   │
│  │  - Gains: N, omega, kappa        │   │
│  └─────────────────────────────┘   │
│              ↓                      │
│  ┌─────────────────────────────┐    │
│  │  Inner: VPP Policy (PPO)     │   │
│  │  - Frozen during gain step  │   │
│  │  - Trained during policy step│   │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### 2.1 Components

| Component | Role | Status |
|---|---|---|
| **VPP Policy** | Neural network predicting pursuit point | ✅ Exists (Stage 6F) |
| **LOS-rate Guidance** | Classical guidance law with VPP | ✅ Exists |
| **Proportional Navigation** | Alternative guidance law | ✅ Exists (Stage 6G) |
| **Hybrid Guidance** | Hysteresis + blended switch | ✅ Exists (Stage 6G) |
| **CEM Gain Optimizer** | Cross-entropy method for gain optimization | ⏳ Not implemented (6H.0) |
| **Score Function** | Evaluates gain configuration | ⏳ Not defined (6H.0) |
| **Rollback Manager** | Reverts to last checkpoint if score drops | ⏳ Not implemented (6I.0) |

---

## 3. Gate Definitions

### 3.1 Gate 6G.1: Probe Hardening + Guidance Limitation Execution

**Acceptance Criteria**:
- [x] `run_stage6g_guidance_limitation_probe.py` has `--smoke`, `--dry-run`, `--output-dir`, `--allow-incomplete`
- [x] Script generates `resolved_config.yaml`, `run_manifest.json`, `raw_episodes.csv`, `scenario_method_summary.csv`, `pairwise_mcnemar.csv`, `paper_safe_claims.md`, `README_result_block.md`
- [x] Failure contract: exit 1 unless all artifacts written
- [x] No silent fallback: missing checkpoint → explicit failure
- [x] `effective_guidance_mode` tracked via `.mode` attribute on guidance laws
- [x] McNemar exact p-value using `scipy.stats.binomtest`
- [x] Smoke test passes (all 12 cells, 1 episode each)
- [x] Full probe completes (720 episodes) or partial with clear `Status: incomplete`
- [x] Paper-safe claims updated: tail-chase limitation status no longer `Pending`

**Status**: ✅ Complete

### 3.2 Gate 6G.2: Guidance-Law Limitation Analysis

**Acceptance Criteria**:
- [ ] Determination of whether tail-chase failure is:
  - LOS-rate limitation (PN/hybrid succeed where LOS-rate fails)
  - Geometric infeasibility (all guidance laws fail)
  - Policy/predictor limitation (no prediction = GRU frozen, no difference)
- [ ] McNemar p-values for all paired comparisons
- [ ] Cross-seed consistency check (p < 0.05 for all 3 seeds)
- [ ] Physical interpretation documented in `paper_safe_claims.md`

**Status**: ⏳ Pending (blocked by 6G.1 full probe)

### 3.3 Gate 6H.0: Gain-Only CEM Implementation

**Acceptance Criteria**:
- [ ] `CEMGainOptimizer` class exists with unit tests
- [ ] Score function defined and tested:
  - Deterministic: same seed → same score
  - Monotonic: higher score = better tracking
  - Decomposable: Score = success_rate × mean_return + stability_penalty
- [ ] CEM can optimize at least 2 parameters (e.g., `N`, `omega`)
- [ ] Reproducible experiment output: `cem_results.json`

**Status**: ⏳ Not started

### 3.4 Gate 6H.1: Fixed-Policy Gain Optimization

**Acceptance Criteria**:
- [ ] Freeze VPP policy (GRU frozen, weights locked)
- [ ] Optimize guidance gains (LOS-rate or hybrid) over 4 scenarios
- [ ] Multi-seed comparison (3 seeds):
  - Fixed gains (default) vs optimized gains
  - McNemar exact p-value for fixed vs optimized
- [ ] Paper-safe claim: "Optimized gains improve X% over default gains"

**Status**: ⏳ Not started

### 3.5 Gate 6I.0: Alternating Bilevel Training

**Acceptance Criteria**:
- [ ] Explicit alternating schedule:
  ```
  Epoch 1–20:   Train policy (gain frozen)
  Epoch 21–30:  Optimize gains (policy frozen)
  Epoch 31–50:  Train policy (gain frozen)
  Epoch 51–60:  Optimize gains (policy frozen)
  ...
  ```
- [ ] Checkpoint after each phase
- [ ] Rollback strategy: if score drops > 10% for 3 consecutive epochs, revert to last checkpoint
- [ ] `alternating_schedule.yaml` documented

**Status**: ⏳ Not started

### 3.6 Gate 6I.1: Regret and Stability Audit

**Acceptance Criteria**:
- [ ] Regret curve: cumulative regret vs optimal policy
- [ ] Success rate curve: per-epoch success rate
- [ ] Stability metric: variance of score over 5 epochs
- [ ] Failure root cause analysis: crash, OOB, timeout, guidance dead zone
- [ ] Report: `bilevel_audit.md`

**Status**: ⏳ Not started

### 3.7 Gate 7A: JSBSim/F-16 Validation

**Acceptance Criteria**:
- [ ] Transfer Simple backend conclusions to 6DOF JSBSim backend
- [ ] At least 2 scenarios validated in JSBSim
- [ ] Comparison: Simple backend vs JSBSim (same policy, same gains)
- [ ] Paper-safe claim: "Conclusions hold in 6DOF simulation"

**Status**: ⏳ Not started

### 3.8 Gate 7B: Paper Release Package

**Acceptance Criteria**:
- [ ] Frozen configs: `config/release/`
- [ ] Frozen seeds: `seeds.json` (all training and evaluation seeds)
- [ ] CSVs: `raw_episodes.csv`, `scenario_method_summary.csv`, `pairwise_mcnemar.csv`
- [ ] Figures: `figures/` (success rate, capture time, miss distance)
- [ ] Summary: `paper_safe_claims.md`, `README_result_block.md`
- [ ] Commit hash: documented in `run_manifest.json`
- [ ] Environment file: `requirements.txt` with exact versions
- [ ] DOI or arXiv submission

**Status**: ⏳ Not started

---

## 4. Ablation Ladder

Before full bilevel training, we must validate each component in isolation:

| Step | Test | Success Criteria | Status |
|---|---|---|---|
| 1 | LOS-rate only, no prediction | Baseline success rate | ✅ Complete (Stage 6E) |
| 2 | GRU frozen, LOS-rate | Prediction improvement | ✅ Complete (Stage 6F) |
| 3 | PN only, no prediction | PN baseline | ✅ Complete (Stage 6G) |
| 4 | Hybrid only, no prediction | Hybrid baseline | ✅ Complete (Stage 6G) |
| 5 | LOS vs PN vs hybrid, no prediction | Guidance comparison | 🔄 In Progress (Stage 6G) |
| 6 | LOS vs PN vs hybrid, GRU frozen | Prediction + guidance | 🔄 In Progress (Stage 6G) |
| 7 | CEM optimize N only | Single parameter optimization | ⏳ Not started (6H.0) |
| 8 | CEM optimize N + omega | Two parameter optimization | ⏳ Not started (6H.0) |
| 9 | CEM optimize hybrid thresholds | Switching parameter optimization | ⏳ Not started (6H.0) |
| 10 | Alternating: policy → gain → policy | End-to-end bilevel | ⏳ Not started (6I.0) |

---

## 5. Score Contract (6H.0)

The CEM optimizer must produce a **score** that satisfies:

### 5.1 Determinism

```python
score1 = evaluate_gains(gains, seed=42)
score2 = evaluate_gains(gains, seed=42)
assert abs(score1 - score2) < 1e-6
```

### 5.2 Monotonicity

If configuration A has higher success rate and lower crash rate than B, then `score(A) > score(B)`.

### 5.3 Decomposability

```python
score = (
    success_rate * 100.0
    + mean_return * 0.1
    - stability_penalty * 10.0
)
```

Where:
- `success_rate`: Fraction of successful episodes
- `mean_return`: Mean episode return
- `stability_penalty`: Standard deviation of return over 5 episodes

### 5.4 Bounds

```python
0.0 <= score <= 100.0
```

---

## 6. Alternating Schedule (6I.0)

### 6.1 Example Schedule

```yaml
schedule:
  - phase: policy_train
    epochs: 20
    freeze: gains
    learning_rate: 3e-4
  - phase: gain_optimize
    epochs: 10
    freeze: policy
    cem_samples: 50
    cem_elite: 10
  - phase: policy_train
    epochs: 20
    freeze: gains
    learning_rate: 3e-4
  - phase: gain_optimize
    epochs: 10
    freeze: policy
    cem_samples: 50
    cem_elite: 10
```

### 6.2 Rollback Rules

```yaml
rollback:
  trigger: score_drop > 10% for 3 consecutive epochs
  action: revert to last checkpoint
  max_rollbacks: 3
  abort_after: 3 rollbacks
```

---

## 7. Release Checklist (7B)

### 7.1 Code

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No uncommitted changes (`git status --short` is empty)
- [ ] All configs frozen in `config/release/`
- [ ] All scripts documented with `--help`

### 7.2 Data

- [ ] `raw_episodes.csv` for all stages
- [ ] `scenario_method_summary.csv` for all stages
- [ ] `pairwise_mcnemar.csv` for all paired comparisons
- [ ] `figures/` with all plots

### 7.3 Documentation

- [ ] `README.md` with runnable commands and current evidence
- [ ] `docs/stage6g_guidance_limitation_probe.md` with probe documentation
- [ ] `docs/bilevel_final_roadmap.md` with gate definitions
- [ ] `paper_safe_claims.md` with all claim statuses

### 7.4 Environment

- [ ] `requirements.txt` with exact versions
- [ ] `environment.yml` (optional, for conda)
- [ ] Docker image (optional, for reproducibility)

### 7.5 Publication

- [ ] arXiv preprint submitted
- [ ] DOI obtained
- [ ] GitHub release tagged

---

*Last updated: 2026-06-05 | Stage 6G.1 in progress | Full probe executing*
