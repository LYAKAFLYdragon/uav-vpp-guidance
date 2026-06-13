# Method Innovation Branch Validation Report

**Date:** 2026-06-13  
**Branches validated:** `CL_CRPPO_CEMGD` (Track 1) and `Intentional_Updates` (Track 2)  
**Merged comparison base:** `CL_CRPPO_CEMGD` with `Intentional_Updates` merged in  
**Environment:** `simple` backend, CPU  
**Seeds:** 5  
**Steps per run:** 50,000  
**Config:** `config/method_innovation_comparison.yaml`

---

## 1. Experiment Design

We compare five algorithm variants on the same curriculum-learning setup:

| Key | Algorithm | Description |
|-----|-----------|-------------|
| `baseline` | Baseline PPO | Standard clipped PPO + entropy bonus |
| `cr_ppo` | CR-PPO | PPO with complexity regularization (entropy × disequilibrium) |
| `intentional` | Intentional PPO | ICU + IAU + CAIS all enabled |
| `intentional_c` | Intentional PPO-C | ICU only |
| `intentional_a` | Intentional PPO-A | IAU only |

All methods share the same network (`[128, 128]` tanh MLP), PPO hyperparameters, reward, curriculum, and scenarios.

### Scripts created

- `scripts/run_method_innovation_comparison.py` — runs the full comparison matrix.
- `scripts/aggregate_method_innovation_comparison.py` — produces tables, CSV, and plots.
- `config/method_innovation_comparison.yaml` — unified comparison configuration.

---

## 2. Results

### 2.1 Final performance (mean ± std over 5 seeds)

| Algorithm | Return | Success Rate | Crash Rate | OOB Rate | Timeout Rate | Steps to 50% SR |
|-----------|--------|--------------|------------|----------|--------------|-----------------|
| Baseline PPO | -313.2±1.2 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| CR-PPO | -312.4±1.9 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO | -313.3±1.5 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO-C | -313.1±0.5 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO-A | -312.0±0.6 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |

**Observation:** Final success rate, crash rate, out-of-bounds rate, and timeout rate are identical across all algorithms and all random seeds. No method reached the 50% success-rate threshold during training, so the "steps to 50% SR" metric is capped at the final step.

### 2.2 Stability metrics (training update logs, mean ± std over 5 seeds)

| Algorithm | Policy Loss | Value Loss | Approx KL | Clip Fraction | Explained Var |
|-----------|-------------|------------|-----------|---------------|---------------|
| Baseline PPO | -0.0034±0.0015 | 2089.9±368.5 | 0.0053±0.0020 | 6.31%±3.50% | 0.0871±0.0670 |
| CR-PPO | -0.0033±0.0014 | 2091.9±364.7 | 0.0051±0.0019 | 6.05%±3.30% | 0.0873±0.0663 |
| Intentional PPO | -0.0001±0.0014 | 2115.7±361.6 | 0.0006±0.0008 | 0.30%±1.01% | 0.0715±0.0523 |
| Intentional PPO-C | 0.0001±0.0020 | 2120.7±360.3 | 0.0010±0.0010 | 0.63%±1.14% | 0.0655±0.0464 |
| Intentional PPO-A | -0.0000±0.0022 | 2101.2±366.3 | 0.0003±0.0006 | 0.18%±0.63% | 0.0799±0.0622 |

**Observation:** Intentional-update variants dramatically reduce `approx_kl` and `clip_fraction` compared with baseline/CR-PPO, confirming that ICU/IAU are actively scaling the update magnitudes. However, this improved update stability does not translate into different final success rates in this setting.

### 2.3 Method-specific diagnostics

| Algorithm | Extra Metrics |
|-----------|---------------|
| CR-PPO | complexity = 0.0726 ± 0.0027 |
| Intentional PPO | scale_actor = 0.0005±0.0003, scale_critic = 0.0021±0.0015, ema_abs_adv = 0.9516±0.0263 |
| Intentional PPO-C | scale_critic = 0.0025±0.0018, ema_abs_adv = 1.0000±0.0000 |
| Intentional PPO-A | scale_actor = 0.0005±0.0004, ema_abs_adv = 0.9516±0.0263 |

### 2.4 Per-scenario success rate (seed 0, final evaluation)

| Scenario | Baseline | CR-PPO | Intentional | Intentional-C | Intentional-A |
|----------|----------|--------|-------------|---------------|---------------|
| favorable | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neutral | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| disadvantage | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| challenging | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

All algorithms solve exactly the same two scenarios (`neutral`, `challenging`) and fail the same two (`favorable`, `disadvantage`). The identical discrete outcomes drive the zero cross-seed variance in success rate.

---

## 3. Why the Performance Metrics Do Not Diverge

The main experiment and an additional 3-seed tuning run (20k steps, 10× `complexity_coef` and 10× `eta_actor`/`eta_critic`) both show the same 36.67% final success rate. We attribute the lack of performance differentiation to the following factors:

1. **Discrete, low-variance evaluation metric.** Final success rate is a hard threshold over only 30 evaluation episodes (3 fixed seeds × 10 episodes). With deterministic eval scenarios and deterministic policies, different policies often produce the same binary outcome, collapsing variance to zero.

2. **Scenario distribution creates a flat ceiling.** The four scenarios reduce to two easy and two hard cases. All methods reach the same per-scenario success pattern, so the aggregate success rate cannot differ.

3. **Task is too easy/hard for the chosen horizon.** The learning curves are flat from the first evaluation (2,048 steps), indicating that the policies reach their asymptotic behavior very quickly. Either the task is already nearly solved for two scenarios and unsolvable for the other two within 512 steps, or the policy network quickly finds the same dominant strategy.

4. **Simple backend has reduced dynamics diversity.** The simplified flight model and constant-velocity target offer fewer degrees of freedom in which exploration regularization (CR-PPO) or intentional update budgets (Intentional PPO) can create meaningful behavioral differences.

5. **Intentional PPO implementation uses only one update epoch.** Unlike baseline/CR-PPO, which perform `update_epochs=10` passes over the rollout buffer, the current `IntentionalPPOAgent.update()` iterates over the data only once. This is a substantial implementation difference that may under-train the policy and mask any benefit from intentional scaling.

---

## 4. Tuning Recommendations

To make the comparison more discriminative, try the following before declaring a winner:

### A. Make the evaluation more informative

- Increase `eval_episodes` to 30–50 and add more eval seeds.
- Report continuous metrics: mean final range, minimum range, final ATA, and episode return distribution.
- Use a stratified per-scenario success rate with confidence intervals rather than a single aggregate binary rate.

### B. Increase environment difficulty / diversity

- Add more scenarios, e.g., varying initial headings, speeds, altitudes, and target maneuvers.
- Use the `jsbsim` backend, which has richer dynamics and is the eventual target domain.
- Lengthen episodes or relax/tighten success thresholds to move performance off the 0%/100% per-scenario plateau.

### C. Algorithm-specific tuning

- **CR-PPO:** Sweep `complexity_coef` over `[1e-4, 1e-3, 1e-2, 1e-1]` and `cr_n_bins` over `[4, 8, 16]`.
- **Intentional PPO:**
  - Fix the single-epoch issue: add the outer `for epoch in range(self.update_epochs)` loop around the intentional minibatch loop.
  - Sweep `eta_actor` and `eta_critic` over a wider range (e.g., `[1e-3, 1e-2, 1e-1, 1.0]`).
  - Try disabling CAIS initially and only compare ICU vs. IAU vs. ICU+IAU to isolate the source of benefit.
- **Baseline:** Run a learning-rate sweep (`1e-4`, `3e-4`, `1e-3`) to ensure the baseline is well-tuned.

### D. Training length and curriculum

- Try 100k–200k steps; the flat learning curves may be a horizon issue.
- Adjust curriculum stage thresholds. The current gate (50% SR) is never met for `favorable`/`disadvantage`, so the curriculum stays open and may not provide useful staging.

---

## 5. Conclusions and Branch Recommendation

### Track 1: CR-PPO + Curriculum Learning + CEM-GD (`CL_CRPPO_CEMGD`)

- **Status:** Functionally correct and minimally invasive. CR-PPO plugs into the existing PPO update with a single loss-term change.
- **Verdict:** The branch is **safe but not clearly beneficial** under the current simple-backend benchmark. Keep the code, but CR-PPO should be treated as an optional regularizer until tested on JSBSim or a harder scenario set.
- **Action:** Do **not** roll back. Merge is acceptable once the smoke tests pass, but do **not** claim a performance win from this benchmark alone.

### Track 2: Intentional PPO + Combat-Aware Schedule (`Intentional_Updates`)

- **Status:** The intentional-update mechanism is working (lower KL/clip fraction prove the scalars are active), but the current implementation has two issues:
  1. It only performs one epoch per PPO update, unlike the baseline's 10 epochs.
  2. CAIS requires relative-state info to be passed through `store_transition`, which is now supported but adds coupling.
- **Verdict:** The branch shows **genuine training-stability differences** but no performance gain here. The missing update-epochs loop is likely a bug that should be fixed before any claim of superiority.
- **Action:** **Fix the single-epoch issue** and re-run the comparison. If performance still does not improve, consider merging only the ICU/IAU modules as optional utilities rather than the default algorithm.

### Overall recommendation

1. **Neither branch demonstrates a clear performance advantage on this benchmark.**
2. **Both branches are worth preserving** as optional modules because they are well-encapsulated and add no baseline regression.
3. **The baseline PPO remains the default.** Only switch the default after JSBSim-scale experiments with the tuning changes above show a statistically significant win.
4. **Immediate next step:** Fix Intentional PPO's update-epochs loop, then run a 5-seed 100k-step comparison on the JSBSIM backend with expanded eval episodes and continuous metrics.

---

## 6. Artifacts

- Main summary: `outputs/method_innovation_compare/summary.md`
- Main CSV: `outputs/method_innovation_compare/summary.csv`
- Learning curves: `outputs/method_innovation_compare/learning_curves.png`
- Stability bars: `outputs/method_innovation_compare/stability_bars.png`
- Run log: `outputs/method_innovation_compare_run.log`
- Tuning run log: `outputs/method_innovation_tuning_run.log`
