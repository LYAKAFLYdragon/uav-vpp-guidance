# Method Innovation Branch Validation Report

**Date:** 2026-06-13  
**Branches validated:** `CL_CRPPO_CEMGD` (Track 1) and `Intentional_Updates` (Track 2)  
**Merged comparison base:** `CL_CRPPO_CEMGD` with `Intentional_Updates` merged in  
**Environment:** `simple` backend, CPU  
**Seeds:** 5  
**Steps per run:** 50,000  
**Main configs:** `config/method_innovation_comparison.yaml`, `config/method_innovation_comparison_hard.yaml`

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

### Scripts created / modified

- `scripts/train_curriculum_ppo.py` — unified to support `ppo`, `cr_ppo`, and `intentional_ppo`.
- `scripts/run_method_innovation_comparison.py` — runs the full comparison matrix.
- `scripts/aggregate_method_innovation_comparison.py` — produces tables, CSV, t-tests, learning curves, and stability plots.
- `config/method_innovation_comparison.yaml` — standard unified comparison configuration.
- `config/method_innovation_comparison_hard.yaml` — harder config with sinusoidal weaving target and six diverse scenarios.

### Fix applied to Intentional PPO

The original `IntentionalPPOAgent.update()` only performed a single pass over the rollout data, unlike `PPOAgent`/`CRPPOAgent` which respect `update_epochs=10`. We added the outer `for epoch in range(self.update_epochs):` loop with per-epoch index reshuffling. This change is included in the re-run below.

---

## 2. Standard-Scenario Results (original 4 scenarios)

### 2.1 Final performance (mean ± std over 5 seeds)

| Algorithm | Return | Success Rate | Crash Rate | OOB Rate | Timeout Rate | Steps to 50% SR |
|-----------|--------|--------------|------------|----------|--------------|-----------------|
| Baseline PPO | -313.2±1.2 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| CR-PPO | -312.4±1.9 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO | -313.3±1.5 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO-C | -313.1±0.5 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO-A | -312.0±0.6 | 36.67%±0.00% | 13.33%±0.00% | 50.00%±0.00% | 0.00%±0.00% | 49,152±0 |

### 2.2 Stability metrics

| Algorithm | Policy Loss | Value Loss | Approx KL | Clip Fraction | Explained Var |
|-----------|-------------|------------|-----------|---------------|---------------|
| Baseline PPO | -0.0034±0.0015 | 2089.9±368.5 | 0.0053±0.0020 | 6.31%±3.50% | 0.0871±0.0670 |
| CR-PPO | -0.0033±0.0014 | 2091.9±364.7 | 0.0051±0.0019 | 6.05%±3.30% | 0.0873±0.0663 |
| Intentional PPO | -0.0001±0.0014 | 2115.7±361.6 | 0.0006±0.0008 | 0.30%±1.01% | 0.0715±0.0523 |
| Intentional PPO-C | 0.0001±0.0020 | 2120.7±360.3 | 0.0010±0.0010 | 0.63%±1.14% | 0.0655±0.0464 |
| Intentional PPO-A | -0.0000±0.0022 | 2101.2±366.3 | 0.0003±0.0006 | 0.18%±0.63% | 0.0799±0.0622 |

**Observation:** With the single-epoch implementation, Intentional PPO variants show dramatically lower KL/clip fraction, indicating the intentional scalars are active, but final success rates are identical.

---

## 3. Harder-Scenario Results (6 weaving scenarios, after update_epochs fix)

JSBSim was not available in this environment (no `JSBSIM_ROOT`), so we created a harder `simple`-backend benchmark: `target_mode: sinusoidal`, six scenarios covering tail-chase, head-on, close/far crossing, disadvantage, and weaving pursuit, with 5 eval seeds × 20 eval episodes.

### 3.1 Final performance (mean ± std over 5 seeds)

| Algorithm | Return | Success Rate | Crash Rate | OOB Rate | Timeout Rate | Steps to 50% SR |
|-----------|--------|--------------|------------|----------|--------------|-----------------|
| Baseline PPO | -244.7±0.6 | 46.00%±0.00% | 32.00%±0.00% | 22.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| CR-PPO | -244.9±0.8 | 46.00%±0.00% | 32.00%±0.00% | 22.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO | -245.8±0.7 | 46.00%±0.00% | 32.00%±0.00% | 22.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO-C | -244.3±0.7 | 46.00%±0.00% | 32.00%±0.00% | 22.00%±0.00% | 0.00%±0.00% | 49,152±0 |
| Intentional PPO-A | -244.9±0.7 | 46.00%±0.00% | 32.00%±0.00% | 22.00%±0.00% | 0.00%±0.00% | 49,152%±0.00% | 49,152±0 |

### 3.2 Stability metrics

| Algorithm | Policy Loss | Value Loss | Approx KL | Clip Fraction | Explained Var |
|-----------|-------------|------------|-----------|---------------|---------------|
| Baseline PPO | -0.0035±0.0021 | 2134.0±389.7 | 0.0049±0.0015 | 5.76%±2.60% | 0.0831±0.0633 |
| CR-PPO | -0.0034±0.0021 | 2133.0±390.4 | 0.0049±0.0014 | 5.76%±2.39% | 0.0853±0.0646 |
| Intentional PPO | 0.0078±0.0251 | 1909.4±461.6 | 0.0436±0.0938 | 19.24%±18.11% | 0.2083±0.1220 |
| Intentional PPO-C | -0.0035±0.0022 | 2001.4±416.9 | 0.0059±0.0017 | 7.30%±2.90% | 0.1604±0.0899 |
| Intentional PPO-A | -0.0008±0.0062 | 2133.2±387.0 | 0.0091±0.0152 | 7.71%±8.02% | 0.0870±0.0669 |

**Observation:** After adding `update_epochs`, the full Intentional PPO now exhibits **higher** KL and clip fraction on average, with larger cross-run variance. ICU-only (Intentional-C) modestly lowers value loss, while IAU-only (Intentional-A) is closer to baseline. Success rates remain identical across all variants.

### 3.3 Per-scenario success rate (seed 0, final evaluation)

| Scenario | Baseline | CR-PPO | Intentional | Intentional-C | Intentional-A |
|----------|----------|--------|-------------|---------------|---------------|
| crossing_close | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| crossing_far | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| disadvantage | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| head_on | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| tail_chase | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| weaving_pursuit | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Again, all algorithms solve and fail exactly the same scenarios.

---

## 4. Why the Performance Metrics Still Do Not Diverge

Even after fixing the `update_epochs` bug and increasing scenario difficulty, final success rates are identical. Contributing factors:

1. **Deterministic evaluation + deterministic policies + binary metric.** With fixed eval seeds and a hard success threshold, different policies frequently produce the same discrete outcome, collapsing cross-seed variance to zero.

2. **Per-scenario success rates are at 0% or 100% plateaus.** Each scenario is either solved by all methods or by none, so aggregate success rate cannot differ.

3. **Flat learning curves.** Performance asymptotes within the first 2,048 steps; additional training and algorithmic changes do not push scenarios off their 0%/100% plateaus.

4. **Simple backend lacks dynamics diversity.** The simplified flight model and rule-based target motion provide limited room for exploration regularizers or intentional update budgets to create divergent behavior.

5. **Intentional PPO scales are very small.** With default `eta_actor=0.01` and `eta_critic=0.1`, the effective per-update scalars are ~`1e-3`–`1e-2`. Combined with 10 epochs, the full ICU+IAU+CAIS configuration can swing between overly conservative and unexpectedly large steps (high variance seen in the hard benchmark).

---

## 5. Updated Tuning Recommendations

To obtain a decisive comparison, the following changes are needed:

### A. Evaluation

- Increase `eval_episodes` to 50+ and add at least 5–10 eval seeds.
- Report continuous metrics (mean final range, min range, final ATA, return distribution) alongside binary success rate.
- Add eval-time action noise or domain randomization to make the metric sensitive to small policy differences.

### B. Environment

- Run on **JSBSim** as soon as it is available; the simple backend is too constrained.
- Add stochastic target maneuvers (e.g., sinusoidal with random phase/frequency) instead of fixed weaving.
- Avoid 0%/100% per-scenario plateaus by adjusting initial geometries and success thresholds.

### C. Intentional PPO tuning

- **Sweep eta aggressively:** `eta_actor` ∈ `[1e-3, 1e-2, 1e-1, 1.0]`, `eta_critic` ∈ `[1e-2, 1e-1, 1.0, 10.0]`.
- **Consider removing CAIS initially** and compare ICU-only / IAU-only / ICU+IAU without phase coupling.
- **Monitor effective step size:** ensure `scale_actor` and `scale_critic` are in a sensible range (e.g., 0.1–10) rather than `1e-3`.

### D. CR-PPO tuning

- Sweep `complexity_coef` over `[1e-4, 1e-3, 1e-2, 1e-1]` and `cr_n_bins` over `[4, 8, 16]`.

---

## 6. Conclusions and Branch Recommendation

### Track 1: CR-PPO + Curriculum Learning + CEM-GD (`CL_CRPPO_CEMGD`)

- **Status:** Functionally correct, minimally invasive, and stable across both benchmarks.
- **Verdict:** **Safe but not clearly beneficial** on simple backend. No regression, no statistically significant gain.
- **Action:** Do **not** roll back. Merge is acceptable, but treat CR-PPO as an optional regularizer, not a default upgrade.

### Track 2: Intentional PPO + Combat-Aware Schedule (`Intentional_Updates`)

- **Status:** The `update_epochs` bug has been fixed. ICU/IAU scalars are active, but the full ICU+IAU+CAIS combination shows **higher update variance** on the harder benchmark. The individual ICU-only and IAU-only ablations are more stable.
- **Verdict:** Still **no performance advantage** on simple backend. The mechanism is interesting but not yet tuned or proven.
- **Action:** Do **not** make Intentional PPO the default. Keep it as an optional module. Before merging as default, run the eta sweep and JSBSim comparison recommended above.

### Overall recommendation

1. **Neither branch beats the baseline on these benchmarks.**
2. **Both branches can be preserved** as optional, well-encapsulated modules.
3. **Baseline PPO remains the default.**
4. **Next step:** Run the recommended eta sweep and a 5-seed 100k-step JSBSim comparison with expanded, continuous evaluation metrics.

---

## 7. Artifacts

- Standard summary: `outputs/method_innovation_compare/summary.md`
- Standard CSV/plots: `outputs/method_innovation_compare/summary.csv`, `learning_curves.png`, `stability_bars.png`
- Harder summary: `outputs/method_innovation_compare_hard/summary.md`
- Harder CSV/plots: `outputs/method_innovation_compare_hard/summary.csv`, `learning_curves.png`, `stability_bars.png`
- Run logs: `outputs/method_innovation_compare_run.log`, `outputs/method_innovation_compare_hard_run.log`
- Tuning run log: `outputs/method_innovation_tuning_run.log`
