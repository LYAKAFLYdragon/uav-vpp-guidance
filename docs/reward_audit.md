# Reward Design Audit

> **Scope**: `src/uav_vpp_guidance/envs/reward.py` and the reward blocks in
> `config/canonical/reward.yaml`, `config/reward.yaml`, and
> `config/experiment/train_no_prediction_vpp_ppo.yaml`.
>
> **Conclusion**: The implemented reward is a **dense weighted mixture plus
> terminal sparse events**. Earlier reports describing an "extreme sparse reward"
> are obsolete and should be corrected.

---

## 1. Per-Step Dense Terms

| Term | Weight (canonical) | Weight (base `config/reward.yaml`) | Type | Unit | Potential? | Description |
|------|-------------------:|-----------------------------------:|------|------|:----------:|-------------|
| `w_range` | 0.6 | 0.5 | Dense | none | No | Encourage range inside `[ideal_range_min, ideal_range_max]` |
| `w_angle` | 0.9 | 0.8 | Dense | none | No | Penalize large ATA + AA (in degrees) |
| `w_energy` | 0.0 | 0.2 | Dense | none | No | Reserved; currently zero in canonical config |
| `w_safety` | 3.0 | 2.0 | Dense | none | No | Penalize altitude near `min_altitude_m` |
| `w_saturation` | 0.5 | 1.0 | Dense | none | No | Penalize `nz_cmd > 6.5` and `roll_rate_cmd > 1.4` |
| `w_smooth` | 0.2 | 0.1 | Dense | none | No | Penalize command deltas step-to-step |
| `w_turn_rate` | 0.5 | 0.5 | Dense | none | No | Penalize kinematically infeasible heading rates |
| `w_closing` | 0.1 | 0.0 | Dense | none | No | Reward negative range rate (closing) |
| `w_alive` | 0.02 | 0.0 | Dense | none | No | Small per-step survival bonus |
| `w_overshoot` | 0.0 | 0.0 | Dense | none | No | Penalize closing when already inside ideal range |
| `w_boundary` | 0.0 | 0.0 | Dense | none | No | Penalize extreme range / altitude |

## 2. Terminal Sparse Events

| Event | Value (canonical) | Value (base `config/reward.yaml`) | Type | Injected by env? |
|-------|------------------:|----------------------------------:|------|-----------------:|
| `terminal_success` | +400.0 | +200.0 | Sparse | Yes |
| `terminal_failure` | −300.0 | −200.0 | Sparse | Yes |
| `terminal_crash` | −600.0 | −300.0 | Sparse | Yes |

## 3. Potential-Based Shaping (optional ablation)

| Parameter | Canonical | Base config | Status |
|-----------|----------:|------------:|--------|
| `enabled` | `false` | `true` | Disabled in canonical experiments |
| `C` | 0.001 | 0.001 | Constant for distance-gradient shaping |
| `gamma` | 0.99 | 0.99 | Discount used in shaping difference |

The PBS term is **not required** for early-training survival; the dense mixture
already provides sufficient signal. PBS remains available as an ablation
("PBS ablation" in the Phase 3 experiment matrix).

## 4. Recommended Experiments

- **A2' reward ablation**: dense-only vs. dense + PBS vs. terminal-only.
- If dense-only and dense + PBS are statistically equivalent, declare PBS
  redundant in the paper and disable it in canonical configs.

### 4.1 A2' Result (executed — Phase 2) **[FROZEN]**

The A2' ablation has been run on the **simple backend** with **standard PPO**
and the **canonical configuration** (`config/canonical/`), 3 seeds per
condition. All three reward designs are derived solely by toggling existing
canonical reward parameters; no new defaults were introduced.

- `dense_only` (= Baseline): canonical dense mixture + terminal sparse events,
  PBS disabled.
- `dense_pbs`: dense_only + `potential_based_shaping.enabled = true`.
- `terminal_only`: all canonical dense weights set to `0.0`, terminal sparse
  events retained, PBS disabled.

| Condition | Seeds | Final SR (mean ± std) |
|-----------|------:|-----------------------|
| Dense-only (Baseline) | 3 | 25.6% ± 11.7% |
| Dense + PBS | 3 | 28.9% ± 7.7% |
| Terminal-only | 3 | 30.0% ± 3.3% |

Pairwise significance on per-seed final success rate (paired by seed):

| Comparison | Δ SR | p (paired t) | Cohen's d | Mann–Whitney p | Significant @0.05 |
|------------|-----:|-------------:|-----------|---------------:|:-----------------:|
| Dense-only vs Dense + PBS | +3.3% | 0.785 | 0.18 (negligible) | 1.000 | No |
| Dense-only vs Terminal-only | +4.4% | 0.625 | 0.33 (small) | 0.825 | No |
| Dense + PBS vs Terminal-only | +1.1% | 0.742 | 0.22 (small) | 1.000 | No |

**Verdict — PBS is REDUNDANT.** Dense-only and Dense + PBS are not
statistically distinguishable (p = 0.785, negligible effect size). PBS does not
improve the final success rate, confirming the Section 3 expectation that the
dense mixture already provides sufficient learning signal. PBS therefore remains
**disabled** in the canonical configuration and should be reported in the paper
as an optional, non-essential enhancement rather than a necessity.

> **Caveat (honest scope)**: With only 3 seeds the test is low-powered, so this
> establishes the *absence of a detectable benefit* from PBS, not strict
> statistical equivalence. The conclusion is robust to the direction of the
> claim: no condition beats the dense-only baseline at the 0.05 level. The
> dominant failure mode across all conditions is out-of-bounds (~65–87%),
> which is a guidance/geometry limitation independent of the reward design and
> is tracked separately under Stage 6G/6H.

Artifacts: `outputs/ablation_reward_design/` (`summary.md`,
`ablation_results.json`, per-condition resolved configs under `configs/`, and
per-seed training/eval logs). Reproduce with:

```bash
python scripts/run_a2_reward_ablation.py --seeds 3 --device cpu
```
