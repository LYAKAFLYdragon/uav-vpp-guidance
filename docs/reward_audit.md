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
