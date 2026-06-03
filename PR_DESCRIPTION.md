# PR: Deep Numerical Hardening of LOS-Rate Guidance + True 3D PN + Hybrid Guidance

## Summary

This PR delivers four tightly coupled improvements to the guidance subsystem:

1. **Deep numerical hardening** of the existing geometric LOS-rate guidance law.
2. **Command Post-Processor** (`overload_rollrate.py`) for final saturation, terminal-phase protection, load-roll coordination, and energy compensation.
3. **True 3D Proportional Navigation** guidance with filtered LOS-rate estimation.
4. **Hybrid Guidance** with hysteresis/dwell-time anti-chatter, supporting range-based, energy-based, and blended switching modes.

All three guidance laws share the identical `compute_command(own_state, target_state, virtual_point, gains)` interface and integrate the same capture-radius blending and NaN/Inf defense.

---

## 1. Numerical Problems Solved

### Distance Singularity (`d → 0`)

**Before**: `if distance > 1e-6:` branch jump. When `d` crossed the threshold, `los_heading` fell back to `own_heading`, causing a 180° flip in heading error and violent roll reversals.

**After**: `d_safe = max(distance, EPS)` with `EPS = 1e-9`. No branch jump. When `d < capture_radius_m` (default 50 m), commands are smoothly blended to safe hold (`nz → base_nz`, `roll_rate → 0`).

### Arcsin Endpoint Singularity (`θ_los → ±90°`)

**Before**: `np.arcsin(np.clip(r_z / d, -1.0, 1.0))`. Near `±1.0`, floating-point overshoot caused `NaN` or violent `nz_cmd` oscillations.

**After**: `np.arctan2(r_z, d_horiz)`. No endpoint singularity; numerically stable across the full `[-π, π]` range.

### Angle Wrapping Instability

**Before**: `_normalize_angle` used `angle % (2π)`. After many cumulative turns (e.g. 1000π), floating-point drift caused `±π` sign flips.

**After**: `_stable_angle_diff` uses `np.arctan2(np.sin(δ), np.cos(δ))`. Robust to arbitrary cumulative angles and naturally maps to `[-π, π]`.

### Missing Internal Guards

**Before**: `compute_command` returned raw values. NaN/Inf propagated into JSBSim actuators, causing simulator crashes.

**After**: Internal clipping (configurable `enable_internal_clip`), optional first-order filtering (`enable_internal_filter`), and a NaN/Inf fallback that instantly returns safe hold commands.

---

## 2. Capture Radius Blending Strategy

When the UAV enters the capture sphere (`d < capture_radius_m`):

```
capture_ratio = d / capture_radius_m   # ∈ [0, 1)
roll_rate_cmd *= capture_ratio
nz_cmd = (1 - capture_ratio) * base_nz + capture_ratio * nz_cmd
```

At `d = 0`, the output is exactly safe hold: `nz = base_nz` (typically 1.0 g), `roll_rate = 0`, throttle on speed-hold.

### Trade-offs

| Capture Radius | Pros | Cons |
|---|---|---|
| Large (e.g. 100 m) | Very smooth terminal behavior, low exceedance | May delay fine-positioning, feels sluggish |
| Small (e.g. 20 m) | Responsive near target | Higher numerical risk, more command variance |
| Default (50 m) | Balanced | Tuned for F-16 simple backend |

The radius is configurable via `guidance.params.capture_radius_m`.

---

## 3. Backward Compatibility

- **Action space unchanged**: All guidance laws output `{"nz_cmd", "roll_rate_cmd", "throttle_cmd"}`.
- **Config defaults**: `mode: los_rate` remains the default. Existing `config/experiment/*.yaml` files work without modification.
- **tracking_env.py**: Automatically selects the guidance law from `config["guidance"]["mode"]`; falls back to `los_rate` if unspecified.
- **Existing tests**: All 258 pre-existing tests pass. New tests (48 total for PN, Hybrid, PostProcessor) are additive only.
- **Console scripts**: Entry points (`uav-vpp-train-*`, `uav-vpp-eval-*`) are unaffected.

---

## 4. New Components

### `overload_rollrate.py` → `CommandPostProcessor`

Replaces the `NotImplementedError` stub with a fully functional post-processing layer:

- **Saturation**: clips to `config.limits`.
- **Terminal-phase protection**: scales down aggressive commands when `range < terminal_range_m`.
- **Load-roll coordination**: reduces `roll_rate_cmd` when `nz_cmd` nears its limit.
- **Energy compensation**: boosts throttle when high g-load or low speed is detected.

### `proportional_navigation.py`

True 3D PN with:
- LOS-rate estimation via filtered numerical differentiation (`alpha_filter`).
- Navigation constant `N` (configurable, default 3.0–4.0).
- Decomposition into `nz_cmd` (vertical) and `roll_rate_cmd` (horizontal turn).
- Same capture-radius blending and NaN/Inf defense as geometric mode.

### `hybrid_guidance.py`

Three switching strategies:
- **Range mode**: PN for long range, geometric for short range.
- **Energy mode**: switches to geometric when speed drops below threshold.
- **Blended mode**: continuous linear interpolation across a transition zone.

Anti-chatter protection:
- `hysteresis_m` / `energy_speed_hysteresis_mps`: deadband prevents rapid switching.
- `min_dwell_steps`: minimum dwell time in a mode before switching again.

---

## 5. Verification

```bash
# Full test suite
pytest tests/ -v --ignore=tests/test_jsbsim_bridge.py --ignore=tests/test_jsbsim_env_p1.py
# 306 passed

# Lint
ruff check src/uav_vpp_guidance/guidance/ tests/
# All checks passed

# Format
black src/uav_vpp_guidance/guidance/ tests/
```

---

## 6. Files Changed

| File | Change |
|---|---|
| `src/uav_vpp_guidance/guidance/los_rate_guidance.py` | Deep numerical hardening (EPS, arctan2, stable angle diff, capture radius, NaN fallback) |
| `src/uav_vpp_guidance/guidance/overload_rollrate.py` | New `CommandPostProcessor` with saturation, terminal protection, load-roll coordination, energy compensation |
| `src/uav_vpp_guidance/guidance/proportional_navigation.py` | New True 3D PN guidance law |
| `src/uav_vpp_guidance/guidance/hybrid_guidance.py` | New Hybrid guidance with hysteresis and three modes |
| `src/uav_vpp_guidance/guidance/__init__.py` | Export new classes |
| `src/uav_vpp_guidance/guidance/energy_compensation.py` | Minor cleanup |
| `src/uav_vpp_guidance/envs/tracking_env.py` | Dynamic guidance mode selection from config |
| `src/uav_vpp_guidance/evaluation/run_stage6b_simple_benchmark.py` | Terminal-phase stability metrics (variance, exceedance, estimated miss distance) |
| `config/guidance.yaml` | New params: capture_radius, navigation_constant, hybrid_mode, post_process block |
| `README.md` | Guidance mode selection, capture radius explanation, post-processing setup |
| `docs/method_notes.md` | Comparison table: Geometric vs PN vs Hybrid; numerical hardening notes |
| `docs/stage6b_simple_prediction_benchmark.md` | Document guidance mode benchmark support |
| `tests/test_los_guidance.py` | 14 new edge-case stability tests |
| `tests/test_overload_rollrate.py` | New tests for CommandPostProcessor |
| `tests/test_proportional_navigation.py` | New tests for True 3D PN |
| `tests/test_hybrid_guidance.py` | New tests for Hybrid guidance |
