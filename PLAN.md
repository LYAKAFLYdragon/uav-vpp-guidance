# APIC-style PPO+PID Implementation Plan

## Problem
The current `PPO+PID` controller is not a true low-level flight controller: PPO outputs virtual-point offsets plus a single aggressiveness knob. The user wants an APIC-style implementation where **PPO directly outputs PID gain adjustments** while the guidance reference (waypoint / orbit tangent point) is fixed, making the comparison fair against fixed-gain Enhanced PID.

## Proposed Design

### 1. APIC Action Space
- 6-D normalized action in `[-1, 1]`:
  - `[delta_Kp_nz, delta_Ki_nz, delta_Kd_nz, delta_Kp_roll, delta_Ki_roll, delta_Kd_roll]`
- Effective gains computed multiplicatively around the fixed base gains:
  - `K_eff = K_base * (1 + range * delta)`
- Configurable per-gain range (e.g. ±30%) in YAML.

### 2. Low-Level Controller Extension
- Extend `EnhancedLowLevelController.compute_actuator` to accept an optional `pid_gain_deltas` dict/array.
- When provided, it overrides the fixed-gain / aggressiveness path for those six PID gains.
- Keep the existing interface backward-compatible so existing checkpoints and tests still work.

### 3. Environment Changes
- In `CloseRangeTrackingEnv._apply_action`, detect APIC mode (`low_level_controller.apic.enabled: true`).
- In APIC mode:
  - Use `guidance.direct_track_mode: true` and `virtual_point.mode: zero_offset` so the policy no longer steers the virtual point.
  - Extract the 6-D gain-delta vector from the action.
  - Pass it through `_step_jsbsim` / `_step_simple` to `compute_actuator`.
- Store last gain deltas in `info` for logging, similar to `aggressiveness`.

### 4. Training Script & Config
- New config: `config/experiment/train_apic_pid_jsbsim.yaml`
  - `policy.action_dim: 6`
  - `low_level_controller.apic.enabled: true`
  - `guidance.direct_track_mode: true`
  - `virtual_point.mode: zero_offset`
  - Scenarios covering multi_waypoint and sustained-turn geometries.
- New script: `src/uav_vpp_guidance/training/train_apic_pid.py`
  - Based on `train_ppo_pid.py`, but logs the 6 gain deltas instead of VPP offsets / aggressiveness.
  - Smoke mode for quick validation.

### 5. Evaluation
- Update `run_flight_control_comparison.py` / controller adapters to load APIC checkpoints.
- Run APIC training (smoke then short formal run) for both tasks.
- Compare against fixed-gain Enhanced PID and baseline PID using the existing aggregation/plotting pipeline.

## Expected Outcome
A fair low-level comparison: APIC PPO tunes PID gains, while Enhanced PID uses fixed gains. The result will tell us whether RL-based PID gain adaptation helps on these tracking/orbit tasks.

## Key Risks
- Training may be unstable because small gain changes can cause large control swings. Mitigation: tight gain ranges (±30%), multiplicative scaling, and anti-windup already present.
- Need sufficient training time; plan to start with smoke runs and a 200k-step formal run.
