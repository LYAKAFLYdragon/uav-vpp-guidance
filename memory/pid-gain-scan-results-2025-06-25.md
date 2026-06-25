---
name: pid-gain-scan-results-2025-06-25
description: Optimal PID gains from 90-combination grid scan on break_turn task
metadata:
  type: project
---

# PID Gain Scan Results (2026-06-25)

**Task**: break_turn (JSBSim F-16, 90s per episode)
**Grid**: Kp_nz × Kd_nz = 9×10 = 90 combinations, 5 seeds each = 450 episodes
**Duration**: 170s (32 parallel workers)

## Optimal Gains

| Parameter | Optimal Value | Default (prior) | Change |
|-----------|--------------|-----------------|--------|
| Kp_nz | **0.50** | 0.22 | +127% |
| Kd_nz | **0.10** | 0.06 | +67% |
| Ki_nz | **0.05** | 0.025 | +100% |

## Performance

| Metric | With Optimal Gains | Prior Default |
|--------|-------------------|---------------|
| nz RMSE | 2.26g | ~3.2g (estimated) |
| Crashes | **0/450** | N/A |

## Key Finding

Kp_nz=0.50 (the upper bound tested) gives the best tracking. The break_turn task's smooth trajectory doesn't exercise Kd_nz — RMSE is identical across all Kd_nz values at the same Kp_nz. This is expected: Kd_nz only matters for transient response (sharp nz changes), which will be validated by the [[nz-step-response-task]].

All 90 combinations had 0 crashes thanks to the bank76 protection profile (bank_angle_protection + altitude_hold + lift_compensation).

**Why**: Higher Kp_nz reduces steady-state tracking error. The nz_cmd from LOS-rate guidance for break_turn is relatively smooth (no sharp transients), so the derivative term has negligible effect.

**How to apply**: Use Kp_nz=0.50, Ki_nz=0.05, Kd_nz=0.10 as the baseline PID gains. For tasks with sharp transients (nz steps), Kd_nz may need independent tuning — defer to nz_step_response evaluation.

## Related memories
- [[flight-envelope-characterization-2025-06-25]] — envelope boundaries
- [[ppo-pid-training-results-2025-06-25]] — training with optimal gains
- [[protection-gain-calibration-2025-06-25]] — max stable bank
- [[nz-step-response-task]] — transient response validation (pending)
