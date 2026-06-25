---
name: protection-gain-calibration-2025-06-25
description: Protection gain calibration results — all 192 combos stable, max bank=82°
metadata:
  type: project
---

# Protection Gain Calibration (2026-06-25)

**Task**: multi_waypoint tracking (JSBSim F-16)
**Grid**: 4 bank angles × 4 nz_increments × 3 nz_increment_max × 4 altitude_hold_gains = 192 combinations × 5 seeds = 960 episodes
**Duration**: 58s (32 parallel workers)

## Key Result

**ALL 192 combinations are stable** — the bank76 protection framework is robust across a wide range of parameter settings.

| Bank | Best nz_inc | Best nz_inc_max | Best alt_gain | mean_wp | Crashes |
|------|------------|-----------------|---------------|---------|---------|
| 76° | 0.5 | any | 0.01 | 4.80 | 0 |
| 78° | 0.5 | any | 0.01 | 4.80 | 0 |
| 80° | 0.3 | any | 0.01 | 4.00 | 0 |
| 82° | 0.5 | 1.0 | 0.01 | 4.00 | 0 |

## Best Stable Config

```yaml
max_bank_rad: 1.4312  # 82°
bank_protection_nz_increment: 0.5
bank_protection_nz_increment_max: 1.0
altitude_hold_gain: 0.01
```

- Usable bank range: **60°-82°** (22° total)
- Default recommendation: **76°** (most extensively validated with 20-seed formal evals)

**Why**: Bank angle protection with even moderate nz_increment (0.3-0.5) effectively prevents spiral-dive instability. The protection is NOT sensitive to exact parameter values — it works across the full tested range.

**How to apply**: For conservative operations, use bank=76°. For aggressive maneuvering, bank up to 82° is safe with the calibrated protection parameters. Beyond 82° was not tested but may work.

## Related memories
- [[flight-envelope-characterization-2025-06-25]] — full envelope characterization
- [[pid-gain-scan-results-2025-06-25]] — optimal PID gains
- [[2026-06-23-flight-control-archive]] — prior bank76 validation
