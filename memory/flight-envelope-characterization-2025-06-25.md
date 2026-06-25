---
name: flight-envelope-characterization-2025-06-25
description: Flight envelope boundaries for Enhanced PID controller (bank, nz, speed, altitude)
metadata:
  type: project
---

# Flight Envelope Characterization (2026-06-25)

**Platform**: Cloud server (96 vCPU, RTX 2080 Ti, JSBSim F-16)
**Controller**: Enhanced PID with bank76 protection profile
**Config**: Kp_nz=0.50, Ki_nz=0.05, Kd_nz=0.10, bank_protection=76°, altitude_hold_gain=0.01, lift_compensation=enabled

## Bank Envelope (multi_waypoint task)

**ALL bank angles 60°-82° are STABLE** (0/5 crashes, mean_wp ≥ 4.0):

| Bank | Crashes | mean_wp | Stability |
|------|---------|---------|-----------|
| 60°-82° | 0/5 | 4.0-5.0 | stable |

Usable bank range: **22°** (60°-82°). Protection system works across full tested range.

**Why**: Bank angle protection with nz_increment=0.5 prevents spiral-dive crashes. Combined with altitude hold and lift compensation, the controller maintains stable flight even at extreme bank angles.

**How to apply**: When designing upper-level air combat maneuvers, bank angles up to 82° are safe for the Enhanced PID. For safety margin, recommend operational limit of 76° (validated with 20-seed formal eval).

## NZ Envelope (multi_waypoint task)

**ALL nz values 2g-7g are STABLE** (0/5 crashes, mean_wp=4.0):

| NZ | Crashes | mean_wp | Stability |
|----|---------|---------|-----------|
| 2g-7g | 0/5 | 4.0 | stable |

**Why**: The Enhanced PID controller handles the full nz range. nz_max=7.0 is a config limit, not a structural one — the F-16 can pull more.

## Speed × Altitude Envelope (sustained_turn task, 1024-step)

**Only 6/25 points stable — high altitude is a CRITICAL failure mode**:

```
        2000m   4000m   6000m   8000m   10000m
120m/s   ✅      ✅      ❌      ❌      ❌
170m/s   ✅      ✅      ❌      ❌      ❌
220m/s   ⚠️      ✅      ❌      ❌      ❌
270m/s   ❌      ⚠️      ❌      ❌      ❌
320m/s   ⚠️      ✅      ❌      ❌      ❌
```

✅ = stable (0 crash) | ⚠️ = marginal (1-4 crashes) | ❌ = unstable (5/5 crashes)

**Key finding**: Above 6000m, ALL speeds crash (100%). The F-16 engine thrust degrades at altitude, unable to sustain the energy required for continuous turning. The operational envelope is **2000-4000m at 120-220 m/s**.

**Why**: Sustained turning requires continuous lift (≈nz × weight). At high altitude, lower air density means higher AoA needed for the same lift, which increases drag. Combined with reduced engine thrust, the aircraft cannot maintain both speed and altitude — [[sustained-turn-energy-management]].

**How to apply**: Combat maneuvers requiring sustained turning should be planned below 5000m. High-altitude engagements should use boom-and-zoom (energy fighting) rather than sustained turning.

## Related memories
- [[pid-gain-scan-results-2025-06-25]] — optimal PID gains
- [[ppo-pid-training-results-2025-06-25]] — PPO+PID hybrid training
- [[protection-gain-calibration-2025-06-25]] — max stable bank angle
- [[sustained-turn-energy-management]] — energy management in sustained turns
- [[2026-06-23-flight-control-archive]] — prior flight control findings
