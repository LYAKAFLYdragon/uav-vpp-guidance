---
name: sustained-turn-energy-management
description: Why sustained turning causes altitude/speed loss and mitigation strategies
metadata:
  type: project
---

# Sustained Turn Energy Management

**Date**: 2026-06-25
**Scope**: Both Enhanced PID and PPO+PID controllers

## The Problem

Extended sustained turning (>60s) inevitably leads to energy loss: either altitude decay or speed decay. This is a fundamental aerodynamic constraint, not a controller bug.

### Enhanced PID (bank76 profile, 1024-step ST)
- **Failure mode**: Altitude violation (crash)
- **Speed at crash**: ~260 m/s (healthy — speed is maintained)
- **Orbits completed**: ~1.0
- **Mechanism**: Controller prioritizes speed maintenance over altitude. As energy bleeds, altitude drops below 500m limit.

### PPO+PID (200k-trained, 1024-step ST)
- **Failure mode**: Speed stall (149.7 m/s < 150 m/s limit)
- **Max nz**: 10.9g (F-16 structural exceedance!)
- **Orbits completed**: ~2.44
- **Mechanism**: PPO pulls excessive nz to maintain tight orbit → high induced drag → speed decays to stall. PPO trades speed for tracking accuracy because the reward function doesn't penalize energy loss.

## Root Cause

The physics of sustained turning:
```
Required lift = nz × weight
Induced drag ∝ (lift)² / (air density × speed²)
```

To maintain a constant-radius turn at constant speed:
- nz must be sufficient to produce centripetal acceleration
- Engine thrust must overcome induced drag
- At high nz (>4g), induced drag grows quadratically, exceeding available thrust

The F-16 at 250 m/s and 1200m radius needs ~5.3g. This produces induced drag that the engine can only overcome temporarily — energy bleeding is inevitable.

## Why This Wasn't Caught Earlier

- The original sustained_turn task used 450 steps (90s), which only allows ~1 orbit.
- Extended to 1024 steps (204.8s), the energy loss becomes apparent after ~1.5 orbits.
- The acceptance criterion of "mean_orbits ≥ 1.5" is achievable but "0 crashes" is NOT with current controller design.

## Mitigation Strategies

1. **Altitude hold gain tuning**: Current `altitude_hold_gain=0.01` may be too weak. Increasing to 0.02-0.03 could help Enhanced PID maintain altitude longer.

2. **Energy-aware reward for PPO**: Add a speed-preservation term to the reward function: `+w_energy × (current_speed / initial_speed)`.

3. **Larger orbit radius**: A larger orbit requires less nz → less induced drag → more sustainable. Current orbit_radius=1200m. Increasing to 1800-2000m reduces required nz.

4. **Accept the limitation**: Real F-16s cannot sustain max-performance turns indefinitely. The controller correctly reflects this physical constraint. For evaluation, use "mean_orbits ≥ 1.5 with 0 crashes before energy exhaustion" rather than "0 crashes at 204.8s".

## Related memories
- [[flight-envelope-characterization-2025-06-25]] — envelope shows altitude limitation
- [[ppo-pid-training-results-2025-06-25]] — PPO energy management behavior
- [[pid-gain-scan-results-2025-06-25]] — PID gains used
