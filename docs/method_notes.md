# Method Notes

## Virtual Pursuit Point (VPP)

The policy outputs a 5-dimensional normalized action vector, mapped to:
- Longitudinal offset `d_long`
- Lateral offset `d_lat`
- Vertical offset `d_vert`
- Prediction time `tau_pred`
- Speed bias `speed_bias`

These parameters define a virtual point relative to the target aircraft, which the own aircraft attempts to track.

## Guidance Laws

### LOS-Rate Guidance (Geometric)

The default guidance law computes:
- Normal overload command `nz_cmd` from LOS elevation and distance scaling.
- Roll-rate command `roll_rate_cmd` from heading error and current roll damping.
- Throttle command from speed-error feedback.

Strengths: intuitive geometric interpretation, good for midcourse pursuit.
Weaknesses: can produce high command variance in terminal phase when distance → 0.

### Proportional Navigation (True 3D PN)

Classical PN guidance with LOS-rate estimation via filtered numerical differentiation:
- `a_cmd = N * Vc * d(lambda)/dt` perpendicular to the LOS.
- Decomposed into `nz_cmd` (vertical acceleration) and `roll_rate_cmd` (heading turn).

Strengths: theoretically optimal for intercept, smoother terminal-phase commands.
Weaknesses: requires LOS-rate filtering; sensitive to measurement noise.

### Hybrid Guidance

Switches or blends between geometric LOS-rate and PN based on engagement conditions:
- **Range mode**: pure PN for long range (> threshold), pure LOS for short range.
- **Energy mode**: switches to LOS when speed drops below threshold (energy protection).
- **Blended mode**: continuous linear interpolation across a transition zone.

Recommended for robustness: leverages PN efficiency in midcourse and geometric precision in terminal phase.

### Command Post-Processor

Optional final processing layer (enabled via `guidance.post_process.enabled`):
- **Terminal-phase protection**: scales down aggressive commands when range < threshold.
- **Load-roll coordination**: reduces roll rate when `nz_cmd` nears its limit.
- **Energy compensation**: boosts throttle when high g-load or low speed is detected.
- **Saturation**: clips all commands to configured limits.

## Strategy-Gain Bilevel Optimization

Outer loop: optimize guidance gains via CEM using regret.
Inner loop: train VPP policy with fixed gains via PPO.

Alternation continues until convergence or budget exhaustion.

## Terminal-Phase Behavior Comparison

| Aspect | Geometric LOS-Rate | True PN | Hybrid (Blended) |
|--------|-------------------|---------|------------------|
| Midcourse efficiency | Moderate | High | High (PN dominant) |
| Terminal smoothness | Can oscillate | Smoother | Smoother (LOS damping) |
| Energy awareness | No | No | Yes (energy mode) |
| Limit exceedance | Higher near capture | Lower | Lowest |
| Tuning complexity | Low | Medium (filter alpha, N) | Medium |
