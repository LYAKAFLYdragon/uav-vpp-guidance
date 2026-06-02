# Method Notes

## Virtual Pursuit Point (VPP)

The policy outputs a 5-dimensional normalized action vector, mapped to:
- Longitudinal offset `d_long`
- Lateral offset `d_lat`
- Vertical offset `d_vert`
- Prediction time `tau_pred`
- Speed bias `speed_bias`

These parameters define a virtual point relative to the target aircraft, which the own aircraft attempts to track.

## LOS-Rate Guidance

The guidance law computes:
- Normal overload command `nz_cmd` from LOS rate and closing velocity.
- Roll-rate command `roll_rate_cmd` from heading error.
- Throttle command from energy compensation (optional).

## Strategy-Gain Bilevel Optimization

Outer loop: optimize guidance gains via CEM using regret.
Inner loop: train VPP policy with fixed gains via PPO.

Alternation continues until convergence or budget exhaustion.
