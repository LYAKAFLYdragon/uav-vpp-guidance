# F01 proposed law specification — offline evidence only

**Status:** proposed for review; not approved for runtime use. The frozen
`LOSRateGuidance` remains the only runtime behavior. This note names an
offline candidate `vertical_los_rate_pn_with_bounded_geometric_bias` so that it
is not confused with the audited implementation.

## Frame, signals, and signs

All vectors use a local right-handed **NEU** frame: north `x` [m], east `y`
[m], and up `z` [m]. Let `r = p_vp - p_own` [m] and
`v = v_vp - v_own` [m/s]. Horizontal range is `rho = hypot(r_x, r_y)` [m],
range is `R = ||r||` [m], elevation is
`lambda = atan2(r_z, rho)` [rad], and its vertical-plane LOS rate is

```text
lambda_dot = (rho * v_z - r_z * rho_dot) / R^2 [rad/s]
rho_dot    = (r_x v_x + r_y v_y) / rho       [m/s].
```

Positive elevation, LOS rate, and normal-load contribution command an upward
turn in this proposed convention. `range_rate = r·v/R`; closing speed is
`V_c = clamp(-range_rate, 0, 300 m/s)`. A non-closing trajectory therefore
has no PN load contribution. The formula is evaluated only for a finite,
non-singular relative state.

## Proposed offline composition

```text
n_geom = 0.5 * sin(lambda)                              [g]
n_los  = 1.5 * V_c * clamp(lambda_dot, -0.1, 0.1) / g0 [g]
n_raw  = 1.0 + n_geom + n_los                           [g]
n_cmd  = clamp(n_raw, -2.0, 7.0)                        [g]
```

`g0 = 9.80665 m/s²`. The geometric term is bounded by ±0.5 g, is independent
of range when LOS angle is unchanged, and tends to zero as elevation error
tends to zero. The PN term explicitly contains LOS angular rate, is bounded
by the declared closing/rate validity envelope, and reverses under mirrored
vertical geometry. These values are proposed evidence parameters, **not
approved gains**.

## Validity, fallback, and safety interaction

The reference trace covers `800 <= R <= 2500 m`, `0 <= V_c <= 300 m/s`,
`|lambda| <= 30°`, and `|lambda_dot| <= 0.1 rad/s`. For `R < 50 m`,
`rho <= epsilon`, non-finite state, or an undefined elevation plane, the
candidate returns capture hold: `n_raw = n_cmd = 1.0 g`, and both contributions
are zero. The existing F04 normal-load envelope `[-2, 7] g` is retained as a
post-composition saturation boundary; it is neither widened nor replaced.

## Compatibility and migration strategy

No runtime selection, configuration, checkpoint, observation, or backend
behavior changes in task 2. A future approved implementation must expose the
law under a new explicit mode/name, preserve the frozen law as the default,
record any loaded-config override, retain capture/filter/post-processing
semantics unless separately approved, and use an artifact-backed scenario
comparison before default promotion.

## Review gate

The pure geometry checks and offline trace show the candidate has explicit
LOS-rate semantics and retains the stated F04 clipping boundary. They do not
establish aircraft-model safety, tuning adequacy, or a reviewed migration
strategy. Physical-semantics and safety review are therefore **not accepted**;
`f01_evidence_gate.json` records `needs_more_evidence` and retains the frozen
baseline.
