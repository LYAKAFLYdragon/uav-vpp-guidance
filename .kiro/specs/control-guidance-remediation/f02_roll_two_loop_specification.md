# F02 proposed roll two-loop/damping specification — offline evidence only

**Status:** proposed for review; not approved for runtime use. The frozen `LOSRateGuidance` remains the only runtime behavior. This offline-only candidate is named `coordinated_turn_bank_outer_roll_rate_inner`.

## Frame, sources, signals, and signs

The candidate uses the local NEU frame and standard right-hand body-axis convention: positive heading error `e_psi = psi_LOS - psi_own` [rad] requests a positive (right) turn; positive bank `phi` [rad] and roll rate `p` [rad/s] are right-wing-down/right-turn positive. `psi_own` is the horizontal velocity heading, `psi_LOS` is the horizontal virtual-point LOS heading, `phi` is measured roll attitude, `p` is the measured body roll rate, and `V` is measured true airspeed [m/s]. The frozen path currently provides `roll_rad` but not an explicitly validated roll-rate source; a future implementation must map a named body-rate telemetry field with frame, sign, units, freshness, finite-value, and fallback validation before it can select this candidate.

The frozen formula is reproduced only for comparison:

```text
p_cmd,F = clip(k_roll * e_psi - k_damp * phi, -1.5, +1.5) [rad/s]
```

It has no `p` term. Thus states with equal heading error and bank but opposite roll rates receive identical frozen commands.

## Proposed offline two-loop composition and units

The outer loop turns heading error into a bounded coordinated-turn demand:

```text
psi_dot_raw = K_psi * e_psi                                  [rad/s]
psi_dot_cmd = clip(psi_dot_raw, -0.35, +0.35)                [rad/s]
V_used       = clip(V, 80, 350)                              [m/s]
phi_raw      = atan(V_used * psi_dot_cmd / g0)                [rad]
phi_cmd      = clip(phi_raw, -45 deg, +45 deg)               [rad]
```

The inner loop tracks bank with explicit roll-rate damping:

```text
p_att   = K_phi * (phi_cmd - phi)                             [rad/s]
p_damp  = -K_p * p                                            [rad/s]
p_raw   = p_att + p_damp                                      [rad/s]
p_cmd   = clip(p_raw, -1.5, +1.5)                            [rad/s]
```

Proposed evidence parameters are `K_psi = 1.0 s^-1`, `K_phi = 2.0 s^-1`, and `K_p = 0.8` (dimensionless when multiplying `p` [rad/s]). `g0 = 9.80665 m/s²`. The speed bounds protect the coordinated-turn calculation from unvalidated low-speed behavior and cap its reference envelope; they are not a runtime airspeed schedule. For fixed positive heading error, larger valid airspeed produces a larger same-sign bank demand until the 45° cap.

## Saturation and anti-windup

All outer-loop limits occur before the inner loop. F04's existing roll-rate envelope remains the final `[-1.5, +1.5] rad/s` boundary and is neither widened nor moved. The candidate is P/PD only and has no integral state, so no integrator can wind up. A future integral or rate-limited implementation must use back-calculation or conditional integration at the **actual final limiter** and expose its state/anti-windup telemetry; it is not authorized by this proposal. Non-finite input produces a zero roll-rate safe hold in the offline model; a future runtime mapping must preserve its existing finite fallback/capture behavior unless separately approved.

## Responsibility boundary and preservation

Guidance owns `e_psi -> psi_dot_cmd -> phi_cmd -> p_cmd` and reports all intermediate values. It does not issue aileron, rudder, or actuator commands. The low-level controller continues to filter `roll_rate_cmd` and map it to normalized aileron; the actuator interface remains responsible for actuator normalization and saturation. Command post-processing, F04 limits, filtering cadence, capture blending, backend selection, observation schema, configuration, and `record_config_override` semantics are unchanged by task 3.

No runtime selection, config, checkpoint, or backend behavior changes in this task. A future approved implementation must use an explicit mode/name, preserve frozen behavior as the default, record loaded-config overrides, establish source validation for roll rate and airspeed, and evaluate the same frozen scenario/seed/policy/horizon/backend matrix before default promotion.

## Review gate

The offline unit/property evidence shows an explicit rate-feedback term, sign-correct step/reversal/bank-release behavior, speed-aware coordinated-bank demand, F04 clipping, and reset-safe diagnostics. It does not establish aircraft-model stability, tracking, safety, gain adequacy, an engineering review of source conventions, or a closed-loop baseline comparison. The frozen matrix is declared but unexecuted; its three required main-policy checkpoints are absent. Although JSBSim is installed, strict JSBSim evidence cannot be produced without these artifacts. The gate is therefore **not approved**: retain the frozen baseline and record `needs_more_evidence`.
