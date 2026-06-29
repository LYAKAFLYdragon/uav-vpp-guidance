# Air Combat Maneuver Decision VPP Tactical-Basis MVP

## Purpose

This document freezes the intended MVP design for moving VPP from a
"virtual-point tracking interface" toward a "combat-geometry maneuver decision
interface" while preserving the current combat-only scope and the existing
3-dimensional policy action head.

The MVP does not add policy output dimensions. Instead, it reinterprets the
current VPP `x/y/z` action semantics from raw geometric offsets into continuous
tactical-basis coefficients:

- `action[0]`: lead-lag coefficient
- `action[1]`: inside-outside coefficient
- `action[2]`: climb-descent coefficient

## Scope Guardrails

- Mainline baseline remains `reset075_no_mode_switch_longscale00`.
- The MVP ships behind a new `virtual_point.action_semantics` config flag.
- Default behavior remains the current offset-based VPP path.
- Combat readiness scope remains `head_on` and `crossing_feasible`.
- Opponent scope remains `expert` and `end_to_end`.
- No observation schema changes are included in this MVP.
- No reward rewrite is included in this MVP.
- No formal held-out is triggered by this design alone.

## High-Level Design

The MVP keeps the current anchor stack and downstream guidance path:

```text
policy action (3D)
-> tactical basis coefficients
-> tactical-basis world offset
-> existing anchor stack
-> virtual point
-> existing LOS guidance / post-merge guards
-> backend
```

The existing `anchor_mode` stack remains intact:

- `predicted_target`
- `offensive_position`
- close-range anchor rewrites
- post-merge release/recovery/clamp logic

The only first-order semantic change is how `action[0:3]` is converted into the
pre-guidance world offset.

## Tactical-Basis Contract

### 1. Lead-Lag basis

- Symbol: `a_ll`
- Positive meaning: lead
- Negative meaning: lag
- Primary geometric role: closure control and turning-room control
- Recommended frame: `target_velocity`

### 2. Inside-Outside basis

- Symbol: `a_io`
- Positive meaning: outside
- Negative meaning: inside
- Primary geometric role: one-circle vs two-circle bias and merge-side bias
- Recommended frame: `encounter_stable`

### 3. Climb-Descent basis

- Symbol: `a_cd`
- Positive meaning: climb / high-side
- Negative meaning: descent / low-side
- Primary geometric role: vertical separation and energy management
- Recommended frame: `world_neu`

## Tactical-Basis World Offset

For `virtual_point.action_semantics=tactical_basis_v1`, the new offset contract
is:

```text
world_offset =
    a_ll * b_ll_world
  + a_io * b_io_world
  + a_cd * b_cd_world
```

where:

- `b_ll_world` is the lead-lag basis vector in world coordinates
- `b_io_world` is the inside-outside basis vector in world coordinates
- `b_cd_world` is the climb-descent basis vector in world coordinates

The resulting `world_offset` is then added to the selected anchor position to
produce the final VPP, just as in the current system.

## Why Generator First

The MVP should be implemented first in `VirtualPointGenerator`, not in
`tracking_env`, because the generator is already the component responsible for
translating policy actions into VPP geometry. `tracking_env` should remain the
orchestration layer for:

- anchor selection
- direct-track bypass
- post-merge release/recovery/clamp overrides
- telemetry assembly

This keeps the semantic change isolated and minimizes regression risk.

## Config Contract

The following new `virtual_point` fields are introduced.

```yaml
virtual_point:
  action_semantics: cartesian_offset

  tactical_basis_longitudinal_frame: target_velocity
  tactical_basis_lateral_frame: encounter_stable
  tactical_basis_vertical_frame: world_neu

  tactical_basis_lead_lag_extent_m: 1200.0
  tactical_basis_lead_lag_extent_m_by_task:
    head_on: 1200.0
    crossing_feasible: 800.0

  tactical_basis_inside_outside_extent_m: 0.0
  tactical_basis_inside_outside_extent_m_by_task:
    head_on: 1600.0
    crossing_feasible: 300.0

  tactical_basis_climb_descent_extent_m: 600.0
  tactical_basis_climb_descent_extent_m_by_task:
    head_on: 600.0
    crossing_feasible: 300.0

  tactical_basis_lateral_sign_mode: same_side
  tactical_basis_encounter_stable_max_heading_delta_deg: 20.0
```

### Config semantics

- `action_semantics`
  - `cartesian_offset`: current behavior
  - `tactical_basis_v1`: new tactical-basis behavior
- `*_extent_m`
  - converts normalized coefficients into metric basis amplitudes
- `*_extent_m_by_task`
  - allows conservative `crossing_feasible` rollout while enabling stronger
    `inside/outside` semantics in `head_on`
- `tactical_basis_lateral_sign_mode`
  - follows the existing `same_side/fixed_positive/fixed_negative` idea
- `tactical_basis_encounter_stable_max_heading_delta_deg`
  - keeps lateral basis construction stable in near-degenerate geometry

## Telemetry Contract

The MVP adds step-level telemetry so that action semantics and geometry outcome
can be analyzed directly.

Required new step-level fields:

- `action_semantics`
- `configured_action_semantics`
- `tactical_basis_enabled`
- `tactical_basis_action_ll`
- `tactical_basis_action_io`
- `tactical_basis_action_cd`
- `tactical_basis_lead_lag_extent_m`
- `tactical_basis_inside_outside_extent_m`
- `tactical_basis_climb_descent_extent_m`
- `tactical_basis_longitudinal_frame`
- `tactical_basis_lateral_frame`
- `tactical_basis_vertical_frame`
- `tactical_basis_lateral_sign_mode`
- `tactical_basis_lateral_sign`
- `tactical_basis_ll_world_x`
- `tactical_basis_ll_world_y`
- `tactical_basis_ll_world_z`
- `tactical_basis_io_world_x`
- `tactical_basis_io_world_y`
- `tactical_basis_io_world_z`
- `tactical_basis_cd_world_x`
- `tactical_basis_cd_world_y`
- `tactical_basis_cd_world_z`
- `tactical_basis_world_offset_x`
- `tactical_basis_world_offset_y`
- `tactical_basis_world_offset_z`

The current telemetry fields such as `vp_forward_bias_m`,
`vp_lateral_bias_m`, `post_merge_attack_zone_advantage_s`,
`direct_track_mode_effective`, and release/recovery markers remain unchanged and
continue to be the primary comparison outputs.

## Aggregate Diagnostics Contract

The MVP should extend combat geometry aggregation with basis-aware summary
metrics without removing any existing fields.

Required new aggregate metrics:

- `pre_merge_mean_tactical_basis_action_ll`
- `pre_merge_mean_tactical_basis_action_io`
- `pre_merge_mean_tactical_basis_action_cd`
- `pre_merge_positive_tactical_basis_io_fraction`
- `post_merge_mean_tactical_basis_action_ll`
- `post_merge_mean_tactical_basis_action_io`

These metrics are diagnostic only. The primary gate remains the existing
combination of:

- win rate
- damage margin
- crash count
- `vp_forward_bias_m`
- `post_merge_attack_zone_advantage_s`

## MVP Acceptance Criteria

The MVP is considered implemented correctly if:

- legacy configs with `action_semantics=cartesian_offset` are behaviorally
  unchanged
- the new semantics can be enabled via YAML only
- the action head remains 3D
- `head_on` can express non-zero inside-outside bias without changing action
  dimension
- `crossing_feasible` can keep a mild inside-outside amplitude via task-specific
  extents
- existing pilot artifacts can compare basis actions against
  `vp_forward_bias_m` and post-merge outcomes

The MVP is not considered successful merely because it trains or runs. It must
make the geometry choice more explicit and auditable.

## Non-Goals

- Replace the current mainline baseline immediately
- Rewrite the reward function in the same patch
- Redesign direct-track or post-merge clamp logic in the same patch
- Change observation ordering or feature dimension
- Mix sustained-turn or multi-waypoint tasks into the mainline evaluation

## Recommended Rollout Strategy

1. Implement generator semantics and telemetry only.
2. Verify legacy compatibility with targeted tests.
3. Add one head-on-focused experimental YAML derived from the current baseline.
4. Run tactical-basis combat-only pilot on `head_on + crossing_feasible`.
5. Compare basis-action diagnostics with existing geometry metrics before
   changing any reward or clamp mechanism.
