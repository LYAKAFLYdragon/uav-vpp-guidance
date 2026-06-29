# Air Combat Maneuver Decision VPP Redesign: Requirements, Design, Tasks

## 0. Purpose

This document defines a combined requirements, design, and task plan for moving
the current VPP guidance stack closer to an air-combat maneuver decision system.

Current evidence suggests that the existing VPP interface is effective for some
intercept-like geometries, especially `crossing_feasible`, but can fail badly in
`head_on`. The likely cause is not only prediction error. The more fundamental
issue is that the policy chooses a virtual point to track, while close air combat
success depends on shaping future offensive geometry: attack angle, post-merge
position, closure, energy, and bidirectional weapon-zone exposure.

The target outcome is a fair, testable redesign path that answers:

| Question | Required answer |
|---|---|
| Does VPP guide the ego aircraft toward an air-combat advantage position? | Quantified by attack-zone asymmetry, merge geometry, and damage margin. |
| When prediction hurts `head_on`, is it because the future anchor is geometrically harmful? | Quantified by lookahead-trained policies and VP/merge diagnostics. |
| Can a revised VPP action space express offensive position rather than only point tracking? | Verified by frame-aware VPP tests and combat-only pilots. |
| Can the result support a paper claim about task-geometry dependence? | Verified by fair training, held-out seeds, and trajectory diagnostics. |

## 1. Current System Summary

The current environment pipeline is:

```text
policy action -> virtual pursuit point -> LOS-rate guidance
-> low-level controller -> JSBSim/simple backend -> reward/termination
```

Relevant implementation anchors:

| Component | File | Current role |
|---|---|---|
| Environment loop | `src/uav_vpp_guidance/envs/tracking_env.py` | Connects policy action, VPP, guidance, backend, reward, and combat HP. |
| VPP generator | `src/uav_vpp_guidance/virtual_point/generator.py` | Converts normalized action to an offset and adds it to an anchor position. |
| Coordinate transform stub | `src/uav_vpp_guidance/virtual_point/coordinate_transform.py` | Intended target-relative transform, currently not implemented. |
| LOS guidance | `src/uav_vpp_guidance/guidance/los_rate_guidance.py` | Tracks the selected virtual point using heading/elevation errors. |
| Reward | `src/uav_vpp_guidance/envs/reward.py` | Primarily range, angle, safety, energy, saturation, smoothness, and turn-rate shaping. |
| Combat damage | `src/uav_vpp_guidance/envs/attack_zone.py` | Defines bidirectional attack-zone score and HP exchange. |

## 2. Evidence Behind the Redesign

The latest local long-round outputs show a geometry-specific pattern.

| Scope | Observation |
|---|---|
| `expert/head_on` lookahead sweep | Prediction VPP has `win_rate=0.0` for all lookaheads. Damage margin worsens from `-11.55` at `0.0s` to `-46.80` at `0.5s`, then partially recovers to `-37.35` at `1.0s`. |
| `end_to_end/head_on` lookahead sweep | `0.0/0.2/0.5s` remain strong, but `1.0s` collapses to `win_rate=0.222` and negative damage margin. |
| `expert/head_on` combat pilot at `1.0s` | Prediction VPP loses badly against no-prediction VPP: `0.0` vs `1.0` win rate and `-37.35` vs `+51.55` damage margin. |
| `expert/crossing_feasible` combat pilot at `1.0s` | Prediction VPP improves over no-prediction VPP: `1.0` vs `0.8` win rate and `+16.60` vs `+14.85` damage margin. |
| Trajectory-level aggregate signal | In `expert/head_on`, prediction and no-prediction reach similar minimum range and similar envelope time, but damage outcome reverses. This points to harmful combat geometry, not simple failure to close. |

Primary local artifacts:

| Artifact | Path |
|---|---|
| Lookahead conclusion | `outputs/jsbsim_hrl_comparison_long_round1/lookahead_sweep_conclusion.md` |
| Lookahead CSV | `outputs/jsbsim_hrl_comparison_long_round1/lookahead_sweep_head_on.csv` |
| Combat pilot episode records | `outputs/jsbsim_hrl_comparison_long_round1/long_prediction_vpp_jsbsim_compare_long_lh1p0_combat_pilot_expert_seed10/aggregate/episode_records.json` |

## 3. Design Defects

| ID | Defect | Why it matters for air combat |
|---|---|---|
| D1 | VPP offset is added in world coordinates. | A fixed NEU offset does not consistently mean "behind target", "lag pursuit", "high-side", or "beam" as headings change. |
| D2 | The policy action controls a point, not a tactical state. | Air combat decision quality depends on future relative geometry, not merely reaching a target-adjacent point. |
| D3 | Future anchors can pull the ego aircraft through the merge. | In `head_on`, chasing the predicted future target position can preserve closure but worsen post-merge attack-zone asymmetry. |
| D4 | `dynamics_aware` only enforces heading feasibility. | It clips to a feasible heading sector but does not encode offensive geometry or weapon-zone exposure. |
| D5 | LOS-rate guidance ignores target tactical state. | Guidance tracks the VP point; it does not directly optimize attack angle, target attack angle, closure control, or energy trade. |
| D6 | Dense reward is tracking-oriented. | Range and angle shaping do not fully represent bidirectional combat advantage, especially near merge. |
| D7 | Combat HP is a terminal/evaluation mechanism, not a VPP design constraint. | The policy may receive weak or delayed information about whether its chosen point creates offensive or defensive exposure. |
| D8 | Observation and action frames are mismatched. | Observations are relative geometry features, while action offsets are not expressed in the same tactical frame. |
| D9 | Episode trajectory diagnostics are insufficient. | Existing aggregate records can show outcome reversals but often lack step-level VP, attack-zone, and merge-state traces. |
| D10 | "Perfect prediction" is not a true oracle policy. | Without a separately trained oracle-anchor policy, perfect prediction is an inference-time ablation, not an upper bound on achievable policy quality. |

## 4. Requirements

### 4.1 Functional Requirements

| ID | Requirement | Acceptance signal |
|---|---|---|
| R1 | Add a frame-aware VPP offset option. | `virtual_point.offset_frame` supports current behavior plus at least one target-relative or encounter-relative frame. |
| R2 | Preserve legacy behavior by default. | Existing configs without the new key produce unchanged VPP positions and old tests continue to pass. |
| R3 | Express offensive position as a first-class design target. | At least one new VPP mode can place anchors in target-relative offensive regions such as lag, beam, or rear-quarter positions. |
| R4 | Log VPP and combat geometry at step level. | Evaluation artifacts include VP position, anchor position, offset frame, range, own attack angle, target attack angle, attack-zone flags, HP, and merge markers. |
| R5 | Support fair lookahead training. | `0.0/0.2/0.5/1.0s` lookahead variants can be separately trained with identical non-lookahead settings. |
| R6 | Keep combat-only formal scope clean. | Mainline comparison remains limited to `head_on` and `crossing_feasible`, with `expert` and `end_to_end` opponent splits. |
| R7 | Do not silently mutate loaded configs. | Every YAML-loaded mutation is recorded through `record_config_override` or `record_config_override_if_changed`. |
| R8 | Enforce backend transparency. | Evaluation records active backend, fallback status, and fallback reason in provenance. |
| R9 | Keep crossing attack-zone AoA explicit. | Formal/cloud crossing entries pass `--attack-zone-close-range-max-aoa-deg 60`. |

### 4.2 Scientific Requirements

| ID | Requirement | Acceptance signal |
|---|---|---|
| S1 | Separate prediction accuracy from anchor geometry. | Compare current-target, future-target, and tactical-frame anchors under matched training budgets. |
| S2 | Test task-geometry dependence. | Show per-task results for `head_on` and `crossing_feasible` rather than only pooled metrics. |
| S3 | Test opponent dependence. | Report `expert` and `end_to_end` splits separately. |
| S4 | Report damage-based outcomes. | Include win rate, damaging win rate, damage margin, damage dealt, damage taken, crashes, target crash/OOB, and effective engagement rate. |
| S5 | Avoid unsupported oracle language. | Treat `perfect_prediction_vpp` as an inference-time ablation unless separately trained as an oracle-anchor policy. |

### 4.3 Non-Functional Requirements

| ID | Requirement | Acceptance signal |
|---|---|---|
| N1 | Minimal invasive changes. | Changes are isolated to VPP geometry, telemetry, runner config plumbing, and targeted tests. |
| N2 | Observation contract stability. | Any observation-schema change updates `AGENTS.md`, `README.md`, and tests. No base feature order changes without schema migration. |
| N3 | Reproducibility. | Output directories contain resolved config, provenance, command line, git info, and summary artifacts. |
| N4 | Backward compatibility. | Existing no-prediction and prediction comparison tests remain valid. |

## 5. Proposed Design

### 5.1 VPP Offset Frames

Add a config-controlled frame layer before adding the offset to the anchor.

| `virtual_point.offset_frame` | Meaning | Intended use |
|---|---|---|
| `world_neu` | Current behavior: action offset is in global NEU coordinates. | Backward compatibility and baseline. |
| `target_velocity` | Longitudinal axis follows target velocity, lateral axis is horizontal normal. | Express rear/lead/beam positions relative to target motion. |
| `los_relative` | Longitudinal axis follows own-to-target LOS, lateral axis is horizontal normal. | Express lead/lag relative to current engagement line. |
| `encounter` | Axes derived from target velocity and LOS, with stable fallback when degenerate. | Express merge-aware combat geometry. |

Default must be `world_neu`.

Implementation target:

```text
normalized action -> metric offset in selected local frame
-> frame transform -> world offset
-> anchor position + world offset -> virtual point
```

### 5.2 Tactical Anchor Modes

Keep existing anchor modes, but add tactical modes only after frame telemetry is stable.

| Anchor mode | Meaning | Status |
|---|---|---|
| `current_target` | Anchor at current target position. | Existing. |
| `predicted_target` | Anchor at predicted target position. | Existing. |
| `oracle_future_position` | Constant-velocity true-state ablation. | Existing, not an oracle-policy upper bound. |
| `offensive_position` | Anchor offset from target into a configured offensive region. | New candidate. |
| `merge_escape_or_lag` | Pre-merge head-on anchor that biases toward survivable post-merge geometry. | New candidate, experimental. |

The new tactical modes should be introduced behind config flags and should not
replace the current mainline until validated.

### 5.3 Combat Geometry Telemetry

Add a diagnostic artifact for every combat evaluation episode.

Required per-step fields:

| Field group | Fields |
|---|---|
| Time and task | `step`, `time_s`, `task`, `opponent_stage`, `method`, `seed`, `episode` |
| Positions | `ego_pos_x/y/z`, `target_pos_x/y/z`, `vp_pos_x/y/z`, `anchor_pos_x/y/z` |
| VPP metadata | `anchor_mode`, `offset_frame`, `offset_x/y/z`, `lookahead_time_s`, `prediction_valid`, `prediction_fallback` |
| Relative geometry | `range_m`, `range_rate_mps`, `ata_deg`, `aa_deg`, `ego_attack_aoa_deg`, `target_attack_aoa_deg` |
| Combat state | `ego_in_attack_zone`, `target_in_attack_zone`, `ego_hp`, `target_hp`, `ego_attack_score`, `target_attack_score` |
| Merge markers | `pre_merge`, `post_merge`, `min_range_so_far_m`, `first_pass_complete` |
| Control state | `nz_cmd`, `roll_rate_cmd`, `throttle_cmd`, saturation flags |

Derived episode-level diagnostics:

| Metric | Purpose |
|---|---|
| `first_ego_attack_time_s` | Whether ego gets first offensive window. |
| `first_target_attack_time_s` | Whether opponent gets first offensive window. |
| `post_merge_attack_zone_advantage_s` | Difference between ego and target post-merge attack-zone time. |
| `vp_forward_bias_m` | Whether VP lies ahead of target in the target-velocity frame. |
| `vp_lateral_bias_m` | Whether VP encourages bracket/beam/lag geometry. |
| `merge_min_range_m` | Merge severity and collision/overshoot risk. |
| `damage_margin` | Direct combat outcome signal. |

### 5.4 Reward and Evaluation Separation

Do not immediately rewrite reward. First add telemetry and compare whether the
current policy already chooses harmful geometry.

After diagnostics, consider a small optional combat-geometry shaping block:

| Candidate reward term | Purpose | Guardrail |
|---|---|---|
| Offensive attack-zone advantage | Reward ego attack score minus target attack score. | Keep disabled by default. |
| Post-merge survivability | Penalize target-only attack-zone exposure after first pass. | Only for combat tasks. |
| Closure control near merge | Penalize excessive closure if no offensive angle is achieved. | Avoid weakening crossing performance. |
| Energy floor near merge | Preserve maneuverability through the merge. | Use existing energy terms where possible. |

### 5.5 Fair Experiment Design

The next fair validation should use:

| Dimension | Values |
|---|---|
| Tasks | `head_on`, `crossing_feasible` |
| Opponents | `expert`, `end_to_end` |
| Methods | `no_prediction_vpp`, `prediction_vpp_jsbsim_compare`, new frame-aware VPP variants |
| Lookahead training | `0.0`, `0.2`, `0.5`, `1.0s` separately trained |
| Backend | `jsbsim`, `strict_backend=True` |
| Crossing AoA | Explicit `60 deg` |
| Primary metrics | win rate, damaging win rate, damage margin, damage dealt/taken, effective engagement rate |
| Diagnostic metrics | attack-zone timing, post-merge advantage, VP frame bias, merge markers |

## 6. Task Plan

### T0: Freeze Current Baseline Evidence

| Item | Deliverable | Acceptance |
|---|---|---|
| T0.1 | Archive current long-round summaries. | Local output paths listed in the experiment note. |
| T0.2 | Create a baseline readout table. | Includes lookahead sweep and combat pilot results for both opponent splits. |
| T0.3 | Mark `perfect_prediction_vpp` language as ablation-only. | No document or summary calls it an oracle upper bound without separate training. |

### T1: Add VPP Geometry Diagnostics

| Item | Deliverable | Acceptance |
|---|---|---|
| T1.1 | Add step-level VP/combat telemetry collection in combat evaluations. | JSONL or CSV emitted per run without changing policy behavior. |
| T1.2 | Add summary builder for merge diagnostics. | Produces per-method, per-task, per-opponent table. |
| T1.3 | Add targeted tests for telemetry schema. | Tests verify required fields exist and are finite for a smoke episode. |

### T2: Implement Frame-Aware VPP Offsets

| Item | Deliverable | Acceptance |
|---|---|---|
| T2.1 | Implement `target_velocity` and `los_relative` transforms. | Unit tests verify offset rotation for canonical headings. |
| T2.2 | Add `virtual_point.offset_frame` config with default `world_neu`. | Existing configs preserve old VPP positions. |
| T2.3 | Record offset-frame provenance in episode info. | Evaluation artifacts show selected frame and no silent config override. |
| T2.4 | Add targeted tests for backward compatibility. | `world_neu` output equals old `anchor + offset` behavior. |

### T3: Train Fair Lookahead Policies

| Item | Deliverable | Acceptance |
|---|---|---|
| T3.1 | Create explicit configs for `0.0/0.2/0.5/1.0s` lookahead. | Only prediction/lookahead settings differ; non-lookahead settings match. |
| T3.2 | Train each lookahead separately. | Checkpoints and resolved configs exist for every lookahead. |
| T3.3 | Run combat-only pilot. | `head_on + crossing_feasible`, `expert + end_to_end`, common seeds. |
| T3.4 | Build lookahead conclusion table. | Reports monotonicity or non-monotonicity of head-on degradation. |

### T4: Prototype Tactical VPP Modes

| Item | Deliverable | Acceptance |
|---|---|---|
| T4.1 | Add `offensive_position` anchor mode behind config. | Disabled by default and covered by unit tests. |
| T4.2 | Add head-on merge-aware candidate. | Pilot shows whether it reduces target-only attack-zone exposure. |
| T4.3 | Compare against no-prediction and current prediction VPP. | Same combat-only matrix and metrics. |

### T5: Formal Validation

| Item | Deliverable | Acceptance |
|---|---|---|
| T5.1 | Run larger held-out seed evaluation. | At least 480 held-out seeds per main method/scope when compute budget allows. |
| T5.2 | Generate top-level formal summaries. | Overall executive, no-prediction mainline, and dedicated prediction/VPP comparison summaries. |
| T5.3 | Produce paper-safe conclusion. | Claims are scoped by task, opponent, training budget, and anchor design. |

## 7. Acceptance Criteria

### 7.1 Engineering Acceptance

| Gate | Criterion | Pass condition |
|---|---|---|
| G1 | Backward compatibility | All existing targeted tests pass; `world_neu` reproduces old VPP behavior. |
| G2 | Provenance | Any YAML-loaded config mutation appears in `provenance["config_overrides"]`. |
| G3 | Backend transparency | Formal outputs record `backend=jsbsim`, strict backend, and no silent fallback. |
| G4 | Telemetry completeness | Combat runs include VP, anchor, offset frame, attack-zone, HP, and merge diagnostics. |
| G5 | Summary reproducibility | Dedicated comparison summary can be regenerated from local outputs. |

### 7.2 Scientific Acceptance

| Gate | Criterion | Pass condition |
|---|---|---|
| G6 | Geometry diagnosis | `head_on` failures can be explained by quantified attack-zone asymmetry or VP frame bias, not only aggregate win rate. |
| G7 | Lookahead claim | If degradation grows with lookahead, conclusion is based on separately trained lookahead policies. |
| G8 | Task dependence | `head_on` and `crossing_feasible` are reported separately with opponent splits. |
| G9 | Fair comparison | Prediction and no-prediction methods share backend, VPP settings, training budget, seeds, and opponent matrix except for the studied variable. |
| G10 | Oracle language | No upper-bound claim is made for `perfect_prediction_vpp` unless an oracle-anchor policy is separately trained and evaluated. |

### 7.3 Success Thresholds for the Redesign

| Outcome | Threshold | Interpretation |
|---|---|---|
| Diagnostic success | Step-level analysis shows whether losing policies give opponent earlier or longer attack-zone exposure. | Confirms or rejects the harmful-geometry hypothesis. |
| Frame-aware VPP success in `head_on` | Damage margin improves by at least `+15` points over current prediction VPP without reducing crossing win rate by more than `0.05`. | Frame-aware action space is useful. |
| Tactical VPP success in `head_on` | Win rate gap to no-prediction VPP closes by at least half under equal budget. | VPP can be made combat-position aware. |
| Geometry-dependence claim | Prediction helps or remains neutral in crossing but hurts or requires redesign in head-on under fair training. | Supports a paper claim about task-geometry dependence. |

## 8. Immediate Next Steps

| Priority | Task | Reason |
|---|---|---|
| P0 | Add geometry diagnostics before changing reward. | The current result already hints at harmful merge geometry; telemetry will make it falsifiable. |
| P0 | Implement `offset_frame=world_neu` default and tests. | Establishes a safe extension point without changing legacy behavior. |
| P1 | Add `target_velocity` and `los_relative` frames. | These are the smallest changes that make VPP offsets correspond to tactical geometry. |
| P1 | Re-run small combat-only pilot with frame-aware VPP. | Quick check before spending large compute. |
| P2 | Train separate lookahead policies for geometry-dependence claims. | Avoids over-interpreting inference-time sweeps. |
| P2 | Prototype tactical anchor modes only after diagnostics. | Keeps design changes grounded in measured failure modes. |

## 9. Non-Goals for This Redesign Pass

| Non-goal | Reason |
|---|---|
| Rewrite the whole environment or policy stack. | The current stack is useful and already produces task-dependent evidence. |
| Mix `sustained_turn` or `multi_waypoint` into combat readiness. | This pass is specifically about combat tasks. |
| Treat `perfect_prediction_vpp` as an oracle upper bound. | It is currently an inference-time ablation unless separately trained. |
| Change the base observation feature order. | This would break the observation schema contract and downstream checkpoints. |
| Use pooled metrics as the main claim. | Pooled metrics hide the key task-geometry dependence. |

## 10. Paper Claim Guardrails

A defensible paper statement after this redesign should be scoped like:

> In close-range JSBSim combat tasks, prediction-aware VPP improves geometries
> where future target displacement aligns with interception, but can degrade
> head-on merge outcomes when the VPP action interface represents future point
> tracking rather than tactical offensive position. Frame-aware or tactical VPP
> anchors are therefore necessary to convert prediction accuracy into combat
> maneuver advantage.

Claims that require more evidence:

| Claim | Additional evidence needed |
|---|---|
| Prediction is universally beneficial. | Multi-task, multi-opponent results where all task splits improve. |
| Perfect prediction is an upper bound. | Separately trained oracle-anchor policies. |
| VPP is a general air-combat decision representation. | Evidence that it can express and learn offensive post-merge positioning. |
| Head-on failure is due only to prediction error. | Diagnostics showing VP geometry is harmless and prediction error dominates. |

## 11. New Conversation Handoff Prompt

Use the following prompt when opening a fresh implementation thread.

```text
You are working in E:\uav-vpp-guidance on the VPP air-combat maneuver
decision redesign.

First, read and obey:
- E:\uav-vpp-guidance\AGENTS.md
- E:\uav-vpp-guidance\docs\air_combat_maneuver_decision_vpp_redesign.md

Hard constraints:
- Any mutation of YAML-loaded config must be recorded with
  record_config_override or record_config_override_if_changed.
- Do not silently override backend, strict_backend, opponent, attack-zone, or
  VPP settings.
- Keep changes minimal and backward compatible.
- Preserve old tests and old behavior wherever possible.
- If observation schema changes, update AGENTS.md, README.md, and targeted
  tests. Do not change the base feature order without a schema migration.
- Combat readiness scope for this line of work is only:
  tasks=head_on,crossing_feasible and opponents=expert,end_to_end.
- Do not mix sustained_turn or multi_waypoint into the mainline combat
  conclusion.
- Crossing combat/formal entries must explicitly use
  --attack-zone-close-range-max-aoa-deg 60.
- Do not describe perfect_prediction_vpp as an oracle upper bound unless an
  oracle-anchor policy is separately trained.

Current hypothesis:
The current VPP interface is a point-tracking action abstraction, not a true
air-combat maneuver decision interface. It often helps crossing/intercept-like
geometries, but can hurt head_on because future anchors can pull the ego aircraft
through the merge instead of shaping post-merge offensive position. The main
design gap is that VPP offsets are currently world-coordinate point offsets,
while air combat advantage is target/encounter-relative geometry.

Relevant code:
- src/uav_vpp_guidance/envs/tracking_env.py
- src/uav_vpp_guidance/virtual_point/generator.py
- src/uav_vpp_guidance/virtual_point/coordinate_transform.py
- src/uav_vpp_guidance/guidance/los_rate_guidance.py
- src/uav_vpp_guidance/envs/reward.py
- src/uav_vpp_guidance/envs/attack_zone.py
- scripts/run_jsbsim_hrl_comparison.py
- scripts/build_formal_gate_summary.py

Relevant local evidence:
- outputs/jsbsim_hrl_comparison_long_round1/lookahead_sweep_conclusion.md
- outputs/jsbsim_hrl_comparison_long_round1/lookahead_sweep_head_on.csv
- outputs/jsbsim_hrl_comparison_long_round1/long_prediction_vpp_jsbsim_compare_long_lh1p0_combat_pilot_expert_seed10/aggregate/episode_records.json

Known result pattern:
- prediction_vpp helps expert/crossing_feasible in the pilot.
- prediction_vpp is much worse than no_prediction_vpp on expert/head_on.
- In expert/head_on, prediction_vpp and no_prediction_vpp reach similar minimum
  range and envelope time, but damage outcome reverses. This suggests harmful
  merge/offensive geometry, not simply failure to close.

Primary objective for this new thread:
Start converting VPP from "track a virtual point" toward "choose an air-combat
maneuver geometry" through a small, testable first slice.

Recommended first slice:
1. Add combat geometry diagnostics before changing reward or policy semantics.
   Capture per-step VP, anchor, offset, attack-zone, HP, range, ATA/AA,
   ego/target attack AoA, and merge markers for combat evaluations.
2. Add a diagnostic summary builder that produces per-task, per-opponent,
   per-method merge/attack-zone tables.
3. Add targeted pytest for telemetry schema and summary behavior.
4. Only after diagnostics are stable, add frame-aware VPP offsets:
   virtual_point.offset_frame = world_neu | target_velocity | los_relative.
   Default must be world_neu and must reproduce old behavior.
5. Add unit tests for canonical frame transforms and backward compatibility.
6. Run a small combat-only pilot, not a full formal batch, to check whether
   frame-aware VPP improves head_on without damaging crossing.

Expected implementation guardrails:
- Prefer extending existing runner/evaluation artifacts over introducing a new
  parallel pipeline.
- Keep new telemetry optional or scoped to combat evaluations so old outputs do
  not churn unexpectedly.
- Add summary metrics that directly test the hypothesis:
  first_ego_attack_time_s, first_target_attack_time_s,
  post_merge_attack_zone_advantage_s, vp_forward_bias_m, vp_lateral_bias_m,
  merge_min_range_m, damage_margin.
- For config-driven changes, expose explicit CLI/config fields and record
  provenance for any loaded-config mutation.

Acceptance before ending the first implementation turn:
- Targeted tests pass.
- Existing behavior is preserved when offset_frame is omitted or world_neu.
- A smoke/pilot run can emit the new diagnostic artifact.
- The final response reports changed files, tests run, output paths if any, and
  any remaining risks.
```

