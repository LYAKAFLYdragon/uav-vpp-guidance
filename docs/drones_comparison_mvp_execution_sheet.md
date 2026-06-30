# Drones Comparison MVP Execution Sheet

## 1. Purpose

This sheet converts the current comparison-baseline MVP into an execution-level
checklist for the `Drones` submission.

It is intentionally narrower than
`docs/drones_comparison_mvp_implementation_checklist.md`:

- it tracks what is already created
- it lists what still must be created
- it assigns each baseline to a paper table role
- it keeps the `legacy hierarchical Aerospace baseline` in scope

Paper wording should use `geometry-basis`.
Repo implementation names may keep existing `tactical_basis` identifiers.

## 2. Frozen Protocol

Every baseline below must stay on this same protocol:

- anchor baseline remains `reset075_no_mode_switch_longscale00`
- do not promote `directtrack2000`
- combat-only scope remains `head_on + crossing_feasible`
- opponent split remains `expert + end_to_end`
- crossing evaluation must pass `--attack-zone-close-range-max-aoa-deg 60`
- backend remains `jsbsim`
- pilot budget remains 10 seeds: `480-489`
- do not launch formal held-out beyond the 10-seed pilot
- any YAML mutation must be recorded through the provenance contract
- do not change observation schema for this comparison MVP

## 3. Paper Table Allocation

### Table 1: paper-core interface family

Only these four rows belong in the main paper table:

1. `reset075_no_mode_switch_longscale00` baseline
2. broad geometry-basis combat finetune
3. narrow-extents geometry-basis combat finetune
4. mixed crossing-restored geometry-basis combat finetune

### Table 2: reviewer-facing external comparators

These rows answer the strongest external-baseline objections:

1. end-to-end direct-command PPO
2. PN zero-offset guidance
3. legacy hierarchical Aerospace baseline

Optional rows that should stay out of the fastest manuscript path unless they
are cleaned up:

1. LOS zero-offset guidance
2. zero-offset / no-VPP comparator

### Appendix or rebuttal-only comparators

These do not block the fastest submission route:

1. SAC-cartesian
2. SAC-mixed geometry-basis
3. semantics-only geometry-basis MVP
4. old incompatible legacy outputs copied directly from `E:\CloseAirCombat_control`

## 4. Submission Implementation Board

This is the shortest file-level checklist for the current submission MVP.
Each row states:

- which config trains the baseline
- which config evaluates it under the frozen paper protocol
- which launcher should be used
- which new files are still missing
- whether the result belongs in the manuscript main tables

### Main manuscript rows

1. `Cartesian VPP baseline`
   - training config:
     `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml`
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
   - launcher:
     `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
   - new files still needed:
     none
   - manuscript destination:
     Table 1
   - result source:
     the unified geometry-family 10-seed pilot output for `expert` and
     `end_to_end`

2. `Geometry-basis broad finetune`
   - training config:
     `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.yaml`
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
   - launcher:
     `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
   - new files still needed:
     none
   - manuscript destination:
     Table 1
   - result source:
     existing 10-seed pilot outputs on 2026-06-29, or the unified rerun bundle

3. `Geometry-basis narrow finetune`
   - training config:
     `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.yaml`
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
   - launcher:
     `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
   - new files still needed:
     none
   - manuscript destination:
     Table 1
   - result source:
     existing 10-seed pilot outputs on 2026-06-29, or the unified rerun bundle

4. `Geometry-basis mixed finetune`
   - training config:
     `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.yaml`
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
   - launcher:
     `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
   - new files still needed:
     none unless a newer mixed checkpoint replaces the current one
   - manuscript destination:
     Table 1
   - result source:
     existing 10-seed pilot outputs on 2026-06-29, or the unified rerun bundle

### Reviewer-facing comparator rows

5. `End-to-end direct-command PPO`
   - training config:
     `config/experiment/train_end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat.yaml`
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_end_to_end_direct_command_best.yaml`
   - launchers:
     `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command.ps1`
     `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command_10seed_pilot.ps1`
   - new files still needed:
     no new files; the remaining gate is a paper-protocol checkpoint if one is
     not already present
   - manuscript destination:
     Table 2
   - result source:
     one 10-seed pilot output per opponent stage after the checkpoint is ready

6. `PN zero-offset guidance`
   - training config:
     none
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_pn_zero_offset.yaml`
   - launcher:
     `scripts/run_reset075_no_mode_switch_longscale00_rule_pn_zero_offset_10seed_pilot.ps1`
   - new files still needed:
     none
   - manuscript destination:
     Table 2
   - result source:
     one 10-seed pilot output per opponent stage

7. `Legacy hierarchical Aerospace baseline`
   - training config:
     none new; reuse the frozen checkpoint
     `E:\CloseAirCombat_control\baseline_results\checkpoints\proposed_ppo.zip`
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge.yaml`
   - launchers:
     `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_smoke.ps1`
     and
     `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_10seed_pilot.ps1`
   - new files still needed:
     none for the smoke path; the next gate is whether to spend compute on the
     10-seed pilot
   - manuscript destination:
     Table 2 if the bridge passes smoke cleanly; otherwise Discussion-only
   - result source:
     3-seed smoke reevaluation already completed under the current frozen paper
     protocol; a 10-seed pilot is still pending

### Appendix-only rows

8. `Zero-offset target-anchor comparator`
   - training config:
     `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_no_vpp_combat.yaml`
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_no_vpp_combat_best.yaml`
   - launchers:
     `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat.ps1`
     `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat_10seed_pilot.ps1`
   - new files still needed:
     none
   - manuscript destination:
     appendix unless the no-VPP path is redesigned to be action-sensitive
   - result source:
     appendix-only sanity evidence
   - current readiness note:
     smoke training is now validated, but the formal paper-protocol checkpoint
     is still missing

9. `LOS zero-offset guidance`
   - training config:
     none
   - evaluation config:
     `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_los_zero_offset.yaml`
   - launcher:
     `scripts/run_reset075_no_mode_switch_longscale00_rule_los_zero_offset_10seed_pilot.ps1`
   - new files still needed:
     none
   - manuscript destination:
     appendix or rebuttal backup
   - result source:
     optional 10-seed pilot output per opponent stage
   - current readiness note:
     3-seed smoke evaluation is now validated for both `expert` and
     `end_to_end`

## 5. Result Fields Required for All Paper Tables

For every row admitted to Table 1 or Table 2, extract and archive:

- `win_rate`
- `damage_margin`
- `ego_crashes`
- `target_crash_or_oob`
- `pre_merge_vp_forward_bias_m`
- raw termination semantics

Extraction rule:

- use `aggregate/method_task_summary.json` for outcome counts
- use `aggregate/combat_geometry_diagnostics.json` for geometry-health metrics
- use raw episode JSON for termination audit
- remember that aggregate `crashes` means
  `ego_crashes + target_crash_or_oob`

## 6. Execution Status Snapshot

### Already created

- unified Table 1 comparison config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- unified Table 1 10-seed pilot launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- no-VPP combat training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_no_vpp_combat.yaml`
- no-VPP comparison config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_no_vpp_combat_best.yaml`
- no-VPP training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat.ps1`
- no-VPP 10-seed pilot launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat_10seed_pilot.ps1`
- end-to-end combat training config:
  `config/experiment/train_end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat.yaml`
- end-to-end comparison config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_end_to_end_direct_command_best.yaml`
- end-to-end training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command.ps1`
- end-to-end 10-seed pilot launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command_10seed_pilot.ps1`
- rule-guidance bridge:
  `src/uav_vpp_guidance/evaluation/rule_guidance_policy.py`
- rule-guidance comparison-runner support:
  `scripts/run_jsbsim_hrl_comparison.py`
- PN comparison config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_pn_zero_offset.yaml`
- LOS comparison config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_los_zero_offset.yaml`
- PN 10-seed pilot launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_rule_pn_zero_offset_10seed_pilot.ps1`
- LOS 10-seed pilot launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_rule_los_zero_offset_10seed_pilot.ps1`
- legacy hierarchical bridge policy:
  `src/uav_vpp_guidance/evaluation/legacy_hierarchical_policy.py`
- legacy hierarchical bridge unit tests:
  `tests/test_legacy_hierarchical_policy.py`
- legacy hierarchical comparison-runner support:
  `scripts/run_jsbsim_hrl_comparison.py`
- legacy hierarchical evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge.yaml`
- legacy hierarchical smoke launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_smoke.ps1`
- legacy hierarchical 10-seed launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_10seed_pilot.ps1`
- registry entries:
  `config/checkpoint_registry.yaml`
  - `training.no_vpp_combat`
  - `training.end_to_end_combat`

### Dry-run validated on 2026-06-30

- Table 1 unified comparison config builds successfully
- no-VPP comparison config builds successfully
- end-to-end comparison config builds successfully
- PN zero-offset comparison config builds successfully
- LOS zero-offset comparison config builds successfully
- legacy hierarchical comparison config builds successfully
- the no-VPP comparison config required an evaluation-side pin to keep
  `virtual_point.mode: zero_offset` because the comparison runner defaults PPO
  methods back to `virtual_point.mode: normal`
- current missing items are checkpoint generation for
  `end_to_end_combat` plus any optional SAC work, not comparison-config syntax

### Smoke validated on 2026-06-30

- no-VPP smoke training completed:
  `outputs/experiments/smoke_validate_no_vpp_combat_fix1/`
- end-to-end smoke training completed:
  `outputs/experiments/smoke_validate_end_to_end_combat/`
- PN expert smoke completed:
  `outputs/jsbsim_hrl_comparison/reset075_rule_pn_expert_3seed_smoke_20260630/`
- PN end-to-end smoke completed:
  `outputs/jsbsim_hrl_comparison/reset075_rule_pn_e2e_3seed_smoke_20260630/`
- LOS expert smoke completed:
  `outputs/jsbsim_hrl_comparison/reset075_rule_los_expert_3seed_smoke_20260630/`
- LOS end-to-end smoke completed:
  `outputs/jsbsim_hrl_comparison/reset075_rule_los_e2e_3seed_smoke_20260630/`
- legacy hierarchical expert smoke completed:
  `outputs/jsbsim_hrl_comparison/reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_expert_3seed_smoke_20260630_fix1/`
- legacy hierarchical end-to-end smoke completed:
  `outputs/jsbsim_hrl_comparison/reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_end_to_end_3seed_smoke_20260630_fix1/`
- no-VPP training path required a `NoVPPGuidance` compatibility fix so the
  zero-offset generator accepts the current override keywords
- smoke produced complete artifact bundles with no bridge/runtime failures after
  the command-override compatibility fix in `tracking_env.py`
- smoke outcome mix was entirely timeout-dominated with zero ego crashes and
  zero target crash/out-of-bounds in all four lanes
- smoke evidence is enough to say the bridge is protocol-compatible
- smoke evidence is not yet enough to promote the legacy row into the main
  reviewer-facing table without a 10-seed pilot

### Not yet created

- SAC training bridge and evaluation config(s)

### Important current limitations

`scripts/run_jsbsim_hrl_comparison.py` currently supports only:

- `agent_type: ppo`
- `agent_type: end_to_end`
- `agent_type: oracle_task_gate`
- `agent_type: rule_guidance`

It does not yet support:

- `sac`

### Important interpretation risk

The current `no_vpp` path is **not** a clean learned-policy ablation.

`src/uav_vpp_guidance/virtual_point/no_vpp_guidance.py` always returns the
target position and ignores the policy action, so the current
`no_vpp_combat` assets should be described as a
`zero-offset target-anchor comparator`, not as a strong learned baseline.

For the fastest credible submission route:

- keep `no_vpp_combat` as appendix / internal sanity evidence
- do not rely on it as a main reviewer-facing learned comparator unless the
  no-VPP path is redesigned to be action-sensitive

## 7. Baseline-by-Baseline Implementation Sheet

### A. Baseline: Cartesian VPP anchor model

- Paper label:
  `Cartesian VPP baseline`
- Reviewer question answered:
  anchor point for all interface deltas
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml`
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- Training launcher:
  existing repo launcher for the baseline training path
- 10-seed evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- New files still needed:
  none
- Paper table role:
  Table 1
- Result source:
  rerun through the unified Table 1 config for the cleanest artifact bundle
  under the frozen paper protocol

### B. Broad geometry-basis combat finetune

- Paper label:
  `Geometry-basis broad finetune`
- Reviewer question answered:
  does a semantically reinterpreted 3-D action head reshape geometry at all
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.yaml`
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.ps1`
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- 10-seed evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- New files still needed:
  none
- Paper table role:
  Table 1
- Locked evidence:
  `outputs/diagnostics/tactical_basis_combat_finetune_10seed_pilot_report.md`
- Result source:
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_best_expert_10seed_20260629/`
  and
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_best_end_to_end_10seed_20260629/`

### C. Narrow geometry-basis combat finetune

- Paper label:
  `Geometry-basis narrow finetune`
- Reviewer question answered:
  can geometry-health repair recover the broad variant's head-on failure
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.yaml`
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.ps1`
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- 10-seed evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- New files still needed:
  none
- Paper table role:
  Table 1
- Locked evidence:
  `outputs/diagnostics/tactical_basis_combat_finetune_narrow_extents_10seed_pilot_report.md`
- Result source:
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_expert_10seed_20260629/`
  and
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_end_to_end_10seed_20260629/`

### D. Mixed crossing-restored geometry-basis combat finetune

- Paper label:
  `Geometry-basis mixed finetune`
- Reviewer question answered:
  can task-selective extents preserve head-on repair while recovering crossing
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.yaml`
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.ps1`
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- 10-seed evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- New files still needed:
  none unless a new mixed checkpoint supersedes the current one
- Paper table role:
  Table 1
- Current narrative role:
  preferred main-candidate interface variant
- Result source:
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_expert_10seed_20260629/`
  and
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_end_to_end_10seed_20260629/`

### E. Zero-offset / no-VPP PPO

- Paper label:
  `Zero-offset target-anchor comparator`
- Reviewer question answered:
  internal sanity check for removing VPP offset geometry
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_no_vpp_combat.yaml`
- Comparison config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_no_vpp_combat_best.yaml`
- Important evaluation guard:
  this config now pins `virtual_point.mode: zero_offset` through
  `method_def.config_overrides` so evaluation does not silently fall back to
  normal VPP mode
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat.ps1`
- 10-seed evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat_10seed_pilot.ps1`
- Comparison-runner support needed:
  none, already works as `agent_type: ppo`
- New files still needed:
  no additional files required before smoke validation
- Mandatory wording guard:
  do not describe this row as a strong learned comparator unless the no-VPP
  path is redesigned so the policy action matters
- Validation to run:
  dry-run comparison config
  smoke training seed 0 is now validated
  appendix-only pilot if you still want a zero-offset sanity row
- Paper table role:
  appendix by default
  promote only if redesigned into a true action-sensitive ablation

### F. End-to-end direct-command PPO

- Paper label:
  `End-to-end direct-command PPO`
- Reviewer question answered:
  is the interface layer necessary, or can a direct control policy match it
- Training config:
  `config/experiment/train_end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat.yaml`
- Comparison config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_end_to_end_direct_command_best.yaml`
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command.ps1`
- 10-seed evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command_10seed_pilot.ps1`
- Comparison-runner support needed:
  none, already works as `agent_type: end_to_end`
- New files still needed:
  no additional files required before smoke validation
- Validation to run:
  dry-run comparison config
  smoke train seed 0 is now validated
  10-seed pilot only after the formal checkpoint exists
- Paper table role:
  Table 2

### G. Zero-offset PN guidance

- Paper label:
  `PN zero-offset guidance`
- Reviewer question answered:
  can a classical rule-guidance law match or exceed the learned interface
- Existing reusable assets:
  `src/uav_vpp_guidance/guidance/proportional_navigation.py`
  `scripts/run_stage6g5d_pn_mode_switch_probe.py`
- Existing comparison support:
  `src/uav_vpp_guidance/evaluation/rule_guidance_policy.py`
  `scripts/run_jsbsim_hrl_comparison.py`
- Existing config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_pn_zero_offset.yaml`
- Existing launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_rule_pn_zero_offset_10seed_pilot.ps1`
- Training needed:
  no
- Validation order:
  config dry-run
  3-seed smoke bridge is now validated
  10-seed pilot
- Paper table role:
  Table 2
- Priority:
  highest among not-yet-implemented external baselines

### H. Zero-offset LOS guidance

- Paper label:
  `LOS zero-offset guidance`
- Reviewer question answered:
  how much of the gain comes from learning rather than deterministic geometry
  tracking
- Existing reusable assets:
  current VPP / guidance stack
- Existing comparison support:
  `src/uav_vpp_guidance/evaluation/rule_guidance_policy.py`
  `scripts/run_jsbsim_hrl_comparison.py`
- Existing config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_los_zero_offset.yaml`
- Existing launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_rule_los_zero_offset_10seed_pilot.ps1`
- Training needed:
  no
- Validation order:
  config dry-run
  3-seed smoke bridge is now validated
  10-seed pilot
- Paper table role:
  Table 2 only if PN is skipped; otherwise appendix or rebuttal backup

### I. Legacy hierarchical Aerospace baseline

- Paper label:
  `Legacy hierarchical pursuit-strategy baseline`
- Reviewer question answered:
  is the new geometry-basis interface materially better than the author's
  previous hierarchical action-space method
- Legacy assets already present:
  `E:\CloseAirCombat_control\baseline_results\checkpoints\proposed_ppo.zip`
  `E:\CloseAirCombat_control\baseline_results\logs\proposed_ppo_eval_log.csv`
  `E:\CloseAirCombat_control\baseline_results\run_config_table5.json`
- Important restriction:
  do not copy old metrics directly into the new paper tables
- Legacy source-of-truth scripts:
  `E:\CloseAirCombat_control\run_table5_r3_7_fixed.py`
  `E:\CloseAirCombat_control\run_table5_r3_7_complete.py`
- Legacy reference artifacts:
  `E:\CloseAirCombat_control\baseline_results\checkpoints\proposed_ppo.zip`
  `E:\CloseAirCombat_control\baseline_results_final\checkpoints\proposed_ppo.zip`
  `E:\CloseAirCombat_control\baseline_results_final\logs\proposed_ppo_eval_log.csv`
  `E:\CloseAirCombat_control\baseline_results_final\table5_comparison.md`
- Legacy bridge contract:
  reconstruct the old 16-D high-level observation from simulator state using
  the legacy `extract_high_level_obs` semantics, not from the current
  `observation_vector` wholesale
- Legacy action contract:
  `0 = lag`, `1 = lead`, `2 = pure`
- Existing bridge code already created:
  `src/uav_vpp_guidance/evaluation/legacy_hierarchical_policy.py`
- Existing unit tests already created:
  `tests/test_legacy_hierarchical_policy.py`
- Comparison-runner support already created:
  `scripts/run_jsbsim_hrl_comparison.py`
  now supports `agent_type: legacy_hierarchical`
- Runner hardening already created:
  SB3 `.zip` checkpoints now bypass the `torch.load` config/dim audit path and
  fall back to `config_path`
- Config already created:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge.yaml`
- Smoke launcher already created:
  `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_smoke.ps1`
- 10-seed launcher already created:
  `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_10seed_pilot.ps1`
- Training needed:
  no new training if `proposed_ppo.zip` can be loaded cleanly
- Smoke status:
  the 3-seed smoke checklist is now satisfied for both `expert` and
  `end_to_end`
- Paper table role:
  eligible for Table 2 10-seed piloting
  not yet table-ready from smoke evidence alone

### J. SAC-cartesian

- Paper label:
  `SAC Cartesian VPP`
- Reviewer question answered:
  is the observed gain tied to PPO, or does it persist under another
  continuous-control algorithm
- Current blocker:
  `src/uav_vpp_guidance/agents/sac_agent.py` is a stub
- Reusable precedent:
  `E:\CloseAirCombat_control\run_v5_1m.py`
  `E:\CloseAirCombat_control\run_v6_1m.py`
- New Python support required:
  `src/uav_vpp_guidance/training/train_sac_guidance.py`
  `src/uav_vpp_guidance/agents/sb3_sac_agent.py`
- Comparison-runner change required:
  extend `scripts/run_jsbsim_hrl_comparison.py` with `agent_type: sac`
- Config to create:
  `config/experiment/train_sac_prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml`
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_sac_cartesian_best.yaml`
- Launcher to create:
  `scripts/run_reset075_no_mode_switch_longscale00_sac_cartesian.ps1`
  `scripts/run_reset075_no_mode_switch_longscale00_sac_cartesian_10seed_pilot.ps1`
- Training needed:
  yes
- Paper table role:
  appendix or rebuttal-first, not on the fastest submission-critical path

## 8. Immediate Build Order

### Phase A: validate what already exists

1. dry-run unified Table 1 config
2. dry-run end-to-end comparator config
3. dry-run PN comparator config
4. dry-run LOS comparator config
5. smoke-train end-to-end seed 0

### Phase B: finish the minimum external comparator pack

1. generate the end-to-end direct-command checkpoint
2. run the end-to-end 10-seed pilot
3. run PN 3-seed smoke
4. promote PN to the 10-seed pilot if smoke passes
5. implement legacy hierarchical bridge
6. run legacy 3-seed smoke
7. promote legacy to the 10-seed pilot if smoke passes

### Phase C: only if time and compute permit

1. LOS 10-seed pilot as backup / appendix support
2. no-VPP redesign if you want a true learned ablation
3. SAC-cartesian
4. SAC-mixed geometry-basis

## 9. Main-Table Result Routing

### Table 1 must include

- baseline
- broad
- narrow
- mixed

Each row should report, for both `expert` and `end_to_end`, and for both
`head_on` and `crossing_feasible`:

- `win_rate`
- `damage_margin`
- `ego_crashes`
- `target_crash_or_oob`
- `pre_merge_vp_forward_bias_m`

### Table 2 should include, in this priority order

1. end-to-end direct-command PPO
2. PN zero-offset guidance
3. legacy hierarchical Aerospace baseline

If one of these is not ready by submission freeze:

- keep the row out of the main manuscript tables
- mention it in Discussion / Limitations as ongoing validation

### Appendix should hold

1. zero-offset target-anchor comparator (`no_vpp_combat`) unless redesigned
2. LOS zero-offset guidance
3. semantics-only geometry-basis MVP
4. any copied legacy outputs that were not reevaluated under the current
   protocol

## 10. Minimal Submission-Credible MVP

If the goal is the fastest credible submission without opening a larger compute
campaign, stop at:

1. Table 1 locked from existing internal family outputs
2. end-to-end PPO validated and piloted
3. PN bridge piloted
4. legacy hierarchical bridge smoke-tested and, if clean, piloted

This is the smallest pack that still answers:

- interface family trade-offs
- direct-control ablation
- classical guidance comparison
- self-evolution beyond the Aerospace hierarchical method

## 11. Bottom Line

The fastest credible route is:

1. keep Table 1 fixed as baseline + broad + narrow + mixed
2. finish only the minimum reviewer-facing comparator pack
3. keep the current `no_vpp` row out of the main reviewer story unless it is
   redesigned into a true action-sensitive ablation
4. treat `legacy hierarchical Aerospace baseline` as a bridge-validation task,
   not as a copy-paste baseline
5. leave SAC for appendix, rebuttal, or a later revision unless extra compute
   becomes available
