# Drones Comparison MVP Implementation Checklist

## 1. Purpose

This file is the current implementation bill of materials for the fastest
credible `Drones` submission route.

It answers three practical questions for every baseline:

1. which training and evaluation assets are the source of truth
2. whether any new config or script still needs to be created
3. whether the resulting row belongs in `Table 1`, `Table 2`, or the appendix

Paper wording should use `geometry-basis`.
Repo config and file names may keep the existing `tactical_basis` identifiers.

## 2. Frozen Paper Protocol

All baselines below must stay on this same protocol:

- main baseline remains `reset075_no_mode_switch_longscale00`
- do not promote `directtrack2000`
- combat-only scope remains `head_on + crossing_feasible`
- opponent split remains `expert + end_to_end`
- crossing evaluation must explicitly use
  `--attack-zone-close-range-max-aoa-deg 60`
- backend remains `jsbsim`
- pilot budget remains 10 seeds: `480-489`
- do not launch the formal held-out beyond this pilot path
- do not change the observation schema for this comparison MVP
- every YAML-loaded mutation must be recorded through the provenance contract

Every table-eligible run must archive:

- `aggregate/method_task_summary.json`
- `aggregate/combat_geometry_diagnostics.json`
- raw episode JSON
- a termination-mix audit derived from raw episodes

Interpretation guard:

- `crashes` in `method_task_summary.json` means
  `ego_crashes + target_crash_or_oob`
- never read `crashes` as ego crashes alone

## 3. Which Rows Belong in the Paper

### Table 1: main interface-family table

Only these four rows belong in the main paper table:

1. `Cartesian VPP baseline`
2. `Geometry-basis broad finetune`
3. `Geometry-basis narrow finetune`
4. `Geometry-basis mixed finetune`

### Table 2: reviewer-facing comparator table

These are the minimum external rows for the fastest credible submission:

1. `End-to-end direct-command PPO`
2. `Zero-offset PN guidance`
3. `Legacy hierarchical Aerospace baseline`

### Appendix or rebuttal only

Keep these out of the main manuscript path:

1. `Zero-offset LOS guidance`
2. `Zero-offset target-anchor comparator`
3. `Semantics-only geometry-basis MVP`
4. `SAC` variants unless a separate compute campaign is opened

## 4. Required Metrics for Every Table Row

For every row admitted to `Table 1` or `Table 2`, extract and preserve:

- `win_rate`
- `damage_margin`
- `ego_crashes`
- `target_crash_or_oob`
- `pre_merge_vp_forward_bias_m`
- raw termination semantics

Extraction rule:

- take outcome counts from `aggregate/method_task_summary.json`
- take geometry-health metrics from
  `aggregate/combat_geometry_diagnostics.json`
- verify termination semantics from raw episode JSON

## 5. Baseline-by-Baseline Implementation Cards

### 5.1 Cartesian VPP baseline

- Paper label:
  `Cartesian VPP baseline`
- Status:
  ready now
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml`
- Unified evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- Evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- New files still needed:
  none
- Remaining gate:
  no code gate; optionally rerun through the unified Table 1 config for a clean
  paper bundle
- Paper destination:
  `Table 1`
- Result source:
  the baseline row in the unified geometry-family 10-seed pilot bundle

### 5.2 Geometry-basis broad finetune

- Paper label:
  `Geometry-basis broad finetune`
- Status:
  ready now
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.yaml`
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.ps1`
- Unified evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- Evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- New files still needed:
  none
- Remaining gate:
  none
- Paper destination:
  `Table 1`
- Locked result bundles:
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_best_expert_10seed_20260629/`
  and
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_best_end_to_end_10seed_20260629/`
- Locked report:
  `outputs/diagnostics/tactical_basis_combat_finetune_10seed_pilot_report.md`

### 5.3 Geometry-basis narrow finetune

- Paper label:
  `Geometry-basis narrow finetune`
- Status:
  ready now
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.yaml`
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.ps1`
- Unified evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- Evaluation launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
- New files still needed:
  none
- Remaining gate:
  none
- Paper destination:
  `Table 1`
- Locked result bundles:
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_expert_10seed_20260629/`
  and
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_end_to_end_10seed_20260629/`
- Locked report:
  `outputs/diagnostics/tactical_basis_combat_finetune_narrow_extents_10seed_pilot_report.md`

### 5.4 Geometry-basis mixed finetune

- Paper label:
  `Geometry-basis mixed finetune`
- Status:
  ready now and currently the preferred main candidate
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.yaml`
- Training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.ps1`
- Unified evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
- Evaluation launchers:
  `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
  and
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_10seed_pilot.ps1`
- New files still needed:
  none
- Remaining gate:
  none unless this checkpoint is intentionally replaced
- Paper destination:
  `Table 1`
- Locked result bundles:
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_expert_10seed_20260629/`
  and
  `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_end_to_end_10seed_20260629/`
- Locked report:
  `outputs/diagnostics/tactical_basis_combat_finetune_mixed_crossing_restored_10seed_pilot_report.md`

### 5.5 End-to-end direct-command PPO

- Paper label:
  `End-to-end direct-command PPO`
- Status:
  code path ready; smoke training passed; formal checkpoint still missing
- Training config:
  `config/experiment/train_end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat.yaml`
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_end_to_end_direct_command_best.yaml`
- Launchers:
  `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command.ps1`
  and
  `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command_10seed_pilot.ps1`
- New files still needed:
  none
- Remaining gate:
  generate
  `outputs/experiments/end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat/checkpoints/best.pt`
  and then run the 10-seed pilot for both opponents
- Smoke evidence:
  `outputs/experiments/smoke_validate_end_to_end_combat/`
- Paper destination:
  `Table 2`
- Result source after completion:
  one 10-seed output directory for `expert` and one for `end_to_end`

### 5.6 Zero-offset PN guidance

- Paper label:
  `Zero-offset PN guidance`
- Status:
  bridge implemented; dry-run and 3-seed smoke already validated
- Training config:
  none
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_pn_zero_offset.yaml`
- Launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_rule_pn_zero_offset_10seed_pilot.ps1`
- New files still needed:
  none
- Remaining gate:
  run the 10-seed pilot for `expert` and `end_to_end`
- Smoke evidence:
  `outputs/jsbsim_hrl_comparison/reset075_rule_pn_expert_3seed_smoke_20260630/`
  and
  `outputs/jsbsim_hrl_comparison/reset075_rule_pn_e2e_3seed_smoke_20260630/`
- Paper destination:
  `Table 2`
- Result source after completion:
  one 10-seed output directory for `expert` and one for `end_to_end`

### 5.7 Legacy hierarchical Aerospace baseline

- Paper label:
  `Legacy hierarchical pursuit-strategy baseline`
- Status:
  bridge implemented; unit-tested; dry-run and 3-seed smoke completed
- Legacy checkpoint:
  `E:\CloseAirCombat_control\baseline_results\checkpoints\proposed_ppo.zip`
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge.yaml`
- Launchers:
  `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_smoke.ps1`
  and
  `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_10seed_pilot.ps1`
- Supporting code already in place:
  `src/uav_vpp_guidance/evaluation/legacy_hierarchical_policy.py`
  and
  `scripts/run_jsbsim_hrl_comparison.py`
- Supporting tests already in place:
  `tests/test_legacy_hierarchical_policy.py`
  and the legacy-related cases in
  `tests/test_jsbsim_hrl_comparison_runner.py`
- New files still needed:
  none
- Remaining gate:
  run the full 10-seed pilot for `expert` and `end_to_end`
- Smoke evidence:
  `outputs/jsbsim_hrl_comparison/reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_expert_3seed_smoke_20260630_fix1/`
  and
  `outputs/jsbsim_hrl_comparison/reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_end_to_end_3seed_smoke_20260630_fix1/`
- Paper destination:
  `Table 2` only after the 10-seed pilot
- If the 10-seed pilot is skipped:
  keep it in `Discussion` as compatibility evidence, not as a main table row

### 5.8 Zero-offset LOS guidance

- Paper label:
  `Zero-offset LOS guidance`
- Status:
  bridge implemented; dry-run and 3-seed smoke already validated
- Training config:
  none
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_los_zero_offset.yaml`
- Launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_rule_los_zero_offset_10seed_pilot.ps1`
- New files still needed:
  none
- Remaining gate:
  only if appendix or rebuttal support is wanted
- Paper destination:
  appendix or rebuttal backup

### 5.9 Zero-offset target-anchor comparator

- Paper label:
  `Zero-offset target-anchor comparator`
- Status:
  smoke training path validated; formal checkpoint still missing; still
  action-insensitive by design
- Training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_no_vpp_combat.yaml`
- Evaluation config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_no_vpp_combat_best.yaml`
- Launchers:
  `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat.ps1`
  and
  `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat_10seed_pilot.ps1`
- Supporting compatibility fix:
  `src/uav_vpp_guidance/virtual_point/no_vpp_guidance.py`
- New files still needed:
  none
- Remaining gate:
  if you still want an appendix row, generate
  `outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_no_vpp_combat/checkpoints/best.pt`
- Smoke evidence:
  `outputs/experiments/smoke_validate_no_vpp_combat_fix1/`
- Mandatory wording guard:
  do not describe this as a strong learned ablation because the current
  `no_vpp` path ignores the policy action
- Paper destination:
  appendix only

## 6. File-Creation Inventory Still Open

For the fastest submission MVP, there are no remaining must-create config or
script files for:

- `Table 1` internal rows
- `End-to-end direct-command PPO`
- `Zero-offset PN guidance`
- `Legacy hierarchical Aerospace baseline`

The remaining blockers are compute and checkpoint generation, not missing file
plumbing.

New files become necessary only if the scope is expanded beyond the fastest
submission route, especially for `SAC`.

## 7. Results Routing Into the Manuscript

### Main paper table

Use exactly these rows:

1. `Cartesian VPP baseline`
2. `Geometry-basis broad finetune`
3. `Geometry-basis narrow finetune`
4. `Geometry-basis mixed finetune`

### Reviewer-facing comparator table

Use these rows once their pilot outputs are ready:

1. `End-to-end direct-command PPO`
2. `Zero-offset PN guidance`
3. `Legacy hierarchical pursuit-strategy baseline`

### Appendix

Keep these outside the main story:

1. `Zero-offset LOS guidance`
2. `Zero-offset target-anchor comparator`
3. `Semantics-only geometry-basis MVP`

## 8. Fastest Execution Order From Today

1. Keep `Table 1` fixed as `baseline + broad + narrow + mixed`.
2. Do not open a new algorithm branch yet.
3. Generate the formal `end_to_end` checkpoint.
4. Run the PN 10-seed pilot.
5. Run the legacy hierarchical 10-seed pilot.
6. Run the `end_to_end` 10-seed pilot.
7. Add LOS or no-VPP only if appendix space or rebuttal support is needed.

## 9. Explicit Non-Goals For This MVP

Do not spend the fastest submission window on:

- formal held-out expansion beyond the 10-seed pilot
- promoting `directtrack2000`
- turning `no_vpp` into a headline learned baseline
- SAC implementation
- APN or no-prediction redesign unless the paper is deferred into a larger
  compute cycle

## 10. Bottom Line

The current MVP is no longer blocked by missing comparison scripts or configs.
It is now a compute-allocation problem:

- `Table 1` is already file-complete
- `PN` is evaluation-ready
- `legacy hierarchical` is bridge-ready
- `end_to_end` is code-ready but still needs the formal checkpoint

That is the smallest implementation set that still supports a credible
interface-design and geometry-diagnostics paper for `Drones`.
