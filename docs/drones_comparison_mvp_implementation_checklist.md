# Drones Comparison MVP Implementation Checklist

## 1. Purpose

This checklist turns the current comparison-baseline discussion into an
execution-ready MVP plan for the `Drones` submission.

Paper terminology should use `geometry-basis`.
Implementation terminology in the current repo should keep the existing
`tactical_basis` config and file names unless a separate repo-wide rename is
scheduled.

## 2. Frozen Protocol

All new baselines must be brought onto the same paper protocol:

- main baseline remains `reset075_no_mode_switch_longscale00`
- do not promote `directtrack2000`
- combat-only scope is `head_on + crossing_feasible`
- opponent split is `expert + end_to_end`
- crossing entry must explicitly use `--attack-zone-close-range-max-aoa-deg 60`
- backend must be `jsbsim`
- pilot budget remains 10 seeds: `480-489`
- do not launch a formal `480-seed` held-out
- every YAML mutation must be recorded through the provenance override contract
- output pack must include:
  - `aggregate/method_task_summary.json`
  - `aggregate/combat_geometry_diagnostics.json`
  - raw episode JSONs
  - termination-mix audit

## 3. What Should Go Where

### Table 1: Main paper table

Keep the paper-core interface family compact:

- baseline:
  `prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00`
- broad finetune:
  `..._tactical_basis_headon_mvp_combat_finetune_best`
- narrow finetune:
  `..._tactical_basis_headon_mvp_combat_finetune_narrow_extents_best`
- mixed finetune:
  `..._tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best`

Do not put the semantics-only MVP into Table 1.
Keep it in method-bridge text or appendix.

### Table 2: External comparator table

This should answer reviewer objections without bloating the main narrative:

- end-to-end direct-command PPO
- zero-offset PN guidance
- legacy hierarchical Aerospace baseline

Optional appendix / rebuttal rows:

- zero-offset LOS guidance
- zero-offset target-anchor comparator (`no_vpp_combat`)
- SAC-cartesian
- SAC-mixed geometry-basis

### Appendix / method bridge only

- semantics-only tactical/geometry-basis MVP
- failed mixed candidates that do not beat the current mixed-crossing-restored
  checkpoint
- any legacy result that was not re-evaluated under the current paper protocol

## 4. Required Metrics for Every Baseline

Every baseline that is allowed into Table 1 or Table 2 must report, per lane:

- `win_rate`
- `damage_margin`
- `ego_crashes`
- `target_crash_or_oob`
- `pre_merge_vp_forward_bias_m`
- raw termination semantics

Recommended extraction rule:

- take `win_rate`, `ego_crashes`, `target_crash_or_oob`, `crashes`, `timeouts`
  from `aggregate/method_task_summary.json`
- take `pre_merge_vp_forward_bias_m` from
  `aggregate/combat_geometry_diagnostics.json`
- take `damage_margin` from the same aggregated output if present; otherwise
  compute it as `mean_damage_dealt - mean_damage_taken` and label the derivation
  explicitly in the paper/source-data sheet
- always verify termination mix from raw episode JSON because aggregate
  `crashes` means `ego_crashes + target_crash_or_oob`

## 5. Execution Order

### P0: already paper-critical and mostly available

1. unify the four internal interface variants into one comparison config
2. lock Table 1 from the existing 10-seed pilot outputs
3. clean the raw termination audit for the mixed variant

### P1: minimum external comparator pack

1. end-to-end direct-command PPO
2. PN guidance
3. legacy hierarchical Aerospace baseline

### P2: algorithm-choice pack

1. SAC-cartesian
2. SAC-mixed geometry-basis

### P3: non-blocking appendix support

1. LOS guidance
2. zero-offset target-anchor comparator (`no_vpp_combat`) unless redesigned

If compute or coding bandwidth collapses, P1 is higher priority than P2 except
that at least one SAC comparator is still strongly recommended for rebutting the
algorithm-choice objection.

## 6. Baseline-by-Baseline Implementation Matrix

### 6.1 Internal paper-core family

#### A. Main baseline

- Status: already exists and remains the anchor
- Existing training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml`
- Existing evaluation method key:
  `prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00`
- Existing result role:
  Table 1
- New files to create:
  - `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_geometry_family_main_table.yaml`
    This should contain all four Table 1 methods in one config.
  - `scripts/run_reset075_no_mode_switch_longscale00_geometry_family_10seed_pilot.ps1`
    This should call `run_jsbsim_hrl_comparison.py` once for `expert` and once
    for `end_to_end`.
- New training needed: no
- Result source for Table 1:
  existing baseline pilot outputs plus the unified rerun config if you want one
  artifact bundle

#### B. Broad finetune

- Status: already exists
- Existing training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.yaml`
- Existing training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.ps1`
- Existing eval config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_best.yaml`
- Existing pilot outputs:
  - `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_best_expert_10seed_20260629/`
  - `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_best_end_to_end_10seed_20260629/`
- Existing summary report:
  `outputs/diagnostics/tactical_basis_combat_finetune_10seed_pilot_report.md`
- Result role:
  Table 1
- New files to create:
  none beyond the unified Table 1 config and runner
- New training needed: no

#### C. Narrow finetune

- Status: already exists
- Existing training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.yaml`
- Existing training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.ps1`
- Existing eval config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents_best.yaml`
- Existing pilot outputs:
  - `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_expert_10seed_20260629/`
  - `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_narrow_extents_best_end_to_end_10seed_20260629/`
- Existing summary report:
  `outputs/diagnostics/tactical_basis_combat_finetune_narrow_extents_10seed_pilot_report.md`
- Result role:
  Table 1
- New files to create:
  none beyond the unified Table 1 config and runner
- New training needed: no

#### D. Mixed finetune

- Status: already exists and is the current best main-candidate checkpoint
- Existing training config:
  `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.yaml`
- Existing training launcher:
  `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.ps1`
- Existing eval config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best.yaml`
- Existing 10-seed pilot launchers:
  - `scripts/run_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_10seed_pilot.ps1`
- Existing pilot outputs:
  - `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_expert_10seed_20260629/`
  - `outputs/jsbsim_hrl_comparison/tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best_end_to_end_10seed_20260629/`
- Current observed headline:
  - `expert/head_on`: `win_rate 0.60 -> 1.00`, `ego_crashes 2 -> 0`
  - `expert/crossing_feasible`: `win_rate 0.30 -> 0.90`, `ego_crashes 1 -> 0`
  - `end_to_end/head_on`: `win_rate 0.90 -> 0.90`, `ego_crashes 1 -> 1`
  - `end_to_end/crossing_feasible`: `win_rate 1.00 -> 1.00`, but still heavily
    target-failure dominated
- Result role:
  Table 1
- New files to create:
  none beyond the unified Table 1 config and runner
- New training needed: no, unless you choose to supersede this checkpoint

#### E. Semantics-only geometry-basis MVP

- Status: already exists
- Existing eval config:
  `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp.yaml`
- Result role:
  appendix / method bridge only
- New files to create:
  none unless you want a dedicated appendix runner
- New training needed: no

### 6.2 External comparator pack

#### F. Zero-offset / no-VPP PPO

- Reviewer question answered:
  appendix-only sanity check for removing VPP offset geometry
- Existing reusable assets:
  - `config/experiment/train_no_vpp_ppo.yaml`
  - `config/experiment/train_no_vpp_direct_command.yaml`
  - `scripts/run_no_vpp_baseline.py`
  - `scripts/run_p0a_vpp_ablation.sh`
  - current comparison runner can load it as ordinary `agent_type: ppo`
- Important implementation note:
  `src/uav_vpp_guidance/virtual_point/no_vpp_guidance.py` ignores the policy
  action and always returns the target position, so the current `no_vpp_combat`
  path should not be sold as a strong learned comparator without redesign
- Important note:
  do not reuse the old simple-backend result directly; train and evaluate under
  the current combat-only JSBSim protocol
- New config to create:
  - `config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_no_vpp_combat.yaml`
    Derived from the main baseline training config, but force:
    `virtual_point.mode: zero_offset`
  - optional alias config:
    `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_no_vpp_combat_best.yaml`
- New launcher to create:
  - `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat.ps1`
  - `scripts/run_reset075_no_mode_switch_longscale00_no_vpp_combat_10seed_pilot.ps1`
- Python code changes needed:
  none if the zero-offset mode already behaves correctly under
  `train_prediction_vpp_ppo`
- Result role:
  appendix by default
  promote only if redesigned into a true action-sensitive learned ablation

#### G. End-to-end direct-command PPO

- Reviewer question answered:
  is the guidance/VPP layer necessary, or does direct command work as well?
- Existing reusable assets:
  - `config/experiment/train_end_to_end_ppo.yaml`
  - `scripts/train_end_to_end_baseline.py`
  - `scripts/run_end_to_end_ppo_multi_seed.py`
  - comparison runner already supports `agent_type: end_to_end`
- Important note:
  this repo's end-to-end baseline is already a cleaner 3-D direct-command
  ablation, not raw 4D actuator output
- New config to create:
  - `config/experiment/train_end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat.yaml`
    Derived from the current paper protocol, not from the legacy simple-backend
    config
  - `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_end_to_end_direct_command_best.yaml`
- New launcher to create:
  - `scripts/run_end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat_multi_seed.py`
  - `scripts/run_reset075_no_mode_switch_longscale00_end_to_end_direct_command_10seed_pilot.ps1`
- Python code changes needed:
  probably none beyond the new config and registry entries
- Result role:
  Table 2

#### H. Zero-offset LOS guidance

- Reviewer question answered:
  how much of the gain comes from learning at all, versus a deterministic
  geometry-tracking law?
- Existing reusable assets:
  - `src/uav_vpp_guidance/guidance/`
  - comparison runner infrastructure
- Existing created support:
  - `src/uav_vpp_guidance/evaluation/rule_guidance_policy.py`
  - `scripts/run_jsbsim_hrl_comparison.py`
  - `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_los_zero_offset.yaml`
  - `scripts/run_reset075_no_mode_switch_longscale00_rule_los_zero_offset_10seed_pilot.ps1`
- Training needed:
  no
- Result role:
  appendix or rebuttal backup

#### I. Zero-offset PN guidance

- Reviewer question answered:
  does the proposed interface outperform a classical PN-style guidance baseline?
- Existing reusable assets:
  - `src/uav_vpp_guidance/guidance/proportional_navigation.py`
  - `scripts/run_stage6g5d_pn_mode_switch_probe.py`
- Existing created support:
  - `src/uav_vpp_guidance/evaluation/rule_guidance_policy.py`
  - `scripts/run_jsbsim_hrl_comparison.py`
  - `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_rule_pn_zero_offset.yaml`
  - `scripts/run_reset075_no_mode_switch_longscale00_rule_pn_zero_offset_10seed_pilot.ps1`
- Training needed:
  no
- Result role:
  Table 2
- Priority note:
  if time is extremely tight, LOS is cheaper; PN is the stronger review-facing
  classical baseline

#### J. SAC-cartesian

- Reviewer question answered:
  is PPO special here, or does the interface still help under another
  continuous-control RL algorithm?
- Current repo status:
  `src/uav_vpp_guidance/agents/sac_agent.py` is still a stub and cannot be used
- Reusable precedent:
  old SB3-SAC code exists in `E:\\CloseAirCombat_control\\run_v5_1m.py`
  and `run_v6_1m.py`
- New Python files to create:
  - `src/uav_vpp_guidance/training/train_sac_guidance.py`
  - `src/uav_vpp_guidance/agents/sb3_sac_agent.py`
    or replace the current stub in `agents/sac_agent.py`
- Existing Python files to modify:
  - `scripts/run_jsbsim_hrl_comparison.py`
    Add `agent_type: sac`
- New config to create:
  - `config/experiment/train_sac_prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml`
  - `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_sac_cartesian_best.yaml`
- New launcher to create:
  - `scripts/run_reset075_no_mode_switch_longscale00_sac_cartesian.ps1`
  - `scripts/run_reset075_no_mode_switch_longscale00_sac_cartesian_10seed_pilot.ps1`
- Training needed:
  yes
- Result role:
  appendix or rebuttal-first, not on the fastest submission-critical path

#### K. SAC-mixed geometry-basis

- Reviewer question answered:
  is the mixed geometry-basis advantage tied to PPO, or does the interface
  transfer to SAC as well?
- Reuses the same new SAC training/eval bridge as SAC-cartesian
- New config to create:
  - `config/experiment/train_sac_prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_mixed_crossing_restored.yaml`
  - `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_sac_mixed_crossing_restored_best.yaml`
- New launcher to create:
  - `scripts/run_reset075_no_mode_switch_longscale00_sac_mixed_crossing_restored.ps1`
  - `scripts/run_reset075_no_mode_switch_longscale00_sac_mixed_crossing_restored_10seed_pilot.ps1`
- Training needed:
  yes
- Result role:
  appendix or rebuttal-first, not on the fastest submission-critical path
- Priority note:
  if only one SAC baseline fits the schedule, do SAC-cartesian first

#### L. Legacy hierarchical Aerospace baseline

- Reviewer question answered:
  what is the real gain over the author's prior hierarchical discrete-to-
  continuous interface?
- Existing assets:
  - paper/project source tree:
    `E:\\CloseAirCombat_control`
  - old PPO result bundles:
    `E:\\CloseAirCombat_control\\baseline_results*`
  - old model/checkpoint assets such as:
    `E:\\CloseAirCombat_control\\baseline_results\\checkpoints\\proposed_ppo.zip`
- Legacy bridge contract that must be preserved:
  - reconstruct the old 16-D observation with legacy
    `extract_high_level_obs` semantics from simulator state
  - keep the original discrete action meaning:
    `0 = lag`, `1 = lead`, `2 = pure`
- Important warning:
  old results are not protocol-compatible by default and must not be copied into
  the new paper table without reevaluation
- Existing bridge code already created:
  - `src/uav_vpp_guidance/evaluation/legacy_hierarchical_policy.py`
    This wrapper reconstructs the legacy 16-D observation, decodes
    `lag/lead/pure`, and emits a current-comparison compatible command
    override
- Existing unit tests already created:
  - `tests/test_legacy_hierarchical_policy.py`
- Existing Python files to modify:
  - `scripts/run_jsbsim_hrl_comparison.py`
    Add a new `agent_type`, for example `legacy_hierarchical`
- Additional runner hardening still required:
  - make `.zip` SB3 checkpoints bypass the current `torch.load` config/dim
    audit path and fall back to `config_path`
- New config to create:
  - `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge.yaml`
- New launcher to create:
  - `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_smoke.ps1`
  - `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_10seed_pilot.ps1`
- Training needed:
  no new training if the old checkpoint can be loaded directly
- Validation step required before counting it:
  - confirm action mapping matches the old paper semantics
  - confirm observation bridge is not leaking unavailable state
  - run a 3-seed smoke bridge before the 10-seed pilot
- Result role:
  Table 2 if the bridge works cleanly
  otherwise Discussion-only self-evolution evidence

## 7. Remaining Files to Create Next

Live file-by-file status should be read from
`docs/drones_comparison_mvp_execution_sheet.md`.

From the current repo state, the next genuinely missing files on the fastest
credible route are:

1. `config/experiment/jsbsim_hrl_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge.yaml`
2. `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_smoke.ps1`
3. `scripts/run_reset075_no_mode_switch_longscale00_legacy_hierarchical_bridge_10seed_pilot.ps1`
4. runner integration in `scripts/run_jsbsim_hrl_comparison.py` for
   `agent_type: legacy_hierarchical`

If you later decide to extend beyond the minimum submission route, the next
missing files after legacy are:

1. `src/uav_vpp_guidance/training/train_sac_guidance.py`
2. `src/uav_vpp_guidance/agents/sb3_sac_agent.py`
3. `config/experiment/train_sac_prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml`
4. `config/experiment/train_sac_prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_mixed_crossing_restored.yaml`

If you later want a true no-VPP learned ablation rather than an appendix
zero-offset comparator, schedule a separate redesign pass for:

1. an action-sensitive no-VPP guidance path
2. its paired train/eval configs
3. a new 10-seed pilot launcher

Current non-file blocker:

1. `stable_baselines3` is not available in the active Python environment, so
   the legacy bridge cannot yet load `proposed_ppo.zip` locally

## 8. Fastest Submission-Grade MVP

If you want the smallest plan that can still survive a serious methods review,
stop at:

- Table 1 internal family:
  baseline + broad + narrow + mixed
- Table 2 external comparators:
  end-to-end PPO + PN + legacy hierarchical

Then add SAC-cartesian if time allows.
Only add SAC-mixed after SAC-cartesian is running.
Keep `no_vpp_combat` in the appendix unless it is redesigned into a true
action-sensitive ablation.

## 9. Recommended Artifact Layout

For each new baseline, keep one evaluation directory per opponent:

- `outputs/jsbsim_hrl_comparison/<run_id>_expert_10seed_<date>/`
- `outputs/jsbsim_hrl_comparison/<run_id>_end_to_end_10seed_<date>/`

Each run should be traced back to:

- one train config
- one eval config
- one launcher
- one checkpoint
- one pilot report markdown

## 10. Bottom Line

The fastest credible route is not to explode the comparison matrix.
It is to lock the four-row internal family first, then add exactly the
review-facing baselines that answer:

- is the geometry-basis interface better than raw VPP scaling?
- is it better than direct-command PPO?
- is it better than a classical guidance law?
- is it materially different from the earlier hierarchical Aerospace method?
- is the observed gain still visible under at least one non-PPO algorithm?
