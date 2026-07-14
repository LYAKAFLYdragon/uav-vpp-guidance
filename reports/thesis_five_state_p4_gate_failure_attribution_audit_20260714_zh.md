# P4 几何门失败归因审计 (GEO-20260713-R1)

Generated (UTC): 2026-07-13T23:41:46.290403+00:00

## 结论

在既有的预注册门限下，**不允许**启动 combat finetune（`combat_finetune_NOT_allowed`）；本审计无权更改该结论。

## Preflight 2a: declared-vs-emitted reconciliation

- declared = 26, emitted = 26, diff = 0, extra = 0
- diff set is EMPTY; the historical 26/22/4 premise is unverified / disproven / structurally impossible under `thesis_shared_skill_geometry.py:511-546` (full Cartesian product, absent filled with 0).

## 真实失败归因 (Real failure attribution)

1. `geometry_phase_coverage = false` 完全由 `post_merge` / `re_entry` 阶段中**已声明且以数值 0 发出**的单元（`present_value_0`）驱动，**并非**由任何缺失/不存在（`absent`）的单元造成。历史上 26/22/4 （`crossing_entry missing`）的前提**不成立 / 已被反证 / 在结构上不可能**（`thesis_shared_skill_geometry.py:511-546` 发出完整笛卡尔积，缺失组合以 0 填充）：已声明的 `crossing_entry` 单元均为存在且非零（`crossing_entry:pre_merge=1092`, `crossing_entry:post_merge=182`, `crossing_entry:post_merge=181`, `crossing_entry:re_entry=5`）。

2. 各技能 intent 进度汇总（阈值 0.55；分数为每技能常量，取自 `selected_gate.intent_progress_fraction`）：
   - `defensive_extension`: intent_progress_fraction = 0.5325 (threshold 0.55) -> skill_intent_progress_passed = false (FAIL)
   - `lead_intercept`: intent_progress_fraction = 0.4977 (threshold 0.55) -> skill_intent_progress_passed = false (FAIL)
   - `pursuit_conversion`: intent_progress_fraction = 0.5193 (threshold 0.55) -> skill_intent_progress_passed = false (FAIL)
   - `reentry_recovery`: intent_progress_fraction = 0.5820 (threshold 0.55) -> skill_intent_progress_passed = true (PASS)

3. `reentry_recovery` 通过 intent 进度检查（`skill_intent_progress_passed = true`），但仍因 `geometry_phase_coverage` 失败而被阻断。

4. 各技能的 `present_value_0` dev-gate 单元（`dev_gate_field_status == present_value_0`，即驱动阶段覆盖失败的零单元）：
   - `defensive_extension`: `disadvantage:post_merge`, `disadvantage:re_entry`, `neutral:post_merge`, `neutral:re_entry`
   - `lead_intercept`: `advantage:post_merge`, `neutral:post_merge`
   - `pursuit_conversion`: `advantage:post_merge`, `neutral:post_merge`
   - `reentry_recovery`: `advantage:post_merge`, `advantage:re_entry`, `neutral:post_merge`, `neutral:re_entry`

5. 全部 4 个技能均通过 `profile_conditioning_coverage` 与 `safety` 检查（证据：`selected_gate.checks.profile_conditioning_coverage` / `selected_gate.checks.safety`），此为予以保留的通过性证据（Req 3.9），不得被重述为失败。

## Gate semantics

- coverage unit = policy_steps (policy steps)
- gate_min_steps = 2 (`thesis_shared_skill_geometry.py:544` / `thesis_shared_skill_geometry.py:561`)
- intent_progress = intent_progress_steps / policy_steps; threshold = 0.55 (`thesis_shared_skill_geometry.py:564`; numerator `thesis_shared_skill_geometry.py:451`)

## Frozen provenance

- provenance_unverified inputs: training_plan, resolved_geometry_config, dev30_manifest, run_commit
- run-level contract_or_provenance_mismatch: true
- run_commit: provenance_unverified
  - `gate_json` available=true sha256=144df30e2151d1f6d15fb3704432bec149c5f5e510a3388e6fe709e9e3b6c56a
  - `registry` available=true sha256=2a06c21b9a8af4a94b7e0c3b644b2e3298d2f5b5c95dbf992d61c0cbfd7b4f92 (provenance_unverified)
  - `training_plan` available=false sha256=n/a (provenance_unverified)
  - `resolved_geometry_config` available=false sha256=n/a (provenance_unverified)
  - `dev30_manifest` available=false sha256=n/a (provenance_unverified)
  - `summary_pursuit_conversion` available=true sha256=7f0cdcda23f5e750bf64208831f0af43484eeaae1dae3f1430648eee1347e9a4
  - `summary_lead_intercept` available=true sha256=e6046432c75bc3de5835f9967e480c9414e8bedf58239a16253b2e96c7ad7f38
  - `summary_defensive_extension` available=true sha256=a92abf7d9f0686349128ea44b1fcdfa86bfaa8714d58ad8eb1e2bca2b908c24a
  - `summary_reentry_recovery` available=true sha256=69f95662f1d4fbae98c1388cb6922552e2be196be650b10e79b9faad4a2abe48
- working_tree (SEPARATE from frozen run facts): head=6dd61d54ca86775d6d0dd4c6ed32e7e7cdb733e4, branch=research/thesis-five-state-shared-intent-v1, dirty=True
- host/interpreter note: audit executed on Python 3.11.14 (Windows AMD64); stdlib only, no third-party dependencies

## initial-class x phase limitation

Coverage is recorded keyed on the scenario's fixed `initial_class` (`thesis_shared_skill_geometry.py:451`, call sites 677/806). This audit proves ONLY initial-class x phase coverage and CANNOT prove dynamic five-state posture during post_merge / re_entry.

## Attribution matrix

| skill | initial_class | phase | train_field_status | dev_gate_field_status | emission_status | train_steps | dev_steps | cell_gate_passed | attribution_label | reason_code | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| defensive_extension | disadvantage | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 100110 | 1278 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| defensive_extension | disadvantage | post_merge | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| defensive_extension | disadvantage | re_entry | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in re_entry is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| defensive_extension | neutral | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 99890 | 1278 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| defensive_extension | neutral | post_merge | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| defensive_extension | neutral | re_entry | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in re_entry is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| lead_intercept | advantage | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 63721 | 1278 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| lead_intercept | advantage | post_merge | present_value_gt_0 | present_value_0 | emitted | 2920 | 0 | false | not_observed_cannot_assess | dev_coverage_gap | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| lead_intercept | neutral | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 66662 | 1278 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| lead_intercept | neutral | post_merge | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| lead_intercept | crossing_entry | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 41053 | 1092 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| lead_intercept | crossing_entry | post_merge | present_value_gt_0 | present_value_gt_0 | emitted | 24321 | 182 | true | mixed_or_inconclusive | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| pursuit_conversion | advantage | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 66882 | 1278 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| pursuit_conversion | advantage | post_merge | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| pursuit_conversion | head_on | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 3603 | 198 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| pursuit_conversion | head_on | post_merge | present_value_gt_0 | present_value_gt_0 | emitted | 55705 | 948 | true | mixed_or_inconclusive | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| pursuit_conversion | neutral | pre_merge | present_value_gt_0 | present_value_gt_0 | emitted | 66456 | 1278 | true | mixed_or_inconclusive |  | moderate: coverage proven for fixed initial_class x phase (pre_merge); no dynamic-posture claim required |
| pursuit_conversion | neutral | post_merge | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | advantage | post_merge | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | advantage | re_entry | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in re_entry is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | head_on | post_merge | present_value_gt_0 | present_value_gt_0 | emitted | 40567 | 951 | true | mixed_or_inconclusive | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | head_on | re_entry | present_value_gt_0 | present_value_gt_0 | emitted | 5721 | 129 | true | mixed_or_inconclusive | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in re_entry is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | neutral | post_merge | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | neutral | re_entry | absent | present_value_0 | emitted | absent | 0 | false | train_coverage_gap | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in re_entry is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | crossing_entry | post_merge | present_value_gt_0 | present_value_gt_0 | emitted | 24427 | 181 | true | mixed_or_inconclusive | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in post_merge is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |
| reentry_recovery | crossing_entry | re_entry | present_value_gt_0 | present_value_gt_0 | emitted | 1184 | 5 | true | mixed_or_inconclusive | initial_class_phase_limitation | low: coverage proven only for fixed initial_class x phase; dynamic five-state posture in re_entry is UNPROVEN (initial_class_phase_limitation, thesis_shared_skill_geometry.py:451) |

