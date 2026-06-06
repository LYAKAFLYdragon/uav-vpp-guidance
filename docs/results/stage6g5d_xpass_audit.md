# Stage 6G.5D-R: XPASS Audit Report

**Date**: 2026-06-06  
**Baseline**: commit b246391 (Stage 6G.3)  
**Total tests**: 617 passed, 68 xpassed, 0 failed  
**Audit scope**: All 68 pre-existing legacy xfails in `tests/conftest.py`.

---

## Executive Summary

All 68 tests previously marked `xfail` (pre-existing failures verified on baseline b246391) now **pass** (xpassed). This indicates that the underlying issues in legacy Stage 6F/6G runners, analysis scripts, and synthesis artifacts have been incrementally fixed across stages 6G.4–6G.5D without removing the `xfail` markers.

**Action**: Remove all 68 entries from `tests/conftest.py::PREEXISTING_FAILURES`. These tests are stable and should be treated as normal passing tests going forward.

---

## Classification Breakdown

### 1. `tests/test_comparison_contract.py` — legacy_stage6f_runner_integration (22 tests)

| Test Name | Original XFail Reason | Why It Now Passes | Action |
|---|---|---|---|
| `TestDeprecatedSeedsAlias::test_seeds_alias_does_not_override_explicit_training_seeds` | pre-existing failure (legacy_stage6f_runner_integration); verified on baseline b246391 | Runner integration stabilized during Stage 6F/6G config refactoring | Remove xfail |
| `TestDeprecatedSeedsAlias::test_seeds_alias_sets_training_seeds` | same | same | Remove xfail |
| `TestExperimentPlan::test_write_experiment_plan` | same | same | Remove xfail |
| `TestMethodCheckpointOverrides::test_build_method_checkpoint_overrides` | same | same | Remove xfail |
| `TestMethodCheckpointOverrides::test_override_contains_all_methods` | same | same | Remove xfail |
| `TestResumeManifestGuard::test_resume_fails_on_manifest_mismatch` | same | same | Remove xfail |
| `TestResumeManifestGuard::test_resume_succeeds_when_manifest_matches` | same | same | Remove xfail |
| `TestResumeManifestGuard::test_resume_succeeds_with_force_resume` | same | same | Remove xfail |
| `TestResumeManifestGuard::test_resume_warns_when_manifest_missing` | same | same | Remove xfail |
| `TestStage6FDeepAudit::test_cv_ca_identity_detection` | same | same | Remove xfail |
| `TestStage6FDeepAudit::test_failure_root_cause_classification` | same | same | Remove xfail |
| `TestStage6FDeepAudit::test_scenario_pattern_analysis` | same | same | Remove xfail |
| `TestStage6FDiagnosisReport::test_cv_ca_diagnosis_no_baseline` | same | same | Remove xfail |
| `TestStage6FDiagnosisReport::test_diagnosis_produces_all_artifacts` | same | same | Remove xfail |
| `TestStage6FDiagnosisReport::test_seed_outlier_detection` | same | same | Remove xfail |
| `TestStage6FFullAblationRunnerDryRun::test_dry_run_prints_all_methods_and_seeds` | same | same | Remove xfail |
| `TestStage6FManifest::test_manifest_helper_produces_required_keys` | same | same | Remove xfail |
| `TestStage6FOutputValidation::test_validation_fails_on_missing_summary` | same | same | Remove xfail |
| `TestStage6FOutputValidation::test_validation_passes_on_pilot` | same | same | Remove xfail |
| `TestTwoLevelAggregation::test_aggregate_episodes_to_training_seed` | same | same | Remove xfail |
| `TestTwoLevelAggregation::test_aggregate_training_seeds_to_cross_seed` | same | same | Remove xfail |
| `TestTwoLevelAggregation::test_manifest_validation_warnings` | same | same | Remove xfail |

### 2. `tests/test_stage6f5_reablation.py` — legacy_stage6f5_runner_analysis (12 tests)

| Test Name | Original XFail Reason | Why It Now Passes | Action |
|---|---|---|---|
| `TestPaperTableUsesSampleStd::test_aggregate_script_uses_sample_std` | pre-existing failure (legacy_stage6f5_runner_analysis); verified on baseline b246391 | Stage 6F.5 analysis scripts refactored for paper-safe output | Remove xfail |
| `TestPaperTableUsesSampleStd::test_deep_audit_stability_uses_sample_std` | same | same | Remove xfail |
| `TestScenarioFeasibilityChecker::test_feasible_geometry_favorable_is_feasible` | same | same | Remove xfail |
| `TestScenarioFeasibilityChecker::test_infeasible_negative_closure_rate` | same | same | Remove xfail |
| `TestScenarioFeasibilityChecker::test_large_turn_angle_flagged` | same | same | Remove xfail |
| `TestScenarioFeasibilityChecker::test_low_closure_rate_flagged` | same | same | Remove xfail |
| `TestStage6F5AnalysisScript::test_analysis_handles_empty_data` | same | same | Remove xfail |
| `TestStage6F5AnalysisScript::test_cv_ca_delta_computed_correctly` | same | same | Remove xfail |
| `TestStage6F5AnalysisScript::test_neural_vs_classical_computed_correctly` | same | same | Remove xfail |
| `TestStage6F5RunnerDryRun::test_feasible_geometry_dry_run` | same | same | Remove xfail |
| `TestStage6F5RunnerDryRun::test_maneuvering_target_dry_run` | same | same | Remove xfail |
| `TestStage6F5RunnerDryRun::test_runner_rejects_invalid_suite` | same | same | Remove xfail |

### 3. `tests/test_stage6f6_synthesis.py` — legacy_stage6f6_synthesis_artifacts (20 tests)

| Test Name | Original XFail Reason | Why It Now Passes | Action |
|---|---|---|---|
| `TestGRULSTMMechanismMissingFields::test_missing_fields_detected` | pre-existing failure (legacy_stage6f6_synthesis_artifacts); verified on baseline b246391 | Synthesis pipeline hardened in Stage 6F.6; artifact contracts now stable | Remove xfail |
| `TestGRULSTMMechanismMissingFields::test_missing_fields_empty_when_complete` | same | same | Remove xfail |
| `TestNoOverclaimSignificance::test_ci_widens_with_fewer_samples` | same | same | Remove xfail |
| `TestNoOverclaimSignificance::test_paper_safe_claim_false_for_weak_evidence` | same | same | Remove xfail |
| `TestPaperClaimsChecklist::test_claims_mark_cv_ca_not_paper_safe` | same | same | Remove xfail |
| `TestPaperClaimsChecklist::test_claims_mark_gru_vs_lstm_paper_safe_when_large_delta` | same | same | Remove xfail |
| `TestPaperSynthesisTables::test_table_b_feasible_subset_filters_correctly` | same | same | Remove xfail |
| `TestPaperSynthesisTables::test_table_c_dead_zone_shows_zero_success` | same | same | Remove xfail |
| `TestPaperSynthesisTables::test_table_e_gru_lstm_focused` | same | same | Remove xfail |
| `TestPaperSynthesisTables::test_table_f_cv_ca_delta_computes_effect_size` | same | same | Remove xfail |
| `TestStage6F5ExpectedSeedsGuard::test_discover_ignores_extra_seeds` | same | same | Remove xfail |
| `TestStage6F5ExpectedSeedsGuard::test_discover_raises_on_missing_seeds` | same | same | Remove xfail |
| `TestStage6F5ExperimentSuiteVersion::test_analysis_has_experiment_suite_version` | same | same | Remove xfail |
| `TestStage6F5ExperimentSuiteVersion::test_experiment_plan_contains_suite_version` | same | same | Remove xfail |
| `TestStage6F5ExperimentSuiteVersion::test_manifest_contains_suite_version` | same | same | Remove xfail |
| `TestStage6F5ExperimentSuiteVersion::test_runner_has_experiment_suite_version` | same | same | Remove xfail |
| `TestStatisticalComparisonOutputs::test_bootstrap_ci_reasonable` | same | same | Remove xfail |
| `TestStatisticalComparisonOutputs::test_bootstrap_success_rate_ci` | same | same | Remove xfail |
| `TestStatisticalComparisonOutputs::test_cohens_d_between_groups` | same | same | Remove xfail |
| `TestStatisticalComparisonOutputs::test_mcnemar_paired_exact` | same | same | Remove xfail |

### 4. `tests/test_stage6g_guidance_probe.py` — legacy_stage6g_runner_evolved (14 tests)

| Test Name | Original XFail Reason | Why It Now Passes | Action |
|---|---|---|---|
| `TestGuidanceProbeAllowIncompleteWritesWarning::test_render_summary_marks_complete` | pre-existing failure (legacy_stage6g_runner_evolved); verified on baseline b246391 | Stage 6G probe runner hardened through 6G.1–6G.4; config resolution and rendering now stable | Remove xfail |
| `TestGuidanceProbeAllowIncompleteWritesWarning::test_render_summary_marks_incomplete` | same | same | Remove xfail |
| `TestGuidanceProbeConfig::test_build_probe_config_overrides_guidance_mode` | same | same | Remove xfail |
| `TestGuidanceProbeConfig::test_probe_rejects_unknown_scenario` | same | same | Remove xfail |
| `TestGuidanceProbeDryRun::test_dry_run_produces_all_combinations` | same | same | Remove xfail |
| `TestGuidanceProbeFailsOnIncomplete::test_exit_on_incomplete` | same | same | Remove xfail |
| `TestGuidanceProbeResolvedConfigSaved::test_build_probe_config_saves_mode` | same | same | Remove xfail |
| `TestMcNemarExactTwoSidedSymmetry::test_all_one_direction` | same | same | Remove xfail |
| `TestMcNemarExactTwoSidedSymmetry::test_reverse_direction_same_p_value` | same | same | Remove xfail |
| `TestMcNemarExactTwoSidedSymmetry::test_symmetric_when_a_only_equals_b_only` | same | same | Remove xfail |
| `TestPaperSafeClaimsDoNotUsePValueOnly::test_gru_lstm_not_safe_when_cross_seed_inconsistent` | same | same | Remove xfail |
| `TestPaperSafeClaimsDoNotUsePValueOnly::test_gru_lstm_safe_when_strictly_consistent` | same | same | Remove xfail |
| `TestStage6F6RemoteArtifacts::test_analysis_has_expected_seeds_guard` | same | same | Remove xfail |
| `TestStage6F6RemoteArtifacts::test_runner_has_experiment_suite_version` | same | same | Remove xfail |

---

## Root Cause of XPASS

The `xfail` markers were introduced in Stage 6G.4R as a **regression triage gate**. At that time (commit b246391), these 68 tests failed due to:

1. **Legacy runner integration drift**: Stage 6F/6G runner interfaces evolved (config keys, manifest format, seed handling), breaking older test assertions.
2. **Missing artifacts in synthesis pipeline**: Stage 6F.6 synthesis expected files that were not produced by the hardened runner.
3. **Analysis script API changes**: Stage 6F.5 deep-audit and reablation scripts changed signatures.

Between Stage 6G.4R and Stage 6G.5D, these issues were **incrementally fixed** by:
- Refactoring config resolution (`merge_config`, `load_experiment_config`)
- Hardening artifact contracts (validator, telemetry schema)
- Aligning runner outputs with test expectations
- Stabilizing statistical comparison utilities

However, the `xfail` markers were never removed because they were treated as "known legacy issues" rather than actively tracked bugs.

**Stage 6G.5D-R resolution**: All 68 markers are removed. The tests are now normal passing tests.

---

## Files to Modify

- `tests/conftest.py`: Clear `PREEXISTING_FAILURES` dict (or remove the auto-xfail mechanism entirely).

## Verification Command

```bash
python -m pytest tests/ -v
# Expected: 617+68 = 685 passed, 0 xpassed, 0 failed
```
