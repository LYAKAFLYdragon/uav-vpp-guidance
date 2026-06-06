# Stage 6G.5D-R Final Report

**Date**: 2026-06-06  
**Branch**: `feature/los-guidance-deep-hardening`  
**Commit**: `a2c9e01`  
**Previous**: `1a7abd2`

---

## Summary

Stage 6G.5D-R completed all seven tasks (A–G). The critical mode-switch latch bug was fixed, all 68 legacy xfail markers were cleared, robustness tests and smoke runners were added, and the threshold optimization pre-gate plan was drafted. Remote sync verified.

---

## Task A: Remote Sync & Origin Status

| Check | Result |
|---|---|
| `git status` | Clean working tree after commit |
| Current branch | `feature/los-guidance-deep-hardening` |
| Local log | `a2c9e01` (6G.5D-R) on top of `1a7abd2` (6G.5C) |
| Origin log | Now synchronized to `a2c9e01` |
| Latch fix on origin | ✅ Confirmed (`mode_switch_latched` present in tracking_env.py) |
| README on origin | ✅ Confirmed (6G.5D Complete, 6G.5D-R In Progress, 6H.0-lite Pre-unblocked) |
| Tests on origin | ✅ Confirmed (new test files present) |

**Push status**: Successfully pushed `a2c9e01` to `origin/feature/los-guidance-deep-hardening`.

---

## Task B: README & Paper-Safe Claims

### Stage Table Updates

| Stage | Status |
|---|---|
| 6G.5A–6G.5D | ✅ Complete |
| 6G.5D-R | 🧪 In Progress |
| 6H.0-lite | ⏳ Pre-unblocked |
| 6H (full) | ⏳ Gated |

### Claims Modified

1. **Removed**: "Tail-chase remains infeasible across guidance laws" (already marked ❌, no longer relevant).
2. **Updated**: "Pure PN without VPP succeeds on three high-energy tail-chase candidates" → merged with latched mode-switch claim into a single scoped claim.
3. **Added**: "Pure PN without VPP and latched PN mode-switch succeed on three tested high-energy tail-chase candidate geometries" — scope explicitly limited to pt20/pt29/pt38, seeds 0/1/2, 10 episodes each.
4. **Added**: "Mode-switch with PN latch rescues VPP-based architectures" — backed by 90/90 success on `mode_switch_vpp_elsewhere`.
5. **Added**: "Mode-switch threshold 15°/3000m/100mps is sufficient for tested candidates" — backed by gate firing step 1 on all 90 episodes.

### Constraints Enforced

- No claim of universal tail-chase feasibility.
- No claim of full bilevel validation.
- No claim that VPP is universally harmful.

---

## Task C: 68 XPASS Tests

### Action Taken

- Generated `docs/results/stage6g5d_xpass_audit.md` with full classification and rationale.
- Cleared `PREEXISTING_FAILURES` dict in `tests/conftest.py`.
- All 68 legacy xfail markers removed.

### Verification

```
Before: 617 passed, 68 xpassed
After:  685 passed, 0 xpassed
```

### Root Cause

The 68 tests were marked xfail in Stage 6G.4R as "pre-existing failures" on baseline `b246391`. Between 6G.4R and 6G.5D, the underlying runner integration, synthesis pipeline, and analysis scripts were incrementally hardened. The tests now pass consistently.

---

## Task D: Latch Robustness Tests

New file: `tests/test_stage6g5d_latch_robustness.py` (9 tests, all passing)

| Test | Purpose | Status |
|---|---|---|
| `test_latch_persists_when_aspect_exceeds_threshold` | Gate active step 1; latch persists even when aspect grows beyond threshold | ✅ PASS |
| `test_latch_resets_on_env_reset` | `_mode_switch_latched` cleared on `env.reset()` | ✅ PASS |
| `test_pn_guidance_state_resets_on_env_reset` | `_guidance_pn._prev_los_vec` cleared on `env.reset()` | ✅ PASS |
| `test_latch_does_not_activate_for_high_aspect` | 90° aspect → gate inactive | ✅ PASS |
| `test_latch_does_not_activate_for_low_closing_speed` | closing speed < threshold → gate inactive | ✅ PASS |
| `test_latch_does_not_activate_for_long_range` | range > threshold → gate inactive | ✅ PASS |
| `test_vpp_before_gate_pn_after_latch` | Monkey-patched delayed gate: VPP before activation, direct-track PN after latch | ✅ PASS |
| `test_default_policy_is_hold_for_episode` | No implicit exit; latch holds for full episode | ✅ PASS |
| `test_telemetry_contains_all_latch_fields` | All required telemetry keys present | ✅ PASS |

---

## Task E: Small Robustness Smoke

New file: `scripts/run_stage6g5d_latch_robustness_smoke.py`

Run: 13 scenarios × 6 variants × 3 seeds × 5 episodes = **1,170 episodes**

### Key Results

| Scenario | pure_pn | mode_switch_latched | mode_switch_hysteresis | mode_switch_min_hold | vpp_los | vpp_pn |
|---|---|---|---|---|---|---|
| candidate_pt20 | 15/15 | 15/15 | 15/15 | 15/15 | 0/15 | 0/15 |
| candidate_pt29 | 15/15 | 15/15 | 15/15 | 15/15 | 0/15 | 0/15 |
| candidate_pt38 | 15/15 | 15/15 | 15/15 | 15/15 | 0/15 | 0/15 |
| near_aspect_10 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| near_aspect_15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| near_aspect_20 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| near_range_1800 | 15/15 | 15/15 | 15/15 | 15/15 | 0/15 | 0/15 |
| near_range_2400 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| neg_aspect_60 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| neg_aspect_90 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| neg_low_ego | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| neg_low_closing | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| regression_favorable | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |

### Interpretation

1. **Candidate success confirmed**: All mode-switch variants replicate pure-PN success on pt20/pt29/pt38.
2. **Near-threshold aspect (10°–20°)**: Even pure PN fails. This indicates the tail-chase feasible region is **extremely narrow** — aspect must be effectively 0°.
3. **Near-threshold range (1800m OK, 2400m fail)**: Pure PN success is range-sensitive.
4. **Negative controls**: All variants fail as expected. Gate does not activate on high aspect / low speed / low closing speed.
5. **Regression_favorable**: All variants fail (0/15). This specific geometry (ego=220, target=180, range=2000) is **not feasible under current simple-backend parameters**. This is **not** a mode-switch regression — baseline VPP+LOS and pure PN also fail.

> ⚠️ **Important caveat**: The `regression_favorable` geometry was selected from Stage 6F config, but Stage 6F may have used different success thresholds or backend parameters. A true regression test requires identifying a geometry that is actually feasible with VPP+LOS in the current setup.

---

## Task F: Threshold Optimization Pre-Gate Plan

New file: `docs/stage6h0_lite_mode_switch_threshold_optimization_plan.md`

Key elements:
- Search space: aspect_enter (10–25°), range_enter (1500–3000m), closing_speed_enter (80–160mps), hold_policy variants.
- Evaluation protocol: candidate + near-threshold + negative control + regression geometries.
- Acceptance criteria before Stage 6H:
  1. Candidates ≥95% success
  2. Negative controls ≤5% false activation
  3. Stage 6F feasible geometries ≤5pp degradation
  4. 100% telemetry audit coverage
  5. No random policy fallback

---

## Task G: Answers to 10 Questions

| # | Question | Answer |
|---|---|---|
| 1 | Origin feature branch contains latch fix? | **Yes.** Verified via `git show origin/...:tracking_env.py \| grep mode_switch_latched`. Commit `a2c9e01` pushed successfully. |
| 2 | README synced to 6G.5D/6G.5D-R? | **Yes.** Stage table updated; 6G.5D Complete, 6G.5D-R In Progress, 6H.0-lite Pre-unblocked, 6H full Gated. Last-updated footer revised. |
| 3 | Old paper-safe claims corrected? | **Yes.** Removed/superseded outdated "infeasible across guidance laws" claim. Added scoped claims for pure PN + latched mode-switch. No universal claims. |
| 4 | 68 xpassed handled? | **Yes.** All 68 legacy xfail markers removed from `tests/conftest.py`. Audit report generated. Test suite now reports **685 passed, 0 xpassed**. |
| 5 | Latch resets? | **Yes.** `test_latch_resets_on_env_reset` confirms `_mode_switch_latched = False` after `env.reset()`. |
| 6 | PN guidance state resets? | **Yes.** `test_pn_guidance_state_resets_on_env_reset` confirms `_guidance_pn._prev_los_vec = None` after `env.reset()`. |
| 7 | Near-threshold robustness passes? | **Partial.** Candidates and negative controls behave as expected. However, pure PN itself fails on near_aspect_10 and near_range_2400, revealing the **tail-chase feasible region is narrower than the gate threshold** — the gate does not cause these failures, but it also cannot rescue geometries where pure PN is inherently unstable. This is a guidance-law limitation, not a latch bug. |
| 8 | Mode-switch harms non-tail-chase feasible geometries? | **Inconclusive from current smoke.** The `regression_favorable` geometry (ego=220, target=180) fails on **all** variants including pure PN and VPP+LOS baseline, suggesting it is not feasible under current parameters. A true regression test requires a confirmed feasible non-tail-chase geometry. |
| 9 | Can enter Stage 6H.0-lite threshold optimization? | **Yes, with caveat.** The latch mechanism is robust and auditable. The threshold space is well-defined. However, 6H.0-lite should first identify a **genuine feasible non-tail-chase geometry** for regression testing before claiming the search space is safe. |
| 10 | Full bilevel still needs to wait? | **Yes.** Full bilevel (joint VPP policy + gain optimization) remains gated. 6H.0-lite threshold grid search must pass acceptance criteria first. |

---

## Next Recommended Actions

1. **Identify true feasible non-tail-chase geometry**: Run a small sweep on Stage 6F geometries with current backend/config to find at least one where VPP+LOS succeeds. Use this as the regression baseline.
2. **Run Stage 6H.0-lite threshold grid search**: Execute the plan from `docs/stage6h0_lite_mode_switch_threshold_optimization_plan.md`.
3. **Lock threshold config**: Once grid search passes acceptance criteria, write `config/experiment/stage6h0_locked_mode_switch.yaml`.
4. **Proceed to full bilevel only after 6H.0-lite acceptance**.

---

*Report generated: 2026-06-06 | Commit: a2c9e01 | Branch: feature/los-guidance-deep-hardening*
