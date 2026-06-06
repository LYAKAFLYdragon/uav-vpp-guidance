# Stage 6G.5D-R Final Report

**Date**: 2026-06-06  
**Branch**: `feature/los-guidance-deep-hardening`  
**Commit**: `a2c9e01`  
**Tests**: 685 passed, 0 failed, 0 xpassed

---

## Task A: Remote Sync & Origin Status

### 1. Does origin feature branch contain the latch fix?

**Yes.**

```bash
$ git show origin/feature/los-guidance-deep-hardening:src/uav_vpp_guidance/envs/tracking_env.py | grep -n "mode_switch_latched"
142:        self._mode_switch_latched = False
229:        self._mode_switch_latched = False
443:                self._mode_switch_latched = True
444:            if self._mode_switch_latched:
```

Commit `a2c9e01` pushed to origin successfully.

---

## Task B: README & Paper-Safe Claims

### 2. Is README synced to 6G.5D/6G.5D-R?

**Yes.** Stage table updated:

| Stage | Status |
|---|---|
| 6G.5D | ✅ Complete |
| 6G.5D-R | 🧪 In Progress |
| 6H.0-lite | ⏳ Pre-unblocked |
| 6H (full) | ⏳ Gated |

Last updated footer reflects: "Stage 6G.5D-R in progress | Mode-switch threshold optimization pre-unblocked; full bilevel remains gated until near-threshold robustness smoke passes"

### 3. Are old paper-safe claims corrected?

**Yes.**

- Removed/superseded: "Tail-chase remains infeasible across guidance laws" (now marked ❌ Not paper-safe, superseded by 6G.5C).
- Updated: Combined "Pure PN without VPP" and "latched PN mode-switch" into a single scoped claim.
- Added: "Mode-switch with PN latch rescues VPP-based architectures" with explicit scope (pt20/pt29/pt38, 90/90).
- Added: "Mode-switch threshold 15°/3000m/100mps is sufficient for tested candidates".

All claims adhere to paper-safe rule: scope-limited, cross-seed verified, no universal language.

---

## Task C: 68 XPASS Tests

### 4. How were the 68 xpassed tests handled?

**Cleared all legacy xfail markers.**

- Root cause: Pre-existing failures from Stage 6G.4R baseline (b246391) were incrementally fixed across stages 6G.4–6G.5D but markers were never removed.
- Action: Emptied `PREEXISTING_FAILURES` dict in `tests/conftest.py`.
- Audit document: `docs/results/stage6g5d_xpass_audit.md` records all 68 tests, their original xfail reasons, why they now pass, and the recommended action.
- Verification: `pytest tests/ -q` → **685 passed, 0 xpassed, 0 failed**.

---

## Task D: Latch Robustness Tests

### 5. Does the latch reset correctly?

**Yes.** Test `test_latch_resets_on_env_reset` passes.

- `env.reset()` sets `self._mode_switch_latched = False`.
- Verified across 3 seeds and multiple episodes.

### 6. Does PN guidance state reset correctly?

**Yes.** Test `test_pn_guidance_state_resets_on_env_reset` passes.

- `env.reset()` calls `self._guidance_pn.reset()` (fixed in 6G.5D).
- `_prev_los_vec`, `_prev_time`, `_filtered_los_rate` are all set to `None` after reset.
- Prevents stale LOS filter state from contaminating subsequent episodes.

---

## Task E: Small Robustness Smoke

### 7. Does near-threshold robustness pass?

**Partial. Key findings from smoke (13 scenarios × 6 variants × 15 episodes = 1,170 total):**

| Scenario | pure_pn | mode_switch_latched | mode_switch_vpp_elsewhere | vpp_los | vpp_pn |
|---|---|---|---|---|---|
| candidate_pt20/pt29/pt38 | 15/15 | 15/15 | 15/15 | 0/15 | 0/15 |
| near_range_1800 | 15/15 | 15/15 | 15/15 | 0/15 | 0/15 |
| near_range_2400 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| near_aspect_10/15/20 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| neg_aspect_60/90 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| neg_low_ego / low_closing | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| regression_favorable | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |

**Interpretation:**
- ✅ **Candidates confirmed**: 100% success on pt20/pt29/pt38 across all mode-switch variants.
- ✅ **Latch does not cause false activations**: Negative controls (aspect 60°/90°, low speed) correctly do not trigger latch; all variants crash as expected.
- ⚠️ **Pure PN success region is very narrow**: Even aspect=10° or range=2400m causes pure PN to fail. This is a **guidance-law limitation of pure PN itself**, not a latch bug.
- ⚠️ **regression_favorable is infeasible under current config**: ego=220, target=180, range=2000 fails for ALL variants (pure PN, VPP+LOS, VPP+PN). This suggests either: (a) the simple backend dynamics differ from Stage 6F's JSBSim runs, or (b) success thresholds have changed. **This is NOT a mode-switch regression** because the baseline itself fails.

### 8. Does mode-switch harm non-tail-chase feasible geometries?

**Inconclusive from current smoke.**

- The chosen `regression_favorable` geometry (ego=220, target=180, range=2000, aspect=0) fails for ALL variants including pure_pn_no_vpp and vpp_policy_los.
- This means it is **not a feasible geometry under the current simple-backend configuration**, so we cannot measure "regression" vs baseline.
- **Action needed**: Identify a geometry that is genuinely feasible with VPP+LOS in the simple backend, then re-test with `mode_switch_vpp_elsewhere` to confirm no degradation.
- However, `mode_switch_vpp_elsewhere` uses VPP+LOS when the gate is inactive. For geometries where gate does NOT activate (high aspect, low closing speed), its behavior is **identical** to `vpp_policy_los`. The smoke confirms this: both crash on the same negative controls.

---

## Task F: Threshold Optimization Pre-Gate

### 9. Can we enter Stage 6H.0-lite threshold optimization?

**Yes, with a precondition.**

The mode-switch latch mechanism is **robust and ready for threshold optimization**:
- Latch activates correctly on eligible geometries.
- Latch persists for the full episode.
- Latch resets cleanly between episodes.
- Negative controls do not false-trigger.
- Telemetry is complete and auditable.

**Precondition before starting 6H.0-lite grid search:**
Find and validate at least one **genuinely feasible non-tail-chase geometry** in the simple backend. Only then can we confirm that `mode_switch_vpp_elsewhere` does not degrade performance on feasible geometries when the gate remains inactive.

Plan document: `docs/stage6h0_lite_mode_switch_threshold_optimization_plan.md`

---

## Task G: Full Bilevel Status

### 10. Does full bilevel still need to wait?

**Yes. Full bilevel remains gated.**

Reasons:
1. **6H.0-lite must complete first**: Threshold optimization acceptance criteria (≥95% candidate success, ≤5pp regression on feasible geometries, auditable telemetry) must be met.
2. **No validated non-tail-chase feasible geometry in simple backend yet**: Need to confirm VPP+LOS can succeed on at least one simple-backend geometry before bilevel can optimize over it.
3. **Bilevel formulation requires stable initialization**: The outer-level optimizer needs a known-good threshold region to start from. 6H.0-lite provides this.

---

## Summary of Files Changed

| File | Action | Description |
|---|---|---|
| `src/uav_vpp_guidance/envs/tracking_env.py` | Modified | Episode latch + PN reset fix |
| `tests/conftest.py` | Modified | Cleared 68 legacy xfail markers |
| `tests/test_stage6g5d_pn_mode_switch.py` | Added | 9 contract tests (gate, bypass, latch) |
| `tests/test_stage6g5d_latch_robustness.py` | Added | 9 robustness tests (persistence, reset, telemetry, negative controls) |
| `scripts/run_stage6g5d_pn_mode_switch_probe.py` | Added | Stage 6G.5D runner |
| `scripts/run_stage6g5d_latch_robustness_smoke.py` | Added | Stage 6G.5D-R robustness smoke runner |
| `scripts/analyze_stage6g5c_vpp_offset_mechanism.py` | Added | VPP offset diagnostic |
| `docs/results/stage6g5d_xpass_audit.md` | Added | XPASS audit report |
| `docs/stage6h0_lite_mode_switch_threshold_optimization_plan.md` | Added | Threshold optimization plan |
| `README.md` | Modified | Stage table, claims, last updated |
| `memory/2026-06-05.md` | Modified | Completion notes |

---

*Report generated: 2026-06-06 | Commit: a2c9e01 | Branch: feature/los-guidance-deep-hardening*
