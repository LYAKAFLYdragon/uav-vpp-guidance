# Stage 6H.0-F: Pre-Existing Failures Audit

**Audit date**: 2026-06-06
**Commit**: `fd91ec8` (revert) + working-tree fixes
**Full suite result**: **716 passed, 0 failed, 0 xpassed**

---

## 1. Executive Summary

All previously reported pre-existing failures have been **resolved**.

- **Previously reported**: 68 failures (or 12 in subset runs)
- **Root cause**: `tests/conftest.py` did not include the project root in `sys.path`, causing `ModuleNotFoundError` for all tests that import from `scripts/` (which lacks `__init__.py`)
- **Fix**: Add `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` to `tests/conftest.py`
- **Result after fix**: 716 passed, 0 failed

---

## 2. Failure Breakdown (Before Fix)

| Test File | # Failed | Failure Pattern | Root Cause |
|---|---|---|---|
| `tests/test_comparison_contract.py` | 21 | `ModuleNotFoundError: scripts.run_stage6f_full_ablation` | Missing project root in sys.path |
| `tests/test_stage6f5_reablation.py` | 12 | `ModuleNotFoundError: scripts.run_stage6f5_reablation` | Missing project root in sys.path |
| `tests/test_stage6f6_synthesis.py` | 17 | `ModuleNotFoundError: scripts.analyze_stage6f5_results` | Missing project root in sys.path |
| `tests/test_stage6g_guidance_probe.py` | 18 | `ModuleNotFoundError: scripts.run_stage6g_guidance_limitation_probe` | Missing project root in sys.path |

**Total**: 68 failures, all identical pattern.

---

## 3. Fix Details

**File**: `tests/conftest.py`

**Change**:
```python
import os
import sys

# Allow `from scripts.xxx import ...` in tests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

**Rationale**: The `scripts/` directory contains runnable Python modules but no `__init__.py`. Pytest's default `sys.path` does not include the project root when tests are invoked from the repository root on all platforms. Adding the project root ensures `from scripts.xxx import ...` resolves correctly.

---

## 4. Verification

```bash
pytest tests/ -q
# Result: 716 passed, 0 failed, 0 xpassed
```

---

## 5. Action Log

| Test | Action | Status |
|---|---|---|
| All 68 previously-failing tests | Fix environment (sys.path) | ✅ Resolved |
| No xfail markers needed | N/A — all pass | ✅ Verified |

---

## 6. Paper Impact

> **None.** These were environment/import issues, not logic failures. The fix does not change any test assertions, behavior, or paper-safe claims.

---

*Last updated: 2026-06-06 | Stage 6H.0-F*
