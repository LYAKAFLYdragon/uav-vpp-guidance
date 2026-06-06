# Stage 6H.0-R: Config Drift Audit — 6F.5A vs 6H.0-lite

**Audit date**: 2026-06-06T08:58:47.386474

## 1. Executive Summary

- **Total differences**: 15
- **Critical (affects physics / success criteria)**: 0
- **Moderate (naming, logging, output)**: 2

## 2. Critical Differences

| Config Path | Stage 6F.5A | Stage 6H.0-lite | Impact |
|---|---|---|---|

## 3. Stage 6F Scenarios vs 6H.0 Search Space

| Scenario | Range (m) | Aspect (°) | Closure (m/s) | Covered by 6H Search? |
|---|---|---|---|---|
| favorable | 800.0 | 0.0 | -70.0 | ❌ No |
| neutral | 2000.0 | 180.0 | -400.0 | ❌ No |
| disadvantage | 721.1 | 30.0 | 68.9 | ❌ No |
| challenging | 2121.3 | 180.0 | -410.0 | ❌ No |

## 4. Key Finding

**ALL Stage 6F scenarios fall OUTSIDE the Stage 6H.0-lite search space.**
The baseline search did not evaluate any geometry similar to those that succeeded in 6F.5A.
This is the root cause of the zero-candidate result — not a regression in VPP performance.

## 5. Recommendations

1. **Expand 6H.0 search** to include smaller ranges (800–1200m) and tail-chase aspects (0°).
2. **Replay Stage 6F scenarios** with the current checkpoint to confirm no code regression.
3. **Do not claim VPP has no non-tail-chase feasible region** until the search space covers historical successes.
