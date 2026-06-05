# Stage 6G Guidance-Law Limitation Probe

**Version**: 6G.1 hardened  
**Date**: 2026-06-05  
**Status**: ✅ Complete (full probe executed, 720 episodes, 12 cells, McNemar exact)

---

## 1. Research Question

Is the tail-chase / stern-conversion dead zone observed in Stage 6E a **guidance-law limitation** (specific to LOS-rate guidance) or a **geometric/physics infeasibility** (any guidance law would fail)?

### 1.1 Background

Stage 6E showed that neural prediction (GRU, LSTM) improves tracking in feasible-geometry scenarios (favorable geometry, weaving pursuit). However, in **tail-chase / stern-conversion** engagements, the policy consistently fails regardless of prediction method. This raises the question: is the failure due to the **guidance law** (LOS-rate) or the **geometry itself** (stern conversion is inherently difficult)?

### 1.2 Hypotheses

| Hypothesis | Prediction | Test |
|---|---|---|
| H1: LOS-rate limitation | PN or hybrid would succeed where LOS-rate fails | Compare success rates across guidance laws |
| H2: Geometric infeasibility | All guidance laws fail in tail-chase | Compare success rates across guidance laws |
| H3: Policy limitation | Even with better guidance, the policy cannot track | No prediction vs GRU frozen under same guidance |

---

## 2. Experimental Matrix

### 2.1 Variables

- **Independent Variable 1**: Guidance law (`los_rate`, `proportional_navigation`, `hybrid`)
- **Independent Variable 2**: Prediction method (`no_prediction`, `gru_frozen`)
- **Independent Variable 3**: Scenario (`favorable`, `disadvantage`, `weaving_pursuit`, `weaving_disadvantage`)
- **Independent Variable 4**: Evaluation seed (`0`, `1`, `2`)

### 2.2 Fixed Parameters

- Training seed: `0`
- Episodes per scenario: `10`
- Backend: `simple` (flat-earth, point-mass, 6-DOF)
- High-level dt: `0.2` s

### 2.3 Total Episodes

```
3 guidance × 4 scenarios × 2 methods × 10 episodes × 3 seeds = 720 episodes
```

### 2.4 Probe Cells

Each cell is a unique combination of `(guidance_mode, scenario, method)`:

| Cell | Guidance | Scenario | Methods | Episodes | Seeds |
|---|---|---|---|---|---|
| 1 | los_rate | favorable | no_prediction, gru_frozen | 10 | 0,1,2 |
| 2 | los_rate | disadvantage | no_prediction, gru_frozen | 10 | 0,1,2 |
| ... | ... | ... | ... | ... | ... |
| 12 | hybrid | weaving_disadvantage | no_prediction, gru_frozen | 10 | 0,1,2 |

---

## 3. Output Artifacts

### 3.1 Required Files

```
<run_id>/
  resolved_config.yaml          ← Final effective config (paths, modes, seeds, git commit)
  run_manifest.json             ← Audit: start/end, hostname, python version, git commit,
                                    dirty status, command line, run status
  raw_episodes.csv              ← One row per episode
  scenario_method_summary.csv   ← Aggregated per scenario × method × guidance
  pairwise_mcnemar.csv          ← Exact McNemar p-values for paired comparisons
  paper_safe_claims.md          ← Claim status table with reasons
  README_result_block.md        ← Copy-paste ready block for README embedding
  run.log                       ← Full execution log
```

### 3.2 raw_episodes.csv Schema

| Column | Type | Description |
|---|---|---|
| `scenario` | str | Scenario name |
| `method` | str | `no_prediction` or `gru_frozen` |
| `guidance_mode_requested` | str | The guidance mode requested in config |
| `effective_guidance_mode` | str | The actual guidance mode instantiated |
| `training_seed` | int | Training seed (always 0) |
| `evaluation_seed` | int | Evaluation seed (0, 1, 2) |
| `episode_index` | int | Episode index within the cell |
| `success` | bool | Whether the episode was a success |
| `termination_reason` | str | `success`, `crash`, `out_of_bounds`, `timeout`, `unknown` |
| `capture_time` | float | Episode length × dt (seconds) |
| `miss_distance` | float | Final range (m) |
| `min_range` | float | Minimum range achieved (m) |
| `oob` | bool | Out of bounds flag |
| `crash` | bool | Crash flag |
| `fallback_used` | bool | Whether prediction fallback was used |
| `prediction_error` | float | Mean prediction error (m) |

### 3.3 scenario_method_summary.csv Schema

| Column | Type | Description |
|---|---|---|
| `scenario` | str | Scenario name |
| `method` | str | Prediction method |
| `guidance_mode` | str | Effective guidance mode |
| `n_episodes` | int | Total episodes |
| `success_rate` | float | Fraction of successes |
| `crash_rate` | float | Fraction of crashes |
| `oob_rate` | float | Fraction of out-of-bounds |
| `timeout_rate` | float | Fraction of timeouts |
| `fallback_rate` | float | Fraction using fallback |
| `mean_return` | float | Mean return |
| `std_return` | float | Std return |
| `mean_miss_distance_m` | float | Mean final range |
| `mean_capture_time_s` | float | Mean capture time |
| `failure_root_causes` | str | JSON dict of failure reasons |

### 3.4 pairwise_mcnemar.csv Schema

| Column | Type | Description |
|---|---|---|
| `scenario` | str | Scenario name |
| `method` | str | Method name (for guidance comparisons) or method pair |
| `comparison` | str | E.g., `no_prediction_vs_gru_frozen`, `los_rate_vs_proportional_navigation` |
| `n_pairs` | int | Number of paired episodes |
| `a_success_b_failure` | int | Discordant: A success, B failure |
| `a_failure_b_success` | int | Discordant: A failure, B success |
| `mcnemar_exact_p` | float | Exact two-sided McNemar p-value |
| `a_success_rate` | float | Success rate of method A |
| `b_success_rate` | float | Success rate of method B |

---

## 4. Statistical Method

### 4.1 Primary Comparison: Exact McNemar Test

For paired outcomes (same scenario, same seed, same episode index), we compare two methods or two guidance laws using the **exact two-sided McNemar test**.

```python
from scipy.stats import binomtest

def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """
    b: A success, B failure
    c: A failure, B success
    """
    b = int(b)
    c = int(c)
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
```

### 4.2 Significance Threshold

- `p < 0.05`: Claim a statistically significant difference.
- `p >= 0.05`: No evidence for a difference.

### 4.3 No P-Value-Only Claims

A claim is **paper-safe** only if:
1. The full experimental matrix is completed (no `Pending` cells).
2. The difference is statistically significant (`p < 0.05`).
3. The difference is consistent across evaluation seeds.
4. The difference has a physical interpretation (e.g., LOS-rate has a dead zone in stern conversion, PN does not).

---

## 5. Result Interpretation Rules

| Pattern | Interpretation | Claim Status |
|---|---|---|
| All guidance laws show 0% success in a scenario | Scenario is geometrically infeasible | "Yes — infeasible" |
| LOS-rate fails, PN/hybrid succeed | LOS-rate has a guidance limitation | "Yes — guidance limitation" |
| Partial success under LOS-rate | Scenario-dependent; geometry may be borderline | "Mixed" |
| No significant difference (McNemar p > 0.05) | No evidence for a difference | "Pending / No evidence" |
| No prediction = GRU frozen (p > 0.05) | Predictor has no effect on guidance limitation | "No effect" |

### 5.1 Paper-Safe Claim Update Protocol

After each full run:
1. Review `paper_safe_claims.md`.
2. Update `README.md` Section 4 with new claim statuses.
3. If a claim moves from `Pending` to `Yes`, ensure the full matrix + McNemar + cross-seed consistency are satisfied.
4. If a claim must remain `Pending`, document the blocker.

---

## 6. Running the Probe

### 6.1 Dry Run (no simulation)

```bash
python scripts/run_stage6g_guidance_limitation_probe.py \
    --dry-run \
    --output-dir outputs/stage6g_probe/dryrun
```

### 6.2 Smoke Test (fast validation)

```bash
python scripts/run_stage6g_guidance_limitation_probe.py \
    --smoke \
    --output-dir outputs/stage6g_probe/smoke
```

### 6.3 Full Probe

```bash
python scripts/run_stage6g_guidance_limitation_probe.py \
    --output-dir outputs/stage6g_probe/full
```

### 6.4 Custom Parameters

```bash
python scripts/run_stage6g_guidance_limitation_probe.py \
    --guidance-modes los_rate proportional_navigation \
    --scenarios favorable weaving_pursuit \
    --episodes-per-scenario 5 \
    --eval-seeds 0 1 \
    --output-dir outputs/stage6g_probe/custom
```

---

## 7. Known Limitations

- **Simple backend only**: Flat-earth, point-mass, no sensor noise, no actuator dynamics. Conclusions may not transfer to JSBSim/F-16 without validation (Stage 7A).
- **Simplified guidance laws**: LOS-rate, PN, and hybrid are 2D-3D simplified. No 3D engagement or autopilot lag.
- **Synthetic scenarios**: Scenarios are hand-designed and may not represent all real-world engagement geometries.
- **Single training seed**: Only training seed 0 is tested. Cross-seed training robustness is not evaluated.
- **Limited methods**: Only `no_prediction` and `gru_frozen` are compared. LSTM, CA, CV, and other predictors are excluded to limit scope.

---

*Last updated: 2026-06-05 | Full probe executing | See `outputs/stage6g_guidance_limitation_probe/full_run/` for latest results*
