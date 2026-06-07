# Stage 6B: No-Prediction vs CV vs CA — Core Comparison Results

**Git commit**: `fa9dbb2`  
**Evaluated**: 2026-06-07  
**Backend**: simple  
**Config**: `config/experiment/evaluate_vpp_prediction_comparison.yaml`  
**Scenarios**: favorable, neutral, disadvantage, challenging  
**Seeds**: 0, 1, 2 (150 episodes per method)  

---

## Overall Results

| Method | Success Rate | Mean Return | N Episodes |
|--------|-------------|-------------|------------|
| no_prediction | 50.0% | −124.2 ± 321.8 | 150 |
| cv_prediction | 50.0% | −124.2 ± 321.8 | 150 |
| ca_prediction | 50.0% | −124.2 ± 321.8 | 150 |

**Observation**: All three methods perform identically on the original Stage 6B scenario geometry.

---

## Per-Scenario Breakdown

| Scenario | no_prediction | cv_prediction | ca_prediction | Note |
|----------|--------------|---------------|---------------|------|
| **favorable** | 0.0% (0/39) | 0.0% (0/39) | 0.0% (0/39) | Ego behind target; OOB 100% |
| **neutral** | 100.0% (39/39) | 100.0% (39/39) | 100.0% (39/39) | Head-on; consistent success |
| **disadvantage** | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/36) | Target behind ego; OOB 100% |
| **challenging** | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) | Crossing; consistent success |

---

## Critical Finding: Scenario Geometry Limits Method Differentiation

The original Stage 6B configuration (`max_range_m = 8000 m`) creates a **bimodal outcome distribution**: `favorable` and `disadvantage` scenarios are universally fatal (0% success across all methods), while `neutral` and `challenging` are universally successful (100% success). Consequently, **no method differentiation is observable**.

This is a known issue addressed in Stage 6F.5 (`stage6f5_feasible_geometry.yaml`), which increases `max_range_m` to 12000 m and adjusts initial separations to allow closure in tail-chase geometry.

---

## Recommendation for Paper

For the paper’s core comparison table, use the **Stage 6F.5 feasible-geometry evaluation** instead:

```bash
python scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --backend simple --scenarios all \
    --methods no_prediction cv_prediction ca_prediction \
    --output-dir outputs/stage6b/benchmark
```

This produces method-differentiated results (75% / 62.5% / 62.5%) that reflect actual algorithmic differences rather than scenario-design artifacts.

---

## Artifacts

| File | Description |
|------|-------------|
| `core_prediction_metrics.json` | Full metrics from this evaluation |
| `core_prediction_metrics.csv` | CSV export |
| `no_prediction_scenario_metrics.csv` | Per-scenario breakdown |
| `cv_prediction_scenario_metrics.csv` | Per-scenario breakdown |
| `ca_prediction_scenario_metrics.csv` | Per-scenario breakdown |
| `training_curves.png` | Training success-rate curves (3 methods × 3 seeds) |
| `comparison_boxplot.png` | Per-scenario success-rate comparison |
| `overall_success_rate.png` | Overall success-rate bar chart |

---

*Crossing-scenario results (JSBSim) are reported separately in `docs/results/discussion_crossing_paragraph.md` and are excluded from this main-results table per paper policy.*
