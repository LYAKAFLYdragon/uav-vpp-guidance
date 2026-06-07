# Table 1: Main Comparison (Stage 6B, Feasible Geometry)

| Method | Success Rate | Mean Return | p vs Baseline | Cohen's d |
|--------|-------------|-------------|---------------|-----------|
| No-Prediction | 75.00% | −7.26 ± 355.84 | — | — |
| CV Prediction | 62.50% | −110.57 ± 400.28 | 0.0013* | −0.373 (small) |

**Notes:**
- 80 episodes per method (10 seeds × 8 scenarios from `stage6f5_feasible_geometry.yaml`).
- CA Prediction produced numerically identical results to CV Prediction; omitted pending bug investigation.
- * indicates statistical significance at α = 0.05 (paired t-test).
