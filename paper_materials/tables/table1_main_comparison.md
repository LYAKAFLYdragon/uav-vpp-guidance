# Table 1: Main Comparison (Stage 6B, Feasible Geometry)

| Method | Success Rate | Mean Return | N Episodes | p vs Baseline | Cohen's d |
|--------|-------------|-------------|------------|---------------|-----------|
| No-Prediction | 75.00% | −7.26 ± 355.84 | 80 | — | — |
| Parametric Prediction† | 62.50% | −110.57 ± 400.28 | 160 | 0.0013* | −0.373 (small) |

**Notes:**
- † CV and CA predictors produced identical predictions under constant-velocity target motion. When target acceleration is zero, the CA model's additional degree of freedom converges to zero, reducing its prediction to the CV form mathematically ($\hat{p}_{CA} = p_0 + v\Delta t + \frac{1}{2}\hat{a}\Delta t^2 = p_0 + v\Delta t = \hat{p}_{CV}$ when $\hat{a} \to 0$). This is expected behavior under the chosen target kinematics, not an implementation bug.
- * indicates statistical significance at α = 0.05 (paired t-test).
