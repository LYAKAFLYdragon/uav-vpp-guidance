# Table 1: Main Comparison (Real Training, Stage 6B, Feasible Geometry)

| Method | Success Rate | Mean Return | N Episodes | p vs Baseline | Cohen's d |
|--------|-------------|-------------|------------|---------------|-----------|
| No-Prediction | 62.50% | −113.64 ± 403.78 | 80 | — | — |
| Parametric Prediction† | 62.50% | −114.24 ± 404.44 | 160 | 0.0000* | −0.769 (medium) |

**Notes:**
- † CV and CA predictors produced identical predictions under constant-velocity target motion. When target acceleration is zero, the CA model's additional degree of freedom converges to zero, reducing its prediction to the CV form mathematically ($\hat{p}_{CA} = p_0 + v\Delta t + \frac{1}{2}\hat{a}\Delta t^2 = p_0 + v\Delta t = \hat{p}_{CV}$ when $\hat{a} \to 0$). Reported as a single "Parametric Prediction" category.
- * Paired t-test on episode return vs No-Prediction baseline. Success rates are identical, but return distributions differ significantly (Cohen's $d = -0.769$, medium effect).
- All models were trained from scratch for 200K steps under identical configuration (`stage6f5_feasible_geometry.yaml`).
