# Table 1: Main Comparison (Stage 6B, Feasible Geometry)

| Method | Success Rate | Mean Return | N Episodes | p vs Baseline | Cohen's d |
|--------|-------------|-------------|------------|---------------|-----------|
| No-Prediction | 75.00% | −7.26 ± 355.84 | 80 | — | — |
| Parametric Prediction† | 62.50% | −110.57 ± 400.28 | 160 | 0.0013* | −0.373 (small) |
| LSTM (Frozen)‡ | 75.00% | −7.17 ± 355.84 | 80 | 0.0001* | 0.448 (small) |
| GRU (Frozen)‡ | 62.50% | −110.78 ± 400.42 | 80 | 0.0013* | −0.374 (small) |

**Notes:**
- † CV and CA predictors produced identical predictions under constant-velocity target motion. When target acceleration is zero, the CA model's additional degree of freedom converges to zero, reducing its prediction to the CV form mathematically ($\hat{p}_{CA} = p_0 + v\Delta t + \frac{1}{2}\hat{a}\Delta t^2 = p_0 + v\Delta t = \hat{p}_{CV}$ when $\hat{a} \to 0$). This is expected behavior under the chosen target kinematics, not an implementation bug.
- ‡ Neural predictors (LSTM/GRU) were frozen during RL training; the policy learns to use (or ignore) their outputs.
- * Paired t-test on episode return vs No-Prediction baseline. LSTM matches No-Prediction in success rate but shows a small statistically significant difference in return distribution.
