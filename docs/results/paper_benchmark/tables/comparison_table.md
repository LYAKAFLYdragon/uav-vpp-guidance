| Method | Success Rate | Mean Return | N Episodes | vs Baseline p | Cohen's d |
| --- | --- | --- | --- | --- | --- |
| no_prediction | 75.00% | -7.26 ± 355.84 | 80 | nan | nan |
| parametric_prediction | 62.50% | -110.57 ± 400.28 | 160 | 0.0013* | -0.373 (small) |
| gain_only | 62.50% | -82.24 ± 363.56 | 80 | 0.0113* | -0.290 (small) |

**Footnote**: CV and CA predictors produced identical predictions under constant-velocity target motion ($\hat{p}_{CA} = p_0 + v\Delta t + \frac{1}{2}\hat{a}\Delta t^2 = p_0 + v\Delta t = \hat{p}_{CV}$ when $\hat{a} \to 0$). Reported as a single "Parametric Prediction" category ($N = 160$ episodes).
