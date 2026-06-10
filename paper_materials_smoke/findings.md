# Automated Findings for Paper
## Architecture Comparison
The full hierarchical architecture (VPP + LOS-rate guidance) achieved the highest success rate of **75.0%** (95% CI: [71.9%, 77.3%]), outperforming the No-VPP baseline (61.8%), the End-to-End direct-control baseline (56.7%), and the No-Prediction baseline (58.6%).

Statistically significant pairwise comparisons (paired t-test, p < 0.05) include: **VPP vs No-Pred**, **VPP vs No-VPP**, **VPP vs End-to-End**.

## Predictor Benefit by Maneuver Intensity
Under weak target maneuvers, the LSTM predictor provided a modest benefit of **+9.96%**. Under strong maneuvers, this benefit increased to **+26.88%**, confirming the conditional advantage hypothesis: predictor value stratifies with target maneuver intensity.

## Gain Optimization
CEM-optimized gains improved success rate by **+14.47%** over default fixed gains (74.7% vs 60.2%). Convergence speedup was approximately 2.1x. Heuristic tuning achieved 69.2%, falling between default and CEM-optimized performance.

