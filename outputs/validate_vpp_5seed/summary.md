# 5-Seed VPP Validation Summary

Trained: `train_no_prediction_vpp_ppo` for 200 k steps on canonical scenarios.
Evaluated: 30 episodes per seed (3 eval seeds × 10 episodes).

## Pooled results

| Metric | Value |
|---|---:|
| Success rate | 70.0 % (105 / 150) |
| Crash rate | 16.7 % |
| Out-of-bounds rate | 13.3 % |
| Timeout rate | 0.0 % |

## Per-seed overall

| Seed | Success rate | Mean return |
|---:|---:|---:|
| 0 | 70.0 % | −5.28 |
| 1 | 70.0 % | −0.52 |
| 2 | 70.0 % | +0.13 |
| 3 | 70.0 % | +0.31 |
| 4 | 70.0 % | −0.54 |

All five seeds produce exactly 21 successes out of 30 episodes because the
success-rate granularity is 3.33 percentage points per seed.
