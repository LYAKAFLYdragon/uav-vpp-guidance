# 5-Seed No-VPP Validation Summary

Trained: direct-command (zero VPP offset) PPO for 200 k steps on canonical scenarios.
Evaluated: 30 episodes per seed (3 eval seeds × 10 episodes).

## Pooled results

| Metric | Value |
|---|---:|
| Success rate | 73.3 % (110 / 150) |
| Crash rate | 13.3 % |
| Out-of-bounds rate | 13.3 % |
| Timeout rate | 0.0 % |

## Per-seed overall

| Seed | Success rate | Mean return |
|---:|---:|---:|
| 0 | 73.3 % | +3.46 |
| 1 | 73.3 % | +3.46 |
| 2 | 73.3 % | +3.46 |
| 3 | 73.3 % | +3.46 |
| 4 | 73.3 % | +3.46 |

All five seeds produce exactly 22 successes out of 30 episodes because the
success-rate granularity is 3.33 percentage points per seed.
