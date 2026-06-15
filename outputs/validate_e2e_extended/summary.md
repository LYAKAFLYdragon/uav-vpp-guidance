# End-to-End Extended Training Summary

Reward function aligned with the hierarchical baseline (`--align-reward`).
Two independent seeds per budget.

## Results by budget

| Budget | Seed 0 SR | Seed 1 SR | Mean SR | Notes |
|---:|---:|---:|---:|---|
| 200 k | 36.7 % | 0.0 % | 18.3 % | Seed 0 crash-free; seed 1 crashes. |
| 500 k | 0.0 % | 0.0 % | 0.0 % | Both seeds fail. |
| 1 M | 36.7 % | 0.0 % | 18.3 % | Seed 0 OOB; seed 1 crashes. |
| 2 M | 36.7 % | 0.0 % | 18.3 % | Seed 0 OOB; seed 1 OOB. |

## Interpretation

Increasing the training budget up to 2 M steps does not reliably improve the
E2E baseline. One seed reaches ~37 % success, but the other remains at 0 %
across all budgets. This indicates high seed sensitivity and instability,
supporting the claim that the hierarchical decomposition is easier to train,
but not that it is strictly necessary.
