# Experiment Protocol

## Baselines

1. **Fixed-gain pursuit (no VPP)**: Direct LOS guidance to target position.
2. **Fixed-gain VPP**: Policy generates VPP, but gains are fixed.
3. **Gain-only CEM**: Policy frozen, gains optimized.
4. **Proposed bilevel**: Joint optimization.

## Evaluation Metrics

- Success rate
- Non-crash success rate
- Crash rate
- Timeout rate
- Mean episode return
- Mean time-to-success
- Command saturation rate

## Ablation Studies

- Remove regret-aware gain update
- Remove gain observation from policy input
- Remove safety penalty from reward
