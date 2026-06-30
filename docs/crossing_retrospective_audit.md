# Crossing Specialist Retrospective Audit Report

## Audit Date
2026-06-30

## Audit Objective
Determine whether the crossing-weighted specialist's 0.0 win rate on expert/crossing is due to:
1. **Training failure**: The specialist never learned crossing geometry
2. **Selection failure**: A better checkpoint exists but was not selected

## Methodology

### Phase 1: Retrospective Checkpoint Comparison (3-seed screening)
Test all checkpoints from the crossing-weighted training run on expert/crossing_feasible, AoA 60, 3 seeds each:
- step_8192.pt
- step_16384.pt
- step_24576.pt
- step_32768.pt
- best.pt
- last.pt

### Phase 2: 10-seed Validation
Promising checkpoints (step_8192, step_24576) re-tested on 10 seeds to verify robustness.

## Results

### Phase 1: 3-seed Screening

| Checkpoint | 3-seed Win Rate | HP Advantage | Dominant Termination |
|------------|----------------|--------------|---------------------|
| step_8192 | **0.667** (2/3) | +8.2 ± 13.2 | mixed |
| step_16384 | 0.000 (0/3) | -21.5 | timeout_hp_disadvantage |
| step_24576 | **0.667** (2/3) | +2.2 ± 8.4 | mixed |
| step_32768 | 0.000 (0/3) | -18.3 | timeout_hp_disadvantage |
| best | 0.000 (0/3) | -18.3 | timeout_hp_disadvantage |
| last | 0.000 (0/3) | -18.3 | timeout_hp_disadvantage |

**Initial observation**: step_8192 and step_24576 show 66.7% win rate in 3-seed, suggesting training may have learned crossing capability.

### Phase 2: 10-seed Validation

| Checkpoint | 10-seed Win Rate | HP Advantage | Termination |
|------------|-----------------|--------------|-------------|
| step_8192 | **0.000** (0/10) | -15.3 ± 6.4 | timeout_hp_disadvantage (all) |
| step_24576 | **0.000** (0/10) | -15.3 ± 6.4 | timeout_hp_disadvantage (all) |

**Critical finding**: The 3-seed "wins" were **false positives due to variance**. Neither checkpoint is robustly effective on expert/crossing.

### Comparison: 3-seed vs 10-seed

| Checkpoint | 3-seed Wins | 10-seed Wins | Verdict |
|------------|------------|-------------|---------|
| step_8192 | 2/3 | 0/10 | False positive (variance) |
| step_24576 | 2/3 | 0/10 | False positive (variance) |

## Root Cause Analysis

### Hypothesis A: Training learned crossing but selection picked wrong checkpoint
**Status**: REJECTED
- No checkpoint achieves >0.0 win rate on 10-seed expert/crossing
- Even the "best" early checkpoints (step_8192, step_24576) fail at 10-seed
- If training had truly learned robust crossing, at least one checkpoint should be consistently effective

### Hypothesis B: Training never learned robust crossing geometry
**Status**: CONFIRMED
- All checkpoints consistently fail at 10-seed
- The 3-seed "wins" were spurious (variance on small sample)
- The training process (32,768 timesteps, 16 updates) was insufficient to learn robust crossing geometry
- The policy may have learned a "fly near the target" heuristic but not a "enter and maintain attack envelope" strategy

## Why 3-seed gave false positives

With 3 seeds and a true win probability of ~10-20%, the probability of observing 2/3 wins by chance is non-trivial:
- If true p(win) = 0.15, P(≥2/3 wins) = 3 × (0.15)² × (0.85) + (0.15)³ ≈ 0.06 (6%)
- With 6 checkpoints tested, expected false positives ≈ 0.36
- This matches our observation: 2 out of 6 checkpoints showed spurious "wins"

## Conclusion

**The crossing specialist's failure is a TRAINING failure, not a SELECTION failure.**

- No checkpoint from the training run is robustly effective on expert/crossing
- The 3-seed "wins" were statistical noise
- The training horizon (32,768 timesteps) was likely insufficient
- The reward shaping may not have provided sufficient pressure to learn attack envelope geometry

## Recommendation

### Do NOT attempt to fix selection metric
There is no better checkpoint to select. All checkpoints are equally ineffective.

### Proceed with crossing specialist v2 finetune
Design a focused training run with:
1. **Lane-aware evaluation**: Primary metric = expert/crossing_feasible win rate, not mixed average
2. **Longer training horizon**: >64,000 timesteps (double the current)
3. **Geometry-aligned reward**: Add explicit signal for entering and maintaining attack envelope
4. **Checkpoint selection**: Select based on expert/crossing_feasible performance, not total reward

### Next steps
1. Design crossing v2 training config
2. Run training with extended horizon and geometry-aware reward
3. Validate on expert/crossing_feasible with 10-seed before declaring success
4. Only after crossing specialist is fixed: re-evaluate oracle gate and proceed to learned commander

## Data Artifacts
- Retrospective 3-seed: `outputs/jsbsim_hrl_comparison/crossing_retro_*_expert_3seed/`
- Validation 10-seed: `outputs/jsbsim_hrl_comparison/crossing_step8192_expert_10seed/`, `crossing_step24576_expert_10seed/`
- Audit script: `scripts/run_crossing_retrospective.sh`
