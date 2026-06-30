# Crossing Specialist Audit Report

## Audit Date
2026-06-30

## Audit Objective
Determine why the crossing-weighted specialist (obs_dim=18) achieves 0.0 win rate on crossing_feasible against the expert opponent in AoA 60 scope.

## Methodology
1. **Raw trajectory analysis**: Analyze all 10 expert/crossing_feasible episodes from oracle_task_gate_a60_expert_10seed
2. **De-confounding run**: Run the same crossing specialist with matched observation dimensions (obs_dim=18, no truncation) to isolate whether the obs_dim mismatch is causing failure

## Key Findings

### Finding 1: Crossing specialist never enters the attack zone envelope

From all 10 expert/crossing episodes:
- **Envelope fraction**: 0.0039 (2 steps out of 512 = 0.4s)
- **First envelope time**: NaN (never entered)
- **Min range**: ~554m (consistent across all seeds)
- **All terminations**: `timeout_hp_disadvantage`
- **All outcomes**: `loss`

**Interpretation**: The crossing specialist is not guiding the aircraft into the attack zone. The aircraft approaches the target to ~550m but never establishes a position inside the attack envelope.

### Finding 2: Virtual Point is far from target at closest approach

- **VP error at min range**: ~686m (average across all episodes)
- **Close range anchor active**: 98.44% of the time

**Interpretation**: Despite the close_range_anchor being active almost constantly, the VP is placed ~686m away from the target. This means the VPP is not generating a pursuit point that leads the aircraft into the attack zone. The close_range_anchor mode is "predicted_target" for crossing, but the VP is still far from the target.

### Finding 3: Ego takes more damage than target in all episodes

- **Ego damage**: 48.4 ± 6.1 (mean ± std)
- **Target damage**: 33.0 ± 4.4 (mean ± std)
- **HP advantage**: -15.3 ± 7.5 (mean ± std)

**Interpretation**: The expert opponent is consistently dealing more damage than the crossing specialist. This could be because:
1. The ego is not in the attack zone, so the ego's attack zone score is low/zero
2. The expert opponent is maneuvering to keep the ego in its own attack zone
3. The crossing specialist's maneuver is not aggressive enough

### Finding 4: Truncation is NOT the cause (de-confounding run confirms)

**De-confounding run**: Used obs_dim=18 config (no `include_task_type`, matched to specialist training) with the same crossing specialist checkpoint.

Results are **IDENTICAL** to the canonical run (obs_dim=19 with truncation):
- Win rate: 0.0 (both runs)
- All losses with `timeout_hp_disadvantage`
- Min range: ~553-558m (identical)
- Envelope fraction: 0.0 (identical)
- Ego damage > target damage (identical)

**Conclusion**: The truncation from 19 to 18 is **not** the cause of failure. The crossing specialist itself is ineffective on the crossing task against the expert opponent.

## Root Cause Analysis

### Hypothesis 1: The crossing specialist was not trained to solve the crossing task

Evidence:
- The crossing specialist was trained with `task_weights: {head_on: 1.0, crossing_feasible: 2.0}` — but this is within a mixed training loop where the policy sees both tasks
- The checkpoint shows only 8 updates and 16,384 timesteps (very short training)
- The specialist might have overfitted to the head_on task or failed to learn crossing dynamics

### Hypothesis 2: The crossing task is inherently harder than head-on

Evidence:
- Head-on: 1.0 win rate with head_on specialist
- Crossing: 0.0 win rate with crossing specialist
- The crossing scenario starts with a ~45° angle offset (target heading 225° vs own heading 0°), requiring a turn-to-intercept maneuver
- The expert opponent may have a stronger counter-maneuver for crossing scenarios

### Hypothesis 3: The virtual point configuration for crossing is not optimal

Evidence:
- `offset_frame_by_task[crossing_feasible] = "world_neu"` (vs "target_velocity" for head_on)
- `close_range_anchor_mode_by_task[crossing_feasible] = "predicted_target"` (vs "current_target" for head_on)
- VP error at min range is ~686m, suggesting the VP is not aligned with the target

## Recommendation

### Immediate: Do NOT train a learned commander yet

A learned commander cannot solve a problem that the specialists themselves cannot solve. The crossing specialist's 0.0 win rate means:
- Even a perfect oracle gate (perfect task identification) cannot improve crossing performance
- The learned commander would need to either:
  a) Find a way to make the crossing specialist work (unlikely if the specialist itself is broken)
  b) Fallback to a different strategy for crossing (but no better specialist exists)

### Next Steps (in order of priority)

1. **Audit crossing specialist training logs**: Check the training curves for the crossing-weighted specialist to see if it ever learned to solve the crossing task during training
2. **Retrain crossing specialist with adjusted config**:
   - Longer training horizon (more than 16,384 timesteps)
   - Different reward shaping for crossing scenarios
   - Potentially separate head-on and crossing training phases
3. **Alternative: Train a multi-task specialist** that can handle both head-on and crossing with a single network, rather than task-specific specialists
4. **Only after crossing specialist is fixed**: Re-evaluate the oracle gate and then proceed to learned commander training

## Data Artifacts

- **Canonical run (obs_dim=19)**: `outputs/jsbsim_hrl_comparison/oracle_task_gate_a60_expert_10seed/`
- **De-confounding run (obs_dim=18)**: `outputs/jsbsim_hrl_comparison/oracle_task_gate_crossing_matched_obs_10seed/`
- **Analysis script**: `analyze_crossing_audit.py`

## Summary

| Metric | Canonical (obs=19) | Matched (obs=18) | Conclusion |
|--------|-------------------|------------------|------------|
| Win rate | 0.0 | 0.0 | Truncation not the cause |
| Min range | 554.5m | 553.1m | Identical |
| Envelope fraction | 0.0039 | 0.0000 | Never enters envelope |
| Ego damage | 48.4 | 48.4 | Identical |
| Termination | timeout_hp_disadvantage | timeout_hp_disadvantage | Identical |

**Verdict**: The crossing specialist is fundamentally ineffective on the crossing task. The failure is in the specialist training, not in the oracle gate or the observation mismatch. Fix the specialist before training the commander.
