# Disadvantage Feasibility Gradient Analysis

## 1. Executive Summary

The standard `disadvantage` scenario (ego 200 m/s, target 220 m/s, target behind-left at heading 30°) appears to sit at or beyond the feasibility boundary of the current VPP-LOS guidance stack under the F-16 JSBSim model and the 200-step (40 s) episode horizon.  The failure is not purely a reward or policy problem: even a hand-designed aggressive lead-turn action times out 100% of the time once crashes are prevented.

This document proposes a **feasibility gradient** of five `disadvantage_bridge_*` scenarios that monotonically increase difficulty along the four dominant geometric parameters:

1. speed ratio `λ = V_ego / V_target`
2. initial lateral offset `|y_target|`
3. target heading `ψ_target`
4. effective turn angle required to get nose-on

The gradient is designed to let the policy first learn the **lead-turn maneuver shape** in an energy-rich, forgiving regime, then gradually transfer that skill toward the standard geometry.

## 2. Why the Standard Disadvantage Is Hard

### 2.1 Initial Geometry

```
Ego:    (0, 0, 5000),    V = 200 m/s, ψ = 0°
Target: (-600, 400, 5000), V = 220 m/s, ψ = 30°
```

- Range to target: **721 m** (already inside the 1000 m success radius).
- LOS heading from ego to target: **146°** (left-rear).
- Target is **faster** and its velocity vector has a forward component, so the range initially **increases**.
- To achieve success (`range < 1000 m` AND `|ATA| < 30°`), ego must turn left roughly **115–140°** while simultaneously managing energy/altitude and closing the lateral gap.

### 2.2 The `dynamics_aware` Clipping Trap

With `dynamics_aware: true`, the VPP generator clips the virtual point to a forward cone of only ~11.5°.  For a target in the rear hemisphere, the shortest signed heading error points **away** from the target, so the clip forces the aircraft to turn in the wrong direction.  This explains the baseline behavior: the policy drifts right and times out.

### 2.3 Energy and Turn-Rate Limits

Disabling the clip allows the required left turn, but the F-16 at 200 m/s needs several seconds to turn 120°.  During that time a 220 m/s target travels ~1.5–2 km, so the range after the turn is often already >2 km.  Because the target is faster, the aircraft cannot close the gap by chasing from behind; it must perform a **lead turn** that cuts across the target’s future path.  This requires both a tight turn and enough excess speed to convert the geometry into a tail-chase or nose-on position.

With altitude-hold and 60° roll-angle protection (v5) the aircraft stays alive, but fixed-action lead/left policies still time out 100% of the time.  This is the strongest evidence that the standard geometry is near the physical boundary.

## 3. Feasibility Gradient Design

### 3.1 Parameter Dimensions

| Parameter | Symbol | Effect on Difficulty |
|-----------|--------|----------------------|
| Speed ratio | `λ = V_ego / V_target` | Higher `λ` → more energy to close/turn; lower `λ` → target pulls away. |
| Lateral offset | `\|y_target\|` | Larger offset → larger required heading change. |
| Target heading | `ψ_target` | Larger angle → target moves more leftward, reducing required turn but also pulling away faster if `λ < 1`. |
| Longitudinal offset | `x_target < 0` | More negative → target is further behind. |

### 3.2 The Bridge Scenarios

| Scenario | `V_ego` | `V_target` | `λ` | Target position | `ψ_target` | Physical Intuition | Predicted Learnability |
|----------|---------|------------|-----|-----------------|------------|--------------------|------------------------|
| `disadvantage_bridge_1` | 260 | 180 | **1.44** | (-300, 100) | 10° | Large energy margin, small offset. Ego can turn left easily and get behind. | **> 90%** |
| `disadvantage_bridge_2` | 240 | 190 | **1.26** | (-400, 150) | 15° | Comfortable lead-turn. Still clear energy advantage. | **70–90%** |
| `disadvantage_bridge_3` | 230 | 200 | **1.15** | (-450, 200) | 20° | Moderate lead-turn. Policy must commit to the turn. | **50–70%** |
| `disadvantage_bridge_4` | 220 | 210 | **1.05** | (-500, 250) | 25° | Narrow margin. Geometry is forgiving (offset still moderate) but energy is tight. | **30–50%** |
| `disadvantage_bridge_5` | 210 | 215 | **0.98** | (-550, 300) | 28° | Near-standard. Slight speed disadvantage; requires precise lead-turn timing. | **10–30%** |
| `disadvantage` (standard) | 200 | 220 | **0.91** | (-600, 400) | 30° | Target is faster and further behind-left. At feasibility boundary. | **0–20%** |

All scenarios keep the same altitude (5000 m) and use the same JSBSim F-16 model and maneuver-library target policy.

## 4. Feasibility Hypothesis

### 4.1 Learnable Region

Under the current VPP-LOS stack with `dynamics_aware: false` and altitude/roll protections:

- **`λ > 1.15` + moderate offset (`|y| < 250 m`)**: clearly learnable. The aircraft has enough energy to turn and close.
- **`1.05 < λ < 1.15`**: learnable but sensitive to turn timing and VPP placement. The policy must learn to place the virtual point ahead-left of the target, not just left.
- **`0.98 < λ < 1.05`**: boundary region. Success is possible only with an efficient lead-turn and some luck in target maneuver selection.
- **`λ < 0.98` with target behind-left**: very difficult/impossible within 200 steps unless the target maneuver library happens to turn toward ego.

### 4.2 Unlearnable Region (Current Stack)

The standard `disadvantage` (`λ ≈ 0.91`) is hypothesized to be **outside the stable learnable region** for the current stack.  Even if the policy discovers a near-optimal lead turn, the target’s speed advantage and the F-16 turn-rate limit make it unlikely to satisfy the success criterion (`range < 1000 m`, `|ATA| < 30°`, 0 s hold) before timeout.

### 4.3 Actionable Consequences

If the empirical results match this hypothesis, the project faces two choices:

1. **Stop at `disadvantage_bridge_4` or `bridge_5`** and treat them as the practical hard case, while explicitly documenting that the standard `disadvantage` is beyond the current method’s envelope.
2. **Relax the standard `disadvantage` geometry** (e.g. ego 210 m/s, target 210 m/s equal speed, or reduce offset) so that it falls into the boundary-but-learnable region.

## 5. Curriculum Rationale

The 5-stage curriculum in `config/experiment/disadvantage_curriculum.yaml` follows the gradient:

1. **Stage 0–1**: Master the lead-turn shape in `bridge_1` and `bridge_2`.  These are so energy-rich that even random exploration quickly finds a successful left turn.
2. **Stage 2**: Transfer to `bridge_3`, where the policy must start timing the turn.
3. **Stage 3**: Introduce `bridge_4` and `bridge_5` together, forcing the policy to generalize across the narrow-margin regime.
4. **Stage 4**: Mix `bridge_5` with the standard `disadvantage`.
5. **Stage 5**: Full distribution with heavy weight on `disadvantage`.

The curriculum keeps `favorable` and `neutral` as anchors so the policy does not forget how to close in easy geometries.

## 6. Suggested Validation Experiments

1. **Single-scenario smoke tests**: Train a small PPO run (10k steps) on each bridge in isolation and record success rate.  This validates the ordering.
2. **Curriculum run**: Train the full 60k-step curriculum and compare per-stage per-scenario SR.
3. **Ablate the protections**: Confirm that without altitude-hold/roll-protection, the bridges also crash when `dynamics_aware: false`.
4. **Horizon sensitivity**: If `bridge_4`/`bridge_5` time out, increase `max_steps` from 200 to 300 and re-evaluate.
5. **Success-threshold sweep**: Temporarily relax `success_ata_deg` to 45° or `success_range_m` to 1500 m for `bridge_5`/`disadvantage` to check whether the issue is geometry closure or criterion tightness.

## 7. Fallback Options

If even `disadvantage_bridge_4` cannot be learned reliably:

- **Reduce target maneuver aggressiveness** in the bandit config for the bridge scenarios.
- **Use scripted warm-start**: initialize the policy with 5–10 seconds of hard left-turn demonstrations on `bridge_1`/`bridge_2`.
- **Relax the standard scenario**: redefine `disadvantage` with `V_target = 210` and `V_ego = 210` (equal speed, λ = 1.0).  This moves it from the unlearnable region to the boundary region.

---

**Files produced**

- `config/experiment/disadvantage_bridge_scenarios.yaml`
- `config/experiment/disadvantage_curriculum.yaml`
- `docs/disadvantage_feasibility_analysis.md`
