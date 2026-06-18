# Curriculum Learning v2 Design Rationale

**Author**: AI coding assistant  
**Date**: 2026-06-18  
**Scope**: `config/experiment/close_range_curriculum_v2.yaml` and `close_range_curriculum_stage3_v2.yaml`

## 1. Goal

The original `close_range_curriculum.yaml` already provides a working three-stage curriculum.  
The v2 revision keeps that structure but hardens **Stage 3** so that it can actually separate:

- **VPP** (virtual-pursuit-point guided PPO)
- **No-VPP** (zero-offset VPP, i.e. pure LOS-rate pursuit)
- **E2E** (end-to-end direct control)

If Stage 3 is too easy, all three methods saturate and the experiment cannot demonstrate the value of prediction / virtual-point planning.

## 2. Analysis of the Original Stage 3

| Aspect | v1 value | Issue |
|--------|----------|-------|
| Bandit difficulty | `medium` | Only 7 maneuvers allowed; omits `loop`, `split_s`, `immelmann`, `vertical_scissors`, `defensive_spiral`, `extension`, `jink`. |
| Reaction time | `3.0 s` | Long reaction time gives the pursuer a large planning advantage. |
| Selector noise | `0.1` | Low noise makes the bandit exploitable. |
| Parameter perturbation | `0.1` | Maneuver parameters are too predictable. |
| Success range | `700 m` | Loose for a dogfight metric. |
| Success ATA | `18°` | Still permissive. |
| Success hold time | `2.0 s` | Reasonable, but combined with loose geometry it is easy to satisfy by accident. |

With these settings the bandit is essentially a "scripted medium-agility target".  
VPP and No-VPP can both perform well by reactive LOS-rate tracking, so the Stage 3 success-rate gap is small.

## 3. Changes in v2

### 3.1 Bandit difficulty: `medium` → `hard`

```yaml
difficulty: hard
allowed_maneuvers:
  - straight_level
  - coordinated_turn
  - barrel_roll
  - dive
  - loop
  - high_yoyo
  - low_yoyo
  - scissors
  - split_s
  - immelmann
  - break_turn
  - displacement_roll
  - vertical_scissors
  - defensive_spiral
  - extension
  - jink
```

`hard` unlocks the complete maneuver library.  Crucially, it adds:

- `vertical_scissors`, `defensive_spiral` – high-rate defensive reversals that punish pure pursuit.
- `extension` – energy-retaining escape that forces the pursuer to manage closure rate.
- `jink` – rapid lateral offset that stresses LOS-rate tracking.
- `loop`, `split_s`, `immelmann` – vertical-plane maneuvers that require altitude/energy anticipation.

These maneuvers are exactly where a predictive VPP offset (placing the aim point ahead of the current target position) should outperform a zero-offset / direct LOS tracker.

### 3.2 Reaction time: `3.0 s` → `1.5 s`

A shorter reaction time means the bandit can switch defenses faster when the pursuer closes in.  
This reduces the value of ``chase the current position'' and increases the value of anticipating where the target will be.

### 3.3 Stochasticity increased

```yaml
selector_noise: 0.25        # v1: 0.1
param_perturb_scale: 0.25   # v1: 0.1
history_switch_prob: 0.6    # v1: 0.5
```

Higher noise makes the target less deterministic; the pursuer cannot memorize a single counter-maneuver.  
Parameter perturbation varies the geometry of each maneuver execution, widening the state distribution.

### 3.4 Success criteria tightened

| Criterion | v1 | v2 | Rationale |
|-----------|-----|-----|-----------|
| Range | `700 m` | `600 m` | Requires genuine close approach, not a lucky fly-by. |
| ATA | `18°` | `15°` | Requires better angular alignment. |
| Hold time | `2.0 s` | `1.0 s` | Shorter hold still filters noise but lowers the chance of accidental success. |
| Hysteresis range | `750 m` | `650 m` | Consistent with tightened range. |
| Hysteresis ATA | `22°` | `18°` | Consistent with tightened ATA. |

The hold time was reduced from 2.0 s to 1.0 s because, with a harder bandit, holding for 2.0 s while staying inside 600 m / 15° is very difficult even for a strong VPP policy.  The combination of 600 m + 15° + 1.0 s is calibrated to be achievable by a well-trained VPP agent but challenging for No-VPP and E2E.

### 3.5 Gate threshold unchanged

`stage_gate_sr: 0.40` is kept.  Tightening the bandit and the criteria already makes the gate harder to pass; lowering the threshold further would risk skipping stages.  If pilot runs show that even 40% is unreachable within the budget, the threshold can be reduced to 0.35 or a Stage 2.5 can be inserted.

## 4. Expected Effects

| Method | Expected Stage 3 behavior |
|--------|---------------------------|
| **VPP** | Can place the virtual point ahead of/around the bandit during reversals and scissors; should reach the highest SR. |
| **No-VPP** | Pure LOS-rate pursuit lags behind high-rate reversals; should lag VPP by 10–30 absolute SR points. |
| **E2E** | Must relearn low-level control while tracking; expected to crash more often and finish with lower SR. |

If the v2 Stage 3 still does not separate VPP and No-VPP, the next knobs are:

1. Increase `selector_noise` / `param_perturb_scale` further.
2. Reduce `reaction_time_s` to `1.0 s`.
3. Add `target_init` speed/heading randomization (domain randomization scale > 1.0).
4. Tighten range to `500 m` and ATA to `12°`.

## 5. Validation Plan

1. Run `scripts/train_curriculum_ppo.py --config config/experiment/close_range_curriculum_v2.yaml --smoke`.
2. Run one full seed and inspect `logs/curriculum_log.csv` to confirm the Stage 3 gate is reachable.
3. Run VPP / No-VPP / E2E each on `close_range_curriculum_stage3_v2.yaml` and compare per-scenario SR.

## 6. Files Changed

- `config/experiment/close_range_curriculum_v2.yaml` (new)
- `config/experiment/close_range_curriculum_stage3_v2.yaml` (new)
- `docs/curriculum_v2_design_rationale.md` (this file)
