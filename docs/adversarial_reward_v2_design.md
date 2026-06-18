# Adversarial Reward v2 Design

**Author**: AI coding assistant  
**Date**: 2026-06-18  
**Scope**: `config/adversarial/train_target_v2.yaml`, `config/adversarial/train_pursuer_v2.yaml`, and `src/uav_vpp_guidance/envs/adversarial_jsbsim_env.py`

## 1. Problem Diagnosis

The original adversarial reward (`config/adversarial/train_target.yaml`) is dominated by two terms:

```yaml
w_range: 0.5      # farther is better
w_escaping: 0.1   # positive range_rate is good
```

This gives the target agent a strong, dense, monotonic signal to **fly away in a straight line**.  The optimal behavior under this reward is essentially "point away from the pursuer and accelerate", not "maneuver inside the combat envelope to avoid capture".  As a result:

- The target never learns to use break turns, scissors, or jinks.
- The pursuer is chasing a receding point that quickly exceeds `max_range_m`.
- Capture rate drops below 20%, making the adversarial loop degenerate.

The v2 redesign changes the target reward from "escape distance" to "survival + engagement-aware maneuvering", and simultaneously makes the pursuer reward more aggressive.

## 2. Target Reward v2

File: `config/adversarial/train_target_v2.yaml`

| Term | v1 | v2 | Rationale |
|------|-----|-----|-----------|
| `w_survival` | 0.01 | 0.01 | Keep small per-step bonus for staying alive. |
| `w_escaping` | 0.1 | **0.05** | Halved: do not reward pure running away. |
| `w_energy` | 0.05 | **0.1** | Increased: high energy is needed for hard turns. |
| `w_range` | 0.5 | **0.2** | Reduced: distance alone is not the goal. |
| `w_maneuver` | — | **0.1** | **NEW**: reward body yaw rate → encourages turning/jinking. |
| `w_proximity` | — | **0.05** | **NEW**: penalty when `range > 6400 m` → keeps target in arena. |
| `terminal_captured` | -200 | **-500** | Stronger penalty for being captured. |
| `terminal_oob` | -200 | **-100** | Strategic escape is acceptable. |
| `terminal_timeout` | 50 | **100** | Big reward for surviving the full episode. |

### 2.1 New reward terms (code)

Implemented in `src/uav_vpp_guidance/envs/adversarial_jsbsim_env.py` inside `TargetRewardCalculator`:

- **Maneuver reward**: `w_maneuver * min(|r| / maneuver_ref_rate, 1.0)` where `r` is the target's body yaw rate.  This makes turning/jinking valuable without requiring an explicit action input.
- **Proximity penalty**: `-w_proximity * min(max(0, range - proximity_range_m) / proximity_range_m, 1.0)`.  Default `proximity_range_m = 6400 m` (≈ 80% of the default 8000 m arena radius).

### 2.2 Expected target behavior

With v2, the target should learn that:

1. Flying straight away earns less dense reward.
2. Hard turns and jinks earn immediate `maneuver` reward and can avoid capture.
3. Staying inside the arena avoids the `proximity` penalty and keeps the chance of a timeout victory.
4. Crashing or being captured is costly; surviving to timeout is valuable.

## 3. Pursuer Reward v2

File: `config/adversarial/train_pursuer_v2.yaml`

The pursuer reuses the existing `RewardCalculator` but overrides the weights in the `reward:` block:

| Term | Default | v2 | Rationale |
|------|---------|-----|-----------|
| `w_closing` | 0.0 | **0.3** | Enable closing-velocity reward; critical against evasive targets. |
| `w_safety` | 2.0 | **1.0** | Reduce conservative altitude/range behavior. |
| `w_angle` | 0.8 | **1.2** | More weight on ATA/AA alignment. |
| `w_range` | 0.5 | 0.5 | Keep range interval reward. |
| `w_adversarial_bonus` | — | **0.5** | **NEW**: extra bonus for closing on a maneuvering target. |

### 3.1 Adversarial bonus (code)

Implemented in `src/uav_vpp_guidance/envs/reward.py` inside `RewardCalculator`:

```python
if self.w_adversarial_bonus > 0.0 and info.get("adversarial_maneuvering", False):
    range_rate_mps = rel.get("range_rate_mps", 0.0)
    closing_signal = max(0.0, -range_rate_mps / 200.0)
    reward_adversarial = self.w_adversarial_bonus * closing_signal
```

The flag `adversarial_maneuvering` is set by `AdversarialJSBSimEnv.step`:

- For `target_controller_type == "maneuver_library"`: true when current maneuver ≠ `straight_level`.
- For RL target: true when `|yaw_rate| > 0.05 rad/s`.

This makes the pursuer value not just closing, but closing on a target that is actively evading.

## 4. Reward Debug Logging

To verify the new reward terms produce the intended behavior, the environment can log per-step reward decomposition.

Configuration:

```yaml
adversarial_debug:
  log_rewards: true
  reward_log_path: "outputs/adversarial/debug/reward_terms.csv"
```

Columns:

| Column | Description |
|--------|-------------|
| `step` | High-level decision step |
| `p_reward_total` | Total pursuer reward |
| `p_reward_geometry` | `reward_range + reward_angle` |
| `p_reward_range` | Pursuer range reward |
| `p_reward_angle` | Pursuer angle reward |
| `t_reward_total` | Total target reward |
| `t_reward_survival` | Target survival bonus |
| `t_reward_escaping` | Target escaping bonus |
| `t_reward_energy` | Target energy reward |
| `t_reward_maneuver` | Target maneuver diversity reward |

The CSV is created automatically when `log_rewards: true`.

## 5. Validation Plan

1. Smoke test both v2 configs:
   ```bash
   python scripts/train_adversarial_target.py --config config/adversarial/train_target_v2.yaml --smoke
   python scripts/train_adversarial_pursuer.py --config config/adversarial/train_pursuer_v2.yaml --smoke
   ```
2. Inspect `outputs/adversarial/debug/reward_terms.csv`:
   - Target `maneuver` term should be non-zero and correlated with turns.
   - Target `proximity` term should stay near zero (target not running away).
   - Pursuer `adversarial` bonus should be positive during closing phases.
3. Train target v2 for ~50K steps and evaluate capture rate vs the frozen pursuer.  If capture rate is > 30%, the reward is in the right regime.
4. Fine-tune pursuer v2 against the trained target.  Capture rate should improve over the frozen pre-trained pursuer.

## 6. Files Changed

- `config/adversarial/train_target_v2.yaml` (new)
- `config/adversarial/train_pursuer_v2.yaml` (new)
- `src/uav_vpp_guidance/envs/adversarial_jsbsim_env.py` (target reward terms + debug logging)
- `src/uav_vpp_guidance/envs/reward.py` (pursuer `w_adversarial_bonus`)
- `docs/adversarial_reward_v2_design.md` (this file)
