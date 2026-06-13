# Method Innovation Track 2: Intentional Updates for Air-Combat RL

Branch: `Intentional_Updates`

This branch implements the second method-innovation track, adapting Sutton et
al.'s *Intentional Updates for Streaming Reinforcement Learning* to the
UAV-VPP close-range air-combat guidance problem. The implementation follows
the research guide in
`C:\Users\赖宇\Downloads\IntentionalRL实验计划与论文撰写.md`.

## Core idea

Instead of fixing a parameter-space learning rate, Intentional Updates fix the
**functional change** caused by each update:

- Critic: control ``Delta V(s)`` — how much the value prediction moves toward
  the TD target.
- Actor: control ``Delta log pi(a|s)`` — how much the policy probability of the
  sampled action changes.

This is especially attractive for air combat because:

- Rewards are sparse and terminal-heavy.
- Long trajectories with delayed feedback make value overshooting likely.
- Tactical policies should not jump abruptly between maneuvers.

## Three switchable components

### 1. Intentional Critic Update (ICU)

Approximates the batch-compatible step size

```
alpha_v = eta_v * mean(|R_t - V(s_t)|) / (||grad_theta V(s_t)||^2 + eps)
```

and scales the critic loss gradient so that the effective optimizer step is
``alpha_v``. This limits value-function drift regardless of network depth or
feature scale.

File: `src/uav_vpp_guidance/agents/intentional_ppo_agent.py`

### 2. Intentional Actor Update (IAU)

Approximates

```
alpha_pi = eta_pi * (mean(|A_t|) / EMA(|A|)) / (||grad_theta log pi(a_t|s_t)||^2 + eps)
```

and scales the actor/entropy loss gradient so the effective step is
``alpha_pi``. This keeps the policy update budget interpretable in action-space
terms rather than parameter-space terms.

File: `src/uav_vpp_guidance/agents/intentional_ppo_agent.py`

### 3. Combat-Aware Intentional Schedule (CAIS)

A rule-based phase classifier maps geometry features to actor/critic budget
scalers:

| Phase | Geometry | eta_actor | eta_critic | Intuition |
|-------|----------|-----------|------------|-----------|
| search_approach | far range | 1.5x | 1.0x | encourage exploration |
| merge_maneuver | medium range / changing angles | 1.0x | 1.0x | stable learning |
| advantage_position | own nose on target | 0.5x | 0.5x | do not break a good position |
| disadvantage_defense | target behind ownship | 0.5x | 1.5x | learn danger fast, act conservatively |
| terminal | very close to termination | 0.3x | 0.5x | absorb outcome slowly |

File: `src/uav_vpp_guidance/training/combat_aware_schedule.py`

## Configuration

`config/method_innovation_track2.yaml` exposes all switches independently so
you can run the full ablation matrix:

- PPO (baseline)
- I-PPO-C (critic only)
- I-PPO-A (actor only)
- I-PPO (actor + critic)
- CAI-PPO (I-PPO + combat-aware schedule)

## Tests

- `tests/test_intentional_ppo_agent.py`
- `tests/test_combat_aware_schedule.py`

Run:

```bash
python -m pytest tests/test_intentional_ppo_agent.py tests/test_combat_aware_schedule.py -v
```

## Wiring status

The agent is implemented and tested in isolation. To use it in training you
need to:

1. Replace `PPOAgent` with `IntentionalPPOAgent` in the training script, or
   add a CLI flag such as `--algorithm intentional_ppo`.
2. Pass the observation dict (`info`) to `agent.store_transition(..., info=obs)`
   so that combat-aware phase features can be extracted from
   ``obs["relative_state"]``.
3. Log the additional diagnostics returned by `agent.update()`:
   ``scale_actor``, ``scale_critic``, ``ema_abs_adv``.
