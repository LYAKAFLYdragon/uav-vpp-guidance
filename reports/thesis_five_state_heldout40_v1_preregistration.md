# Independent Five-State Heldout40 Preregistration

**Source ID:** `THESIS-FIVE-STATE-HELDOUT40-V1`  
**Family:** `noncanonical_thesis_five_state_heldout_envelope_v1`  
**Status:** evaluation-only; no result has been inspected for this envelope.

## Frozen Protocol

- Forty explicit JSBSim initial conditions: five initial geometry states, two
  target-minus-own altitude offsets, mirror pairs, and two unseen
  distance/speed packages.
- Four frozen methods: canonical two-skill PPO, privileged static task-gate
  reference, always-head-on, and always-crossing. MVP-2 is excluded because its
  fixed-budget pilot is frozen negative evidence.
- Three opponents are reported separately: rule-based expert, end-to-end
  neural, and independently trained PPO/VPP. The two neural opponents receive
  only an explicit feature-name projected, role-reversed base 16-D observation.
- Each scenario uses its preregistered seed identically across methods within
  an opponent. The episode horizon is 160 high-level steps (32 seconds).

## Head-On Discriminability Gate

For each opponent, use only the eight scenarios with `initial_class=head_on`.
The head-on terminal win rate is interpretable for routing comparisons only if:

1. the canonical PPO resolved ratio is at least `0.60`; and
2. at least one fixed specialist's resolved ratio is at least `0.60`.

Otherwise, emit `terminal_win_rate_not_discriminative` for that opponent and
report terminal outcomes descriptively only. The gate is not a performance
target and does not permit a rerun or scenario replacement.

## Required Episode Evidence

The analysis must retain raw episode telemetry and separately report terminal
reason, win/loss/draw/unresolved status, final HP differential, ego and target
attack-zone occupancy and first entry, ego crash/OOB, phase occupancy,
dynamic-taxonomy occupancy, and the three VPP bias components. Results from
different opponents must never be pooled.

## Stop Rule

No training, tuning, checkpoint replacement, reward/VPP/guidance/PID change,
scenario replacement, seed replacement, or result-driven rerun is authorized.
Any failure of the discriminability gate narrows interpretation; it does not
authorize a repair cycle.
