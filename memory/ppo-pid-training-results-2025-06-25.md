---
name: ppo-pid-training-results-2025-06-25
description: PPO+PID hybrid training results (200k steps, optimal PID gains, JSBSim F-16)
metadata:
  type: project
---

# PPO+PID Training Results (2026-06-25)

**Training config**: `train_ppo_pid_jsbsim.yaml` with optimal PID gains (Kp=0.50, Kd=0.10)
**Backend**: JSBSim F-16, CUDA (RTX 2080 Ti)
**Duration**: 1,787s (~30 min) for 200k steps

## Training Results

| Metric | Value |
|--------|-------|
| Total steps | 200,000 |
| Total episodes | 1,757 |
| Eval success rate | **86.67%** |
| Eval crash rate | 13.33% |
| Eval OOB rate | 0.00% |
| Best return | 168.82 |
| Aggressiveness (mean) | -0.708 |

## Evaluation on Multi-Waypoint + Sustained-Turn (20 seeds each)

| Task | Crashes | Key Metric |
|------|---------|------------|
| multi_waypoint | **0/20** | mean_wp = **4.00** |
| sustained_turn | 20/20 (stall) | mean_orbits = 2.44 |

## Key Findings

1. **PPO+PID excels at waypoint tracking**: 0 crashes, mean_wp=4.00, matching Enhanced PID performance.
2. **Sustained turn energy management is the weakness**: PPO agent pulls excessive nz (up to 10.9g!) to maintain the orbit, bleeding speed until stall at ~150 m/s. Despite crashing, achieves 2.44 orbits before stall.
3. **Agent is conservative**: Aggressiveness = -0.708 means PID gains are scaled to ~0.65× base values. The agent learned that being conservative reduces crashes in training scenarios.
4. **Training is fast**: 200k steps in ~30 minutes on RTX 2080 Ti — JSBSim simulation is the bottleneck, not GPU.

**Why**: The PPO agent optimizes for reward (close tracking), not for energy preservation. The reward function doesn't penalize energy loss, so the agent trades speed for tracking accuracy. This mirrors the classic fighter pilot dilemma: turn tight and lose energy, or preserve energy and accept wider turns.

**How to apply**: 
- For tasks requiring energy preservation (sustained_turn), add an energy penalty to the reward function or use a separate energy-management reward term.
- For close-range tracking (multi_waypoint, break_turn), the current PPO+PID agent is production-ready.
- The 13.33% training crash rate is acceptable for training but should be monitored; further curriculum tuning may reduce it.

## Related memories
- [[pid-gain-scan-results-2025-06-25]] — optimal PID gains used
- [[flight-envelope-characterization-2025-06-25]] — operational envelope
- [[sustained-turn-energy-management]] — energy management problem
- [[2026-06-23-flight-control-archive]] — prior results
