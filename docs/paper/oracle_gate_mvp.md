## Oracle Task Gate: A Static Specialist Selection Baseline

### 3.1 Motivation
The Oracle Task Gate serves as a hard-wired control baseline for the hierarchical commander architecture. It routes each task to a pre-trained frozen specialist without any learning, establishing a performance reference for what a perfect learned commander could achieve if it had perfect task identification and the specialists were already effective.

### 3.2 Implementation
The OracleTaskGatePolicy class implements a static mapping from task names to frozen PPO specialists. Each specialist is loaded from its training checkpoint with its original configuration. The gate handles observation dimension mismatches via padding or truncation.

```python
class OracleTaskGatePolicy:
    def __init__(self, specialists_config, device="cpu"):
        self._specialists = {}
        for task_name, spec_cfg in specialists_config.items():
            specialist = PPOAgent(...)
            specialist.load(spec_cfg["checkpoint"])
            self._specialists[task_name] = specialist

    def set_task_name(self, task_name):
        self.current_task_name = task_name

    def get_deterministic_action(self, obs):
        specialist = self._specialists[self.current_task_name]
        # Handle obs_dim mismatch via padding/truncation
        ...
        return specialist.get_deterministic_action(obs)
```

### 3.3 Experimental Results (AoA 60 scope)

#### Expert Opponent Stage

| Method | head_on | crossing_feasible |
|--------|---------|-------------------|
| Oracle Task Gate | **1.0** | **0.0** |

The head-on specialist achieves a perfect 1.0 win rate on the head-on task, demonstrating that the specialist training is effective for its designated task. However, the crossing specialist fails to achieve any wins on the crossing task (0.0), indicating a fundamental limitation in the current crossing specialist training.

#### End-to-end Opponent Stage

| Method | head_on | crossing_feasible |
|--------|---------|-------------------|
| Oracle Task Gate | **0.8** | **0.5** |

The oracle gate shows improved performance against the end-to-end opponent on the crossing task (0.5), but the expert-stage result remains concerning.

### 3.4 Analysis
The failure of the crossing specialist to perform on the crossing task (expert stage: 0.0) reveals a critical gap in the current training pipeline. Despite task-weighted training (crossing weight = 2.0), the specialist does not generalize to the crossing scenario against the expert opponent.

Possible explanations include:
1. **Task difficulty imbalance**: The crossing task may be inherently more difficult than the head-on task against the expert opponent.
2. **Training configuration**: The combat finetune process may have overfitted to the head-on task or failed to capture the crossing task's dynamics against the expert.
3. **Observation mismatch**: The crossing specialist was trained with obs_dim=18 (without task_type), while the oracle gate environment provides obs_dim=19. Truncation of the `is_crossing` feature may degrade performance.

**Note**: We do not claim the oracle gate as an "upper bound" for the learned commander because the crossing specialist itself is ineffective. A learned commander cannot outperform a specialist pool that contains an ineffective member.

### 3.5 Implications for the Hierarchical Commander
The oracle gate results establish two key observations:
- **Head-on specialist is effective**: 1.0 win rate against expert opponent
- **Crossing specialist is ineffective**: 0.0 win rate against expert opponent

Before training a learned commander, we must first understand why the crossing specialist fails. If the crossing specialist can be improved, the oracle gate would provide a stronger baseline. If not, the learned commander may need to compensate for the weak crossing specialist through dynamic adaptation.

### 3.6 Next Steps
1. Audit the crossing specialist's training logs and hyperparameters to identify the failure cause.
2. Evaluate whether the crossing specialist needs retraining with different configuration.
3. Only proceed with learned commander training after the crossing specialist issue is resolved or acknowledged.
