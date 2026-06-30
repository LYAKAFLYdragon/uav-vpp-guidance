## Oracle Task Gate: A Static Specialist Selection Baseline

### 3.1 Motivation
The Oracle Task Gate serves as a hard-wired control baseline for the hierarchical commander architecture. It routes each task to a pre-trained frozen specialist without any learning, establishing an upper bound on what a perfect learned commander could achieve if it had perfect task identification.

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

### 3.3 Experimental Results

#### Expert Opponent Stage

| Method | head_on | crossing_feasible | min(ho, cr) |
|--------|---------|-------------------|-------------|
| Oracle Task Gate | **1.0** | **0.0** | **0.0** |
| Baseline (single PPO) | 0.7 | 0.2 | 0.2 |

The head-on specialist achieves a perfect 1.0 win rate on the head-on task, demonstrating that the specialist training is effective for its designated task. However, the crossing specialist fails to achieve any wins on the crossing task (0.0), indicating a fundamental limitation in the current crossing specialist training.

#### End-to-end Opponent Stage

| Method | head_on | crossing_feasible | min(ho, cr) |
|--------|---------|-------------------|-------------|
| Oracle Task Gate | 0.8 | 0.444 | 0.444 |
| Baseline (single PPO) | ~0.9 | ~0.8 | ~0.8 |

The oracle gate shows improved performance against the end-to-end opponent, particularly on the crossing task (0.444), but still falls short of the baseline's balanced performance.

### 3.4 Analysis
The failure of the crossing specialist to perform on the crossing task reveals a critical gap in the current training pipeline. Despite task-weighted training (crossing weight = 2.0), the specialist does not generalize to the crossing scenario. Possible explanations include:

1. **Task difficulty imbalance**: The crossing task may be inherently more difficult than the head-on task, requiring more training steps or a different architecture.
2. **Observation mismatch**: The crossing specialist was trained with obs_dim=18 (without task_type), while the oracle gate environment provides obs_dim=19. Truncation of the `is_crossing` feature may degrade performance.
3. **Training configuration**: The combat finetune process may have overfitted to the head-on task or failed to capture the crossing task's dynamics.

### 3.5 Implications for the Hierarchical Commander
The oracle gate results establish two key bounds:
- **Upper bound on head-on performance**: 1.0 (perfect)
- **Lower bound on crossing performance**: 0.0 (current specialist limitation)

A learned commander cannot exceed the oracle gate's performance unless it can identify and switch to better specialists. Since the current crossing specialist is ineffective, the learned commander's value lies in its ability to dynamically adapt or in providing a pathway to train better specialists.

### 3.6 Next Steps
1. Investigate the crossing specialist's training logs to identify why it fails on the crossing task.
2. Consider retraining the crossing specialist with adjusted hyperparameters or a longer training horizon.
3. Evaluate whether a multi-task policy (single network with task-conditioned heads) could outperform the specialist combination.
4. Proceed with learned commander training, using the oracle gate as a static baseline for comparison.
