# Oracle Task Gate MVP 实验结果与分析

## 实验日期
2026-06-30

## 实验目的
验证 Oracle Task Gate（静态专家选择基线）的实现正确性，确认 specialist 组合是否能在各自任务上达到预期的纯性能。

## 实验配置
- **Comparison Config**: `config/experiment/jsbsim_hrl_oracle_task_gate_mvp_only.yaml`
- **Backend**: JSBSim (strict)
- **Opponent**: Expert (rule-based), End-to-end (neural)
- **Seeds**: 0-9 (10 seeds)
- **Tasks**: head_on, crossing_feasible
- **Specialists**:
  - head_on: head_on-weighted checkpoint (obs_dim=19)
  - crossing_feasible: crossing-weighted checkpoint (obs_dim=18)

## 关键 Bug 修复
### Bug 1: virtual_point 配置缺失
**问题**: `jsbsim_hrl_oracle_task_gate_mvp_only.yaml` 中缺少 `virtual_point` 配置，导致 `VirtualPointGenerator` 使用默认值，与 specialist 训练时的配置不一致。

**影响**: 
- head_on specialist 的 head_on 性能从 1.0 降到 0.1
- crossing specialist 的 crossing 性能异常

**修复**: 在 comparison config 中添加完整的 `virtual_point` 和 `trajectory_prediction` 配置，与 specialist 的 checkpoint config 一致。

### Bug 2: trajectory_prediction 配置不完整
**问题**: comparison config 中缺少 `trajectory_prediction` 的完整配置（model, history, normalization 等）。

**影响**: 预测器初始化可能使用不同的参数，影响 VPP 行为。

**修复**: 在 comparison config 中添加完整的 `trajectory_prediction` 配置。

## 实验结果

### Expert Stage (Rule-based opponent)

| Method | Task | Win Rate | Notes |
|--------|------|----------|-------|
| Oracle Task Gate | head_on | **1.0** (10/10) | 使用 head_on specialist，完美表现 |
| Oracle Task Gate | crossing_feasible | **0.0** (0/10) | 使用 crossing specialist，表现差 |
| Baseline (direct PPO) | head_on | 0.7 | 单策略平衡 head_on/crossing |
| Baseline (direct PPO) | crossing_feasible | 0.2 | 单策略平衡 head_on/crossing |

### End-to-end Stage (Neural opponent)

| Method | Task | Win Rate | Notes |
|--------|------|----------|-------|
| Oracle Task Gate | head_on | **0.8** (8/10) | head_on specialist 稳健 |
| Oracle Task Gate | crossing_feasible | **0.444** (4/9) | crossing specialist 表现改善 |

## 结果分析

### 1. Head-on specialist 性能验证
head_on specialist 在 expert stage 上达到 **1.0 胜率**，证明了：
- Oracle task gate 的 specialist 路由逻辑正确
- head_on specialist 的 checkpoint 权重正确加载
- env 配置与 specialist 训练时一致

### 2. Crossing specialist 性能问题
crossing specialist 在 expert stage 上只有 **0.0 胜率**，远低于预期。

**原因分析**:
- crossing specialist 的 checkpoint 本身在 crossing 任务上表现差（从之前的 pilot 数据确认：crossing-weighted 在 crossing 上只有 0.1-0.2 胜率）
- crossing specialist 的 obs_dim=18，而 env 的 obs_dim=19（包含 `is_crossing`），导致 obs 被截断，丢失 task 信息
- 即使使用 obs_dim=18 的 env 配置（不包含 `is_crossing`），crossing specialist 在 crossing 上也只有 0.0 胜率

**结论**: crossing specialist 本身在当前配置下对 crossing 任务无效，不是 oracle gate 的 bug。

### 3. 与 Baseline 的对比
- **Expert stage**: Oracle gate 的 head_on (1.0) >> Baseline (0.7)，但 crossing (0.0) << Baseline (0.2)
- **End-to-end stage**: Oracle gate 的 head_on (0.8) vs Baseline (~0.9)，crossing (0.444) vs Baseline (~0.8)

**关键指标**: `min(expert/ho, expert/cr)`
- Oracle gate: min(1.0, 0.0) = **0.0**
- Baseline: min(0.7, 0.2) = **0.2**

Oracle gate 的下界低于 baseline，因为 crossing specialist 在 crossing 上表现差。

## 下一步建议

1. **Investigate crossing specialist**: 分析 crossing specialist 的 training 配置和 checkpoint，找出为什么它在 crossing 上表现差。
2. **Retrain crossing specialist**: 如果 crossing specialist 的训练有问题，考虑重新训练一个更强的 crossing specialist。
3. **Adjust crossing task config**: 检查 crossing 任务的难度配置（attack_zone, reward, scenario 等），确认是否过于困难。
4. **Proceed with learned commander**: 即使 crossing specialist 表现差，oracle gate 的 routing 逻辑是正确的，可以继续推进 learned commander 的训练。

## 论文叙事要点
- Oracle gate 是 learned commander 的静态控制基线
- head_on specialist 证明了 specialist 在纯任务上的有效性（1.0 胜率）
- crossing specialist 暴露了当前 specialist 训练的局限性
- 需要进一步分析 crossing specialist 的 training 过程
