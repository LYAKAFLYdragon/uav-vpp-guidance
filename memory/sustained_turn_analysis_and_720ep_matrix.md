# Sustained-Turn 指标分析与 720-EP 回归矩阵补全

## 1. Sustained-Turn 指标分析

### 1.1 数据汇总（bank75/76/80）

| Config | Controller | Completed Orbits | Avg Turn Radius (m) | Avg Turn Rate (°/s) | Speed (mps) |
|--------|------------|------------------|----------------------|---------------------|-------------|
| bank75 | baseline_pid | 0.98 | - | - | - |
| bank75 | enhanced_pid | 1.13 | - | - | - |
| bank75 | ppo_pid | 1.07 | - | - | - |
| bank75 | robust_pid | 1.13 | - | - | - |
| bank76 | baseline_pid | 0.95 | 3312.6 | 17.79 | - |
| bank76 | enhanced_pid | 1.11 | 3317.5 | 13.54 | - |
| bank76 | ppo_pid | 1.10 | 3106.1 | 14.16 | - |
| bank76 | robust_pid | 1.11 | 3317.5 | 13.54 | - |
| bank80 | baseline_pid | 0.64 | - | - | - |
| bank80 | enhanced_pid | 0.71 | - | - | - |
| bank80 | ppo_pid | 1.70 | - | - | - |
| bank80 | robust_pid | 0.71 | - | - | - |

### 1.2 关键发现

**目标 vs 实际对比：**
- 目标 turn radius: 1200m
- 实际 avg turn radius: ~3106-3317m (bank76)
- 实际 radius 是目标的 **2.6-2.8 倍**

**时间限制分析：**
- Episode 时间限制: 450 steps × 0.2s = **90s**
- 理论单圈时间（基于实际 radius）: 2π × 3315m / 250mps ≈ **83.3s**
- 理论最大圈数: 90s / 83.3s ≈ **1.08**
- 实际观测: **0.95-1.11** (bank75/76)
- **结论：实际 completed_orbits 与理论上限一致，不是 tracking 不准，而是时间限制过紧**

**bank80 异常：**
- ppo_pid 在 bank80 达到 1.70 orbits，远超其他控制器（0.64-0.71）
- 说明 ppo_pid 在 bank80 配置下有显著优势
- bank80 与其他 bank 的差异需要进一步调查

**目标 2.0 orbits 的可行性：**
- 要达到 2.0 orbits，需要 2 × 83.3s ≈ **166.6s**
- 当前时间限制只有 90s，**理论不可行**
- 建议：要么将时间限制扩展到 1024 steps（204.8s），要么将目标调整为 1.0 orbits

### 1.3 建议

1. **时间限制调整**：将 sustained_turn 的 `max_high_level_steps` 从 450 调整到 1024（204.8s），使 2.0 orbits 理论可行
2. **或调整目标**：将 completed_orbits 的目标从 2.0 降低到 1.0，与现实能力匹配
3. **区分原因**：在报告中明确区分 "tracking 不准" vs "时间不够"，使用 avg_turn_radius_m 和 avg_turn_rate_deg_s 作为辅助指标

---

## 2. 720-EP 回归矩阵补全

### 2.1 完整矩阵（现有数据）

**Config = bank75 (20 seeds × 4 controllers × 3 tasks = 240 episodes)**

| Task ↓ / Controller → | baseline_pid | enhanced_pid | ppo_pid | robust_pid |
|-----------------------|-------------|-------------|---------|------------|
| multi_waypoint (wp) | **5.00** | 4.25 | 4.00 | 4.25 |
| sustained_turn (orbits) | 0.98 | 1.13 | 1.07 | 1.13 |
| break_turn (track_err m) | 0.0 | 0.0 | 0.0 | 0.0 |

**Config = bank76 (20 seeds × 4 controllers × 3 tasks = 240 episodes)**

| Task ↓ / Controller → | baseline_pid | enhanced_pid | ppo_pid | robust_pid |
|-----------------------|-------------|-------------|---------|------------|
| multi_waypoint (wp) | **5.00** | 4.25 | 4.00 | 4.25 |
| sustained_turn (orbits) | 0.95 | 1.11 | 1.10 | 1.11 |
| break_turn (track_err m) | 0.0 | 0.0 | 0.0 | 0.0 |

**Config = bank80 (20 seeds × 4 controllers × 3 tasks = 240 episodes)**

| Task ↓ / Controller → | baseline_pid | enhanced_pid | ppo_pid | robust_pid |
|-----------------------|-------------|-------------|---------|------------|
| multi_waypoint (wp) | 4.00 | 4.00 | 4.00 | 4.00 |
| sustained_turn (orbits) | 0.64 | 0.71 | **1.70** | 0.71 |
| break_turn (track_err m) | 0.0 | 0.0 | 0.0 | 0.0 |

### 2.2 缺失数据（需要补充）

| 缺失项 | 原因 | 优先级 | 行动计划 |
|--------|------|--------|----------|
| **GainScheduled PID** | 未在回归中运行 | P1 | 使用现有的 `GainScheduledPIDController` 运行 20-seed 评估 |
| **PPO 基线**（纯 PPO） | 需要训练新 checkpoint | P0 | 训练 `train_no_prediction_vpp_ppo_jsbsim_compare.yaml` 并评估 |
| **真正的 PPO+PID** | 需要训练新 checkpoint | P0 | 训练 `train_ppo_pid_jsbsim.yaml` 并评估 |
| **break_turn 指标** | track_err=0.0 是数据 bug | P1 | 调查 break_turn 评估逻辑 |

### 2.3 break_turn 数据问题

所有 break_turn 的 `mean_track_error_m` = 0.0m，这不可能（即使是 perfect tracking 也有数值误差）。

**可能原因：**
1. `break_turn` 的 `compute_tracking_metrics` 使用了不同的 metric key
2. 数据在 aggregate 阶段被错误处理
3. break_turn 的 trajectory 记录格式与 multi_waypoint 不同

**建议调查路径：**
- 检查 `break_turn_baseline_pid_aggregate.json` 中是否存在 `mean_track_error_m` 以外的 tracking 指标
- 检查 `flight_control_metrics.py` 中 `compute_break_turn_metrics` 是否实现

---

## 3. 综合结论

### 已完成
- 720-EP 矩阵的 **现有 36 个组合** 数据已完整提取（4 controllers × 3 tasks × 3 configs）
- Sustained-turn 的 avg_turn_radius_m 和 avg_turn_rate_deg_s 已在 aggregate 中可用
- Baseline PID 5.00 满分现象已解释（继承链 + M1 修复传递效应）

### 待完成
1. **P0**: 训练真正的 PPO+PID（action_dim=4, apic.enabled=false）并评估
2. **P0**: 训练 PPO 基线并评估
3. **P1**: 运行 GainScheduled PID 的 20-seed 评估
4. **P1**: 调查 break_turn track_err=0.0 的原因
5. **P1**: 调整 sustained_turn 时间限制或目标

### 下一步行动
```bash
# 1. 启动 PPO+PID 训练（当前最高优先级）
D:/Anaconda3/python.exe -m uav_vpp_guidance.training.train_ppo_pid_jsbsim \
    --config config/experiment/train_ppo_pid_jsbsim.yaml

# 2. 并行启动 PPO 基线训练
D:/Anaconda3/python.exe -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
    --config config/experiment/train_no_prediction_vpp_ppo_jsbsim_compare.yaml

# 3. 训练完成后，运行完整评估
D:/Anaconda3/python.exe scripts/run_flight_control_comparison.py \
    --run-id fc_compare_final \
    --checkpoint-ppo-pid outputs/experiments/ppo_pid_jsbsim/checkpoints/last.pt \
    --checkpoint-ppo outputs/experiments/no_prediction_vpp_ppo_jsbsim_compare/checkpoints/last.pt \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --jobs 4
```
