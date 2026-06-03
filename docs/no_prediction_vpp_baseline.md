# No-Prediction VPP Baseline

## 概述

No-Prediction VPP Baseline 是后续预测增强方法的基线版本。它关闭了轨迹预测模块，仅使用目标当前位置作为虚拟追踪点的锚点，验证虚拟追踪点自主决策框架本身是否可运行。

核心公式：

```
Pos_Virtual = Pos_T_current + Δp
```

其中 `Δp` 由策略网络（或规则 baseline）输出的归一化 action 映射得到。

## 与预测增强方法的区别

| 特性 | No-Prediction Baseline | 预测增强方法 |
|------|------------------------|--------------|
| 轨迹预测 | 关闭 (`enabled=false`) | 开启 (`enabled=true`) |
| 锚点模式 | `current_target` | `predicted_target` 或 `constant_velocity` |
| 预测模型 | 不使用 | LSTM / GRU / 其他 |
| 虚拟点公式 | `Pos_T_current + Δp` | `Pos_T_pred + Δp` |

## 方法流程

```
action Δp
    ↓
VirtualPointGenerator (anchor_mode=current_target)
    ↓
Pos_Virtual = Pos_T_current + Δp
    ↓
LOSRateGuidance.compute_command()
    ↓
nz_cmd, roll_rate_cmd, throttle_cmd
    ↓
command_limiter + command_filter
    ↓
SimplePointMassEnv / JSBSimEnv
    ↓
observation, reward, done, info
```

## 环境设置

### SimplePointMassEnv

在 JSBSim 完全迁移前，使用简化 3DoF / point-mass 环境验证闭环：

- **本机动力学**：
  - `roll_rate_cmd` 更新滚转角
  - `nz_cmd` 粗略影响俯仰角
  - `throttle_cmd` 影响速度大小
  - 根据速度向量更新位置

- **目标动力学**：
  - `constant_velocity`：匀速直线
  - `sinusoidal`：简单正弦横向机动

### CloseRangeTrackingEnv

高层环境连接所有模块：

1. 读取 `own_state` 和 `target_state`
2. 计算 `relative_state`（`compute_relative_geometry`）
3. 确认 `trajectory_prediction.enabled=false`，不调用 `predictor_adapter`
4. 调用 `VirtualPointGenerator`，`anchor_mode=current_target`
5. 得到 `Pos_Virtual = Pos_T_current + Δp`
6. 调用 `LOSRateGuidance.compute_command()`
7. 调用 `command_limiter` 和 `command_filter`
8. 调用 `SimplePointMassEnv.step(command)`
9. 计算 `reward`
10. 检查 `done`
11. 返回 `obs, reward, done, info`

## Reward

| 奖励项 | 说明 | 权重配置 |
|--------|------|----------|
| `reward_range` | 鼓励进入合理距离区间 [800, 1200]m | `w_range` |
| `reward_angle` | 鼓励 ATA 变小 | `w_angle` |
| `reward_safety` | 高度过低惩罚 | `w_safety` |
| `reward_saturation` | 指令饱和惩罚 | `w_saturation` |
| `reward_smooth` | 指令变化率惩罚 | `w_smooth` |
| 终端奖励 | success / crash / timeout | 固定值 |

## Termination

| 条件 | 规则 |
|------|------|
| **success** | `range_m <= 900` 且 `ata_deg <= 25`，连续保持 `success_hold_time_s` |
| **hysteresis** | 若 `range_m > 950` 或 `ata_deg > 30`，成功计数器清零 |
| **crash** | `altitude_m < min_altitude_m` 或 `altitude_m > max_altitude_m` |
| **timeout** | `step >= max_high_level_steps` |
| **out_of_bounds** | `range_m > max_range_m`（默认 8000m）|

## Observation

展平向量包含：
- 归一化 `range_m`, `range_rate_mps`, `altitude_diff_m`, `speed_diff_mps`
- `sin/cos(los_azimuth)`, `sin/cos(los_elevation)`
- `sin/cos(ata)`, `sin/cos(aa)`
- 归一化 `own_speed`, `target_speed`, `own_altitude`, `target_altitude`

## Metrics

评估指标（`evaluate_no_prediction.py` 输出）：

| 指标 | 说明 |
|------|------|
| `success_rate` | 成功 episode 比例 |
| `crash_rate` | 坠毁 episode 比例 |
| `timeout_rate` | 超时 episode 比例 |
| `out_of_bounds_rate` | 越界 episode 比例 |
| `avg_return` | 平均累积奖励 |
| `avg_episode_length` | 平均 episode 长度 |
| `avg_min_range_m` | 平均最近距离 |
| `avg_final_range_m` | 平均最终距离 |
| `avg_final_ata_deg` | 平均最终 ATA |
| `avg_min_ata_deg` | 平均最小 ATA |

## 运行命令

### Smoke Rollout

```bash
# Random policy smoke
python -m uav_vpp_guidance.training.train_no_prediction_vpp \
    --config config/experiment/no_prediction_vpp.yaml --smoke

# Rule-based policy smoke
python -m uav_vpp_guidance.training.train_no_prediction_vpp \
    --config config/experiment/no_prediction_vpp.yaml --smoke --rule-mode pure_pursuit
```

输出保存至：`outputs/tables/no_prediction_vpp/smoke_summary.json`

### Evaluation

```bash
python -m uav_vpp_guidance.evaluation.evaluate_no_prediction \
    --config config/experiment/no_prediction_vpp.yaml
```

输出保存至：`outputs/tables/no_prediction_vpp/metrics.json`

### Rule-Based Baseline

```bash
python -m uav_vpp_guidance.evaluation.evaluate_no_prediction \
    --config config/experiment/rule_based_pursuit_baseline.yaml \
    --rule-mode pure_pursuit
```

## 后续计划

1. **接入 PPO 训练**：在 `train_no_prediction_vpp.py` 中实现完整 PPO 训练循环
2. **完整 JSBSim 迁移**：从 legacy 项目迁移高保真 F-16 动力学
3. **启用轨迹预测**：切换 `anchor_mode=predicted_target`，运行预测消融实验
4. **增益优化**：实现 gain-only CEM 和 strategy-gain 双层优化
