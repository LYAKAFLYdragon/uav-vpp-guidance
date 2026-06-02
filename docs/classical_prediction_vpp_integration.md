# Stage 6A: Classical CV/CA Prediction VPP Integration with P1 Hardening

## 1. 本阶段目的

在 No-Prediction VPP Baseline（Stage 5）的基础上，接入经典轨迹预测器 Constant Velocity (CV) 和 Constant Acceleration (CA)，使虚拟追踪点锚点从“目标当前位置”升级为“目标预测未来位置”，从而验证预测信息对空战近距跟踪任务的潜在增益。

## 2. 为什么先接 CV/CA，而不是直接接 LSTM/GRU/TSLANet

- **基线对照**：CV/CA 是可解释、零参数的经典模型，能作为后续神经网络预测器的性能下界。
- **闭环验证**：在引入复杂模型前，必须先确认预测-制导-控制闭环的稳定性。
- **数据独立性**：CV/CA 不需要离线数据集训练，可直接在线运行，便于快速迭代。
- **故障隔离**：若后续 LSTM/GRU 出现问题，可立即回退到 CV/CA 确认是预测器问题还是闭环问题。

## 3. CV 模型公式

假设目标在短时预测窗口内速度保持不变：

```
p_pred = p_t + v_t * T
```

其中：
- `p_t`: 目标当前位置（NEU，m）
- `v_t`: 目标当前速度（NED，m/s）
- `T`: 预测前瞻时间（s）

若当前速度向量缺失，则使用标量速度 `v` 和航向角 `ψ` 做简化转换：

```
v = [v * cos(ψ), v * sin(ψ), 0]
```

若速度信息完全缺失，则 fallback 到当前位置。

## 4. CA 模型公式

假设目标在短时预测窗口内加速度保持不变：

```
p_pred = p_t + v_t * T + 0.5 * a_t * T^2
```

其中：
- `a_t`: 目标当前加速度（m/s^2），由历史状态差分估计。

加速度估计策略：
- 优先使用历史序列中最近 3 帧的速度差分。
- 历史不足 3 帧时，fallback 到 CV 模型。
- 速度信息完全缺失时，fallback 到当前位置。

## 5. Predicted Target Anchor 数据流

```
CloseRangeTrackingEnv.step(action):
  1. 获取 own_state, target_state
  2. 计算 relative_state
  3. 若 trajectory_prediction.enabled=true:
       - 更新 TrajectoryStateBuffer
       - 调用 TrajectoryPredictorAdapter.predict(target_state)
       - 得到 predicted_target_position
  4. VirtualPointGenerator.action_to_virtual_point(
       action, own_state, target_state,
       anchor_mode=predicted_target,
       predicted_target_position=predicted_target_position)
  5. LOSRateGuidance.compute_command(...)
  6. 环境动力学 step
  7. reward / termination / info
```

## 6. 与 No-Prediction Baseline 的区别

| 特性 | No-Prediction | CV-Prediction | CA-Prediction |
|---|---|---|---|
| anchor_mode | current_target | predicted_target | predicted_target |
| 预测公式 | — | p + v*T | p + v*T + 0.5*a*T^2 |
| 需要历史 | 否 | 否（当前帧即可） | 是（至少 3 帧） |
| 适用场景 | 低速/直线 | 匀速直线 | 匀加速机动 |
| 失效场景 | — | 高机动转弯 |  jerk 大 / 非匀加速 |

## 7. P1 修复说明

### 7.1 train_log.csv 多 episode 丢失

**问题**：旧版训练脚本在 PPO update 后才写入 `train_row`，导致一个 rollout 内多个 episode 仅最后一个被记录。

**修复**：拆分为两个日志文件：
- `episode_train_log.csv`：每个 episode done 时立即写入 episode 级指标。
- `update_train_log.csv`：每次 PPO update 后写入 update 级指标（policy_loss, value_loss 等）。

### 7.2 config/ppo.yaml key 统一

**问题**：`ppo.yaml` 使用 `clip_range`、`n_steps`、`n_epochs`、`network` 等 key，与代码实际读取的不一致。

**修复**：统一为代码实际使用的 key：
- `clip_coef`
- `rollout_steps`
- `update_epochs`
- `minibatch_size`
- `value_coef`
- `policy`（替代 `network`）

### 7.3 config/env.yaml 补充 max_range_m

**问题**：`env.yaml` 的 `termination` 部分缺少 `max_range_m`。

**修复**：补充 `max_range_m: 8000.0`，与实验配置保持一致。

## 8. 训练命令

### No-Prediction Baseline（验证未破坏）

```bash
python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
  --config config/experiment/train_no_prediction_vpp_ppo.yaml \
  --smoke
```

### CV Prediction

```bash
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
  --config config/experiment/train_vpp_ppo_cv.yaml \
  --smoke
```

### CA Prediction

```bash
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
  --config config/experiment/train_vpp_ppo_ca.yaml \
  --smoke
```

## 9. 评估命令

### Prediction Comparison

```bash
python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
  --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
  --backend simple \
  --episodes 10 \
  --seeds 0 1 2 \
  --save-trajectories
```

## 10. 输出文件说明

训练输出目录结构：

```
outputs/experiments/vpp_ppo_cv_prediction/
  config_snapshot.yaml
  checkpoints/best.pt
  checkpoints/last.pt
  logs/episode_train_log.csv
  logs/update_train_log.csv
  logs/eval_log.csv
  logs/smoke_summary.json
  trajectories/eval/
  figures/
```

评估输出目录结构：

```
outputs/tables/prediction_comparison/simple/
  prediction_metrics.json
  prediction_metrics.csv
  trajectories/no_prediction/
  trajectories/cv_prediction/
  trajectories/ca_prediction/
```

## 11. 如何解释预测误差和闭环性能之间的关系

- **预测误差小 ≠ 闭环性能好**：若 policy 未充分训练，即使预测准确，action 仍可能不合理。
- **闭环性能好 ≠ 预测误差小**：在 favorable 场景下，即使 CV 预测有偏差，policy 仍可能通过调整 Δp 补偿。
- **建议分析维度**：
  - 按 scenario（favorable/neutral/disadvantage/challenging）分别比较。
  - 同时观察 `mean_prediction_error_m` 和 `success_rate` 的相关性。
  - 若 CV/CA 在 challenging 场景下 success_rate 显著高于 No-Prediction，说明预测信息有正向价值。

## 12. 当前局限性

1. **CV/CA 不是最终最优模型**：它们假设目标运动模式简单，在高机动场景（如大 G 转弯）下预测误差会显著增大。
2. **PPO baseline 可能尚未充分收敛**：当前 smoke/短训练下 success_rate 接近 0% 属于正常，需要数十万步以上的充分训练才能看到预测增益。
3. **预测增益必须按场景分析**：不能宣称 CV/CA 在所有场景下都优于 No-Prediction。
4. **CA 加速度估计受噪声影响**：基于历史差分的加速度估计对时间步长和测量噪声敏感。
5. **predictor 当前冻结**：本阶段 predictor 不参与梯度训练，仅作为确定性模块使用。后续阶段可考虑 predictor 与 policy 的联合优化。
