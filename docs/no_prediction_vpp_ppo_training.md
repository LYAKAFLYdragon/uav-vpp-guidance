# Stage 5: No-Prediction VPP PPO Autonomous Decision Baseline

## 1. 本阶段目的

在 **不使用轨迹预测** 的条件下，训练一个 **PPO 策略网络 πθ**，使其根据当前观测输出虚拟追踪点偏移 Δp，形成完整的自主决策闭环基线。

核心目标：
- **策略网络自主决策**：obs → πθ → Δp，替代随机动作；
- **完整 PPO 训练闭环**：rollout 收集 → GAE 计算 → clipped surrogate 更新；
- **统一后端评估**：训练好的策略可同时用于 SimplePointMass 和 JSBSim 后端评估；
- **可复现实验流程**：配置驱动、日志记录、checkpoint 管理、训练曲线绘制。

## 2. 为什么先训练 No-Prediction VPP PPO

在引入轨迹预测之前，必须先确认：**策略网络本身能否学会有效的虚拟追踪点偏移？**

理由：
- **预测器需要基线对比**：没有无预测基线，无法量化预测器的增益；
- **简化问题空间**：No-Prediction 条件下，anchor_mode=current_target，策略只需学习相对于目标当前位置的偏移，问题更简单；
- **验证端到端可训练性**：确认从 obs → Δp → LOS guidance → 环境反馈 → reward 的梯度可以正常回传。

## 3. 策略输入与输出

### 3.1 观测向量 (obs_dim = 16)

| 特征 | 维度 | 说明 |
|------|------|------|
| range_m | 1 | 归一化到 5000m |
| range_rate_mps | 1 | 归一化到 200m/s |
| altitude_diff_m | 1 | 归一化到 10000m |
| speed_diff_mps | 1 | 归一化到 400m/s |
| los_azimuth_sin/cos | 2 | LOS 方位角正弦/余弦 |
| los_elevation_sin/cos | 2 | LOS 俯仰角正弦/余弦 |
| ata_sin/cos | 2 | Aspect Target Angle 正弦/余弦 |
| aa_sin/cos | 2 | Antenna Train Angle 正弦/余弦 |
| own_speed | 1 | 归一化到 400m/s |
| target_speed | 1 | 归一化到 400m/s |
| own_altitude | 1 | 归一化到 10000m |
| target_altitude | 1 | 归一化到 10000m |

### 3.2 动作向量 (action_dim = 3)

| 输出 | 单位 | 范围 |
|------|------|------|
| Δx | m | [-1500, 1500] |
| Δy | m | [-1500, 1500] |
| Δz | m | [-300, 300] |

策略输出经过 `tanh` 压缩后，线性映射到上述物理范围。

### 3.3 虚拟点计算

```python
Pos_Virtual = Pos_Target_Current + Δp
```

`anchor_mode = current_target`，不使用任何预测。

## 4. PPO 训练流程

### 4.1 网络结构

```
Input (16)
    ↓
Shared MLP: [128, 128] + tanh
    ↓
Actor Head (3)        Critic Head (1)
    ↓                        ↓
Mean + log_std          Value
    ↓
Tanh squashing
    ↓
Action scaling to [-1500, 1500] x [-1500, 1500] x [-300, 300]
```

总参数量约 19,207 (hidden_sizes=[128,128])。

### 4.2 超参数

| 参数 | 值 |
|------|-----|
| total_timesteps | 200,000 |
| rollout_steps | 2,048 |
| minibatch_size | 256 |
| update_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_coef | 0.2 |
| value_coef | 0.5 |
| entropy_coef | 0.01 |
| learning_rate | 3.0e-4 |
| max_grad_norm | 0.5 |

### 4.3 训练循环

```
while global_step < total_timesteps:
    # Collect rollout
    for step in range(rollout_steps):
        action, log_prob, value = agent.select_action(obs)
        env.step(action)
        agent.store_transition(obs, action, log_prob, reward, done, value)

    # PPO update
    agent.compute_gae(next_value)
    for epoch in range(update_epochs):
        for minibatch in buffer.get_minibatches(batch_size):
            compute clipped surrogate loss
            compute value loss (clipped)
            compute entropy bonus
            backward + gradient clip + optimizer step

    # Periodic evaluation
    if global_step % eval_interval == 0:
        evaluate_policy(agent, env)
        save best checkpoint
```

## 5. 奖励与终止条件

### 5.1 奖励项

| 项 | 权重 | 说明 |
|----|------|------|
| range_reward | 0.5 | 理想距离区间 [800, 1200]m |
| angle_reward | 0.8 | ATA + AA 越小越好 |
| safety_penalty | 2.0 | 低空惩罚 |
| saturation_penalty | 1.0 | 指令饱和惩罚 |
| smooth_penalty | 0.1 | 指令变化率惩罚 |

### 5.2 终止条件

| 类型 | 条件 |
|------|------|
| success | range ≤ 900m 且 ATA ≤ 25°，持续 0.2s |
| crash | 高度 < 500m 或 > 15000m |
| out_of_bounds | range > 8000m |
| timeout | step ≥ 512 |

## 6. 训练输出文件说明

```
outputs/experiments/no_prediction_vpp_ppo/
├── config_snapshot.yaml          # 实验配置快照
├── checkpoints/
│   ├── best.pt                   # 评估返回最高 checkpoint
│   ├── last.pt                   # 最终 checkpoint
│   └── step_*.pt                 # 周期性保存
├── logs/
│   ├── train_log.csv             # 训练指标（每 episode / update）
│   ├── eval_log.csv              # 评估指标（每 eval_interval）
│   └── smoke_summary.json        # smoke 测试摘要
├── trajectories/
│   └── eval/                     # 评估阶段轨迹
└── figures/                      # 由 plot_training_curves.py 生成
    ├── training_return.png
    ├── training_success_rate.png
    ├── training_score_win_rate.png
    ├── training_loss.png
    ├── eval_range_ata.png
    └── training_kl_clip.png
```

### 6.1 train_log.csv 字段

- `step`, `episode`, `episode_return`, `episode_length`
- `success`, `score_win`, `crash`, `out_of_bounds`, `timeout`
- `mean_range`, `final_range`, `final_ata`
- `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`, `explained_variance`

### 6.2 eval_log.csv 字段

- `step`, `num_episodes`, `mean_return`, `std_return`
- `success_rate`, `crash_rate`, `out_of_bounds_rate`, `timeout_rate`
- `mean_final_range_m`, `mean_final_ata_deg`

## 7. 运行命令

### 7.1 Smoke 测试

```powershell
cd E:\uav-vpp-guidance

python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml `
    --smoke
```

### 7.2 完整训练

```powershell
python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml
```

### 7.3 策略评估（Simple）

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_policy `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml `
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt `
    --backend simple `
    --episodes 10 `
    --seeds 0 1 2 `
    --save-trajectories
```

### 7.4 策略评估（JSBSim）

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_policy `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml `
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt `
    --backend jsbsim `
    --episodes 2 `
    --seeds 0 `
    --save-trajectories
```

### 7.5 绘制训练曲线

```powershell
python -m uav_vpp_guidance.visualization.plot_training_curves `
    --log-dir outputs/experiments/no_prediction_vpp_ppo/logs `
    --output outputs/experiments/no_prediction_vpp_ppo/figures
```

## 8. SimplePointMass 训练与 JSBSim 评估的关系

| 阶段 | 后端 | 目的 |
|------|------|------|
| 训练 | SimplePointMass | 快速迭代、验证策略可学习性 |
| 评估 | SimplePointMass | 与随机策略对比，验证训练效果 |
| 评估 | JSBSim | 验证策略在高保真动力学上的泛化能力 |

注意：当前训练主要在 SimplePointMass 上完成。JSBSim 上直接训练需要更长的计算时间（60 Hz 积分 + 低层控制器延迟），但评估可以快速验证 checkpoint 的迁移能力。

## 9. 当前局限性

1. **尚未使用轨迹预测**：
   - `trajectory_prediction.enabled = false`
   - `anchor_mode = current_target`
   - 策略只能学习基于当前目标位置的偏移

2. **尚未使用增益适配**：
   - `use_gain_adapter = false`
   - 制导增益固定为 k_los=1.0, k_pos=0.5, k_roll=1.0

3. **JSBSim 低层控制器需要进一步校准**：
   - 当前使用简单的线性映射 `elevator = -nz/7`
   - F-16 真实过载-舵面关系随速度、高度变化
   - 策略在 JSBSim 上可能表现不如 SimplePointMass

4. **当前 PPO 结果不要求达到最终最优**：
   - 200k 步训练在 SimplePointMass 上可能只是初步收敛
   - 最终最优性能需要更多训练步数、课程学习、或更精细的奖励 shaping
   - 本阶段目标：**形成可复现的自主决策 baseline**，而非最终 SOTA

## 10. 下一步建议

当前阶段完成后，建议按以下顺序推进：

1. **Constant Velocity (CV) Predictor 接入**：
   - 启用 `trajectory_prediction.enabled = true`
   - 使用 `ConstantVelocityPredictor` 作为第一个预测基线
   - 切换 `anchor_mode = predicted_target`
   - 对比 No-Prediction vs CV-Prediction

2. **LSTM/GRU Predictor 接入**：
   - 训练神经网络预测器
   - 对比 CV vs LSTM/GRU 的预测精度
   - 量化预测精度对策略性能的影响

3. **增益适配器接入**：
   - 让策略同时输出制导增益
   - 或设计双层优化框架（bilevel）

4. **JSBSim 上直接训练**：
   - 使用训练好的 SimplePointMass 策略做 warm-start
   - 在 JSBSim 上 fine-tune
   - 需要更稳定的低层控制器
