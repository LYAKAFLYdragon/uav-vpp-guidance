# No-Prediction VPP Baseline — 实验套件文档

## 1. 本阶段实验目的

本阶段将上一轮的 "最小闭环验证" 扩展为**论文级 No-Prediction VPP Baseline 实验套件**。目标包括：

- 在**多类典型初始场景**下系统评估 no-prediction baseline 的表现；
- 建立一套**可重复、可对比**的评估指标与轨迹记录流程；
- 生成**论文初稿可用的图表**；
- 为后续接入轨迹预测（LSTM/GRU）和完整 JSBSim 提供**可对比的基线数据**。

## 2. 为什么先做 No-Prediction Baseline

在引入任何预测模型之前，必须先回答一个问题：**"如果策略网络只根据当前目标位置来决策虚拟追踪点，性能基线是多少？"**

No-prediction baseline 的意义：
- **隔离变量**：排除预测误差对策略评估的干扰；
- **验证框架**：确认 VPP + LOS-guidance + reward + termination 的闭环本身能跑通；
- **量化预测增益**：后续引入预测模型后，可直接与本 baseline 对比，量化预测带来的提升。

## 3. 环境与任务设定

### 3.1 后端
- 使用 `SimplePointMassEnv`（简化 3DoF 质点运动）进行机制验证；
- 高保真 JSBSim F-16 将在后续阶段接入。

### 3.2 控制周期
- `high_level_dt = 0.2 s`
- `max_high_level_steps = 512`
- 单 episode 最大时长约 102.4 s。

### 3.3 虚拟追踪点
- `anchor_mode = current_target`
- `trajectory_prediction.enabled = false`
- 闭环中**不调用任何 predictor**。

## 4. 四类初始场景说明

| 场景 | 描述 | Ego 初始态势 | Target 初始态势 |
|------|------|-------------|----------------|
| **favorable** | Ego 在目标后方，速度更快 | 位置 `[0,0,5000]`，航向 `0°`，速度 `220 m/s` | 位置 `[2000,0,5000]`，航向 `0°`，速度 `180 m/s` |
| **neutral** | 迎头相遇，速度相近 | 位置 `[0,0,5000]`，航向 `0°`，速度 `200 m/s` | 位置 `[2000,0,5000]`，航向 `180°`，速度 `200 m/s` |
| **disadvantage** | Target 在 Ego 后方且更快 | 位置 `[0,0,5000]`，航向 `0°`，速度 `180 m/s` | 位置 `[-2000,0,5000]`，航向 `0°`，速度 `220 m/s` |
| **challenging** | 大横向偏移，交叉航迹 | 位置 `[0,0,5000]`，航向 `45°`，速度 `200 m/s` | 位置 `[1500,1500,5200]`，航向 `225°`，速度 `210 m/s` |

## 5. 成功判据

**终端成功**（episode 终止时判定）：
- `range_m <= 900`
- `ata_deg <= 25`
- 连续保持 `success_hold_time_s = 0.2 s`

**瞬时成功**（每一步计算，用于 `instant_success_rate`）：
- 单步满足 `range_m <= 900` 且 `ata_deg <= 25`。

## 6. Score-Based Win 判据

除了几何成功判据，我们还引入**双方得分系统**以衡量相对态势优劣：

- **Ego Score**：综合距离分（越接近 900m 越好）、ATA 分（越小越好）、AA 分（越小越好）
- **Target Score**：综合距离分（越远越好）、ATA 分（越大越好，表示目标更难被拦截）

**Score Win**：episode 结束时 `mean_ego_score > mean_target_score`。

## 7. 输出文件说明

### 7.1 汇总指标

```
outputs/tables/no_prediction_vpp/scenario_metrics.json
outputs/tables/no_prediction_vpp/scenario_metrics.csv
```

包含每个场景的统计指标：
- `episode_return`：平均累积奖励
- `success_rate`：终端成功率
- `instant_success_rate`：瞬时成功率
- `score_win_rate`：得分胜率
- `crash_rate` / `timeout_rate` / `out_of_bounds_rate`：各类失败率
- `mean_final_range` / `mean_final_ata_deg`：最终几何状态
- `mean_min_range` / `mean_min_ata_deg`：最近距离/最小 ATA
- `mean_time_to_first_advantage`：首次取得优势的时间
- `mean_advantage_hold_time`：优势保持时长
- `mean_score_ego` / `mean_score_target`：平均得分

### 7.2 轨迹数据

```
outputs/trajectories/no_prediction_vpp/{scenario_name}/seed_{seed}/episode_{episode_id}.csv
```

每行包含：
- `step`, `time`
- `ego_x/y/z`, `target_x/y/z`, `virtual_x/y/z`
- `range_m`, `ata_deg`, `aspect_deg`, `los_rate`
- `nz_cmd`, `roll_rate_cmd`, `throttle_cmd`
- `ego_score`, `target_score`
- `done`, `termination_reason`

### 7.3 图表

```
outputs/figures/no_prediction_vpp/
├── bar_success_score_win.png
├── bar_failure_rates.png
├── trajectory_2d_{scenario}.png
├── range_ata_{scenario}.png
├── scores_{scenario}.png
└── commands_{scenario}.png
```

## 8. 如何运行

### 8.1 场景评估

```powershell
cd E:\uav-vpp-guidance

python -m uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios `
    --config config/experiment/no_prediction_vpp_scenarios.yaml `
    --episodes 5 `
    --seeds 0 1 2 3 4 `
    --save-trajectories
```

可选参数：
- `--rule-mode pure_pursuit`：使用规则 baseline 替代随机策略
- `--save-trajectories`：保存每 episode 的轨迹 CSV

### 8.2 绘图

```powershell
python -m uav_vpp_guidance.visualization.plot_no_prediction_results `
    --metrics outputs/tables/no_prediction_vpp/scenario_metrics.csv `
    --trajectories outputs/trajectories/no_prediction_vpp `
    --output outputs/figures/no_prediction_vpp
```

### 8.3 运行测试

```powershell
pytest tests/ -v
```

## 9. 如何解读结果

### 预期结果（随机策略）

在随机动作下，预期：
- `success_rate` ≈ 0
- `out_of_bounds_rate` 较高（随机动作难以维持跟踪）
- `score_win_rate` ≈ 0

### 基线对比意义

| 后续改进 | 预期变化 |
|----------|----------|
| 接入 PPO 训练 | `success_rate` 和 `score_win_rate` 显著上升 |
| 接入轨迹预测 | 在 `disadvantage` / `challenging` 场景下提升更明显 |
| 接入 JSBSim | 数值会变化，但相对趋势应保持一致 |

### 场景差异分析

- `favorable`：ego 态势最优，应最容易取得 success；
- `neutral`：对称态势，考验策略均衡性；
- `disadvantage`：目标在后方且更快，考验策略的转向和加速决策；
- `challenging`：大横向偏移，考验三维空间机动能力。

## 10. 当前局限性

1. **简化动力学**：`SimplePointMassEnv` 使用 3DoF 质点模型，不包含气动力、力矩、舵面动力学；
2. **随机策略**：当前 baseline 使用随机动作或规则 pursuit，不代表最终策略性能；
3. **无传感器噪声**：观测值直接从仿真状态计算，未加入测量误差；
4. **单目标固定模式**：目标仅支持匀速直线或简单正弦机动，未实现复杂逃逸策略；
5. **不代表最终六自由度物理可信结论**：本阶段目的是验证框架与建立对比基线，高保真结论需待 JSBSim 完全迁移后得出。
