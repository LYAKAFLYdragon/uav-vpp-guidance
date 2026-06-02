# JSBSim No-Prediction VPP High-Fidelity Bridge

## 1. 本阶段目的

将已在 `SimplePointMassEnv` 中验证通过的 **No-Prediction VPP Baseline**，扩展到 **JSBSim F-16 高保真动力学后端**，形成可对比的 high-fidelity bridge。

核心目标：
- **统一后端接口**：同一套 `CloseRangeTrackingEnv` 通过配置切换 `simple` / `jsbsim` 后端；
- **统一评估指标**：同一套 `evaluate_no_prediction_scenarios.py` 和 `plot_no_prediction_results.py` 支持两种后端；
- **统一轨迹格式**：两种后端输出相同格式的 CSV 轨迹；
- **建立对比基线**：为后续接入轨迹预测（CV / LSTM / GRU）提供高保真对比数据。

## 2. 为什么先做 JSBSim No-Prediction Bridge

在引入任何预测模型之前，必须先确认：**虚拟追踪点框架本身能否在真实六自由度动力学上稳定运行？**

理由：
- **SimplePointMass 不等于真实飞机**：质点模型不包含气动力、力矩、舵面动力学，某些策略在简化环境表现好，但在 JSBSim 上可能失速或不可控；
- **控制接口需要校准**：从 `Nz_cmd / roll_rate_cmd / throttle_cmd` 映射到 JSBSim 的 `elevator / aileron / rudder / throttle` 需要实际验证；
- **预测模型的价值体现在高保真环境**：如果只在简化环境对比预测 vs 不预测，结论可信度不足。

## 3. SimplePointMass 与 JSBSim 的区别

| 特性 | SimplePointMass | JSBSim F-16 |
|------|----------------|-------------|
| 自由度 | 3DoF 质点 | 6DoF 完整刚体 |
| 动力学 | 简化运动学 | 非线性气动力 + 推力 + 重力 |
| 控制输入 | 直接改变速度/姿态 | 舵面偏转 + 油门 |
| 低层控制 | 无（直接生效） | 需要 elevator/aileron/rudder 映射 |
| 失速模型 | 无 | 有（alpha 超限会失速） |
| 坐标系 | NEU 局部平面 | 地理坐标（LLA）+ NEU 转换 |
| 计算成本 | 极低 | 中等（60 Hz 积分） |

## 4. 高层 5 Hz / 底层 60 Hz / action_repeat=12 的设计

```
high_level_dt = 0.2 s   (5 Hz 决策频率)
low_level_dt  = 1/60 s  (60 Hz 仿真步长)
action_repeat = 12      (每个高层决策步 = 12 个底层仿真步)
```

设计理由：
- **5 Hz 决策**：策略网络不需要每帧输出，降低计算量；人眼反应时间约 200 ms，5 Hz 合理；
- **60 Hz 仿真**：JSBSim 默认积分频率，保证数值稳定性；
- **action_repeat=12**：`0.2 / (1/60) = 12`，一个高层动作在底层重复执行 12 步。

控制流：
```
Policy action (5 Hz)
    ↓
VirtualPointGenerator
    ↓
LOSRateGuidance → Nz_cmd, roll_rate_cmd, throttle_cmd
    ↓
LowLevelController (滤波 + 映射)
    ↓
ActuatorInterface → elevator, aileron, rudder, throttle
    ↓
重复 12 次 × JSBSim step(1/60 s)
    ↓
新状态
```

## 5. 控制接口说明

### 5.1 输入

| 指令 | 单位 | 范围 |
|------|------|------|
| `nz_cmd` | g | [-2, 7] |
| `roll_rate_cmd` | rad/s | [-1.5, 1.5] |
| `throttle_cmd` | 无量纲 | [0, 1] |

### 5.2 映射到 JSBSim F-16

```
elevator = -nz_cmd / 7.0         # 7g 对应全偏转
aileron  = roll_rate_cmd / 1.5   # 1.5 rad/s 对应全偏转
rudder   = 0.0                   # 当前阶段不用	hrottle  = throttle_cmd        # 直接映射
```

### 5.3 保护机制

- **NaN / inf 防护**：输入非法时自动归零；
- **限幅**：所有输出裁剪到 [-1, 1]（油门 [0, 1]）；
- **一阶滤波**：`alpha=0.3` 平滑指令跳变；
- **饱和标记**：当指令被限幅时，`saturation_flag=True`。

## 6. 成功与终止判据

### 6.1 成功

- `range_m <= 900`
- `ata_deg <= 25`
- 连续保持 `success_hold_time_s = 0.2 s`

### 6.2 滞回退出

- `range_m > 950` 或 `ata_deg > 30` 时，成功计数器清零。

### 6.3 终止类型

| 类型 | 条件 |
|------|------|
| **success** | 满足成功条件并持续保持 |
| **crash** | 高度 < min_altitude_m 或 > max_altitude_m |
| **stall** | （当前框架检测为 crash 的子类，后续可细分） |
| **out_of_bounds** | range_m > max_range_m (默认 8000m) |
| **timeout** | step >= max_high_level_steps |
| **simultaneous** | 双方同时满足成功（当前单目标场景不适用） |

## 7. 输出文件说明

### 7.1 按后端区分的输出路径

```
outputs/
├── tables/no_prediction_vpp/{backend}/
│   ├── scenario_metrics.json
│   └── scenario_metrics.csv
├── trajectories/no_prediction_vpp/{backend}/
│   └── {scenario_name}/seed_{seed}/episode_{episode}.csv
└── figures/no_prediction_vpp/{backend}/
    ├── bar_success_score_win.png
    ├── bar_failure_rates.png
    ├── termination_distribution.png
    ├── trajectory_2d_{scenario}.png
    ├── range_ata_{scenario}.png
    ├── scores_{scenario}.png
    ├── commands_{scenario}.png
    ├── altitude_{scenario}.png        (JSBSim 特有)
    ├── speed_{scenario}.png           (JSBSim 特有)
    └── actuators_{scenario}.png       (JSBSim 特有)
```

### 7.2 轨迹 CSV 字段

通用字段（两种后端都有）：
- `step`, `time`, `backend`
- `ego_x/y/z`, `target_x/y/z`
- `ego_vx/vy/vz`, `target_vx/vy/vz`
- `ego_speed`, `target_speed`
- `range_m`, `ata_deg`, `aspect_deg`, `los_rate`
- `virtual_x/y/z`
- `nz_cmd`, `roll_rate_cmd`, `throttle_cmd`
- `ego_score`, `target_score`
- `done`, `termination_reason`

JSBSim 扩展字段：
- `ego_roll`, `ego_pitch`, `ego_yaw`
- `elevator_cmd`, `aileron_cmd`, `rudder_cmd`
- `throttle_actual`
- `saturation_flag`

## 8. 如何运行

### 8.1 Simple 后端评估

```powershell
cd E:\uav-vpp-guidance

python -m uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios `
    --config config/experiment/no_prediction_vpp_scenarios.yaml `
    --backend simple `
    --episodes 5 --seeds 0 1 `
    --save-trajectories
```

### 8.2 JSBSim 后端评估

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios `
    --config config/experiment/no_prediction_vpp_jsbsim.yaml `
    --backend jsbsim `
    --episodes 2 --seeds 0 `
    --save-trajectories
```

### 8.3 绘图

```powershell
# Simple 后端
python -m uav_vpp_guidance.visualization.plot_no_prediction_results `
    --metrics outputs/tables/no_prediction_vpp/simple/scenario_metrics.csv `
    --trajectories outputs/trajectories/no_prediction_vpp/simple `
    --output outputs/figures/no_prediction_vpp/simple `
    --backend simple

# JSBSim 后端
python -m uav_vpp_guidance.visualization.plot_no_prediction_results `
    --metrics outputs/tables/no_prediction_vpp/jsbsim/scenario_metrics.csv `
    --trajectories outputs/trajectories/no_prediction_vpp/jsbsim `
    --output outputs/figures/no_prediction_vpp/jsbsim `
    --backend jsbsim
```

### 8.4 测试

```powershell
pytest tests/ -v
```

## 9. 如何解读结果

### 9.1 理想基线

No-Prediction + 随机策略在 JSBSim 上的预期表现：
- `success_rate` 可能较低（随机动作难以稳定跟踪）；
- `crash_rate` 可能非零（随机动作可能导致失速或高度超限）；
- `control_saturation_rate` 较高（随机动作频繁触及限幅）。

### 9.2 与 SimplePointMass 的对比意义

| 指标差异 | 可能原因 |
|----------|----------|
| JSBSim success_rate < Simple | 真实动力学有延迟、饱和、失速 |
| JSBSim crash_rate > Simple | 真实飞机有高度/姿态限制 |
| JSBSim control_saturation_rate > Simple | 舵面偏转有物理极限 |
| JSBSim mean_speed 波动大 | 油门-速度映射非线性 |

### 9.3 后续改进预期

接入 PPO 训练后预期：
- `success_rate` 显著上升；
- `crash_rate` 下降；
- `control_saturation_rate` 下降（策略学会平滑控制）；
- `score_win_rate` 上升。

## 10. 当前局限性

1. **低层控制器仍是最小可运行版本**：
   - 当前使用简单的 `elevator = -nz/7` 线性映射，不是 NDI 或自适应控制器；
   - 没有攻角（alpha）保护，飞机可能失速；
   - 没有协调转弯（rudder 为 0）。

2. **JSBSim 参数还需要后续校准**：
   - `nz_to_elevator_gain = 1/7` 是粗略估计；
   - F-16 真实过载-舵面关系随速度、高度变化；
   - 需要系统辨识或 legacy NDI 控制器迁移。

3. **该阶段只验证 No-Prediction VPP 的高保真可运行性**：
   - 不评价预测模型优劣；
   - 不优化策略网络；
   - 只确认框架能在 JSBSim 上跑通、能记录轨迹、能输出指标。

## 11. 下一步建议

当前阶段完成后，建议按以下顺序推进：

1. **PPO 训练接入**：在 `train_no_prediction_vpp.py` 中实现完整 PPO 训练，先在 Simple 后端收敛，再迁移到 JSBSim；
2. **低层控制器优化**：迁移 legacy NDI 控制器，或设计基于 alpha 保护的增益调度；
3. **Constant Velocity (CV) Predictor 接入**：
   - 启用 `trajectory_prediction.enabled=true`；
   - 使用 `ConstantVelocityPredictor` 作为第一个预测基线；
   - 对比 No-Prediction vs CV-Prediction 在 JSBSim 上的差异；
4. **LSTM/GRU Predictor 接入**：
   - 训练并接入神经网络预测器；
   - 切换 `anchor_mode=predicted_target`；
   - 进行消融实验。
