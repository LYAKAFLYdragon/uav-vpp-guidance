# Crossing 专家动作语义与运行配置一致性审计

## 1. 审计对象

- Crossing checkpoint：`E:/uav-vpp-guidance-artifacts/CAN-20260705/assets/outputs/experiments/prediction_vpp_ppo_crossing_v3/checkpoints/best.pt`
- 训练配置快照：`E:/uav-vpp-guidance-artifacts/CAN-20260705/assets/outputs/experiments/prediction_vpp_ppo_crossing_v3/config_snapshot.yaml`
- 物理预检运行配置：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/physical_probe_v1/expert/resolved_config.yaml`
- 机器可读审计：[crossing_runtime_consistency_audit_v1.json](E:/uav-vpp-guidance/reports/crossing_runtime_consistency_audit_v1.json)

## 2. 一致项目

| 项目 | 训练配置 | 预检配置 | 结论 |
|---|---|---|---|
| 观测维度 | 18 | 18 | 一致 |
| 动作维度 | 3 | 3 | 一致 |
| VPP 三轴物理范围 | `[−1500,1500]`、`[−800,800]`、`[−500,500]` | 相同 | 一致 |
| 平滑系数 | 0.3 | 0.3 | 一致 |
| 仿真频率 | 60 Hz | 60 Hz | 一致 |
| 决策频率 | 5 Hz | 5 Hz | 一致 |
| 高层步长 | 0.2 s | 0.2 s | 一致 |
| 动作重复步数 | 12 | 12 | 一致 |

## 3. 关键不一致

| 项目 | Crossing 训练时 | 本次预检实际运行时 | 影响 |
|---|---|---|---|
| 动作语义 | `tactical_basis_v1` | 未配置，使用普通当前目标接口 | 三个归一化动作的几何含义不再相同 |
| VPP 锚点 | `predicted_target` | `current_target` | 相同动作可能指向不同参考点 |
| 轨迹预测 | 启用 | 关闭 | crossing 策略训练时依赖的预测参考被移除 |
| 预测锚点 | `predicted_target` | `current_target` | 进一步改变 VPP 参考几何 |
| 任务相关坐标系 | `world_neu`、`encounter_stable`、`target_velocity` | 未配置 | 纵向、侧向和垂向偏置的解释发生变化 |
| 任务上下文 | 训练侧存在任务相关战术基准配置 | 预检没有显式任务标志 | 任务条件参数不能可靠生效 |
| 战术基准参数 | 使用动态战术基准及任务专属范围 | 未配置 | crossing 的动作输出被另一套接口解释 |

## 4. 结论

观测和动作的形状是对齐的，但动作的物理语义没有对齐。审计判定为：

> 本次 12 条恢复预检可以作为“当前实现的恢复打法”结果保留，但不能作为 crossing 专家在其训练接口下控制能力不足的证据。

换言之，之前得到的“48 步内未恢复”同时包含了恢复策略本身和运行接口不一致的影响。当前不能据此判断 crossing 专家是否能够在 `tactical_basis_v1 + predicted_target` 语义下完成恢复。

## 5. 推荐处理顺序

1. 保留本次 12 条结果，不覆盖、不重命名，标记为 `runtime_contract_mismatch`。
2. 建立与 crossing 训练配置完全一致的只读 matched-contract probe，至少恢复 `tactical_basis_v1`、预测器、`predicted_target` 锚点、任务相关坐标系和范围参数。
3. 在 matched-contract probe 通过 strict-JSBSim 和遥测门禁后，再判断 crossing 专家是否有恢复能力。
4. 在此之前不训练新技能、不调奖励、不扩大 PPO，也不把本次失败写成“crossing 专家能力缺失”。
