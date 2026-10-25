# 最简单双节点动作序列严格运行时探针报告

## 1 实验目的

本次实验从六个离线候选序列中选取非零动作分量最少的一条，检查其在真实连续交接状态下是否能够产生可重复的短时几何改善。实验只改变冻结 H0 策略输出上的两个小幅动作增量，不训练新技能，不改变奖励、制导律、比例积分微分控制器、滤波器、动作范围或 JSBSim 参数。

## 2 固定方案与复现信息

- Source ID：`THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-RUNTIME-PROBE-V1-20260904`
- 配置：[mild_disadvantage_two_node_action_sequence_runtime_probe_v1.yaml](/E:/uav-vpp-guidance/config/experiment/mild_disadvantage_two_node_action_sequence_runtime_probe_v1.yaml)
- 运行脚本：[run_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py](/E:/uav-vpp-guidance/scripts/run_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py)
- 独立验证脚本：[verify_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py](/E:/uav-vpp-guidance/scripts/verify_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py)
- 正式汇总目录：`E:/uav-vpp-guidance/outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1_retry2`
- 配对结果：[paired_response_records.csv](/E:/uav-vpp-guidance/outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1_retry2/paired_response_records.csv)
- 条件结果：[condition_results.csv](/E:/uav-vpp-guidance/outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1_retry2/condition_results.csv)
- 独立验证：[verification.json](/E:/uav-vpp-guidance/outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1_retry2/verification.json)

动作节点相对于交接后的冻结 H0 原动作定义为：第 0 步叠加 `[0, +0.08, 0]`，第 8 步叠加 `[-0.08, 0, 0]`，其余步骤不叠加动作。共使用 12 条已经通过 handoff hash 重放的真实连续轨迹；两个主对手分别运行基线和候选，共 24 条条件、12 组配对。分别读取交接后第 4、12、48 步的距离、距离率、攻击区角、进入角、比能和高度。

## 3 预注册门槛

主要判断时刻为交接后第 12 步。对每个主对手，候选相对于同一场景基线的平均距离至少增加 `0.001 m`，平均距离率至少增加 `0.002 m/s`。同时必须满足：所有配对窗口有效；JSBSim 为最终后端且没有后端回退；响应窗口没有预测器回退；没有动作裁剪；每回合只有一次初始 reset；候选不能新增我方坠毁或越界。第 4 步只用于观察早期方向，第 48 步用于观察持续性，不单独作为主要性能门槛。

## 4 运行完整性结果

独立 verifier 得到 `runtime_integrity_gate=pass`：24/24 条件均为严格 JSBSim，12/12 配对的交接状态 hash 有效，所有响应窗口均包含 4、12、48 步记录；没有后端回退、响应窗口预测器回退、动作裁剪或新增坠毁/越界，两个动作节点在所有候选回合中均被实际执行。该结果证明本次序列的因果配对运行链是完整的。

## 5 配对结果

表中数值为候选减去同一 handoff、同一对手、同一随机种子基线后的平均值。

| 对手 | 时刻 | 距离变化（m） | 距离率变化（m/s） | 攻击区角变化（°） | 进入角变化（°） | 比能变化（m²/s²） | 高度变化（m） |
|---|---:|---:|---:|---:|---:|---:|---:|
| expert | 4 | +0.000068734 | +0.000215096 | -0.000001894 | +0.000101874 | +0.003866125 | -0.000031418 |
| expert | 12 | +0.003717195 | +0.010963242 | -0.000076448 | +0.004563769 | +0.049921052 | -0.001183276 |
| expert | 48 | +0.202013817 | +0.015402333 | -0.002385982 | +0.005855011 | -1.165635753 | -0.187236577 |
| end-to-end | 4 | +0.000249112 | +0.000728311 | -0.000010103 | +0.000410231 | +0.015033655 | -0.000151930 |
| end-to-end | 12 | +0.002618318 | +0.001789377 | -0.000067259 | +0.001095438 | -0.071398600 | -0.001634241 |
| end-to-end | 48 | +0.439330998 | +0.072416005 | -0.006418712 | +0.026807009 | -1.654739430 | -0.292308872 |

## 6 门槛判定与意义

专家对手通过主要改善门：第 12 步平均距离变化为 `+0.003717195 m`，距离率变化为 `+0.010963242 m/s`，且安全门通过。端到端对手的第 12 步距离变化为 `+0.002618318 m`，达到距离门，但距离率变化为 `+0.001789377 m/s`，低于预注册的 `0.002 m/s`，因此端到端对手未通过主要改善门。

最终判定为：运行完整性通过，但跨两个主对手的实用改善门不通过。该序列在两个对手下都显示了正向的短时距离和距离率变化，然而在端到端对手下未达到预先规定的距离率幅度，不能把它写成跨对手稳定的时间扩展编译器收益，也不自动授权训练新技能。当前最稳妥的研究含义是：V4 离线筛选出的候选具有可运行、可测量的方向性，但其改善幅度仍受对手条件影响，尚不足以支撑扩大时间扩展或共享技能训练线。

## 7 执行账本说明

正式统计使用 retry2 目录。初次执行在 runner 资产字典键名错误处停止，未进入物理回合；修正后的一次执行完成了 24 条物理回合，但在汇总阶段缺少比能字段；retry2 是在修正汇总字段后完成的同一预注册 probe，之后只对已落盘轨迹进行离线最终化，没有再次运行 JSBSim。两个工程失败目录均保留，不作为正式统计样本。

后续不应通过放宽门槛、增加动作幅度、修改奖励或重复运行来追求通过；若继续该方向，应先解释端到端对手下距离率响应不足的原因，或将本次结果作为“运行链完整、局部方向可见但跨对手改善门未达成”的边界证据。
