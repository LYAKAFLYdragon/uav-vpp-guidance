# 轻度劣势态势双节点动作序列证据冻结记录

## 1 冻结范围

本记录冻结以下连续证据链：

1. 固定 H0 控制器局部二维经验包线细化；
2. 基于 V4 真实 handoff 局部响应的双节点动作序列离线筛选；
3. 一个最简单双节点序列的独立 strict-JSBSim 配对运行时探针。

本冻结属于当前 dirty 工作树下的研发证据冻结，不等同于 clean SHA 下的论文正式冻结。冻结后不得覆盖原始结果、放宽门槛或重复运行以追求通过。

## 2 离线筛选结果

离线筛选来源标识为：

`THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-OFFLINE-SCREEN-V1-20260904`

筛选只使用 V4 局部响应分析结果，没有运行 JSBSim，也没有训练或修改控制器。第 0 个节点在 handoff 后第 0 步施加，第 1 个节点在 handoff 后第 8 步施加，其余步骤使用冻结 H0 动作。

- 候选组合数：729；
- 两个对手：`expert`、`end_to_end`；
- 共同通过 4 步和 12 步方向门的候选数：6；
- 选取的最简单候选：节点 0 为 `[0,+0.08,0]`，节点 1 为 `[-0.08,0,0]`。

需要特别说明：离线筛选采用线性叠加估计，尚未建模两个动作节点之间的耦合。因此，离线通过只表示该序列值得进行一次独立运行时验证，不表示已经证明系统性能改善。

## 3 运行时探针结果

运行时探针来源标识为：

`THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-RUNTIME-PROBE-V1-20260904`

该探针使用 12 条真实 handoff 源轨迹，分别运行基线和候选，共 24 条严格 JSBSim 条件和 12 组配对，并记录 handoff 后 4、12、48 步响应。

运行完整性门通过：

- 24/24 条件为 JSBSim 最终后端；
- 12/12 组配对的 handoff 状态哈希一致；
- 响应窗口完整；
- 无后端回退、预测器回退、动作裁剪、异常重置或新增我方坠毁/越界。

实用改善门未通过：

| 对手 | 第 12 步平均距离变化（m） | 第 12 步平均距离率变化（m/s） | 安全门 | 实用改善门 |
|---|---:|---:|---|---|
| expert | 0.003717 | 0.010963 | 通过 | 通过 |
| end-to-end | 0.002618 | 0.001789 | 通过 | 未通过 |

预注册要求两个主要对手的距离变化至少为 `0.001 m`、距离率变化至少为 `0.002 m/s`。端到端对手的距离率变化低于门槛，因此最终判定为：

`sequence_not_confirmed_runtime_improvement`

## 4 研究结论

本轮建立了一个可重复的正向工程事实：基于真实 handoff 状态设计的双节点动作序列能够在两个对手下被严格执行，且不会引入新的安全事件；在短时窗口内，两名对手都观察到正向距离变化。

但本轮没有建立跨两个对手的实用几何改善。更准确的结论是：当前接口存在可测量的局部方向响应，但该响应经过完整闭环后仍具有对手依赖，尚不能升级为通用的时间扩展编译器收益。

因此，本轮之后：

- 不启动共享技能库训练；
- 不扩大 PPO；
- 不调奖励、PID、滤波器或动作幅度；
- 不重复运行同一序列以追求通过；
- 不将本轮结果写成“跨对手性能提升”。

论文中可将其作为“局部控制方向可执行性已验证，但系统级改善仍受对手条件影响”的边界结果。若论文需要收口，建议冻结该结论并转入统计表达和章节整合，而不是继续扩大动作序列搜索。

## 5 复现与工件

边界细化配置：

`E:/uav-vpp-guidance/config/experiment/mild_disadvantage_capability_boundary_refinement_v1.yaml`

边界细化结果：

`E:/uav-vpp-guidance/outputs/mild_disadvantage_capability_boundary_refinement_v1/refinement_v1`

离线筛选配置：

`E:/uav-vpp-guidance/config/experiment/mild_disadvantage_two_node_action_sequence_offline_screen_v1.yaml`

离线筛选结果：

`E:/uav-vpp-guidance/outputs/mild_disadvantage_two_node_action_sequence_offline_screen_v1`

运行时探针结果：

`E:/uav-vpp-guidance/outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1_retry2`

关键复核命令：

```powershell
python scripts/verify_mild_disadvantage_capability_boundary_refinement_v1.py `
  --root outputs/mild_disadvantage_capability_boundary_refinement_v1/refinement_v1

python scripts/verify_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py `
  --root outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1_retry2

python -m pytest `
  tests/test_mild_disadvantage_two_node_action_sequence_offline_v1.py `
  tests/test_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py `
  tests/test_mild_disadvantage_two_node_action_sequence_runtime_audit_v1.py -q
```

## 6 关键哈希

| 工件 | SHA-256 |
|---|---|
| 边界细化配置 | `255c76afea8b2a77d7319a8da9d2ae140857684d1c847942bc18056d55d0292a` |
| 边界细化 `pilot_manifest.json` | `b8ab551d019f64684203751a4fbb3120d39edc6a1a092212420b274e3e0650cb` |
| 边界分类表 | `5ed845be64954774a153a6988c91189fd651c52af8a79963da686859a35c0a6a` |
| 离线筛选配置 | `307e4cc561ee1d3f228343a3cbd806ee28120bdb95f99b7ae2a77216847e59ae` |
| 离线筛选结果 | `3d2271908f26811d4bc6ec3c5181ed08e6cb657bd9af8f8f903d0694d6165db4` |
| 运行时 `run_manifest.json` | `bc75241b8eeeac164b00965b0b8e4acd838320c0c4166653dd156196e5c44384` |
| 运行时 `verification.json` | `0f2c656a151409b87427920cfb2883ddfde6f7c290c44f3e93093a556ba1a9c4` |

冻结时 Git 信息：`fix/flight-control-comparison-sync`，`6d6f0f86e537671bab566a61a898d7b4ba52c928`，工作树为 dirty。

