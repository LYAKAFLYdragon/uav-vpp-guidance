# Pressure-Exposed Post-Merge V1 工程无效化记录

**Source：** `THESIS-DISADVANTAGE-PRESSURE-EXPOSED-POSTMERGE-COLLECTOR-V1`  
**状态：** `runtime_valid_taxonomy_gate_invalid_pending_read_only_correction`  
**禁止动作：** 不重跑 V1、不修改 V1 原始输出、不用原始错误标签 gate 做训练、M0--M4 消融或 defensive-extension 结论。

## 已核实事实

- 48 条授权 strict-JSBSim 回合已完成；工程输出路径为 `E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_postmerge_collector_v1`。
- 独立 verifier 重算了 48 个逐回合 SHA、完整 `expert/end_to_end/independent_ppo_vpp × 16` 身份矩阵及 gate；`pressure_gate_verification.json` 通过。这只能证明输出没有丢失、gate 是按当前错误运行链计算的。
- `0.559973--0.819999` 是归一化 observation range，并非物理距离。直接从 JSBSim own/target raw state 读取时，首个场景实际 range 为 `2799.865 m`，与 manifest 的 `2800 m` 一致；场景注入有效。
- manifest 声明 `disadvantage`，但 collector sidecar 的初始标签全为 `advantage`。正式 Taxonomy60 的可复核分析使用 `(AA, ATA)` 参数顺序；该 collector 使用了继承 V5 sidecar 的相反顺序，造成标签合同漂移。

## 正确解释

此前 gate 输出为 `expert 0/16`、`end_to_end 1/16`、`independent PPO/VPP 0/16`。这些数值不是“真实后合流压力不可达”的结论，原因是 taxonomy sidecar 参数顺序错误。对相同冻结 phase trace 作只读 `(AA, ATA)` 重标注时，三对手均为 `16/16` 合格窗口。重标注本身不改变动作、环境或结果，但须形成独立 correction receipt 后才能作为后续 gate 的依据。

## 唯一合理的后续工程动作

1. 固化 V1 的只读 taxonomy correction：对每个已哈希 phase step 以 `(AA, ATA)` 重标注，独立重算 gate，并验证没有改动任何原始输出。
2. 实现 `scenario application preflight`：每个场景 reset 后直接从 JSBSim own/target state 比较实际 `range_m`、own/target speed、altitude difference，以及按 `(AA, ATA)` 得到的初始 taxonomy；每项都输出可哈希 receipt。
3. 由于 V1 phase trace 没有保留 raw physical range/speed/altitude，不能单独支撑完整 opponent capability card。只有完成 raw-state telemetry completion collector（或等价的可验证 sidecar）后，才可在相同 pressure cohort 上做 capability card 和 M0--M4。

在 correction receipt 与 telemetry completion preflight 通过前，不应新增 VPP 单点偏置、reward、训练步数、技能或对手。这样能避免把标签合同错误伪装成算法负结果。
