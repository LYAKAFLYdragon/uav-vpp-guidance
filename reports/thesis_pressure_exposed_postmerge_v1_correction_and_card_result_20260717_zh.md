# Pressure-Exposed Post-Merge V1 纠正与 Capability Card 结果

## 结论

V1 的 48 条 strict-JSBSim 物理回合有效，场景 reset 的实际 state 与 manifest 一致；原始 gate 的失败仅来自 analysis sidecar 将 taxonomy 参数顺序写为 `(ATA, AA)`。正式 Taxonomy60 的冻结约定为 `(AA, ATA)`。

在不改变任何 episode JSON、JSBSim state、action、policy、reward 或 scenario 的前提下，`THESIS-DISADVANTAGE-PRESSURE-EXPOSED-POSTMERGE-TAXONOMY-CORRECTION-R1` 对已哈希 phase trace 重标注并重算 gate。三名对手均通过：每名对手 `16/16` qualifying windows、8 个 distinct signature、正负镜像齐全。

因此，**压力可判定性 gate 已建立**，可以继续准备共同 cohort 的 opponent capability card 和紧凑消融；但不能把 correction receipt 写成原 V1 gate 未经纠正即通过。

## 同 cohort Capability Card

| 对手 | 40 步窗口 target attack exposure | 平均 AA | 平均 ATA | 终端混合 |
| --- | ---: | ---: | ---: | --- |
| expert | 0.902 | 148.5 deg | 31.0 deg | 16 timeout HP disadvantage |
| end-to-end | 1.000 | 149.8 deg | 24.2 deg | 10 target crash/OOB, 6 timeout HP disadvantage |
| independent PPO/VPP | 0.883 | 148.1 deg | 32.7 deg | 15 timeout HP disadvantage, 1 timeout HP advantage |

这些是共同 fixed-head-on 参考控制器下的**压力/接口画像**，不是 Elo、绝对对手强度或现实飞行安全结论。特别是 end-to-end 的 target crash/OOB 只能说明该冻结测试条件下的终端事件，不能被解释为我方气动优势。

生成文件：

- `E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_postmerge_collector_v1/pressure_gate_taxonomy_correction_r1.json`
- `E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_postmerge_collector_v1/capability_cards_taxonomy_corrected_r1/opponent_capability_cards.md`

## 仍未完成的论文缺口

V1 phase trace 保存了正确的几何角、压力、VPP action 和 `n_z` response，但 `range/speed/altitude` 来自归一化 observation，并未保留可直接发表的 raw physical telemetry。因此当前 capability card 只能解决“是否向共同基线施压”的透明度，不能完整回答“压力强度在物理量上如何不同”。

在启动 M0--M4 前，唯一合理的工程步骤是实现并运行非学习 **telemetry-completion preflight**：使用同一 manifest、相同三对手与冻结 reference，逐 reset 记录 raw JSBSim own/target state、physical range/range-rate、speed、altitude、specific energy、AA/ATA、VPP、guidance、AoA、`n_z` command/actual 和 terminal。preflight 必须同时验证 `(AA, ATA)` taxonomy 和 scenario application receipt。通过后才设计新的紧凑 M0--M4 一次性评估 Source；仍不训练。
