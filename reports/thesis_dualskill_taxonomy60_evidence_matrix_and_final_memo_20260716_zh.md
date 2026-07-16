# Taxonomy60 论文证据矩阵与最终 Go/No-Go 备忘录

**正式 Source：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`  
**决定合同：** `THESIS-DUALSKILL-TAXONOMY60-NO-GO-V3`  
**决定：** `GO_FOR_THESIS_WRITING_WITH_BOUNDED_CLAIMS`；`NO_GO_FOR_FURTHER_TRAINING_OR_PATCHES`。

## 证据矩阵

| 论文缺口 / 结论 | 权威证据 | 当前结果 | 可以写入论文的边界 |
| --- | --- | --- | --- |
| 场景覆盖 | Taxonomy60 manifest、formal verification | 60 个冻结且与历史 source 不重合的场景；5 个初始态势各 12 个 | 覆盖一个明确的五态势评估包线，不是完整飞行包线或所有空战态势 |
| 工程有效性 | formal execution manifest、formal verification | 3 个对手 x 5 方法 x 60 场景 = 900/900 strict-JSBSim records，verdict `pass` | 证明本次评估链完整，不证明策略对现实飞行器安全 |
| 对手透明度 | opponent capability cards v3 | expert 为规则对手；end-to-end 与 independent PPO/VPP 为冻结 checkpoint；后者为 provenance-limited | 报告共同参考控制器下的压力画像，不产生 Elo 或绝对强度排名 |
| 双技能和路由消融 | M0--M4 report、paired deltas、mode/guard table | 两个 specialist、static gate、canonical PPO、no-bootstrap PPO 均被拆开；68 条 PPO record 受 guard 混杂 | 把 M3/M4 写作带 deterministic recovery override 的部署接口；不可写成纯 PPO routing 因果优势 |
| 动态几何与阶段 | 900 条 raw phase index、phase summary | 初始 taxonomy 900/900 对齐；185/900 达到 post-merge、155/900 出现 re-entry | phase 是描述性 telemetry，不是新增训练任务；disadvantage 的失败不能直接归因于缺少第三技能 |
| 结果边界 | heatmap、paired forest、代表轨迹 | advantage/crossing-entry 在两个主对手均为满 resolved wins；disadvantage 在 expert 0/12、end-to-end 1/12 | 系统表现是条件化的，不能宣称所有态势或所有对手均占优 |
| 终端/安全语义 | terminal/safety table | 本 formal source 的 terminal 均为 HP timeout 或 draw，没有 crash/OOB | win rate 不是 kill rate、气动优势或安全认证 |

## 最终路线判定

### 论文写作：Go

可以将当前工作组织为“**task-aware dual-skill hierarchical VPP interface 的多态势能力图谱与边界审计**”。核心贡献不是宣称通用五态势智能体已经训练成功，而是用冻结、可复核的场景、对手和模块矩阵，明确分开：

- 已保持的 advantage 与 crossing-entry 条件能力；
- head-on/neutral 的 opponent-dependent 差异；
- disadvantage 的跨主对手失败边界；
- deterministic recovery override 对 PPO 因果解释的限制；
- 三个对手的来源、接口和共同参考压力画像。

### 继续算法研发：No-Go

本 Source 的有效混合/负结果已按 No-Go v3 预注册路由进入 `paper_closure`。禁止的动作是：训练、reward/VPP/guidance/PID 调参、替换场景/seed/checkpoint、关闭 guard 后结果导向重跑，以及将第三对手用于满足主对手重复性。

当前也不满足另起“新接口” memo 的条件：双技能共同失败尚未在两个主对手、双镜像、两高度条件下完成排除 guard、共同执行链与早期对手轨迹差异的机制证明。因此不能把 disadvantage 直接升级为“第三 specialist 已被证明必要”。

## 可复核交付物

- [formal receipt](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/formal_verification.json)：900/900、`pass`。
- [No-Go route decision](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/taxonomy60_route_decision.json)：唯一正式出口为 `paper_closure`。
- [read-only analysis v3](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/analysis_read_only_v3)：分析表、900 条 phase index 和 raw joint SHA。
- [paper materials v2](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/paper_closure_materials_v2)：三张 capability card、M0--M4 report、heatmap、forest plot、代表轨迹。
- [portable slim evidence bundle](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/paper_evidence_bundle_v3)：40 个哈希验证文件；不复制 3.4+ GiB raw archive，但保留全部 raw SHA 收据。

## 论文写作顺序

1. Methods：形式化五态势 taxonomy、M0--M4 组件定义、prediction 的前 9 个 decision step warm-up，以及 recovery guard 的 requested/effective mode 边界。
2. Results：先给出 5 x 3 x 5 heatmap 与 state-specific Wilson 表，再给出对手卡和 pairwise delta；绝不合并 opponent。
3. Mechanism/Discussion：用 dynamic phase table、guard table 和代表轨迹解释能力边界；将 disadvantage 定位为现有双技能接口的失败区域，而非未经证明的新技能需求。
4. Supplement：附上 900 条索引、终端语义、phase 机制表、完整 paired delta、原始 SHA 表和 slim bundle manifest。
