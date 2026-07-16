# 硕士论文证据闭合 Checklist v2：一次正式验证，禁止版本链式试错

**日期：** 2026-07-16  
**状态：** 推荐执行方案，尚未创建正式 Source ID  
**取代关系：** 本文是 [Five-State v2 长期蓝图](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v2_evidence_closure_checklist_20260716_zh.md) 的执行收缩版。长期蓝图保留为研究备忘；不得按其 P0--P7 自动启动训练。  
**目标：** 用一套无训练、可复核的证据包，闭合当前原双技能层级 VPP 硕士论文的场景、对手和消融三项缺口。

## 1. 重新定义成功

本方案不以“所有场景、所有对手均获胜”为目标。成功是：

1. 原双技能 PPO 在明确的五态势包线中的表现和边界都被透明报告。
2. 三个 opponent 的来源、接口、行为压力和终端结构可复核。
3. 两个 specialist、PPO routing、crossing bootstrap/lock 和 deterministic recovery guard 的作用被分离说明。

即使 PPO 在某些态势失败，只要预注册矩阵完整且解释链闭合，三项论文证据质量问题依然得到解决。

## 2. 反试错硬规则

| 类别 | 数量上限 | 规则 |
| --- | ---: | --- |
| 工程 smoke / unit test | 不限 | 不占 Source ID，不产生性能结论，必须在正式运行前完成。 |
| 论文证据闭合 Source | **1** | 仅 `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`；无训练、无调参、无新 policy。 |
| 可选新接口 Source | **0（默认）** | 只有主 Source 完成并经独立决策备忘录批准，才可开放 **1** 个。 |
| 同一接口 family 后续版本 | **0** | 合法负结果后禁止通过 scale、reward、窗口、profile、场景或 action variant 另起 V2/V3。 |

任何新 Source 都必须在预注册中回答下列五问；任一项不能回答则不得运行：

- [ ] 成功和失败各自支持什么**单一**科学结论？
- [ ] 它与 V10/V11/V13/V15、Temporal-Corridor V2、Local-Authority V2 的差异是否是 action envelope 或任务动力学的实质改变，而非参数变化？
- [ ] 负结果是否可直接进入论文边界，而不是只能导向“再试一版”？
- [ ] 真实 JSBSim smoke、telemetry、phase、hash、verifier 是否已在无 Source ID 层通过？
- [ ] episode、磁盘、时间是否在运行前固定，不依赖结果后扩容？

**失败规则：** `implementation_invalid` 只修 smoke/test，未通过独立审查前不重建科学 Source；`valid_negative_mechanism` 永久停止该 family；`valid_positive_mechanism` 只允许起草新决策备忘录，不自动训练。

## 3. 唯一正式 Source：Taxonomy60 × 3 Opponents

| 项目 | 冻结定义 |
| --- | --- |
| Source ID | `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` |
| 性质 | evaluation-only，严格 JSBSim，无训练、无 reward/VPP/guidance/PID 修改 |
| 目标系统 | 当前冻结 canonical dual-skill hierarchical VPP；不伪装为五技能系统 |
| 对手 | expert、end-to-end、independent PPO/VPP；严格分开报告 |
| 场景 | 60 个全新、不重合的五态势场景 |
| 方法 | M0--M4 五个冻结方法 |
| 规模 | `60 × 3 × 5 = 900` 条 episode |
| 结果后动作 | 仅只读分析、论文写作与 artifact bundle；禁止因结果重训或追加 Source |

### 3.1 Heldout60 场景合同

- [ ] 基础层：`5 初始态势 × 3 高度 × 2 镜像 = 30`。
- [ ] 扩展层：同一结构下采用未见的 range、own speed、target speed/closure 组合，再生成 30 场景。
- [ ] 每类初始态势恰有 12 场景；每种高度恰有 20，双镜像各 30。
- [ ] 从位置、速度、heading、LOS elevation 与能量标签重算 taxonomy；场景名称不输入 policy。
- [ ] 与 CAN、RQ1--RQ4、Taxonomy30、Heldout40、V1--V22、calibration 已有场景完全不相交。
- [ ] manifest builder、disjointness checker、镜像 checker、初始 JSBSim validator 均在 Source 创建前通过。

**场景问题的完成判定：** `60/60` 完整且每类 12 个不重合场景，即闭合“场景少”的证据质量缺口；不把 60 场景称作完整飞行包线。

### 3.2 冻结方法矩阵

| 方法 | 保留/移除组件 | 回答的问题 |
| --- | --- | --- |
| M0 `fixed_head_on` | 仅迎头 specialist；无 routing/Oracle/recovery override | 迎头技能的跨态势边界 |
| M1 `fixed_crossing` | 仅交叉 specialist；已验证 adapter；无 routing/override | 交叉技能的跨态势边界 |
| M2 `static_oracle_gate` | 旧静态 task gate；非五态势最优 Oracle | 静态标签映射的作用与局限 |
| M3 `canonical_ppo` | PPO + canonical bootstrap/lock + deterministic recovery guard | 当前部署系统的实际表现 |
| M4 `ppo_unlocked_crossing` | 与 M3 同 checkpoint；仅关 crossing bootstrap/pre-merge lock；保留 recovery guard | crossing 首拍是否来自 task-aware PPO |

- [ ] 五方法固定相同 JSBSim、predictor、VPP、guidance、PID、terminal semantics、scenario seed 和 horizon。
- [ ] M1 通过 observation shape/feature-name preflight；禁止 fallback、padding、截断。
- [ ] M4 不关闭 recovery guard；guard 的作用由 requested/effective mode 和 guard telemetry 解释，而非用无保护策略冒险比较。
- [ ] 每个 PPO macro-step 记录 requested/effective mode、bootstrap/lock reason、guard reason、mode fraction、switch count。

**消融问题的完成判定：** M0--M4 的配对结果和 mode/guard telemetry 完整后，双技能系统的所有真实模块都有可解释对照。该结论不延伸到未训练的“四技能 + profile + encoder”架构。

### 3.3 对手 Capability Card v3

- [ ] 对三个 opponent 分别冻结实现类、checkpoint/config SHA、observation/action/cadence、rule/neural 属性、训练 provenance 与不可恢复字段。
- [ ] 使用 M0/M1/M2 在相同 Heldout60 上的记录，生成每个 opponent 的共同参考压力画像，无需另开 Source。
- [ ] 卡片分别报告 W/L/D、HP advantage、攻击区进入/占比、speed/altitude/specific energy、post-merge/re-entry、terminal mix、ego/target crash-OOB 与 scenario-clustered CI。
- [ ] 明示这些指标只描述“固定参考控制器 × Heldout60”条件下的交互压力；禁止输出 Elo 或绝对强度排行。

**对手问题的完成判定：** 三个 opponent 都有可复现来源和同条件能力卡，所有未知训练信息显式标记；读者能解释 split-specific 结果。

## 4. 正式运行前唯一允许的工程步骤

- [ ] 运行 `5 methods × 3 opponents × 1 smoke scene = 15` 条 disposable strict-JSBSim episode；不占 Source ID，不写入论文结果。
- [ ] Smoke 验证 observation/action shape、crossing adapter、mode/guard telemetry、terminal semantics、prediction receipt、VPP/guidance/PID telemetry、artifact verifier 可读性。
- [ ] Smoke 失败时仅修工程并重做 smoke；不得创建正式 Source、修改 Heldout60 或比较性能。
- [ ] Smoke 通过后，由独立复核人只检查 manifest/asset/config/output hash、900 record universe 与空输出根；不提出调参建议。

**开始条件：** 15/15 smoke pass、disjointness pass、资产 shape pass、输出根为空、磁盘预算通过、预注册签署完成。

## 5. 统计、解释与停止门

- [ ] 每个 `state × opponent × method` 报告 `N_total/N_resolved`、win/loss/draw、Wilson CI、terminal reason 与 ego/target crash-OOB。
- [ ] 同场景同 seed 计算 M3 相对 M0/M1/M2/M4 的 paired delta；不同 opponent 不池化。
- [ ] 分层报告 phase、攻击区、range/range-rate、AA/ATA、能量、高度、VPP、AoA、`n_z` command/actual、clipping/fallback。
- [ ] `-0.05` 只作为 legacy preservation 的实用门，不改写为统计显著优越。
- [ ] target crash/OOB 是合法胜利，但不是飞控安全认证；aggregate win rate 不是 kill rate。
- [ ] Source 完成即冻结：只允许 verifier、只读分析、图表与论文写作；不重跑子集、不换 checkpoint、不追加场景。

## 6. 可选新算法线：默认不启动

这不是闭合论文三缺口的前提。只有 Taxonomy60 已完成、论文证据矩阵已写完，并由独立决策备忘录确认研究价值后才可讨论。

- [ ] 最多创建一个 `THESIS-FIVESTATE-NEW-INTERFACE-GATE-V1`。
- [ ] 它必须是 action envelope 或任务动力学的实质改变，并逐项说明为何不被既有有效负证据覆盖。
- [ ] 先通过无 Source 的 engineering smoke，再只执行一次非学习 local-authority probe。
- [ ] 通过门：至少两 opponent 出现预注册的同方向、实用六维联合响应，且 clipping/fallback/ego crash-OOB 受控。
- [ ] 未通过：停止 Five-State algorithm line；不启动单技能、四技能、profile、encoder 或 PPO 训练。
- [ ] 即使通过：只起草单技能 pilot 的新决策备忘录，不自动授权训练。

## 7. 最终交付物

- [ ] manifest60、scene table、disjointness proof、asset manifest、preflight receipt。
- [ ] 900 raw episode index、summary CSV、paired matrix、terminal/safety table、dynamic taxonomy/phase table。
- [ ] 三张 opponent capability card v3。
- [ ] M0--M4 ablation report、mode/guard telemetry report、声明边界。
- [ ] 五态势×三 opponent×五方法 heatmap、paired-delta forest plot、代表性轨迹。
- [ ] paper evidence matrix、artifact bundle、independent verifier、最终 go/no-go memo。
