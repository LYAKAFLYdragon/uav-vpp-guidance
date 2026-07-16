# Five-State v2：三项证据缺口闭合 Checklist

**日期：** 2026-07-16  
**状态：** design checklist；不构成训练或正式 Source 执行授权  
**适用范围：** 硕士论文的五态势扩展线；不改变 AST 主稿 `CAN-20260705` 的双技能结论。  
**前置阅读：** [V1--V22 尝试账本](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v1_v22_attempt_ledger_20260716_zh.md)。

## 0. 目标与边界

本清单解决三个不同的问题，而不是追求“所有场景、所有对手全部获胜”：

| 论文缺口 | 闭合标准 |
| --- | --- |
| 场景少 | 建立独立五态势 `Heldout60`，覆盖高度、镜像、未见距离/速度，按态势和对手分别报告。 |
| 对手不透明 | 为三个冻结 opponent 建立可复现 capability card：来源、接口、共同参考条件、压力/能量/终端指标与已知未知项。 |
| 消融不完整 | 在同一 observation/action contract 下完成 B0--B4 最小模块矩阵、三 seed、三 opponent、冻结 heldout。 |

**最终可接受表述：** 在三 opponent、冻结 Heldout60 的有限包线内，新系统保持 legacy head-on/crossing 能力，并在预注册的新态势中呈现有边界的实用改善。  
**禁止表述：** 普适空战优越、所有对手获胜、任务标签无关的几何识别、飞行安全认证、Elo/绝对对手强度。

## 1. 全局停止规则

- [ ] 不在旧 V1--V22、Temporal-Corridor V1/V2 或 Local-Authority V1/V2 Source ID 下修改后重跑。
- [ ] 不以增加训练步数、奖励系数、单点 VPP 偏置、短窗口长度或删除失败场景作为“新接口”。
- [ ] 每个新 Source 在运行前冻结 `Source ID`、clean SHA、配置哈希、checkpoint 哈希、manifest 哈希、输出根和磁盘预算。
- [ ] 正式结果按 opponent 分开；不将 expert、end-to-end、independent PPO/VPP 池化为单一优越性统计量。
- [ ] 任何 backend/prediction/checkpoint fallback、非有限 action、异常 reset、未记录 override 或输出不完整，都使该 Source fail-closed。
- [ ] 若 P3 的跨对手控制权威 gate 未通过，停止 Five-State v2 训练线；只完成论文的负证据与边界章节。

## 2. P0：资产、旧证据与输出隔离

### P0.1 建立 v2 clean family

- [ ] 创建新的 clean worktree，例如 `uav-vpp-guidance-five-state-v2`，不得复用含未提交试验改动的目录。
- [ ] 新建输出根 `...-results/five_state_v2/`；按 `source_id/` 隔离，拒绝非空目录。
- [ ] 生成 `v2_asset_manifest.yaml`：JSBSim、aircraft、predictor、VPP、guidance、PID、legacy specialists、canonical PPO、三个 opponent 的 SHA/shape/version。
- [ ] 对旧二技能 baseline 记录 observation/action contract；若 v2 使用新 contract，明确建立 compatibility baseline，禁止静默截断或 padding。
- [ ] 建立容量策略：只保存每个训练 seed 的 top-K checkpoint、完整 config/manifest/hash、正式 raw telemetry；开发期原始 dump 按预注册 retention 清理。

**验收：** asset/shape/preflight/artifact verifier 全部通过；旧 AST canonical 输出不可写；有新 family 的唯一 provenance index。

### P0.2 研究问题与主比较冻结

- [ ] 冻结 RQ-A：新 trajectory-level interface 是否先建立跨对手、可执行的局部几何控制权威？
- [ ] 冻结 RQ-B：在统一 contract 下，扩展技能库是否比双技能 baseline 改善新态势而不损害 legacy 态势？
- [ ] 冻结 RQ-C：routing、intent profile 与 temporal encoder 哪些模块提供独立增益？
- [ ] 为每个 RQ 分配独立 Source ID 前缀、输出路径和 artifact index；禁止跨 RQ 合并胜率或 p 值。

## 3. P1：对手透明度闭合（Capability Card v3）

### P1.1 对手 registry

- [ ] 对 `expert`、`end_to_end`、`independent_ppo_vpp` 各建一张不可变 registry card。
- [ ] 每张卡必须包含：实现类/代码 SHA、checkpoint SHA、观测维度和 feature names、动作维度、决策周期、是否 rule-based、训练 provenance、设备/随机性、可恢复与不可恢复字段。
- [ ] 对不可恢复的训练预算、训练 seed 或内部奖励，显式标记 `provenance_limited`，不得补写推测。
- [ ] 固定对手只在 evaluation 中使用；不为获得更好结果而调 opponent、重训 opponent 或替换 checkpoint。

### P1.2 共同参考 calibration

- [ ] 新建与 Taxonomy30、Heldout40、V1--V22 全部不重合的 calibration manifest60。
- [ ] 使用至少三个冻结 ego reference：fixed head-on、fixed crossing、legacy static task gate；每个 opponent 在相同 60 场景下独立运行。
- [ ] 每个 `opponent × reference` 输出：W/L/D/未判定、Wilson CI、terminal reason、己方/目标 crash-OOB、HP advantage、首次攻击区、攻击区时间、range/range-rate、速度、高度、specific energy、post-merge/re-entry 占比。
- [ ] 以 scenario-clustered bootstrap 给出描述性 CI；不输出 Elo、总强度排行或跨 reference 传递性结论。
- [ ] 完成每对手的行为叙事：它如何产生压力、在哪些 phase 更具攻击性、哪些指标只能在当前 ego reference 下解释。

**验收：** 三张卡完整、输入哈希可复核、共同场景完整、终端语义统一；读者可理解 opponent 条件但不会被诱导为绝对强度排名。

## 4. P2：新动作接口合同（先于任何训练）

当前 raw single-point VPP、短 corridor 和小 pulse 已缺少跨对手实用控制权威。v2 必须提出真正改变 action envelope 的接口，而不是调旧参数。

### P2.1 接口设计冻结

- [ ] 写出新的 action semantics，例如受飞行包线约束的短时参考轨迹参数：相对 range opening、signed-turn、vertical-energy、时间尺度/曲率限制。
- [ ] 明确 compiler 的输入、输出与坐标系；禁止 future state、reset 后 state 注入、隐式 task label 或未记录的 profile 切换。
- [ ] compiler 必须输出 `requested_action`、`projected_action`、逐轴 clipping、投影残差、轨迹 knot、VPP 三轴、guidance receipt、AoA、`n_z` command/actual。
- [ ] 定义 physical envelope：最大位移、最大 slew、最小时间尺度、允许的投影残差；这些上限在运行前冻结。
- [ ] observation contract 变动必须新增 schema version、feature names、shape tests、checkpoint compatibility report 和 `AGENTS.md` 更新。

### P2.2 工程 smoke（不占科学 Source ID）

- [ ] 单场景、三对手各完整运行一回合，真实构造 strict-JSBSim env。
- [ ] 验证 telemetry 非空：raw-SI phase、first-pass/re-entry 顺序、10-frame history、predictor receipt、handoff hash、requested/applied action、VPP/guidance、AoA、`n_z` command/actual、terminal。
- [ ] 验证 observer/warm-up 与 anchor switch 可读；warm-up prediction 不得进入 VPP/guidance/PID/reward/policy observation。
- [ ] 独立 verifier 必须能够读取 smoke output；任一字段缺失即停止，不创建正式 Source。

**验收：** `3/3` smoke pass；无 fallback、非有限值、隐式 override 或收集器默认值。

## 5. P3：非学习的跨对手局部控制权威 Gate

### P3.1 预注册 probe

- [ ] 新建物理不相交的 `InterfaceProbe24`：每个 opponent 各 24 条连续 run-in；场景覆盖至少两镜像、两高度与两个 phase signature。
- [ ] 每个合法 handoff 以同 hash 的 reference 为对照，比较最少三个对称 interface action（例如 neutral、left/extension、right/extension）。
- [ ] 所有对手分别记录六维响应：AA、ATA、range、range-rate、specific energy、altitude；同时记录 attack-zone、VPP/guidance、clipping、AoA、`n_z`、terminal。
- [ ] 使用预注册的短局部窗口和符号一致性规则；禁止根据结果选择 action axis、窗口长度或 handoff cohort。

### P3.2 通过门与 stop rule

- [ ] 每个被比较的 opponent 至少 `6` 个 replay-hash 一致的 qualifying pair，覆盖两镜像和两种场景 signature。
- [ ] candidate 在至少两个 opponent 中均满足：无 backend/prediction/checkpoint fallback；ego crash/OOB delta `<= +0.05`；无未解释 saturation；无裁剪 step 比例 `>= 0.90`。
- [ ] 至少一个 action direction 在至少两个 opponent 上呈现预注册的同方向实用响应：至少两个六维指标改善，且没有高优先级指标显著恶化。
- [ ] 实用幅度必须在预注册前由测量噪声和飞行任务尺度固定；不得使用结果后阈值。推荐同时报告原始单位与 normalized profile delta。
- [ ] 未通过：冻结为 `valid_negative_mechanism`，不训练四技能、不启动高层 PPO、不执行模块消融；论文收束为接口可达性与控制权威边界研究。

**验收：** 只有全部 P3 条件通过，才允许进入 P4。P3 不是胜率实验，也不以击败对手为目标。

## 6. P4：共享技能库的分级验证

### P4.1 单技能可行性 pilot

- [ ] 先只训练一个与 P3 正向动作方向对应的技能；不得一次启动四技能。
- [ ] 冻结 profile、P3 encoder/explicit stats、predictor、VPP、guidance、PID 与三个 opponent registry；关闭高层 routing。
- [ ] 使用与 P3、Taxonomy30、Calibration60 不重合的 train/dev/heldout continuous manifests。
- [ ] geometry pretrain 后进行三 opponent 平衡 combat finetune；每阶段记录 observation/action contract、coverage、intent loss、phase occupancy、clipping、terminal 与 safety。
- [ ] 开发集只用于 checkpoint 选择；未过预注册 dev gate 不执行 heldout。

**单技能验收：** 三对手各自有完整 phase 覆盖；相对固定 head-on 不增加 ego crash/OOB 超过 5pp；至少两个 opponent 出现预注册机制改善；无 fallback 或 checkpoint shape adaptation。

### P4.2 四技能库

- [ ] 仅当单技能通过后，再分别训练 `lead_intercept`、`pursuit_conversion`、`defensive_extension`、`reentry_recovery`。
- [ ] 每个技能有独立 geometry-state/phase 职责、profile validity、训练数据 lineage、checkpoint SHA 和 readiness report。
- [ ] 每个技能必须完成统一 observation -> 3-D/action interface 的 shape/preflight；禁止 checkpoint fallback。
- [ ] 每个技能独立检查：目标 phase coverage、intent progress、action clipping、JSBSim crash/OOB、对手分开覆盖。
- [ ] 任一技能失败不得由其它技能的正结果掩盖；可以降级为“三技能 system”，但必须以新 Source ID 和新主问题重预注册。

**验收：** `4/4` 技能通过各自 gate，或研究目标明确、预注册地降级；不得把“已训练”写成“技能有意义”。

## 7. P5：最小而完整的模块消融

所有方法使用相同环境、action interface、predictor、VPP/guidance/PID、训练预算、opponent 采样、dev selection 规则和三 seed。legacy canonical 双技能结果保留为外部复现基线，但不与新 contract 静默混比。

| 方法 | 固定内容 | 唯一新增模块 | 回答的问题 |
| --- | --- | --- | --- |
| B0 | 统一 contract 下的双技能、固定 task routing、固定 intent | 无 | 新 contract 的最低基线是什么？ |
| B1 | B0 + 四共享技能库 | skill library | 技能容量本身是否有价值？ |
| B2 | B1 + PPO skill routing | learned routing | 高层选择是否优于固定路由？ |
| B3 | B2 + global intent profiles | intent compiler | 技能与几何意图解耦是否有价值？ |
| B4 | B3 + frozen temporal encoder | full system | 时序表征是否贡献独立增益？ |
| B5 | B4 去除非安全 validity mask | diagnostic-only | mask 是安全约束还是隐藏路由？ |

- [ ] B0--B4 各训练 `3` 个独立 seed；B5 仅限预定义安全 sandbox/dev，不进入 formal heldout。
- [ ] 每个 seed 在独立 dev30 上按相同预注册规则选择 checkpoint；禁止按 heldout 结果选择。
- [ ] 每个方法记录 proposed/effective skill/profile、mask reason、switch、phase dwell、profile entropy、VPP、clipping 与 terminal。
- [ ] 逐模块做 paired delta：B1-B0、B2-B1、B3-B2、B4-B3；不使用“完整系统最好”替代模块因果解释。
- [ ] 每个模块若只在单一 opponent 或单 seed 改善，写为条件性发现，不宣称通用增益。

**验收：** 每个模块均有 L1（被正确调用）、L2（中间机制变化）与 L3（预注册结果改善）三层证据；只有 L3 才能写入性能贡献。

## 8. P6：多场景 Heldout60 正式验证

### P6.1 场景与冻结

- [ ] 生成 60 个与 train/dev/calibration/Taxonomy30/V1--V22 不重合的 full-signature JSBSim 场景。
- [ ] 基础层：`5 初始态势 × 3 高度 × 2 镜像 = 30`。
- [ ] 扩展层：另一个 `5 × 3 × 2 = 30`，使用未出现的距离、own/target speed、closure 组合。
- [ ] 所有场景自动重算 taxonomy、LOS elevation、能量/高度标签、镜像关系与 initial-state validity；禁止以 scenario ID 作为策略输入。
- [ ] 对每个 `state × height × mirror` 保留 paired scenario key，所有方法/seed/opponent 共享初始物理状态。

### P6.2 正式矩阵与统计

- [ ] B0--B4 × 3 opponents × 3 system seeds × 60 scenes，最大 `2,700` 条正式 episode；若某方法不适用，必须在预注册中说明而不是事后删除。
- [ ] 每个 cell 报告 `N_total/N_resolved`、win/loss/draw、Wilson 95% CI、terminal reason、ego/target crash-OOB。
- [ ] 同场景同 seed 计算 paired outcome delta；先在 seed 内汇总，再报告跨 seed 均值和离散度；不把不同 opponent 池化。
- [ ] 同时报告机制指标：攻击区、range/range-rate、AA/ATA、能量、高度、phase、VPP、AoA、`n_z` command/actual、clipping/fallback。
- [ ] 保存 raw telemetry index、artifact contract、manifest/config/checkpoint SHA、独立 verifier 与 submission-safe summary。

### P6.3 主验收门

- [ ] **Legacy preservation：** B4 相对 B0 和 canonical 双技能，在 head-on 与 crossing 各 opponent 分开满足 paired resolved-win-rate delta `>= -0.05`；若不通过，不宣称新系统替代旧系统。
- [ ] **新态势价值：** Advantage/Disadvantage/Neutral 至少两个态势在至少两个 opponent 中显示预注册实用改善，并在三 seed 的方向上不矛盾。
- [ ] **安全边界：** B4 ego crash/OOB 不得高于 B0 超过 5pp；任何安全失败均逐 terminal 与 command/actual response 审计。
- [ ] **统计边界：** 不以 CI 重叠与否替代效应量；不把 practical non-inferiority 写成 superiority；所有未通过 gate 原样报告。

## 9. P7：论文证据整合与写作闭合

- [ ] 建立证据矩阵：每个 RQ、Source ID、场景族、对手、方法、seed、主指标、声明边界、不可证明内容。
- [ ] 主文只保留一个 headline formal table；完整 B0--B5、capability cards、terminal tables、轨迹与分析脚本放入补充材料。
- [ ] 方法正式化：`state/history -> temporal representation -> PPO (skill, profile) -> frozen skill -> trajectory compiler/VPP -> guidance -> PID`；明确 deterministic guard 与 validity mask 的角色。
- [ ] 图表至少包括：五态势/飞行包线、opponent capability cards、模块 forest plot、state×opponent×method heatmap、终端/安全语义表、代表性连续轨迹。
- [ ] 在 Discussion 中报告哪些模块未通过，尤其不能将 V1--V22 的工程无效记录误写为方法失败。
- [ ] 进行 paper-code sync：数字、SHA、决策周期、observation/action dim、训练步数、manifest、输出路径和 claim 在主文/补充/README/状态文件中一致。

## 10. 最终决策表

| 关键结果 | 允许的论文定位 | 不允许的论文定位 |
| --- | --- | --- |
| P3 未通过 | 双技能系统边界、接口可达性/控制权威负证据 | 五态势共享技能系统已验证 |
| P3 通过但 P4/P5 失败 | 新接口可控，但技能库或高层学习未获证实 | “训练不足，再训练即可解决” |
| P5 通过但 P6 未通过 | 开发集机制改善，formal 泛化未建立 | 跨场景/跨对手有效 |
| P6 仅单 opponent 通过 | opponent-dependent 五态势结果 | 通用空战决策系统 |
| P6 全部主门通过 | 有限 Heldout60 内的多态势实用改进 | 所有场景、所有对手绝对优势 |

## 11. 当前起点

- [x] Taxonomy30 已解决“没有多态势诊断”的问题，但不是 Five-State v2 的 formal performance evidence。
- [x] Calibration v2 已提供三 opponent 的共同参考压力画像；P1 的任务是把它升级为完全 provenance-aware 的 v3 card。
- [x] V1--V22 已排除多种旧接口与工程误判；它们是 P2/P3 的设计输入，不是训练授权。
- [ ] P0--P7 尚未启动；当前最先允许的工作仅为 P0/P1/P2 的 design、contract、smoke 与非学习 preflight。
