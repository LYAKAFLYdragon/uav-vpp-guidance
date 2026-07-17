# 时域参考轨迹可达性 V1 执行 Checklist

**拟定 Source ID：** `THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1`  
**性质：** 新接口假设下的非学习可控性研究；默认 `formal-small`、`paper_safe=false`。  
**唯一目标：** 在当前单点、4 s、有界 3-D VPP 接口未建立跨对手脱离控制权威后，验证“包线感知的时域参考轨迹编译器”能否在至少两个对手下，安全地产生预注册的联合几何改善。  
**不回答的问题：** 不直接证明最终胜率优势、通用五态势决策、某个新专家必然缺失，或所有对手均可战胜。

---

## 0. 决策总览

| 阶段 | 必须回答的问题 | 通过后才允许 | 失败后的唯一处理 |
|---|---|---|---|
| P0 | 旧 Plan1/Plan2 负证据是否已冻结且可追溯？ | 设计新接口 | 补齐溯源；不运行新实验 |
| P1 | 新接口是否真正改变了 action semantics，而非重新包装单点 pulse？ | 离线包线分析 | 退回接口设计；不训练 |
| P2 | 候选时域轨迹在真实 handoff 状态下是否可被安全编译？ | 工程 smoke | 放弃该轨迹模板；不调 reward/步数 |
| P3 | 三对手的完整运行链和 telemetry 是否有效？ | 一次正式非学习 probe | 修复工程链；不解释性能 |
| P4 | 至少两个对手是否存在安全、方向一致的联合几何控制权威？ | 起草单技能 pilot 预注册 | 停止 temporal-reference/共享技能训练线 |
| P5 | 通过 P4 后，改善是否值得进入训练验证？ | 单一 defensive-extension pilot | 仅报告接口边界，不训练 |

**全程禁止：**

- [ ] 不修改 Taxonomy60、Plan1/Plan2 manifest、旧 checkpoint、旧结果或旧门槛。
- [ ] 不增加训练步数、改 reward、调 PPO、调机械 mode switch、扩大旧单点 pulse 幅度，或事后替换场景。
- [ ] 不使用 target future state、snapshot restore、非法 reset、history padding、backend fallback 或 checkpoint fallback。
- [ ] 不将本 Source 的开发性结果写入 AST 主结果；只有独立、paper-safe 复核后才讨论论文整合。

---

## P0. 冻结基线、问题边界与新 Source

### P0.1 冻结已知负证据

- [x] 记录 clean Git SHA、分支、Python/JSBSim 版本、P3 encoder SHA、三名 opponent checkpoint SHA、预测器 SHA 和 VPP/guidance/PID 配置 SHA。
- [x] 在 provenance 中明确引用 Plan1 V2 与 Plan2 V1 的权威输出，而不复制或改写其数字。
- [x] 声明 Plan2 已排除的解释：动作未进入 VPP/guidance、basis rank 缺失、JSBSim 后端失效。
- [x] 声明尚未排除的解释：更长时域、状态条件化参考、不同 action-envelope 参数化，或对手条件依赖。

**验收：** `source_manifest.json` 能从空目录复核所有输入 SHA；旧结果文件字节哈希不变。

### P0.2 预注册研究边界

- [x] 创建独立 output root：`.../temporal_reference_reachability_v1/`，禁止复用 Plan2 输出目录。
- [x] 生成与 Plan1/Plan2/Taxonomy60 不重合的连续 run-in manifest；三名对手分别固定报告，禁止 pooled gate。
- [x] 将结果标签预先限定为四类：`interface_not_executable`、`physical_envelope_limited`、`opponent_dependent_authority`、`cross_opponent_authority_established`。
- [x] 规定该 Source 仅有一次正式矩阵运行；任何配置、门槛或 manifest 改动都必须新建 Source ID。

**验收：** manifest disjointness test 通过；preregistration、config 和 verifier 中的 Source ID、场景数、方法名完全一致。
**完成记录：** [`source_manifest.json`](temporal_reference_reachability_v1/source_manifest.json) 已在 clean SHA `de1de570732b34868a405d94ddf7d6f66756dbcc` 生成并通过独立 verifier。

---

## P1. 定义真正改变接口语义的时域参考编译器

### P1.1 固定输入、输出与因果边界

新模块必须是“参考生成/编译器”，不是新的 PPO、reward 或隐藏的专家训练。

- [x] **输入仅限：** 当前真实 observation、合法历史、冻结预测器输出、当前 VPP/guidance/PID 状态、固定意图 `defensive_extension`。
- [x] **输出定义为：** 长度为 `K` 的连续 3-D VPP 参考序列；推荐每 `0.2 s` 再编译一次，候选参考时域固定在 `6–10 s`，最终值须在预注册前冻结。
- [x] **禁止输入：** 对手未来真值、episode 终端标签、未来 reward、人工注入的 future state。
- [x] 保留当前 predicted-target VPP 锚点和底层 VPP/guidance/PID；只改变 handoff 之后的参考指令表示。

### P1.2 固定最小候选模板库

只实现以下三类可解释模板，避免组合爆炸：

| 模板 ID | 参考几何意图 | 允许调节的参数 | 不允许改变的内容 |
|---|---|---|---|
| `lateral_extension` | 先建立横向角度与转弯半径裕度，再延长距离 | 左/右符号、侧向目标、固定时域 | VPP/guidance/PID 与对手策略 |
| `vertical_extension` | 用有限高度差换取脱离与能量保护空间 | 上/下符号、垂向目标、固定时域 | 飞行包线与 reset 语义 |
| `combined_break_extension` | 侧向与垂向平滑协同，避免单轴脉冲 | 两轴比例、符号、固定时域 | 额外专家、reward 与未来信息 |

- [x] 每种模板以少量 control knots 或解析曲线定义，并强制 `C1` 连续，禁止阶跃 VPP。
- [x] 参数必须在 config 中显式列出；模板数、knot 数、时域和候选网格在正式运行前冻结。
- [x] 编译器记录 `requested_reference`、`projected_reference`、`projection_residual` 与逐步 action；P3 的 runtime telemetry contract 已预留 VPP、guidance、AoA、`n_z` command/actual、saturation 与预测有效性字段，尚未运行。

**验收：** 对同一状态和随机种子，编译输出逐元素一致；所有输出维度正确、有限，且任何参数均可在 provenance 还原。

---

## P2. 离线包线可执行性筛选（不运行新策略）

**执行状态（2026-07-17）：** `NO-GO, input-contract incompatible`。P2 baseline-only replay 已 fail-closed；历史 12 条连续 handoff 均为 `cartesian_offset`，而 P1 编译器固定要求 `tactical_basis_v1`，故不存在可合法用于离线投影的三轴 basis。详见 [P2 no-go 报告](thesis_temporal_reference_reachability_v1_p2_contract_no_go_20260717_zh.md)。P3 未启动且不允许启动。

### P2.1 选择合法初始状态

- [ ] 仅从已验证连续 run-in 且真实达到 handoff 的 raw episode 中取样；按对手分别保存来源 episode、step、state hash 与 phase/taxonomy 标签。
- [ ] 每个对手至少选取 4 个不同场景 signature，包含双镜像；不能从同一 episode 的相邻步凑数。
- [ ] 只保留 `disadvantage -> post-merge/re-entry` 相关状态；若该 phase 在某对手没有合法样本，记录为 phase-coverage failure，不伪造 reset。

### P2.2 包线投影与筛选

对每个“状态 × 模板”离线计算或使用冻结动力学近似估计：

- [ ] VPP 绝对范围、动作幅度、动作变化率、knot 平滑性与投影残差。
- [ ] 预测 AoA、`n_z`、速度、高度和 specific-energy 的安全裕度。
- [ ] 请求参考与投影参考的几何偏差，尤其是 range-rate、横向逃逸方向与垂向间隔。
- [ ] 每个模板在各对手下的可行状态比例及最常触发限制的 knot/轴。

**P2 go 门：** 至少两名对手中，每个对手至少 4 个合法 handoff 状态存在一个模板，满足：无 action clipping、无硬安全约束违规、投影后仍保留预先定义的最小参考幅度。最小幅度必须从基线噪声和仿真分辨率校准后写入预注册，不能事后按结果降低。

**P2 no-go：**

- [ ] 若候选模板普遍需要大幅投影/裁剪，结论为 `physical_envelope_limited`。
- [ ] 停止该模板与后续训练；可另起 Source 研究新的 action-envelope 参数化，但不得扩大旧 pulse 或直接训练新技能。

**交付物：** `offline_envelope_matrix.csv`、`offline_envelope_summary.json`、逐状态审计索引、独立 verifier。

### P2.3 已确认的研发事实：连续 baseline trajectory 数据契约是新接口的前置条件

**当前判定（2026-07-17）：** 本 Source 在 `p2_input_contract_incompatible` 处封存。P2 的 `NO-GO` 说明历史 continuous-handoff cohort 不能为新接口提供合法输入，**不**说明 temporal-reference 编译器无效、物理不可达、训练不足、reward 不当、对手过强或 JSBSim 后端失效。

新接口的可验证性由两个不可替代的部分共同组成：

1. **新编译器：** P1 的 `tactical_basis_v1` compiler 在每个 handoff 后将时域参考编译为有界的 3-D VPP action sequence。
2. **同一动作语义下的连续 baseline trajectory 数据契约：** 从合法 reset 起，以已冻结、实际启用 `tactical_basis_v1` 的 baseline 连续运行并记录；每个候选 handoff 必须保留可追溯的 basis、frozen reference action、history、prediction 和执行链 telemetry。历史 `cartesian_offset` 记录不得事后转换或重新解释为 tactical basis。

**本 Source 的停止规则：**

- [x] P2 的 12 条历史记录均审计为 `cartesian_offset`，未产生任何合法 `tactical_basis_v1` 输入；P3/P4/P5 因此不得在本 Source 继续。
- [x] 不重跑 P2，不替换或补充 historical cohort，不重解释 `cartesian_offset`，也不改 reward、训练步数、routing、模板幅度或阈值来规避该 no-go。
- [x] P2 no-go 报告、`failure.json` 和 `p2_contract_audit.json` 作为研发性负证据保留；它们不是论文级物理不可达结论。

**唯一允许的后续路线：** 新建独立 Source `THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V1`。其第一目标不是性能改善，而是建立上述连续数据契约；不得复活本 Source 的 P2/P3。

新 collector 必须从合法 reset 起，以当前 `tactical_basis_v1` frozen fixed-head-on baseline 连续采集并保存：

- [x] active/configured action semantics、三个 tactical-basis world vectors、basis rank 与 condition；V3 qualifying steps 均为 rank 3，condition 为 2.67--3.10。
- [x] frozen specialist reference action、prediction validity/fallback、VPP、guidance、AoA、`n_z` command/actual、saturation 已写入 V3 ledger。
- [x] phase、dynamic taxonomy、handoff hash 与 observation/history continuity 已写入；qualifying window 内无 future injection、padding、backend fallback 或 checkpoint fallback。
- [x] 每条 qualifying handoff 均保存完整 40-step baseline reference/basis sequence；共 17 条、680 个 post-merge step。

**新 collector 的 phase + contract gate：**

- [x] V3 qualifying windows 使用 strict JSBSim，action、basis、history 和 telemetry 均有限且可独立读取；预测器 warm-up 仅发生于候选 handoff 之前，window 内为零 fallback/zero padding。
- [x] `end_to_end`（5 条、4 signatures、双镜像）与 `independent_ppo_vpp`（7 条、4 signatures、双镜像）通过；`expert` 有 5 条但仅 3 signatures，按预注册门槛不通过。
- [x] gate 通过，故“不训练”的 stop rule 不触发；但该 gate 只解除新 P2 的**设计资格**，不自动授权训练或 P2 执行。
- [x] 后续只能以新的独立 Source ID 设计新的 post-merge P2 offline envelope screen；不得追溯性地恢复本 Source 的 P2/P3。

**Collector 执行台账（2026-07-17）：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V1` 的 36 条记录均在 `RawSIFirstPassConfig` 初始化时因错误字段名停止，未进入任何 JSBSim step；`V2` 已修复该字段，但在首次 JSBSim reset 时因 JSBSim-ML/PyTorch OpenMP 重复运行时中止，同样未产生任何物理 step。两个 Source 均封存为 engineering-invalid evidence。V3 经不占 Source ID 的三对手 runtime smoke 后完成 36 条采集，并被独立 verifier 复核为 gate pass；这建立的是 post-merge continuous data contract，而非 temporal reference 已改善几何、更非训练收益。预测器在 10-frame history 形成前的 warm-up fallback 仅可记录，不能触发 handoff；每个 qualifying 40-step window 仍必须零 fallback、零 padding。详见下方 V3 结果报告。

---

## P3. 三对手工程 smoke（不判断性能）

每名对手只跑一个完整、合法、连续的代表 episode；此层不占正式结论样本。

- [ ] expert、independent PPO/VPP、end-to-end 都通过合法 reset -> run-in -> handoff -> temporal-reference 接管 -> episode terminal。
- [ ] 预测器 warm-up、anchor switch、history、phase、taxonomy 和 handoff state hash 全部存在且顺序正确。
- [ ] reference request/project/result、3-D VPP、guidance、PID、AoA、`n_z` command/actual、saturation、terminal reason 全部非空、维度正确、数值有限。
- [ ] 无 backend/checkpoint fallback、无异常 reset、无 future-state injection、无 history padding。
- [ ] 独立 verifier 可在不导入 runner 的情况下读取三条记录并验证上述条件。

**P3 no-go：** 任一对手缺 telemetry 或存在连续性违规，停止正式运行，只修复工程链；不得将空 telemetry、短 episode 或 fallback 当作物理结论。

**交付物：** `smoke_records/`、`smoke_verification.json`、字段/维度 contract test。

---

## P4. 一次正式的非学习跨对手可控性 probe

### P4.1 固定实验矩阵

- [ ] 基线：冻结 `fixed_head_on`，与候选模板使用相同 scenario、seed、run-in 和 handoff。
- [ ] 试验方法：P1 冻结的三个 temporal-reference 模板；不加入 PPO routing、Oracle、训练中 checkpoint 或 reward 修改。
- [ ] 三名对手分别报告：`expert`、`independent_ppo_vpp`、`end_to_end`。
- [ ] 每个“对手 × 模板”至少覆盖 12 个不重合场景，双镜像与至少 4 个场景 signature；最终样本数以冻结 manifest 为准。
- [ ] 在 handoff 前后记录相同的固定分析窗口；不得按 terminal 长度事后改变窗口。

### P4.2 预先定义指标

**主要联合几何指标：**

- [ ] 对手攻击区暴露及首次进入时间。
- [ ] range、range-rate、AA、ATA。
- [ ] specific energy、相对高度及垂向分离。

**执行与安全指标：**

- [ ] action clipping、projection residual、VPP 偏置、guidance 输出。
- [ ] AoA、`n_z` command/actual、saturation、fallback 与 reset。
- [ ] ego crash/OOB、target terminal、timeout 与完整 terminal-reason mix。

**统计规则：**

- [ ] 只做同场景配对差值；对手分层报告，禁止合并为一个总胜率或总均值。
- [ ] 最小实用差异（MDE）由 P2 前冻结的基线噪声决定；MDE 与安全容许增量写入 preregistration 和 verifier。
- [ ] “联合改善”至少同时满足：压力指标改善，且 range-rate / 能量 / 高度中的至少一个改善；不能用单个 VPP 变化替代几何改善。

### P4.3 正式 go/no-go 门

**P4 通过：** 同一个模板在至少两名对手中均满足：

- [ ] 完整运行链，无 clipping/fallback/reset/checkpoint 异常。
- [ ] 主要压力指标和至少一个辅助几何指标达到预注册 MDE，方向一致。
- [ ] ego crash/OOB 不高于 paired `fixed_head_on` 超过 `+5 percentage points`。
- [ ] 未出现系统性 AoA、`n_z` 或能量安全恶化。

**P4 失败分类：**

| 观察到的模式 | 唯一结论 | 后续动作 |
|---|---|---|
| 大量投影/裁剪或安全约束频繁触发 | `physical_envelope_limited` | 停止该接口；新 Source 才可研究新的 envelope 参数化 |
| 安全执行但三对手均无联合几何改善 | `behavioral_or_control_authority_not_established` | 停止 temporal-reference/共享技能训练线 |
| 仅一名对手改善 | `opponent_dependent_authority` | 报告条件边界；不训练通用技能 |
| 至少两名对手改善 | `cross_opponent_authority_established` | 只允许起草 P5 的单技能 pilot 预注册 |

**交付物：** 完整 raw records、`paired_geometry_deltas.csv`、`safety_semantics.csv`、`temporal_reference_verification.json`、中文结果与决策报告。

---

## P5. 仅在 P4 通过后：单技能 pilot 的预注册，不自动训练

P5 的目的不是马上证明全面优势，而是检验新接口是否能支持一个明确、可解释的能力学习。

- [ ] 新建 Source ID：`THESIS-DEFENSIVE-EXTENSION-TEMPORAL-REFERENCE-PILOT-V1`；不得沿用 P4 Source ID 或输出目录。
- [ ] 固定唯一技能：`defensive_extension`；固定一个在 P4 通过的模板；冻结 temporal encoder、预测器、VPP/guidance/PID、三对手与 action compiler。
- [ ] 主要 baseline：`fixed_head_on`；辅助 baseline：`fixed_crossing`；不将 Oracle 当作与固定任务重复的主要比较。
- [ ] 重新生成不重合 train/dev/heldout continuous run-in manifests；对手分开报告。
- [ ] 预注册训练预算、checkpoint 选择规则、停止规则和 held-out 一次性评估方式。

**P5 最低结果门：** 至少两名对手下，训练后相对 `fixed_head_on` 不恶化安全性，并在 P4 已证实的几何指标上保留或改善；若失败，只能得出“接口可控但该训练方案未获得额外收益”，不得回调 reward 或补跑。

---

## 完成定义与论文边界

### 本 checklist 的完成定义

- [ ] P0--P4 全部完成，并由独立 verifier 输出唯一的四类决策之一。
- [ ] 所有原始记录、config、manifest、source hash、输出 hash 与结果报告均可移植复核。
- [ ] P4 失败时，形成可引用的负证据闭环；P4 通过时，仅获得 P5 预注册资格，而非自动训练许可。

### 可用于硕士论文的严谨表述

- [ ] 若 P4 失败：在冻结单点 3-D VPP、有限时域与已测对手条件下，尚未建立可安全利用的跨对手脱离几何控制权威；不能将 disadvantage 失败简单归因于 PPO 路由或某个缺失专家。
- [ ] 若 P4 通过：包线感知时域参考接口在限定对手与场景条件下建立了跨对手局部控制权威；仍不得声称全部场景或所有对手均优。
- [ ] 无论结果如何：Plan1/Plan2/P4 均为研发性证据，只有完成独立、paper-safe 的正式复核后才允许进入 AST 主结果。

---

## 执行顺序核对

1. [x] P0：冻结 SHA、输入资产、旧负证据和新 Source。
2. [x] P1：实现并测试时域参考编译器，冻结三模板及参数。
3. [x] P2：已执行为输入契约负证据；历史 cohort 未通过，停止该 Source 的模板筛选。
4. [ ] P3：被 P2 no-go 阻止，未执行；不得绕过门槛。
5. [ ] P4：执行唯一一次非学习正式矩阵，并由独立 verifier 判定。
6. [ ] P5：仅在 P4 通过时起草单技能 pilot 预注册；未通过即停止训练线。

---

## 后继 Source：Post-Merge P2 Offline Envelope Screen V1

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1`。本节是原 Source 被 `p2_input_contract_incompatible` 封存后的独立后继路线，**不**修改原 P2 no-go、P3/P4/P5 的状态，也不允许追溯性恢复原 Source。

- [x] 输入：引用 V3 的 strict-JSBSim 连续同语义窗口；17 条合格记录（expert 5、end-to-end 5、independent PPO/VPP 7），每条含完整 40-step reference/basis sequence。
- [x] 冻结设计：展开 P1 的 8 个 template/sign variants；不重采样、不改模板、动作包线、reward、策略、VPP、guidance 或 PID，且候选 action 不进入环境。
- [x] 预注册门：`end_to_end` 和 `independent_ppo_vpp` 分开报告，每个至少 4 条、4 个 signature、双镜像；box/slew clipping 均为零，平均投影残差不超过 50 m，峰值位移至少 100 m，basis condition 不超过 100。expert 仅诊断，因为其 V3 cohort 只有 3 个 signature。
- [x] 唯一一次离线执行：136/136 state-template rows 完成；所有 row 均为零 box/slew clipping，最大平均残差 `1.95e-14 m`，最小峰值位移 `273.20 m`，最大 basis condition `3.10`。
- [x] Gate：8 个候选在两个 required opponents 都通过。冻结的选择规则选中 `lateral_extension:lat+1:vert+1`；独立 verifier 已重新编译并验证全部 136 行。
- [x] 结果边界：仅确认 post-merge temporal reference 在当前 action envelope 中可以离线无失真编译；不确认运行时几何/胜率/安全改善，不启动训练。

**后继路线的唯一下一步：** 仅允许新建一个独立 Source 来设计 selected-variant paired non-learning runtime probe；必须以固定 `lateral_extension:lat+1:vert+1` 对固定 head-on baseline，另行冻结 manifest、MDE、安全 stop rule、输出根和独立 verifier。详见 [P2 V1 结果与决策](thesis_disadvantage_tactical_basis_postmerge_envelope_p2_v1_result_20260717_zh.md)。

**后继路线设计状态（2026-07-17）：** [Selected Lateral Runtime Probe V1 预注册设计](thesis_disadvantage_tactical_basis_postmerge_lateral_runtime_probe_v1_preregistration_20260717_zh.md) 已完成并通过静态 preflight：12 个 P2 对齐场景、36 个因果配对、72 个计划 episode，候选固定为 `lateral_extension:lat+1:vert+1`，execution 仍为 `false`。该设计不是 runtime 结果，必须另行实现并验证独立 runner/verifier 后才能申请一次性执行。
