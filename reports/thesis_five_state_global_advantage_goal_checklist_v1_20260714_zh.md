# 五态势全场景实用优势 Goal Checklist

**Goal 状态：** Active
**日期：** 2026-07-14
**研发身份：** `noncanonical_thesis_extension`
**目标：** 在预注册、可枚举的五态势 x 高度 x 镜像 x 未见飞行包线 x 三对手 JSBSim 集合内，建立安全、可复现且具有稳定实用优势的五态势共享意图分层飞控方法。

## 0. 先冻结“优势”的可验证含义

“所有可能空战状态均优于所有方法”不可穷尽也不可证伪。本 goal 的正式对象是一个在开始前冻结的 `GLOBAL-ADVANTAGE-V1` evaluation envelope：

| 维度 | 冻结内容 | 数量 |
|---|---|---:|
| 初始态势 | `advantage`、`head_on`、`disadvantage`、`neutral`、`crossing_entry` | 5 |
| 高度条件 | `own_below`、`co_altitude`、`own_above` | 3 |
| 镜像 | `negative`、`positive` | 2 |
| 未见 range-speed 包线 | 两个未出现在训练和 dev 的合法组合 | 2 |
| evaluation seed | 每个几何 cell 四个独立 seed | 4 |
| 对手 | expert、end-to-end、independent PPO/VPP | 3 |

正式 held-out 共 `5 x 3 x 2 x 2 x 4 = 240` episode / method / opponent / policy seed。三组对手永远分开报告，禁止 pooled superiority claim。

### 0.1 两层成功门

| 层级 | 通过条件 | 允许的表述 |
|---|---|---|
| 全 cell 安全非劣 | 每个 `state x height x mirror x envelope x opponent` 聚合 cell 对冻结双技能 baseline 的胜率差 `>= -0.05`，ego crash/OOB 差 `<= +0.05`，且无 contract failure | “在冻结包线内未出现实用退化” |
| 全态势实用优势 | 五个初始态势中每一个均在至少 `2/3` opponents 的 state-level 配对结果上达到胜率差 `>= +0.05`；第三个 opponent 不低于 `-0.05` | “在该包线内跨五态势表现出稳定实用优势” |
| 统计可信度 | 每个 state x opponent 报告 `N_total/N_resolved`、terminal mix、Wilson CI、scenario-clustered paired-bootstrap CI；CI 不支持时降级为实用证据，不称显著优越 | “实用优势，不等同于普适或统计显著优势” |

任何单一 micro-cell 的小样本反例都不得被掩盖；它会进入 failure atlas。该设计追求的是“所有 cell 安全非劣 + 所有态势总体实用占优”，不是不可信的逐 episode 必胜承诺。

## 1. P0：主线隔离与不可变边界

- [ ] 从当前 clean SHA 建立新的 goal worktree 和全新 output root；不得覆盖 `CAN-20260705`、Taxonomy30、P4 或 defensive-extension pilot 输出。
- [ ] 建立 `GLOBAL-ADVANTAGE-V1` source ID、asset manifest、resolved config、容量预算和 retention policy。
- [ ] 固定 canonical 双技能 PPO 为主要 baseline；head-on、crossing、static oracle 仅作为机制参考，不与主要 baseline 混合统计。
- [ ] 冻结三 opponent 的 checkpoint、config、feature contract 和 capability card；缺任一 opponent 即阻塞正式训练。
- [ ] 保持预测器、predicted-target VPP 接口、3-D VPP action head、guidance law、PID、JSBSim、终端规则和 AoA60 限制不变。
- [ ] 禁止 prediction reward、事后修改 taxonomy、用 held-out 选 checkpoint、隐式 task bit、checkpoint fallback 或 backend fallback。

**P0 验收：** 所有输入 SHA/shape 通过，output root 为空，磁盘安全余量 `>= 120 GB`，并可一条命令验证 manifest。

## 2. P1：先修复评价可识别性

这是当前唯一允许进入实现的阶段。此前 pilot 显示：同一 scenario/seed 在方法尚未分叉的 run-in 阶段就可能产生不同 handoff metadata，且 held-out 未保存逐步控制 telemetry。

### 2.1 Run-in deterministic-equivalence preflight

- [ ] 新建仅 dev 的 `GLOBAL-ADVANTAGE-V1-RUNIN-PREFLIGHT`，使用与正式 held-out 不重合的 `5 x 3 x 2 = 30` 场景。
- [ ] 对每个 opponent/scenario/seed，独立运行三次完全相同的 run-in；不加载 candidate、不开启 post-handoff policy。
- [ ] 每次写入 reset state hash、每步 quantized physical-state hash、10-frame history hash、predictor state hash、run-in action hash、handoff state hash 和 terminal reason。
- [ ] 对三种未来 method order permutation 重复 preflight，排查跨 episode state leakage。
- [ ] hash 比较采用冻结序列化和明确量化精度；禁止只比较 handoff step 或 terminal label。
- [ ] 一旦任一 cell 的 reset/handoff hash 不一致，输出可复现故障包并停止，不得启动训练或性能评估。

**P1 gate：** 所有 30 场景 x 3 opponents x 3 repeats 均达到 reset、run-in、handoff 一致；任何 backend/checkpoint/prediction/history fallback 均为零。

### 2.2 Telemetry persistence contract

- [ ] 每个 step 持久化：16-D base state、10-frame history hash、temporal embedding hash、dynamic taxonomy、phase、proposed/effective `(skill, profile)`、mask reason、normalized 3-D action、VPP 三轴偏置、guidance command、`n_z` command/actual、AoA、roll/speed/altitude、PID saturation、prediction valid/fallback、攻击区、HP 和 terminal reason。
- [ ] 每个 episode 存储 handoff snapshot hash 与 raw telemetry path；任何缺字段、padding、NaN/Inf、异常 reset 或 fallback 都要 fail-closed。
- [ ] 训练期只保留抽样 telemetry；dev、ablation 和 held-out 保存完整 telemetry，并写入 artifact manifest 与 SHA-256。
- [ ] 增加 schema test、round-trip reader test、disk-budget test 和 run-to-run hash-equivalence test。

**P1 stop rule：** 不为获得一致性而放松精度、删除 mismatch cell 或改动环境种子；问题先作为 simulator/evaluator reproducibility 缺陷修复并独立复核。

## 3. P2：冻结 GLOBAL-ADVANTAGE-V1 场景与对手压力基线

- [ ] 用连续合法采样器生成 train distribution；train 不得含 dev30 或 heldout240 signature。
- [ ] 固定 dev30 用于 checkpoint 选择：五态势 x 高度 x 镜像各一条，不得用于正式声明。
- [ ] 固定 heldout240：每个几何 cell 两个未见 range-speed package x 四个 evaluation seed；完成 disjointness、镜像、物理可达性和 taxonomy unit tests。
- [ ] 为每个 opponent 建立 capability card：共同 frozen reference policy 下的 target speed/altitude/specific energy、攻击区暴露、first-pass、post-merge/re-entry、terminal mix 与 crash/OOB。
- [ ] 输出 `state x opponent x phase` coverage matrix；任何持久 phase 缺失均阻塞相关技能训练，不能靠增加 episode 数掩盖。
- [ ] 在训练前冻结所有 advantage 判定、paired key、resolved-episode 语义、CI 方法、safety 门和 stop rule。

**P2 验收：** 三 opponent 均有完整 capability card；所有 60 几何 cell 物理可达、metadata 完整、训练/dev/held-out 不相交。

## 4. P3：先证明单个新增机制值得学习

根据 P1 telemetry 的唯一归因选择分支；不得同时开启四技能。

| P1 归因 | 允许的下一步 | 禁止的动作 |
|---|---|---|
| VPP action 导致 command/response 裕度破坏 | 先建立可解释 VPP feasibility projection，并做无学习 step-response/trajectory 测试 | 直接加 reward、加步数或训练四技能 |
| guidance/PID 无法跟随可行 VPP | 修复执行接口并以固定 action replay 验证 | 将问题归咎于 PPO 或 profile |
| 现有 crossing/head-on 已覆盖目标几何 | 设计冻结 phase-aware composition baseline | 声称“缺新技能” |
| 两种既有技能均无法建立目标几何且执行链安全 | 起草一个新 Source ID 的单技能 pilot 预注册 | 自动解锁完整四技能训练 |

- [ ] 每个单技能 pilot 使用新的 train/dev/held-out、固定 profile、固定 P3 encoder、三 opponent 分开门和一次性 held-out。
- [ ] 开始前定义最小安全门、primary intent metric、secondary combat metric、paired handoff hash 条件和停止规则。
- [ ] 仅当至少两个 opponents 同时通过实用改善与安全门，才允许将该技能加入共享库候选。

## 5. P4：四共享技能库与七 profile 的阶段化训练

仅在 P3 已给出至少一个安全、有效的新增机制后启动。

- [ ] 冻结 66-D observation、P3 encoder、四技能 registry、七 profile compiler、validity mask 和三 opponent registry。
- [ ] 每个技能先在其职责态势 x phase 支持域做几何预训练；每个 declared dynamic-state x phase cell 必须有足量真实、连续、无 padding 的样本。
- [ ] 每个技能独立通过：66-D -> 3-D VPP contract、profile-conditioned input、VPP/guidance/PID safety、对手分开覆盖和 intent progress gate。
- [ ] 所有技能通过后才进行三 opponent 平衡 combat finetune；每个 checkpoint 单独冻结 SHA。
- [ ] 任何 skill 未过 gate，冻结该 skill 的负证据并停止完整库训练，不用其他 skill 的成功掩盖。

**P4 验收：** 4/4 skills 分别通过覆盖、可执行性和安全门，且输出中不存在 checkpoint/backend/history/prediction fallback。

## 6. P5：高层五态势共享意图 PPO

- [ ] 仅用已冻结的四技能训练高层 PPO；高层输出 factorized `(skill, profile)`，mask 每步可审计。
- [ ] 训练流程固定为：teacher/oracle imitation warm-start -> balanced multitask PPO -> dynamic-transition finetune。
- [ ] 保留显式短时统计；P3 temporal encoder 默认冻结，仅在已预注册的单独 ablation 中允许小幅微调。
- [ ] 连续 train distribution 三 opponent 平衡采样；dev30 仅用于 checkpoint selection 和 safety stop。
- [ ] 训练三独立 policy seeds；所有 seed 共享相同资产、训练预算和选择规则。
- [ ] 记录 state/phase/profile/skill dwell time、switch reason、mask intervention、VPP/PID telemetry 与 terminal semantics。

**P5 gate：** 三 seed 中至少 `2/3` 通过 dev 安全非劣和五态势覆盖门；否则冻结为负证据，不进入 formal held-out。

## 7. P6：完整消融矩阵

在同一训练分布、同一三 opponent、同一 dev30 checkpoint rule 下完成下列对照；每个变体独立 seed，不跨变体挑 best run。

| 方法 | 回答的问题 |
|---|---|
| 冻结双技能 PPO baseline | 现有主线水平 |
| Full shared-intent | 完整方法 |
| Full minus temporal encoder | 时序表征是否必要 |
| Full minus profile conditioning | intent profile 是否有独立贡献 |
| Fixed-skill / learned-profile | skill selection 的贡献 |
| Learned-skill / fixed-profile | profile selection 的贡献 |
| Best fixed existing specialist | 现有库是否已经足够 |
| 安全 projection（若 P3 触发） | 安全接口是否解释性能变化 |

- [ ] 报告每一 cell 的 paired delta、terminal mix、VPP/PID safety、mode/profile fraction 与 attack-zone/energy mechanism metrics。
- [ ] Oracle 仅作为 privileged upper reference，不与学习方法合并胜率或作可部署基线。
- [ ] 不使用 ablation 结果反向改变 full 方法的 held-out config。

## 8. P7：一次性 GLOBAL-ADVANTAGE-V1 formal held-out

- [ ] 只有 P0--P6 全部通过时才创建独立 clean worktree 和新 output root。
- [ ] 在 heldout240 上对三 policy seeds、三 opponents、主要 baseline 和 full method 执行一次性评估。
- [ ] 每个 episode 先验证 run-in/handoff state hash，再执行 policy；hash 不等价时该 cell 标记 contract failure，不进入 superiority 统计。
- [ ] 输出 per-cell、state-level、opponent-level 和全 envelope 的结果；主文不合并 opponent，补充材料保存完整 telemetry、CI、raw paired outcomes 和 provenance bundle。
- [ ] 预注册 winner rule：先判 safety/contract，再判全 cell 非劣，最后判全态势实用优势；不得只展示 aggregate win rate。
- [ ] 完成后立即冻结结果；无论正负都不得更换场景、seed、训练预算、reward、mask 或重新运行同一 held-out。

## 9. P8：论文与学术价值收口

- [ ] 方法部分明确区分 learned skill/profile、taxonomy validity mask 和任何 deterministic physical safety override。
- [ ] 结果按“五态势 x 三对手 x 关键阶段”展示，不将 dev 或 exploratory 数据写成 formal superiority。
- [ ] 讨论三个层次：可执行性、安全边界、对手依赖；说明未覆盖的真实对抗、武器模型和飞行器模型外推限制。
- [ ] 若 P7 通过：主张“在预注册 JSBSim envelope 内的跨五态势实用优势”。
- [ ] 若 P7 不通过：主张“识别并量化了技能、路由、执行安全或对手压力的边界”，不虚构全场景优势。

## 10. 每周决策板

| 当前证据 | 本周唯一允许推进 | 不允许推进 |
|---|---|---|
| P1 未通过 | run-in reproducibility 与 telemetry contract | 任意性能训练或 held-out |
| P1/P2 通过、P3 未判定 | 单机制 feasibility 预注册 | 四技能并行训练 |
| P3 通过、P4 未通过 | 失败技能的真实 coverage/执行归因 | combat finetune 或 P5 |
| P4 通过、P5 未通过 | 高层三 seed dev 训练 | formal held-out |
| P5/P6 通过 | P7 一次性 formal held-out | 事后调参 |
| P7 完成 | 论文证据整合与复现包 | 重跑以追逐更好数字 |

## 11. 当前状态（2026-07-15）

- [x] 已识别双技能 baseline、三 opponent、五态势 taxonomy、四 skills、七 profiles、P3 encoder 与既有 negative evidence。
- [x] 已完成 defensive-extension pilot 的只读归因；其结论是 candidate-specific terminal safety signal 存在，但冻结 held-out 缺少逐步 telemetry 且 run-in metadata 不一致，不能确定根因。
- [x] P1 R1 已保留为 `implementation_failure_not_experimental_evidence`；其序列化错误不计入任何性能或物理结论。
- [x] P1 R2 已冻结为 `runin_protocol_not_reproducible_do_not_train`：270 个 raw artifacts 均通过 SHA-256 审计，但 90/90 `opponent x scenario` 配对单元均未满足完整 trajectory/boundary 等价性。见 `reports/thesis_global_advantage_p1_r2_audit_20260715_zh.md`、`reports/thesis_global_advantage_p1_r2_matrix_20260715.json` 与对应 CSV。
- [x] 已确认 R2 共同执行链的一项 reset-contract 缺陷：启用的 `CommandPostProcessor` 曾跨 episode 保留 lift-compensation state；修复与边界见 `reports/thesis_global_advantage_p1_r2_reset_leak_diagnosis_20260715_zh.md`。该发现不替代 R3，也不改变 R2 的 NO-GO。
- [x] 已完成 P1 R3 fresh-environment-per-episode 复核的设计预注册：`reports/thesis_global_advantage_p1_r3_fresh_environment_design_20260715_zh.md`。
- [x] 已实现 R3 的默认锁定 config、独立 child runner、完整 reset/runtime/FDM state exporter 与字段敏感性 tests；`execution_permitted=false`，未执行任何 R3 episode。
- [x] 已建立非授权性质的运行前请求清单：`reports/thesis_global_advantage_p1_r3_execution_authorization_request_20260715_zh.md`。
- [x] R3 在首个 reset-runtime snapshot 因 `combat_time_to_kill=NaN` 的未编码语义哨兵终止，`completed_episode_count=0`；已保留为 implementation failure，见 `reports/thesis_global_advantage_p1_r3_implementation_failure_20260715_zh.md`。
- [x] R4 implementation 已冻结于 `5af60fa1a2f7fb0d3722601be393b7008a29b712`；合同测试 `6 passed`，非执行 preflight 已验证 30 场景、三对手、三 repeat 共 270 条计划 episode，且仍为 `execution_permitted=false`。
- [x] R4 已于 2026-07-15 一次性执行，但首个 child 的 raw telemetry 写盘遇到 `numpy.ndarray` serialization `TypeError`，`completed_episode_count=0`；已冻结为 `implementation_failure_not_experimental_evidence`，不得重跑或覆盖。
- [x] R5 JSON-telemetry serialization 修复已实现：raw artifact 写盘前使用 R4 `canonicalize` 递归处理 NumPy arrays；writer regression 已纳入合同测试。
- [x] R5 implementation 已冻结于 `e1e047ef6db2f38cefe009431f29ee8f48c6fa8a`；相关回归为 `156 passed, 2 skipped`，R5 output root 为空。
- [x] R5 已于 2026-07-15 一次性执行并通过：270/270 raw artifact、90/90 cell、每 cell 3 repeat；reset/action/trajectory/boundary/terminal 全等价，无 telemetry 或 fallback 缺失。Gate SHA-256 为 `71d00a4997d277ccc179f3b8252a8078081a39ebe24c3858ee758448581fcc0b`。
- [x] **P1 已通过：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R5-JSON-TELEMETRY-REPRO-V1` 已证明冻结 dev30 x 三对手 run-in 协议可重复；R1--R4 保持独立实现失败或负证据归档，不被覆盖。
- [x] **P2-A manifest freeze：** `THESIS-GLOBAL-ADVANTAGE-V1-HELDOUT240-V1` 已冻结 60 个物理几何 cell x 4 evaluation seed = 240 条独立实例；与 train support、dev30 和旧 heldout60 的实例及物理签名均不相交，payload SHA-256 为 `249234e96e802aa35a72bf2208986a723cba7387109fe1349fd0cff4f0b29526`。
- [x] **P2-B1 capability card：** 从 R5 的共同 frozen run-in specialist、每对手 30 个固定 repeat_0/forward episode 构建了 target speed/altitude/specific energy、攻击区、first-pass、terminal 与 crash/OOB 卡；无 Elo 或总强度排序。
- [x] **P2-B2 input freeze：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1` 使用相同 60 个几何 cell、全新的 per-cell preflight seed；它不消耗 heldout240 的四个 evaluation seed。
- [x] **P2-B2 已一次性执行并关闭：** physical contract 为 `180/180` valid、无 fallback、`44,571` 个有限 step；但所有 180 条 manifest 初始为 `pre_merge` 的记录首个 phase 均为 `post_merge`，三 opponent 的 `pre_merge/re_entry` 都为零，且 raw telemetry 缺少 range/range-rate，故固定为 `physical_pass_phase_observability_no_go`。见 `reports/thesis_global_advantage_p2_physical_preflight_completion_20260715_zh.md`；不解锁训练。
- [x] **P2-B3 已一次性执行并关闭：** 90/90 raw JSBSim scenario-application receipt 通过且 PhaseTracker receipt 可回放，但 tracker 输入来自归一化 policy observation 的 `range_m≈0.760`，而阈值按米解释为 1,000 m；故 90/90 step-0 错标为 `post_merge`。结果固定为 `normalized_phase_input_contract_no_go`，不得重跑或训练。见 `reports/thesis_global_advantage_p2_b3_completion_20260715_zh.md`。
- [x] **P2-B4 已一次性执行并关闭：** `90/90` strict JSBSim、scenario receipt、raw-SI replay、normalized/raw 对照与 step-0 `pre_merge` 全通过；并首次在三个 opponent 下观测到真实 post-merge/re-entry。见 `reports/thesis_global_advantage_p2_b4_completion_20260715_zh.md`。但部分 state×opponent 仍无 re-entry，不解锁训练。
- [x] **P2-B5 设计与实现：** 已建立 `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-R1` 的独立 12 场景 manifest、raw-SI collector、fail-closed runner/preflight 与 tests；设计预检通过。详见 `reports/thesis_global_advantage_p2_b5_phase_feasible_sampler_preregistration_20260715_zh.md`。
- [x] **P2-B5 独立复核：** 已补齐连续 handoff/真实 66-D 路径测试，并修正 dynamic taxonomy 的 `(ATA, AA)` 参数顺序；当前 `14 passed`，不涉及 JSBSim 实验结果。
- [x] **P2-B5 已一次性执行并关闭：** 36/36 strict-JSBSim record 的 scenario receipt、raw-SI replay 和 step-0 phase 均通过，但 expert 为 1/12、independent PPO/VPP 为 0/0，故 `phase_feasible_data_contract_not_established`，`training_unlocked=false`。详见 `reports/thesis_global_advantage_p2_b5_phase_feasible_sampler_completion_20260715_zh.md`。
- [x] **B5 只读 atlas：** `analysis_b5/b5_reachability_atlas_zh.md` 证明 B5 包线没有任何 state x phase 满足三个 opponent 的最小覆盖门槛，禁止从单一 opponent 或 aggregate 反选训练目标。
- [x] **B6 设计预注册：** 已建立 independent opponent-conditional target-geometry reachability atlas 的方法、60 场景范围、逐 opponent candidate gate 与无候选 stop rule。见 `reports/thesis_global_advantage_p2_b6_opponent_conditional_reachability_design_20260715_zh.md`。
- [x] **B6 设计态实现：** manifest builder、profile-free collector、七 profile 66-D finite 检查、candidate gate、runner、read-only analyzer 与 preflight 已实现并在 `execution_permitted=false` 下验证；未运行任何 B6 episode。
- [ ] **当前唯一允许动作：独立复核 B6 的 manifest、profile-free collector、candidate selection rule 和 output capacity，随后才可决定是否一次性授权。** 不得重跑、调参或修补 B5，也不得进入 defensive-extension pilot、四共享技能或 formal held-out。
- [ ] P2--P8 均未授权；尤其不允许以“追求全场景优势”为由绕过 P1/P3 的负证据和 stop rule。
