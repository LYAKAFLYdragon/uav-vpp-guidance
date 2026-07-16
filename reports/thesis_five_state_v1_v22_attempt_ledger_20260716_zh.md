# 五态势扩展 V1--V22 尝试账本

**更新日期：** 2026-07-16  
**适用工作树：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1`  
**范围：** V1--V19、V21、V22 的 21 个已授权执行 Source；V20 是工程设计桥接，未执行，故不计入。  
**论文边界：** 本账本均为五态势/再进入扩展研发证据，不能改写 AST 主稿的 `CAN-20260705` 结果或结论。

## 1. 如何使用本账本

这不是“成功清单”。每条尝试均被永久归为下列三类之一，避免以后把工程故障误认为方法失败，或把局部正向机制误写成性能提升：

| 标签 | 含义 | 可用于什么 | 不能用于什么 |
| --- | --- | --- | --- |
| `valid_positive_mechanism` | 完整运行且独立验证，证明某个数据/接口/因果机制成立 | 约束下一步设计、撰写方法边界 | 胜率优越、训练有效性或安全认证 |
| `valid_negative_mechanism` | 完整运行且独立验证，预注册机制门未通过 | 排除该冻结接口或假设 | 宣称所有未来接口均无效 |
| `implementation_invalid` | 在产生有效比较前被合同、记录或集成错误停止 | 记录工程缺口、指导 smoke/test | 任何物理、策略、对手或奖励结论 |

**历史复现原则：** 已执行的 Source ID 均已消耗，绝不能用相同 ID、相同输出根“修复后重跑”。“复现”优先指：

1. 切换到账本指定的运行提交或从输出根的 `run_manifest.json` 读取精确 SHA、命令与解析配置。
2. 对完整运行使用条目所列 verifier 重放 gate；对零记录运行校验 `failure.json`、授权配置及文件 SHA。
3. 若需再次执行同一科学假设，必须新建 Source ID、不重合 manifest、空输出根、预注册和独立复核；新结果不得覆盖历史 Source。

所有条目中的“结果报告”是该轮的**唯一人类可读入口**；其中给出了输出根、哈希、gate 和原始输出索引。`Runner / config` 是未来进行只读代码复现或建立新 Source 的最短入口。

## 2. 21 轮总览

| 轮次 | 核心问题 | 结果分类 | 最重要事实 | 对后续的约束 |
| --- | --- | --- | --- | --- |
| V1 | 单技能 pilot 是否能起跑 | implementation_invalid | manifest 键名不匹配，0 条有效 episode | 先做真实 manifest round-trip |
| V2 | 修正后的 pilot 是否可训练 | implementation_invalid | 72 条基线完成，首条 train 序列化失败 | train/eval identity 必须分离测试 |
| V3 | runtime-only feasibility 是否闭合 | implementation_invalid | 4 条 partial record 后缺训练 reward 键 | record-only 路径不得依赖训练 reward |
| V4 | 三对手 re-entry handoff 是否可达 | valid_positive_mechanism + gate no-go | 144 条完整；发现环境 first-pass 语义失配 | 改用 raw-SI phase，而不是训练 |
| V5 | 五态势连续 66-D 数据是否可观测 | valid_positive_mechanism | 180/180，15 个状态×对手单元完整 | 仅支持 post-first-pass 可观测性 |
| V6 | 两冻结专家能否产生 re-entry | phase-semantics superseded | 72 条完整，但 legacy re-entry 标签单位错误 | 不得再用其“0 re-entry”作物理负证据 |
| V7 | turn-back teacher 能否产生 re-entry | phase-semantics superseded | 72 条完整，同受 legacy 标签错误影响 | 不能据此否定教师或任务动力学 |
| V8 | raw-SI re-entry 数据合同是否成立 | valid_positive_mechanism | 三对手均有连续 qualifying re-entry | 允许设计训练 preflight，不等于允许训练 |
| V9 | 原始 3-D VPP 脉冲是否可记录 | implementation_invalid | `NaN` 序列化失败，0 条持久化 record | 缺失遥测需显式 receipt，不可伪造数值 |
| V10 | raw additive VPP 是否有实用局部权威 | valid_negative_mechanism | 252/252，动作进入链路但无跨对手实用收益 | 停止在 raw bias head 上调奖励/步数 |
| V11 | relative-frame semantic action 能否改善几何 | valid_negative_mechanism | 144/144，VPP/guidance 响应存在但无多目标候选 | 转向逐分量 six-objective 诊断 |
| V12 | six-objective collector 能否起跑 | implementation_invalid | 缺 `execution.runtime_template`，0 条 record | preflight 必须构造真实 runtime env |
| V13 | 单点语义 VPP 是否有六维下降方向 | valid_negative_mechanism | 288/288，无跨两对手安全多目标方向 | 单点接口不应进入训练 |
| V14 | temporal corridor 合同能否组装 | implementation_invalid | 动态 profile 与固定 66-D 编码冲突，0 条 record | 固定编码锚点与动作 profile 分离 |
| V15 | 短 temporal corridor 是否有效 | valid_negative_mechanism | 144/144，无跨两对手实用下降方向 | 不再微调短走廊、窗口或尺度 |
| V16 | 动态参考路径能否通过执行链 | valid_positive_mechanism | 108/108；expert/independent 机制门通过但裁剪高 | 可研究接口，不可宣称性能改善 |
| V17 | 74-D path context 是否有三对手覆盖 | valid_negative_mechanism | 72/72；end-to-end 覆盖仅 4 episode/80 step | 不训练 path-conditioned PPO |
| V18 | phase telemetry 能否覆盖完整 episode | implementation_invalid | 把 warm-up 未就绪错当 fallback，0 条 record | warm-up 与 analysis-ready 必须分窗 |
| V19 | 分窗后 telemetry 是否正确 | implementation_invalid | warm-up 的预期 anchor fallback 被错误禁止，0 条 record | 必须记录 fallback 原因而非一律拒绝 |
| V21 | current-target warm-up 后能否 release | valid_positive_mechanism | 72/72、三对手均 0 release，确认循环依赖 | 预测器观察更新须与控制消费解耦 |
| V22 | observer-only release 修正是否已生效 | implementation_invalid | 72 条默认值输出；`_observe()` hook 未接入 | 不得从 V22 gate 推断任何物理结论 |

## 3. 逐轮可复核记录

### V1：Neutral post-merge pilot 的 manifest identity 失败

- **Source ID / 状态：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V1`；`implementation_invalid`。
- **假设：** 固定的 continuous run-in 后，单一 reentry-recovery pilot 可在训练前完成合法的 manifest 配对和 telemetry 记录。
- **复现入口：** [结果归档](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_neutral_postmerge_reentry_recovery_pilot_v1_implementation_failure_20260715_zh.md)；运行提交 `5dad3e8444d4092a1204b200ec7fbdcf183bec37`；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_neutral_postmerge_reentry_recovery_pilot_v1.py)；[preflight](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/preflight_thesis_neutral_postmerge_reentry_recovery_pilot_v1.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_pilot_v1`；首个 baseline 的 ledger 写入读取了不存在的 `metadata.scenario_signature`，而新 manifest 只提供 `metadata.geometry_cell_id`；`completed_episode_count=0`。
- **意义：** 只证明 metadata contract 未 round-trip；不说明技能、奖励、JSBSim 或对手有任何问题。
- **留下的规则：** 新 manifest 必须被 runner 的真实序列化路径测试，而不只检查字段存在。

### V2：Pilot 训练身份合同失败

- **Source ID / 状态：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2`；`implementation_invalid`。
- **假设：** 修正 evaluation pairing 后，pilot 可从 dev baseline 进入 train sampling。
- **复现入口：** [结果归档](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_neutral_postmerge_reentry_recovery_pilot_v2_implementation_failure_20260715_zh.md)；运行提交 `8ff8aafe83ade58bbce128b413952ac475444343`；[V2 runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_neutral_postmerge_reentry_recovery_pilot_v2.py)；[V2 manifest builder](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/build_thesis_neutral_postmerge_reentry_recovery_pilot_v2_manifests.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_pilot_v2`；已写入 `72` 条 dev baseline record/telemetry，但首条 train episode 因 train sampler 没有 `geometry_cell_id` 或 integer `scenario_seed` 而失败；无 checkpoint、无 PPO update、无 candidate/heldout 结果。
- **意义：** 72 条 baseline 只能用于检查当时 run-in phase support；不能与不存在的 candidate 作比较。
- **留下的规则：** paired dev/heldout 与不配对 train 必须使用不同 identity key，并以首条 train 的完整序列化做 preflight。

### V3：Record-only runtime 对训练 reward 的隐式依赖

- **Source ID / 状态：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V3`；`implementation_invalid`。
- **假设：** 不训练的 runtime-feasibility collector 可完整运行 48 场景×3 对手。
- **复现入口：** [停止记录](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_neutral_postmerge_reentry_recovery_v3_runtime_failure_20260715_zh.md)；运行提交 `48b55bad4d11e9415bdd162f47f2d3d4bce9bc96`；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py)；[preflight](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/preflight_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_v3_runtime_feasibility`；第 5 条 expert 场景出现 `KeyError('geometry_reward')`，仅有 4 条未到 handoff 的 partial record。
- **意义：** 失败发生在复用训练 episode engine 时，不是 re-entry、飞控或对手的负证据。
- **留下的规则：** pure record path 必须证明不读取训练 reward，或者显式把不影响动作的运行字段列入合同。

### V4：raw-SI first-pass 语义失配的有效诊断

- **Source ID / 状态：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V4`；`valid_positive_mechanism`，但 runtime-feasibility gate 未通过。
- **假设：** 三对手下存在足以支撑 handoff 的 neutral/post-merge family。
- **复现入口：** [结果与相位审计](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_neutral_postmerge_reentry_recovery_v4_result_and_phase_semantics_20260715_zh.md)；运行提交 `884af19f136b7f2588421f53b195d5d1a75dcc3f`；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py)；[verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_v4_runtime_feasibility`；`144/144` strict-JSBSim record 与独立重算通过，但四个 family 无一达到跨三对手 qualifying gate。最关键的是 end-to-end 有 `42/48` raw range-rate 翻转、却有 `0/48` 环境 first-pass，说明环境标记阻断 handoff。
- **意义：** 建立了“collector phase semantics mismatch”这一正向因果诊断，而非低层技能失败。
- **留下的规则：** 后续必须以 raw-SI 的“持续闭合后稳定张开”定义 first-pass，而不是沿用环境布尔字段。

### V5：五态势 post-first-pass 66-D 可观测性

- **Source ID / 状态：** `THESIS-FIVE-STATE-RAW-SI-PHASE-OBSERVABILITY-V5`；`valid_positive_mechanism`。
- **假设：** 五初始态势、三对手均能在连续 JSBSim 中提供有限的 66-D -> 3-D VPP 记录。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_raw_si_v5_observability_result_20260715_zh.md)；[R1 config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_raw_si_v5_r1_observability.yaml)；[record-only contract](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/src/uav_vpp_guidance/evaluation/thesis_five_state_raw_si_v5_record_only.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_raw_si_v5_observability_r1`；`180/180` record、15 个 initial-state×opponent 单元均达到 first-pass 与 post-event 66-D 门，独立 verifier 通过。
- **意义：** 建立连续 post-first-pass 的可观测数据合同，且无 history/prediction/backend/checkpoint failure。
- **重要更正：** legacy `PhaseTracker` 使用归一化观测却用物理单位阈值，因此其“0 re-entry”不能解释为物理 re-entry 缺失；详见 [V5/V7 相位合同审计](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v5_v7_phase_contract_audit_20260715_zh.md)。

### V6：冻结专家 re-entry control matrix（旧 phase 结论已撤销）

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-REACHABILITY-CONTROL-MATRIX-V6`；运行有效，但其 re-entry 零覆盖结论 `phase-semantics superseded`。
- **假设：** 固定 head-on 与兼容 18-D 投影的固定 crossing 可在 first pass 后形成 re-entry。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v6_result_20260715_zh.md)；[授权 config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v6_authorized.yaml)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v6.py)；[verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_five_state_reentry_v6.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v6_control_matrix`；`72/72` strict-JSBSim records，动作、对手分离和 18-D compatibility 均真实执行。
- **意义与边界：** 不能再将“6 个 cell 均为 0 re-entry”作为物理不可达或技能库不足的结论，因为标签计算单位错误；该轮仍保留为兼容 checkpoint 与冻结控制矩阵的工程记录。

### V7：Turn-back teacher（旧 phase 结论已撤销）

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-TEACHER-FEASIBILITY-V7`；运行有效，但 re-entry 零覆盖结论 `phase-semantics superseded`。
- **假设：** 对称 lateral turn-back teacher 可产生可训练的 re-entry 片段。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v7_result_20260715_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v7.py)；[manifest](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/manifests/thesis_five_state_reentry_v7_teacher12.yaml)。
- **结果位置与事实：** 12 个不相交场景、两种 `+/-0.8` turn-back gain、三对手，共 `72` 条 strict-JSBSim record。
- **意义与边界：** 和 V6 一样，运行与 teacher 注入事实可复核；但 legacy phase tracker 使“0 re-entry”不再是有效物理负证据，不能据此否定 teacher 路径。

### V8：物理 re-entry phase contract

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-PHASE-CONTRACT-PROBE-V8`；`valid_positive_mechanism`。
- **假设：** 通过 raw-SI 换算、first-pass 锁存与物理再接近判定，可获得三对手连续 re-entry 数据合同。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v8_phase_contract_result_20260715_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v8.py)；[授权 config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v8_authorized.yaml)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v8_phase_contract_probe`；expert/end-to-end/independent PPO-VPP 分别有 `5/4/6` qualifying episode 与 `330/286/312` re-entry 66-D step；36 条 ledger 的独立 phase replay 通过。
- **意义：** 首次建立了可用于设计训练 preflight 的**物理、连续、可观测** re-entry 输入合同。
- **边界：** 它不训练 policy、不记录 training reward，不说明任何新技能、终端奖励或胜率有效。

### V9：raw VPP action-authority 的 JSON 缺失语义失败

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V9`；`implementation_invalid`。
- **假设：** 7 种冻结小扰动可被 strict JSON telemetry 完整记录。
- **复现入口：** [失败归档](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v9_implementation_failure_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v9_action_authority.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v9_action_authority_design.yaml)。
- **结果位置与事实：** 输出根只含授权配置与 `failure.json`；首个内存 record 因 `NaN` 不符合 JSON 被拒绝，无持久化 episode/gate。
- **意义：** backend 未暴露 vertical VPP bias 是“字段缺失”，不是可写为 `NaN` 的测量值，更不是垂直轴无效。
- **留下的规则：** future telemetry 应区分 `finite value`、`explicit null/unobserved`、`fallback` 三种 receipt。

### V10：raw additive 3-D VPP 无实用 range-extension 权威

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V10`；`valid_negative_mechanism`。
- **假设：** 单轴 raw 3-D VPP 扰动可在至少两对手中安全、实用地降低 range-extension intent loss。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v10_action_authority_result_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v10_action_authority.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v10_action_authority_design.yaml)。
- **结果位置与事实：** `252/252` strict-JSBSim records 与 handoff hash 配对完成；实际 VPP bias 发生数十到数百米变化，但全部方向未达到两对手 `intent-loss delta <= -0.02`。所有 ego crash/OOB delta 为 `0`。
- **意义：** 排除了“动作未注入、clip、fallback 或安全 stop”解释；当前 raw additive head 在 20-step 局部窗口内没有足够实用的 range-extension 权威。
- **留下的规则：** 不再在该 raw bias head 上增加 reward、训练步数或尺度；若继续，需改变动作语义。

### V11：relative-frame semantic compiler 传递有效但无多目标收益

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-RELATIVE-FRAME-SEMANTIC-V11`；`valid_negative_mechanism`。
- **假设：** LOS/range/turn/vertical 的相对坐标语义能在至少两对手中产生安全的几何综合改善。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v11_relative_frame_semantic_result_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v11_relative_frame_semantic.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v11_relative_frame_semantic_design.yaml)；[verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_five_state_reentry_v11_relative_frame_semantic.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v11_relative_frame_semantic`；`144/144` record；语义动作造成 `503--987 m` VPP 位移及 guidance 变化，却没有方法在至少两对手同时达到 range-rate 与 intent-loss 门。
- **意义：** 确认相对语义可穿透 VPP/guidance，而失败点在多目标几何收益，非共同执行链失效。
- **留下的规则：** 下一步必须逐分量测量 AA、ATA、range、range-rate、能量和高度，而不是直接训练。

### V12：six-objective probe 的 runtime mapping 缺失

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-INTENT-CONTROLLABILITY-V12`；`implementation_invalid`。
- **假设：** six-objective 非学习矩阵可构造 record-only JSBSim 环境。
- **复现入口：** [失败归档](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v12_implementation_failure_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v12_intent_controllability.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v12_intent_controllability_design.yaml)。
- **结果位置与事实：** 创建第一个环境前 `KeyError('execution')`；仅 `authorized_config.json` 和 `failure.json`，`completed_records=0`。
- **意义：** 仅发现 `inputs.runtime_template` 与运行时所需 `execution.runtime_template` 不一致。
- **留下的规则：** preflight 不仅校验文件 hash，还必须解析 runtime mapping 并实例化 build-only env。

### V13：single-point VPP 无跨对手 six-objective 控制方向

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-INTENT-CONTROLLABILITY-V13`；`valid_negative_mechanism`。
- **假设：** relative-frame 的 range/turn/vertical 组合存在跨至少两对手、无安全代价的局部多目标下降方向。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v13_intent_controllability_result_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v13_intent_controllability.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v13_intent_controllability_design.yaml)；[verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_five_state_reentry_v13_intent_controllability.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v13_intent_controllability`；`288/288` record、所有动作均被注入且无饱和/新增 ego crash-OOB，但没有组合达到预注册的两对手联合门。
- **意义：** 当前冻结的 20-step single-point VPP 接口存在多目标分辨率/耦合限制；不是训练不足或零动作。
- **留下的规则：** 若继续必须是新的时间扩展接口，而非改单点尺度、窗口、profile 或 reward。

### V14：temporal-corridor 66-D 编码合同失败

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-TEMPORAL-CORRIDOR-V14`；`implementation_invalid`。
- **假设：** 动态 objective profile corridor 可在连续 handoff 后被同一 66-D contract 记录。
- **复现入口：** [失败归档](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v14_implementation_failure_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v14_temporal_corridor.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v14_temporal_corridor_design.yaml)。
- **结果位置与事实：** 首条 handoff 66-D 组装时 `KeyError('profile')`；输出只有授权配置和 failure，0 条 episode。
- **意义：** 动态 action/objective profile 不可隐式替代固定 66-D encoder profile；这不是走廊无效。
- **留下的规则：** 未来接口必须固定可解释的 encoder profile，同时单独记录每步动作/评价 profile。

### V15：固定 encoder 的短 temporal corridor 无实用收益

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-FIXED-ENCODER-TEMPORAL-CORRIDOR-V15`；`valid_negative_mechanism`。
- **假设：** 前半 escape、后半 re-entry 的短 corridor 能在至少两对手中改善预注册 weighted error。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor_result_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor_design.yaml)；[verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v15_fixed_encoder_temporal_corridor`；`144/144` record。expert 与 independent PPO/VPP 的 total/re-entry error 均未改善，end-to-end 仅 3 个 pair；所有方法 ego crash/OOB delta 为 `0`。
- **意义：** 排除“V14 只是编码 bug”与“安全链失效”两种解释；当前短、确定性 corridor 不提供跨对手可学习收益。
- **留下的规则：** 停止对该短走廊做尺度、窗口或 profile 微调。

### V16：动态参考路径执行机制通过，但不等于性能结果

- **Source ID / 状态：** `THESIS-FIVE-STATE-REENTRY-DYNAMIC-REFERENCE-PATH-V16`；`valid_positive_mechanism`。
- **假设：** 三锚点动态参考路径能在不改变 VPP/guidance/PID 的条件下穿透闭环，并在至少两对手形成可审计 action-to-path receipt。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_reentry_v16_dynamic_reference_path_result_20260716_zh.md)；运行 SHA `bfccfc84f9eb24ff88fbe5ccf23b5031410b3bc5`；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_reentry_v16_dynamic_reference_path.py)；[verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_five_state_reentry_v16_dynamic_reference_path.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v16_dynamic_reference_path`；`108/108` record。expert 与 independent PPO/VPP 达到机制门，end-to-end 仅 3 个 pair；VPP/guidance receipt 非零且无新增 ego crash/OOB。
- **重要限制：** action clipping 很高：expert `74/100`、end-to-end `32/60`、independent `81/120` path step 被裁剪；同 handoff 的 terminal reason 完全相同。
- **意义：** 动态路径没有被执行链吞没，但只证明接口可达，不证明路径跟踪、长期 re-entry、胜率或训练收益。

### V17：74-D path context 的 end-to-end 覆盖不足

- **Source ID / 状态：** `THESIS-FIVE-STATE-PATH-CONTEXT-PREFLIGHT-V17`；`valid_negative_mechanism`（coverage gate）。
- **假设：** V16 的 path context 可在三对手中形成足够的 74-D 连续训练前数据覆盖。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v17_path_context_preflight_result_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_v17_path_context_preflight.py)；[verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_five_state_v17_path_context_preflight.py)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_v17_path_context_preflight`；`72/72` record。expert `9/180`、independent `11/220` qualifying episode/step 通过，end-to-end 仅 `4/80`，低于 `8/160` 门。
- **意义：** 74-D 构造、预测、VPP 与飞控链均正确；失败只可归因于该冻结 envelope 的 end-to-end post-first-pass phase occupancy，不可写成路径接口损坏或对手强度排名。
- **留下的规则：** 不得训练 path-conditioned PPO；先独立补足带动力学遥测的可达性诊断。

### V18：未就绪预测被误判为 fallback

- **Source ID / 状态：** `THESIS-FIVE-STATE-V18-PHASE-TELEMETRY-AUDIT`；`implementation_invalid`。
- **假设：** 全 episode phase telemetry 可以严格验证 V17 的覆盖差异。
- **复现入口：** [失败归档](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v18_phase_telemetry_audit_failure_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_v18_phase_telemetry_audit.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_v18_phase_telemetry_audit_design.yaml)。
- **结果位置与事实：** 首个 collector step 即停止、`completed_records=0`；`prediction_valid=false` 且 `prediction_fallback=false` 的 warm-up 未就绪状态被错误认定为 fallback。
- **意义：** 不涉及对手、动力学、VPP 或训练；仅指出 telemetry gate 与冻结 V17 连续接口语义不兼容。
- **留下的规则：** telemetry 要分为允许 predictor 尚未就绪的 warm-up，及 history 满足后才严格的 analysis-ready 窗口。

### V19：warm-up 的预期 anchor fallback 仍被错误禁止

- **Source ID / 状态：** `THESIS-FIVE-STATE-V19-WINDOWED-PHASE-TELEMETRY-AUDIT`；`implementation_invalid`。
- **假设：** 分窗后可记录 warm-up/analysis-ready telemetry，而不会误杀 V17 同源链路。
- **复现入口：** [失败归档](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v19_windowed_phase_telemetry_failure_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_v19_windowed_phase_telemetry.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_v19_phase_telemetry_windowed_design.yaml)。
- **结果位置与事实：** 首条 warm-up episode 停止、`completed_records=0`；在 predicted anchor 尚不可用时，环境按预期退到 `current_target` 并记录 `prediction_fallback=true`，但 V19 仍将其视作禁止事件。
- **意义：** 仅说明 V19 的分窗合同仍不正确，不表示预测器失败或路径接口无效。
- **留下的规则：** warm-up fallback 必须被记录其原因，但不能参与 analysis-ready 的性能/可控性 gate。

### V20：warm-up telemetry wrapper（设计桥接，未计入 21 轮）

- **状态：** 仅完成 [V20 scope/design](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v20_warmup_fallback_telemetry_design_20260716_zh.md) 与 verifier 支撑；没有独立 Source ID 的 JSBSim 执行、输出根或科学结论。
- **作用：** 将 V18/V19 的经验固化为“warm-up 允许 fallback、analysis-ready 禁止 fallback”的工程约束，并为 V21 提供可验证协议。

### V21：current-target warm-up 的 release 循环依赖

- **Source ID / 状态：** `THESIS-FIVE-STATE-V21-WARMUP-ANCHOR-PROTOCOL`；`valid_positive_mechanism`（接口因果诊断）。
- **假设：** 先使用 `current_target`，等预测器有效后一次性切到 predicted anchor，可避免 warm-up fallback 且保持接口连续。
- **复现入口：** [结果报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v21_warmup_anchor_protocol_result_20260716_zh.md)；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_v21_warmup_anchor_protocol.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_v21_warmup_anchor_protocol_design.yaml)。
- **结果位置与事实：** `72` 条 strict-JSBSim record（24 场景×3 对手）全部完成，但三对手均为 `0` anchor release、`0` post-release step。
- **意义：** 确认因果循环：`current_target` warm-up 不进入 predicted-anchor 分支，预测器 history 不更新；release 又要求有效预测，因此永远不能发生。这不是训练不足、对手差异或飞控性能失败。
- **留下的规则：** 若重启该科学假设，observer update 必须与控制链消费彻底隔离，并有 receipt 证明 warm-up prediction 未影响 VPP/guidance/PID/reward/observation。

### V22：observer-release probe 缺少 observation hook

- **Source ID / 状态：** `THESIS-FIVE-STATE-V22-OBSERVER-RELEASE-PROBE`；`implementation_invalid`。
- **假设：** observer-only predictor update 能打破 V21 循环，同时保持 warm-up 控制隔离。
- **复现入口：** [执行有效性审计](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v22_execution_invalidity_audit_20260716_zh.md)；实现 SHA `43f0a04f000b95d46478123eecd3e2c386fa4132`；[runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_five_state_v22_observer_release_probe.py)；[design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_five_state_v22_observer_release_probe_design.yaml)。
- **结果位置与事实：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_v22_observer_release_probe`；虽写入 `72` 条 record，但 `_observe()` override 未接入，`observer_trace` 与 `anchor_switches` 在 `72/72` 条均为空，gate 中的默认 `released=false` 不代表真实测量。
- **意义：** 此轮输出整体 execution-invalid；独立 verifier 的拒绝是正确行为。它不支持“observer-only 有效/无效”，也不能支持“三对手均无法 release”。
- **留下的规则：** 新 Source 在执行前必须以工程 smoke 验证 observer receipt 非空、anchor switch 可读、hook 被实际调用；V22 原输出只可保存为工程负例。

## 4. 未编号但必须一并查看的训练试验

### Terminal-Reward Pilot V1：安全通过，但没有实用改善

- **Source ID：** `THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1`。
- **结果：** [训练后审计](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_reentry_terminal_reward_pilot_v1_post_completion_audit_20260715_zh.md) 确认 `20,000/20,000` step 训练完成，20k checkpoint 被冻结 minimax dev 规则选中；三个对手 ego crash/OOB delta 都是 `0.0`，但没有任一对手达到预注册 `intent_loss` 改善 `<= -0.05`，故 heldout24 未运行。
- **复现入口：** [训练 runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_reentry_terminal_reward_pilot_v1.py)、[completion audit](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/audit_thesis_reentry_terminal_reward_pilot_v1_completion.py)、输出根 `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/thesis_reentry_terminal_reward_pilot_v1`。
- **正确解释：** 这是一条冻结 Dev-only 负/不确定证据：末步胜负/平局/己方 crash-OOB 奖励在这次 20k 预算下没有产生足够的几何准备改善；不证明所有终端奖励无效。

## 5. 账本之后的独立扩展线

以下 Source 不计入 V1--V22 的 21 轮编号，但应与账本共同阅读，因为它们验证了最终 stop decision。

| 独立线 | 结果 | 可保留结论 | 报告 |
| --- | --- | --- | --- |
| Temporal-Corridor Transfer V1 | execution-invalid，`KeyError('execution')`，0 record | 仅说明 runtime mapping 未闭合 | [V1 archive](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_temporal_corridor_transfer_v1_execution_invalid_20260716_zh.md) |
| Temporal-Corridor Transfer V2 | `108/108` 完整、机制负证据 | moderate left/right corridor 可执行且安全持平，但无跨对手六维实用改善 | [V2 result](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_temporal_corridor_transfer_v2_result_20260716_zh.md) |
| Local Action Authority Map V1 | engineering-input incompatible | V15 历史 handoff 不能在当前 runtime replay-match，不能注入 pulse | [V1 archive](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_local_action_authority_map_v1_engineering_input_incompatibility_20260716_zh.md) |
| Local Action Authority Map V2 | `84/84` 完整、机制负证据 | 小幅三轴 pulse 安全，但没有可跨两个对手复现的局部控制方向 | [V2 result](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_local_action_authority_map_v2_result_20260716_zh.md) |
| Runtime smoke + three-knot envelope | smoke 通过；只读包线 no-go | 当前 world-frame action head 没有足够无裁剪余量来诚实测试三锚点路径 | [gate](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_runtime_smoke_and_three_knot_envelope_gate_20260716_zh.md) |

## 6. 跨轮次结论：哪些结论仍然成立

1. **数据与相位层面有真实正向进展。** V5 证明五态势的 post-first-pass 66-D 可观测性；V8 进一步以 raw-SI 规则建立三对手连续 re-entry 合同。V6/V7 的旧“0 re-entry”不再可用，因此不能以它们阻断 V8 之后的结论。
2. **当前单点 VPP 接口的控制边界已被多轮有效实验定位。** V10、V11、V13、V15、Temporal-Corridor V2 和 Local-Authority V2 的问题不同，但一致排除了：在当前冻结 action envelope 中，靠单点偏置、短走廊或小脉冲获得跨两对手、实用且多目标一致控制方向的路径。
3. **工程失败不能被写成科研失败。** V1--V3、V9、V12、V14、V18、V19、V22 都未产生可解释的机制比较；其价值是暴露 manifest identity、runtime mapping、telemetry receipt 与 observer hook 的 smoke 缺口。
4. **V16 是最有价值但必须克制的正向接口发现。** 动态参考路径可以穿透 VPP/guidance/PID 并在两个对手通过预注册机制门，但 action clipping 高、end-to-end 分母不足、terminal 不变；它不是性能结果。
5. **五态势训练线当前应停止，而不是继续调参。** 在未提出真正改变 action envelope 的新 interface 假设前，不应增加训练步数、奖励系数、单点 VPP 尺度、短走廊长度或场景筛选。任何新线必须先通过工程 smoke、phase continuity、无裁剪/可执行性与跨对手局部控制门。

## 7. 下次重新打开该方向时的最小检查表

- 先阅读本账本第 6 节和相应的原始结果报告，不从零记录失败中推导方法结论。
- 用新的 clean SHA 建立并冻结 `Source ID / manifest / output root / config hash / checkpoint hash`。
- 在占用正式 Source ID 前完成不产出科学结论的 engineering smoke：真实 JSBSim env、非空 telemetry、raw-SI phase 单位、handoff hash、observer/anchor receipt、verifier 可读。
- 明确说明新接口改变的是哪一个 action-envelope 假设；禁止把“更大 VPP offset、更长窗口、更多奖励、更多步数”伪装成新科学假设。
- 按 opponent 分开设 gate；至少两个对手中出现同方向、足够幅度、无安全代价的机制改善后，才允许创建训练 pilot。
- 训练 pilot 只准使用新的 dev/heldout split；开发集未通过时不执行 heldout，不在原 Source ID 下调参重跑。

## 8. 快速索引

- 五态势执行主清单：[thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md)
- V5/V7 单位语义更正：[thesis_five_state_v5_v7_phase_contract_audit_20260715_zh.md](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_five_state_v5_v7_phase_contract_audit_20260715_zh.md)
- 当前 runtime/路径包线停止门：[thesis_runtime_smoke_and_three_knot_envelope_gate_20260716_zh.md](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_runtime_smoke_and_three_knot_envelope_gate_20260716_zh.md)
- 当前局部控制权威结论：[thesis_local_action_authority_map_v2_result_20260716_zh.md](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_local_action_authority_map_v2_result_20260716_zh.md)
