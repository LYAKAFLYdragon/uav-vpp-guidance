# Taxonomy60 No-Go 路线决策器 v2

**适用 Source：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`  
**机器可读合同：** [thesis_dualskill_taxonomy60_no_go_v2.yaml](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/decision_contracts/thesis_dualskill_taxonomy60_no_go_v2.yaml)  
**状态：** P1 预注册补强；正式运行前冻结。

## 审核结论

v1 的方向正确，但还不足以防止结果后的解释漂移：它没有量化“重复”和有效记录，未限制 provenance-limited 的独立对手在跨对手结论中的权重，也未处理 M3/M4 的 deterministic recovery guard 对 routing 因果归因的混杂。v2 将这三项固定为合同；它**不改变** Heldout60 场景、五个方法、冻结资产或任何性能变量。

本决策器的唯一目的，是使有效负结果也能平稳转入论文写作路线，而非触发新一轮训练。Taxonomy60 完成后默认的终点永远是 `paper_closure`；“新接口”最多是论文闭合后的独立 decision memo，不是自动研发授权。

## 1. 先判运行是否有效

只有同时满足以下条件，900 条记录才可进入结果分类：

- `900/900` episode 均在严格 JSBSim 后端完成，且无 backend fallback。
- 每条均有 scenario/seed/method/opponent、终端语义、requested/effective mode、guard reason、VPP、guidance、AoA、`n_z` command/actual、clipping、prediction receipt。
- manifest、资产、config 与统计代码的哈希均可由独立 verifier 复算；所有遥测为有限值。

若任一条件失败，整套正式 universe 为 `implementation_invalid`，不可从 partial records 提取性能结论。只允许修复可由 disposable smoke 复现且有测试覆盖的工程根因；最多两个 smoke 修复周期。若正式 Source 已启动且完整记录无效，可执行一次科学变量完全不变的 `-R1`；R1 仍无效则退回 **Taxonomy30 + Calibration v2 的双对手写作路线**，不形成虚假的三对手结论。

## 2. 证据优先级与“重复”的可验收定义

`expert` 与 `end_to_end` 是两个**主对手**。`independent_ppo_vpp` 只可作 provenance-limited 的支持或反例，不能单独构成，也不能替代任一主对手来满足跨对手重复。

一个机制或边界模式必须同时出现在两个主对手中；对每个主对手还必须覆盖双镜像、至少两种高度、至少四条完整场景记录，并完成 fallback、共同执行链、早期目标轨迹与终端语义审计。否则只标注为 `opponent-dependent / non-discriminative`。

所有 cell classification 先后顺序固定为：运行有效性 → 主对手覆盖 → guard 混杂 → 对手依赖/无区分度 → routing/容量/lock 机制 → 终端语义限定。一个场景可有多个**描述标签**，但整项研究只能输出一个 `selected_study_route`，且默认是 `paper_closure`。

## 3. 各类结果应去往哪里

| 可验收模式 | 可说的结论 | 唯一后续动作 |
| --- | --- | --- |
| M3 失败、M0 或 M1 同场景同 seed 获胜；两个主对手均满足重复定义；M3 guard 未改变 mode | `routing-sensitive candidate`，现有低层行为在该有限条件下可能已足够 | 写 mode entry/effective mode、VPP 和 terminal 的机制分析；不 patch、不重训 routing |
| M0、M1、M3 同场景均为 resolved loss；两个主对手均满足重复定义；无 guard/执行链/早期对手轨迹混杂 | `two-skill capacity boundary candidate` | 在论文中收紧为当前双技能接口的能力边界；不把它写成“已证明必须有第三技能” |
| 任何用于 M3 routing 比较的场景中 guard 改写 requested mode | `mixed-interface guard-bound` | 仅报告部署系统的混合接口表现和 guard telemetry；不得将差值归因于 PPO routing |
| 模式仅见于一个主对手，或 `independent_ppo_vpp` 是唯一第二支持 | `opponent-dependent sensitivity` | 更新 capability card，使用 split-specific 文字；不做绝对强弱或普适系统缺陷主张 |
| 五方法 outcome 相同且 terminal/攻击区路径实质一致 | `ceiling/non-discriminative cell` | 保留在热图与补充材料中，说明它没有组件区分能力；不删场景、不补新场景 |
| M3/M4 在 crossing-entry 一致，且 M4 首拍仍为 crossing | task-aware 条件下的 crossing 首拍不依赖 bootstrap/lock | 仅报告有限的 task-aware PPO selection，不扩展为 task-label-free 几何识别 |
| M3/M4 有重复差异或安全差异 | bootstrap/lock 是 deterministic preservation constraint 的候选原因 | 将系统定位为混合接口；不声称纯 learned routing，也不在本 Source 移除 guard |
| target crash/OOB 主导某方法胜率或方法间 terminal mix 不同 | win rate 受 terminal pathway 影响 | 报告 terminal decomposition；不改写为 kill rate、气动优势或安全认证 |

## 4. 唯一可能的算法分支

任何上述 cell-level 负结果都**不会**自动打开算法研发。只有在 `expert` 和 `end_to_end` 都出现已审计的 two-skill capacity boundary，且逐回合 VPP/guidance/AoA/`n_z`/clipping/terminal 审计排除了共同执行链异常与早期对手分歧时，才可在论文证据矩阵完成后起草一次 `optional_new_interface_memo`。

该 memo 必须提出 action envelope 或任务动力学的实质改变，并逐项说明它不被 V10/V11/V13/V15、Temporal-Corridor V2 与 Local-Authority V2 的既有负证据覆盖。独立复核人签字前，不得启动 pilot、训练或任何 V2/V3 变体。任何一个条件不足，正确路线是 `paper_closure`。

## 5. 运行后的固定交付物

P2 verifier 必须生成 `taxonomy60_route_decision.json`。它应包含合同和 manifest 哈希、formal-validity verdict、主/支持对手的分开证据、cell classifications、所有 guard-confounded episode ID、选定 study route、允许/禁止动作和独立复核签字。该 JSON 是结果解释的唯一入口；论文、图表与后续 memo 只能引用其中已通过的结论。

**路线切换摘要：** 工程无效 → smoke 修复或一次同变量 R1；R1 无效 → Taxonomy30 双对手论文；正式有效但表现混合/负面 → Taxonomy60 论文边界与能力图谱；只有双主对手的共同、已排除混杂的容量边界 → 一份新的接口 decision memo，仍不自动训练。
