# Taxonomy60 Formal No-Go 独立复核与路线切换

**审查范围：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` 的冻结 60 场景、3 对手、5 方法正式矩阵。  
**审查立场：** 同行评审、领域专家、研究生导师、AST 编辑与通讯作者的联合证据审查。  
**结论日期：** 2026-07-16。

## 审查结论

正式矩阵工程上有效：三条严格 JSBSim 子运行各 300 条，共 900/900 条记录，独立 verifier 为 `pass`。正式 No-Go 合同也已正确触发唯一的后续路线：`paper_closure`。这不是“实验失败后再训练一次”的 No-Go，而是一个**研究决策 No-Go**：该 Source 不允许用混合、负面或局部正面结果授权训练、调参、换场景、换 checkpoint 或结果导向重跑。

因此，路线应立即且平滑地切换为：只读分析、可移植证据包、论文能力边界与机制透明度写作。不会丢弃 900 条记录，也不会把 guard 混杂或单对手差异伪装为 PPO 的普适因果能力。

**独立复核签核：** `approved_for_paper_closure`。该签核是研究工程与论文证据复核，不替换冻结的 `taxonomy60_route_decision.json` 中机器生成的 `pending_independent_reviewer` 字段，避免事后改写正式运行产物。

## 独立核对

| 检查项 | 结果 | 论文含义 |
| --- | --- | --- |
| 完整性与后端 | 900/900，严格 JSBSim，三个子运行均完成 | 可作为正式评估证据，而非 pilot |
| 对手层级 | `expert`、`end_to_end` 为主证据；`independent_ppo_vpp` 为 provenance-limited 支持证据 | 不可合并对手，也不可用第三对手满足重复性门槛 |
| Recovery guard | 68 条 PPO 记录由 deterministic guard 改变 requested mode | 只能报告为 deployed mixed-interface；不能归因于纯 PPO routing |
| 路由候选 | primary split 中仅出现少量 routing-sensitive candidate | 可做逐回合 mode/VPP/terminal 审计，不能据此训练 routing patch |
| 双技能容量边界候选 | 在部分 cell 出现，但尚未满足“新专家必要”的完整独立条件 | 可作为现有双技能库的边界，而非第三技能已被证明必要 |
| 大量同结果 cell | 每个对手约 54--55 个 cell 为 non-discriminative 或 opponent-dependent | 必须完整报告；它们说明场景区分度和对手条件是结果边界，不是可被删掉的坏数据 |

## 只读导出后的定量核对

`analysis_read_only_v1` 仅读取三份 300-record `summary.csv`、正式 verifier receipt、route decision 与冻结 manifest。它没有读取后再写回原始 episode，也没有启动 JSBSim。

| 主对手 | Canonical PPO 胜/负/平 | resolved win rate [Wilson 95% CI] | Guard 混杂 PPO 记录 |
| --- | ---: | ---: | ---: |
| `expert` | 38 / 22 / 0 | 0.633 [0.507, 0.744] | 10 / 60 |
| `end_to_end` | 43 / 14 / 3 | 0.754 [0.629, 0.848] | 12 / 60 |

- `advantage` 和 `crossing_entry`：两个主对手均为 12/12 resolved wins。这是包线内的正面保持性证据，不是五态势通用优越性。
- `disadvantage`：`expert` 为 0/12、`end_to_end` 为 1/12。它是最清晰的能力边界，但在当前双技能固定接口下尚不能证明“只需新增一个第三技能”。
- `head_on`：`expert` 为 7/12、`end_to_end` 为 7/9（另有 3 平局）；其中大部分 PPO 记录受 guard 混杂，不能据此主张纯 PPO routing 的得失。
- `neutral`：`expert` 为 7/12、`end_to_end` 为 11/12，说明该态势对 opponent 条件敏感，不能合并为单一系统结论。
- 4 个 routing-sensitive cell 中，两个 primary cell 受 guard 混杂；唯一未混杂的 primary cell 是 `expert / taxonomy60_wide_balanced_neutral_below_pos`。这不足以开启 routing patch 或新训练。
- 所有 formal terminal 均为 timeout（HP advantage/disadvantage 或 draw），没有 crash/OOB。该事实仅说明这个冻结包线的 terminal mix，绝不构成飞行安全认证。

## 路线切换

1. 运行 `scripts/analyze_thesis_dualskill_taxonomy60_formal.py`，从冻结的三份 `summary.csv` 生成 900 条 episode index、按态势/对手/方法分层的 Wilson 结果表、配对差值、终端语义表与 guard 透明度表。
2. 在主文中将系统称为“具有 deterministic recovery override 的 task-aware dual-skill hierarchical VPP interface”，而非五态势已训练决策器、纯学习几何识别器或全面优于基线的控制器。
3. 在补充材料中保留 60 × 3 × 5 完整矩阵。`expert` 与 `end_to_end` 分开报告；第三 PPO/VPP 对手只作支持性/限定性对照。
4. 对 routing-sensitive cell 仅做只读的 requested/effective mode、guard、VPP 与 terminal 机制审计；对 capacity candidate 仅做 VPP/guidance/AoA/`n_z`/clipping/terminal 的边界审计。
5. 不启动训练。只有论文闭合后，且 No-Go v3 的 optional-new-interface 条件全部独立满足时，才可另起一份 decision memo；该 memo 不自动授权 pilot。

**已经完成的路线切换交付物：**

- [read-only analysis package](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/analysis_read_only_v1)：900 条索引、分层 Wilson 表、配对 delta、terminal/safety 表、mode/guard 表、routing candidate 索引及 SHA-256 manifest。
- [formal verification receipt](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/formal_verification.json)：已再次从原始 aggregate 复核，仍为 `pass`、900 records、68 guard-confounded records。
- [route decision](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/taxonomy60_route_decision.json)：唯一允许的下一步仍为 `read_only_analysis_artifact_bundle_and_thesis_writing`。

## 审稿式边界

- 该矩阵能支持：冻结双技能接口在明确的 Taxonomy60 包线、多个对手条件下的条件化能力与失效边界。
- 该矩阵不能支持：五态势共享技能库已经有效、所有态势均优、PPO 在无标签情况下学习几何识别、胜率等同于击杀率/气动优势/飞行安全认证。
- 这条切换路线的价值在于把 No-Go 变成可发表的负面与边界证据，而不是用不断增加训练轮次稀释证据可信度。
