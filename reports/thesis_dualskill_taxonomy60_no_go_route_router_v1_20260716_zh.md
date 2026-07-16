# Taxonomy60 No-Go 路线决策器 v1

**适用 Source：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`  
**作用：** 在正式运行前冻结“失败后去哪里”，防止把每个 no-go 自动转化为下一轮调参或新版本。  
**原则：** 路由选择的目标是获得正确、可写入论文的结论，不是寻找下一次可能获胜的设置。

## 1. 工程 no-go：最多一次科学恢复

| 触发 | 允许动作 | 禁止动作 | 终点 |
| --- | --- | --- | --- |
| Smoke 失败 | 只修复对应的 unit/integration test 后重做 smoke；最多两个根因修复周期 | 创建 Source、查看比较指标、改 Heldout60 | 两次后仍失败：停止 Taxonomy60，使用 Taxonomy30 + Calibration v2 写作并报告第三 opponent 未闭合 |
| 正式 Source 在生成完整可解释记录前无效 | 仅允许一次 `...-R1` 技术恢复；完整重跑 900 record universe | 改场景、方法、checkpoint、对手、horizon、统计门、奖励或控制器 | R1 必须与原科学变量完全一致；不是新假设 |
| R1 仍无效，或原运行已有不可解释 partial data | 不再重建；归档 failure manifest，走现有双 opponent 证据写作路线 | R2/V2、挑选 partial record、以日志推断性能 | 论文声明降级为“Taxonomy30 双 opponent 诊断”，不虚构三 opponent formal 结论 |

`R1` 的前提是：故障根因已被无 Source 的 smoke 复现并以测试覆盖；独立 diff 审计证明唯一改动是该工程修复；full manifest、methods、assets、opponents、seeds、terminal protocol 和统计代码哈希不变。

## 2. 有效正式结果的路线表

下表中的“重复”指同一模式至少覆盖两个 opponent、双镜像、两个高度条件，且逐回合审计排除 fallback、共同执行链异常和终端语义误读。未达到重复条件，一律归为 `opponent-dependent / non-discriminative`，不产生算法路线。

| 已观察模式 | 允许的结论 | 正确后续路线 | 明确禁止 |
| --- | --- | --- | --- |
| M3 在 legacy head-on/crossing 不劣于 M0/M1，且五态势表现混合 | 双技能接口在有限包线内具备条件化保持性，不是全态势优越 | **论文写作路线**：报告能力热图、边界、对手卡和失败机制 | 以整体均值宣称五态势通用成功；追加训练 |
| M3 输、但 M0 或 M1 在相同场景赢，且模式重复 | `routing-sensitive candidate`；现有低层行为可能已足够 | **机制写作路线**：报告 mode entry/guard/VPP 差异；保留为未来高层接口问题 | 在 canonical PPO 上强制首拍、加 hold/clamp、立即重训 routing |
| M0 与 M1 在同一场景均输，M3 也输，且模式重复 | `two-skill capacity boundary candidate`，不是“第三专家已被证明需要” | **论文边界路线**：结合 V10/V13/V15 等接口负证据，说明当前 action interface 不能支持直接增技能 | 训练新专家、加奖励、加时长；把共同失败说成训练不足 |
| 共同失败只出现在一个 opponent，或终端/轨迹早期已明显分歧 | `opponent-dependent sensitivity` | **对手条件路线**：更新 capability card 与 split-specific claim | 给对手排绝对强弱；把单 split 失败泛化为系统缺陷 |
| M0--M4 都同胜或同负，攻击区/终端也几乎一致 | `ceiling/non-discriminative cell` | **报告路线**：保留为无区分度场景，避免用它支撑组件价值 | 删掉场景、替换场景直到出现差异 |
| M3 与 M4 一致，且 M4 在 crossing-entry 首拍仍选 crossing | 在 task-aware 条件下，crossing 首拍不依赖 bootstrap/lock | **机制写作路线**：有限地表述 task-aware PPO selection | 写成 task-label-free geometry recognition 或普适 learned routing |
| M3 与 M4 差异显著，尤其 M4 退化或安全变差 | bootstrap/lock 是 deterministic preservation constraint 的候选原因；系统为混合接口 | **混合系统路线**：明确 guard/lock 的工程角色与边界 | 把 M3 说成纯 learned routing；在本 Source 后立即移除 guard |
| target crash/OOB 主导胜率或不同方法 terminal mix 不同 | 胜率受 terminal pathway 影响 | **终端语义路线**：报告 terminal decomposition 和 command/actual response | 将 win rate 改写为 kill rate、气动优势或安全认证 |

## 3. Recovery guard 的诚实边界

M0--M4 可以完整消融两个 specialist、静态 gate、PPO routing 与 crossing bootstrap/lock；但 recovery guard 在 M3/M4 中保留，因而只能通过 requested/effective mode、guard reason 与安全遥测实现**机制透明**，不能声明其带来因果性能增益。

- [ ] 若 guard 在主 Source 中从未触发：明确报告其在该包线中不活跃，不做 guard 成效主张。
- [ ] 若 guard 触发：报告触发频率、进入状态、替换的 mode、后续 terminal 和 command/actual response；表述为 deterministic override。
- [ ] 不在 Taxonomy60 正式矩阵中关闭 guard。若未来确需研究 guard 因果性，只能在论文闭合后设计独立、预先限定的 safety sandbox；它不是当前论文三缺口的前提。

## 4. 可选新接口线的唯一开门条件

Taxonomy60 的任何单一失败都不自动开启新算法线。只有同时满足以下条件，才允许在论文闭合后起草一个新接口 decision memo：

- [ ] M0/M1/M3 的共同失败在至少两个 opponent 重复，覆盖双镜像与两个高度条件。
- [ ] raw VPP、guidance、AoA、`n_z` command/actual、clipping 和 terminal 审计排除了共同执行链、fallback 与早期对手分歧。
- [ ] 新假设修改的是 action envelope 或任务动力学，并逐条说明为何不被 V10/V11/V13/V15、Temporal-Corridor V2 和 Local-Authority V2 覆盖。
- [ ] 新 Source 成功或失败都能形成独立论文边界；若失败，不存在预设的 V2/V3 追随路线。

不满足以上全部条件时，正确路线始终是论文收束，而不是继续算法研发。
