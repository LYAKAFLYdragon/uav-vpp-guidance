# Taxonomy60 P1 Manifest 冻结记录

**Checklist：** [thesis_master_evidence_closure_checklist_v2_20260716_zh.md](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_master_evidence_closure_checklist_v2_20260716_zh.md)

**范围：** 仅冻结 Heldout60 场景合同、方法合同与 no-go 决策合同；未创建正式输出根，未运行 JSBSim 比较，未训练、未调参。

**决定：** `PASS - P1 contract correction complete. P2 的 15 条 disposable strict-JSBSim engineering smoke 获准开发；900 条正式 Source 仍未获准执行。`

## 冻结结果

| 项目 | 已验证值 |
| --- | --- |
| Manifest | `config/experiment/manifests/thesis_dualskill_taxonomy60_v1.yaml` |
| Source ID | `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` |
| 生成 Git SHA | `61180c9d53918d5e83370ae8ec6db30ce192c6db` |
| Manifest version / payload SHA-256 | `2` / `664ef7880c7b95a336848972d91b973904c0c6ea2516516e087940351c4d8630` |
| 场景与计划 records | `60` 场景；`60 x 3 opponents x 5 methods = 900` episodes |
| 几何场景层 | Advantage、Head-on、Disadvantage、Neutral 四象限各 12；Crossing-entry 交战上下文 12 |
| 高度与镜像 | own-below/co-altitude/own-above 各 20；negative/positive 各 30 |
| 物理包 | wide-balanced 30；fast-closure 30 |
| 历史不重合验证 | 49 个历史来源，full-state signature intersection `0` |

`crossing_entry` 被明确保留为交战任务/阶段上下文，而不是被误写为第五个互斥几何象限。旧 19-D PPO 只接收冻结的 legacy task bit：crossing-entry 为 1，四个几何象限为 0；场景名称和新 taxonomy 标签均不输入 policy。

## 可移植的零交集证明

此前有若干 RQ YAML 已从当前工作树清理。P1 没有降低证明标准或仅保留旧口头结论，而是扩展 manifest builder 以从不可变 `git:<revision>:<path>` blob 重建 YAML include 链、场景签名和 SHA-256。49 个 source file records 均为 `git_blob`，不包含机器盘符或临时 worktree 路径。

这使后续复核只需仓库 Git 历史即可复算 `0` intersection；若任一历史 blob 缺失、哈希不一致或新的 Heldout60 场景发生碰撞，builder 必须失败。

## 已冻结的解释边界

- M2 是 legacy static task gate baseline，不是动态五态势 Oracle：仅 crossing-entry 选 frozen crossing，其余四类选 frozen head-on。
- resolved-config 审计发现 canonical 配置没有启用 pre-merge lock；历史 RQ3 的 `crossing_initial_task_bootstrap.enabled: false` 也未被 policy 消费。已以默认行为不变的最小补丁使该字段生效，并为其补齐 per-step telemetry 与回归测试；manifest 以 version 2 重冻。
- 更正后 M3/M4 使用相同 canonical PPO checkpoint；仅 M3 启用 canonical crossing initial-task bootstrap，M4 显式关闭它；两者均不启用 pre-merge lock，并保留 recovery guard。
- 若 guard 改写 M3/M4 的 requested mode，该 episode 只可解释为 deployed mixed-interface evidence，不能归因于 PPO routing。
- expert 与 end-to-end 是跨对手重复的两个主证据；independent PPO/VPP 只能作 provenance-limited 支持或反例。
- 所有有效/负结果的唯一出口由 [No-Go 决策器 v2](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_dualskill_taxonomy60_no_go_route_router_v2_20260716_zh.md) 和其 YAML 哈希合同固定。

## P2 开始门

P2 只能实现并运行 `5 methods x 3 opponents x 1 smoke scene = 15` 条 disposable episode。每条必须严格 JSBSim，并验证 adapter、asset/config hash、requested/effective mode、bootstrap/lock/guard、VPP/guidance/PID、prediction receipt、terminal semantics 与 artifact verifier。smoke 只检查工程链路，不读取比较性能；任何失败仅进入已冻结的工程 no-go 路由。
