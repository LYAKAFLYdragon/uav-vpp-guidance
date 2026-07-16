# Taxonomy60 P3 独立预检记录

**范围：** 只读合同、资产、P2 receipt、磁盘和正式输出根检查；未启动 simulator，未创建任何 `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` episode。

**机器收据：** [P3 independent preflight JSON](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_dualskill_taxonomy60_p3_independent_preflight.json)

**决定：** `PASS FOR FORMAL-RUNNER/VERIFIER DEVELOPMENT ONLY - formal execution remains unauthorized.`

## 已核验事实

| 门 | 结果 |
| --- | --- |
| P1 manifest | version 4、60 场景、49 个 Git blob 历史来源、intersection `0`；No-Go v3 将全部 10 个分类映射到唯一研究路线 |
| No-Go 路由闭合 | `THESIS-DUALSKILL-TAXONOMY60-NO-GO-V3`，SHA-256=`f104...8e10`；P4 formal verifier 必须生成含路线候选、切换理由、memo 资格与恢复状态的决策 JSON |
| 场景平衡 | 四个几何象限和 crossing-entry 各 12；三高度各 20；双镜像各 30 |
| 正式配置矩阵 | 3 opponents x 5 methods x 2 task groups = 30 个 config cell 均可构造，均 strict JSBSim |
| P2 receipt | `15/15` disposable records、无 backend fallback、独立 verifier pass |
| 预测边界 | 前 9 个 high-level steps 为固定 warm-up fallback；第 10 step 起 fallback 禁止 |
| 冻结资产 | 6 个 checkpoint 文件存在、SHA-256/shape 均匹配 |
| JSBSim data | `E:/uav-vpp-guidance-can20260705/envs/JSBSim/data`，SHA-256=`18fa...c10a` |
| 输出与容量 | 正式输出根不存在；可用空间约 290 GiB，高于 50 GiB reserve |

资产目录是 portable bundle 而非 Git worktree，因此其文件 SHA-256 是权威证据；预检不会把缺失 `.git` 误判为资产变化。

## P3 边界

- 此收据不包含任何胜负、HP、攻击区或比较性能统计。
- 它不授权 900 条正式 episode；`formal_execution_authorized=false` 是预期值。
- 任何后续 runner 必须将 `--jsbsim-root` 指向已验证数据目录，保留 5 方法、3 对手、60 场景和同一 scenario seed。
- 任何第 10 个 high-level step 之后的 prediction fallback、backend fallback、checkpoint substitution、缺失 telemetry 或不完整 record universe 必须由 formal verifier 判为 invalid，而非被挑选性分析。
- P4 只能按 No-Go v3 的 `classification_routes` 输出唯一 `selected_study_route`；有效的混合或负面结果默认进入 `paper_closure`，不自动开启训练。

## 下一步

P4 只实现并测试正式 runner、900-record artifact contract 与 independent verifier；实现完成后再由 P3 receipt、formal config/asset hash 和空输出根共同决定是否执行一次正式 Source。
