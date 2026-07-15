# P2-B2 Physical Preflight60 Input Freeze

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1`
**状态：** `implementation_frozen_execution_not_authorised`

`config/experiment/manifests/thesis_global_advantage_v1_p2_physical_preflight60.yaml` 从 heldout240 提取 60 个唯一 geometry cell，但将每条 `scenario_seed` 改为独立的 `preflight_seed`（12001--12060）。它保留几何、态势、高度、镜像和 package，却不使用 heldout240 的 4 个 evaluation seed，因此仅可用于物理可达性预检。

该 preflight 的目标是对每个 geometry cell x 三对手执行 common frozen run-in specialist，检查 strict JSBSim reset、有限动作和状态、无 backend/prediction fallback、最小连续积分步数、phase/first-pass/terminal telemetry。它不加载候选五态势 policy，不报告 win rate，也不产生 heldout 性能结论。

严格 runner 已建立，但仍保持 `execution_permitted=false`。它在执行前验证 canonical config、clean worktree、冻结实现 ancestor、仅授权材料差异、输入 manifest/runtime/registry SHA、三对手与 run-in specialist checkpoint SHA、fresh output root 和最小磁盘余量。物理可达性与 phase 覆盖分开记录：任何 phase gap 都会阻塞相关技能训练，但不会被误写为 JSBSim 物理执行失败。

定向合同测试现为 `6 passed`；非执行 dry-run 已解析 60 个 geometry cell 和 180 条计划记录。一次性授权、实际 180 条运行和结果归档均尚未发生，因此 P2 仍未通过，也不允许训练、调参或启动 formal held-out。
