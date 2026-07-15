# P2-B2 Physical Preflight60 Input Freeze

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1`
**状态：** `manifest_frozen_runner_not_yet_implemented`

`config/experiment/manifests/thesis_global_advantage_v1_p2_physical_preflight60.yaml` 从 heldout240 提取 60 个唯一 geometry cell，但将每条 `scenario_seed` 改为独立的 `preflight_seed`（12001--12060）。它保留几何、态势、高度、镜像和 package，却不使用 heldout240 的 4 个 evaluation seed，因此仅可用于物理可达性预检。

该 preflight 的目标是对每个 geometry cell x 三对手执行 common frozen run-in specialist，检查 strict JSBSim reset、有限动作和状态、无 backend/prediction fallback、最小连续积分步数、phase/first-pass/terminal telemetry。它不加载候选五态势 policy，不报告 win rate，也不产生 heldout 性能结论。

静态 manifest 和 capability-card 测试共 `6 passed`。实际 runner 与一次性授权尚未建立；在其完成前，P2 仍未通过。
