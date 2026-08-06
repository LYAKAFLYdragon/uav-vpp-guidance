# E 盘 UAV-VPP 工程整合索引

生成日期：2026-08-06

## 整合目标

本次以 `E:/uav-vpp-guidance` 为主工程目录，当前目标分支为
`fix/flight-control-comparison-sync`，整合结果提交为 `be698f5`。Git 的
`main` 分支仍保持在原有 `b906abd`，本次没有擅自改写远程主分支。

## 已整合

- `codex/control-guidance-remediation-v1` 与
  `codex/control-guidance-remediation-integration` 已在干净临时 worktree 中
  合并并通过 77 项整改测试。
- 整合内容包括控制方法整改证据、F01/F02 配对回归工具、F03/F06/F11 严格
  运行预检、F07 动作契约和 F16 状态机证据。
- `scripted_evasive_vpp` 实现和干跑证据保留，但没有注册到主比较配置；它仍是
  独立候选，不能被当作正式主线对手。

## 脏工作树保护

主目录原有的 69 个已跟踪修改和约 2708 个未跟踪文件已保存到：

- Git stash：`stash@{0}`，对象 `44fdcdc0d5e5c92fb7bd97589cf7c86d8c586335`
- 状态清单：`E:/uav-vpp-guidance-archives/main_worktree_status_pre_consolidation_20260806.txt`
- tracked diff：`E:/uav-vpp-guidance-archives/main_worktree_tracked_diff_pre_consolidation_20260806.patch`

以下仍有未提交内容，未删除：

- `uav-vpp-guidance-control-guidance-remediation-integration`
- `uav-vpp-guidance-clean-formal`
- `uav-vpp-guidance-disadvantage-capability-v1`
- `uav-vpp-guidance-explore-rl`
- `uav-vpp-guidance-five-state-router-mvp-v0`
- `uav-vpp-guidance-thesis-five-state-v1`

## 已移除的干净 worktree

只删除了 worktree 目录，分支或提交仍保留：

- `uav-vpp-guidance-can20260705`
- `uav-vpp-guidance-control-guidance-remediation-v1`
- `uav-vpp-guidance-thesis-evidence-freeze-v1`

`uav-vpp-guidance-thesis-ablation30-formal` 已解除 Git worktree 注册，但目录中
仍有未跟踪的 `memory`、`paper_materials`、临时文献文本和脚本，因此没有强制删除。

## 结果目录策略

原始结果不复制进 Git，继续保留在 E 盘独立目录，并以已有 manifest、配置和哈希
作为复现入口：

- 主论文证据：`uav-vpp-guidance-artifacts`、`uav-vpp-guidance-thesis-evidence`
- Taxonomy60、P4 和能力分析：`uav-vpp-guidance-five-state-heldout-envelope-v1-results`、
  `uav-vpp-guidance-disadvantage-capability-v1-results`
- 路由和选项价值探索：`uav-vpp-guidance-ppo-routing-v2-results`、
  `uav-vpp-guidance-option-value-router-v1-results`
- supporting evidence：`uav-vpp-guidance-evidence-results`、
  `uav-vpp-guidance-tactical-basis-v2-results`、
  `uav-vpp-guidance-preview-router-mvp-results`
- 非标准或溯源受限结果：`uav-vpp-guidance-noncanonical-results`，只作归档，
  不进入论文 headline。

## 暂不删除

`ind-headon`、`noncanonical-geometry-only` 和 `noncanonical-v14b-clean` 各自含有
主目录没有的报告或脚本；它们应先作为 supporting archive，不能按“代码重复”直接
删除。空目录 `uav-vpp-guidance-prev` 已确认无内容，但当前删除操作受到本机安全
策略阻止，未绕过策略。

## 已知验证差异

整合后的整改测试为 `117 passed, 2 failed`。剩余失败只涉及整合前已有的三份几何
配置与历史保护哈希不一致；本次没有覆盖配置或更新基线来掩盖差异。正式修复这些
历史配置前，应单独建立配置变更说明和新的批准基线。

完整的只读盘点见：
`E:/uav-vpp-guidance-archives/e_drive_consolidation_inventory_20260806.json`。
