# P2-B3 Phase-Observability Preflight 一次性执行授权

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-OBSERVABILITY-PREFLIGHT-V1`
**授权状态：** `completed_execution_closed`
**冻结实现 commit：** `d1ed7de3765f1801a98248bf4e994532b2ceacdd`

## 允许的唯一操作

执行以下命令一次：

```powershell
python scripts/run_thesis_global_advantage_p2_b3_phase_observability.py --execute
```

它只使用 frozen `run_in_head_on` specialist，在 30 条新的诊断几何与 `expert`、`end_to_end`、`independent_ppo_vpp` 三个分开报告 opponent 上生成 90 条 fresh-environment strict-JSBSim 记录。它不训练、不调参、不加载候选五态势策略、不选择 checkpoint，也不输出任何性能、优越性或 opponent 总体强度结论。

## 冻结输入与 gate

| 项目 | SHA-256 / 值 |
|---|---|
| B3 manifest | `d1e7c0e0750ec36eab6ef3ab7b6a353f7ceed8d88211d6d6535344c13e8a384c` |
| runtime config | `3a050de1eac5258fb68ae52dea71fc9cc8c9ad1bea69a73c0cd403a56e40b1f2` |
| runtime registry | `c87edfed4be69a8293a889c9caa55c520fdab7dd93c44c9c6e057be869f435c5` |
| P2-B3 runner | `808096729748c015f2d348dc09f2acacc3063afa427ed0e5090e49535c03d5ee` |
| run-in specialist | `0aeadd11cd8c7723c8963bcc5da21dd97c365e64f234d07b86c2b8f0d7ec5fb6` |
| output root | `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p2_b3_phase_observability` |

执行前必须为 clean worktree、输出根不存在、E 盘空闲空间不少于 120 GB。执行器还会验证冻结 ancestor、授权文件 SHA、manifest/runtime/registry/checkpoint SHA 与三个 opponent 的顺序。

## 判定与停止规则

- 通过 phase-observability gate 要求：90/90 strict-JSBSim valid record、scenario application receipt 通过、完整 PhaseTracker receipt 可回放、且全部 reset 声明为 `pre_merge` 的场景在 step-0 也得到 `pre_merge`。
- `re_entry` 的覆盖只记录，不作为本轮通过门槛；它不能被 episode 数量、调参或重跑“补齐”。
- 任一 hash、fallback、非有限 action/state、scenario receipt、phase replay 或 output-root gate 失败，立即归档为负证据。不得更改场景、阈值、控制器或 reward 后重跑同一 Source ID。
- 运行后无论正负，都将 `execution_permitted` 复位为 `false`。B3 通过也只允许设计下一阶段的 phase-feasible sampler，不直接解锁共享技能训练或 formal held-out。

## 已执行结果

本授权已于 2026-07-15 执行一次并关闭。`90/90` record 均完成 strict JSBSim 结构合同、scenario application receipt 和 PhaseTracker receipt replay；raw backend state 在 reset 时保留了约 `3,800 m` 请求距离。gate SHA-256 为 `e61e223288c85a721e43b6a1a28e31463d5e30a1e1c20142826a3987ef9e10bf`，run manifest SHA-256 为 `0fafadb7dce642fd6c1a2beb2561c5f3d23a0ebe1f610b4ec87da8c7ce48ebb0`。

但 step-0 tracker 输入的 `range_m` 为约 `0.760`，而 persisted raw receipt 的 requested range 为约 `3,800 m`。根因是 policy observation 的 `range_m` 已按 `5,000 m` 归一化，B3 runner 仍将其传给以米为单位阈值的 `PhaseTracker`。因此 90/90 在 step-0 进入 `post_merge`，本轮为 `normalized_phase_input_contract_no_go`，不解锁训练。完整归档见 `reports/thesis_global_advantage_p2_b3_completion_20260715_zh.md`。
