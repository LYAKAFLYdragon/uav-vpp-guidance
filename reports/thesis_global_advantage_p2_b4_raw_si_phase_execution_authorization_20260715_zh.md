# P2-B4 Raw-SI Phase Preflight 一次性执行授权

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-RAW-SI-PHASE-PREFLIGHT-V1`
**授权状态：** `one_shot_execution_authorized`
**冻结实现 commit：** `1a2f854583179b2f17917f60475dbb30aa6f29e3`

唯一允许命令：

```powershell
python scripts/run_thesis_global_advantage_p2_b4_raw_si_phase.py --execute
```

该命令在新的 30 条 B4 diagnostic 场景、三个分开报告 opponent 上运行 frozen `run_in_head_on` specialist，共 90 条 strict-JSBSim record。它不训练、调参、加载候选五态势策略、选择 checkpoint 或报告性能结论。

## 核心冻结合同

- B4 manifest SHA-256：`e709cdafd3ba6ba822454cdbd88ffa49988962cedeed310b15f071b87004fa81`。
- runtime config / registry SHA-256：`3a050de1eac5258fb68ae52dea71fc9cc8c9ad1bea69a73c0cd403a56e40b1f2` / `c87edfed4be69a8293a889c9caa55c520fdab7dd93c44c9c6e057be869f435c5`。
- PhaseTracker 的唯一输入是 `observation.relative_state.range_m/range_rate_mps`（raw SI）；`observation_vector` 的 normalized 值只能写入 diagnostic 字段。
- gate 需要 90/90 structural、scenario receipt、normalization diagnostic、phase replay，以及 step-0 `pre_merge` 语义均通过。
- `re_entry` 覆盖不作为本轮通过门槛。输出根必须不存在，worktree 必须 clean，E 盘空闲空间至少 120 GB。

任一 gate 失败都冻结为新 Source ID 的负证据；不得重跑同一 Source ID、改 reward、增训练步数或进入技能训练。无论结果如何，运行后把 `execution_permitted` 复位为 `false`。
