# Disadvantage Capability Plan 2 V1：三对手 Engineering Smoke 通过记录

**性质：** 工程 smoke，不占正式 Source 结果，不用于性能、胜率或论文主张。  
**正式 Source：** `THESIS-DISADVANTAGE-CAPABILITY-PLAN2-V1`。

## 执行

在固定 Plan2 config 下，对 `expert`、`end_to_end` 与 `independent_ppo_vpp` 分别运行一个 manifest 场景。每个 opponent 包含五种预注册方法，合计 `15` 条 strict-JSBSim 完整 episode。输出根：

`E:/uav-vpp-guidance-disadvantage-capability-v1-results/plan2_motif_capacity_smoke/`

每个 run 的 artifact contract 均为 valid，`failures.json` 均为空。

## 通过条件

- fixed head-on、fixed crossing 与三个 motif policy 都完成完整 episode；
- 三个 motif 在 warm-up 后第 11 个 high-level step 正确进入 intervention；
- `agent_step_metadata.motif_probe_receipt` 在 raw trajectory 中可读；
- runtime tactical basis 为 rank 3；expert smoke 的 lateral-break 首个 receipt condition number 为约 `4.22`；
- 目标机速度与高度、VPP action/world offset、guidance command、actual `n_z`、prediction validity 和 terminal telemetry 均写入；
- 无 JSBSim backend fallback、checkpoint fallback 或缺失 asset。

该 smoke 仅证明正式运行链可执行。它不证明某个原语有效，也不放宽 Plan2 的独立 verifier gate。
