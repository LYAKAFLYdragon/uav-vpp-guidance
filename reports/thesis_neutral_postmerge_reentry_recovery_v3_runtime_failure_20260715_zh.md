# V3 Runtime-Feasibility 执行停止记录（2026-07-15）

## 身份与判定

- Source ID：`THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V3`
- 授权提交：`48b55bad4d11e9415bdd162f47f2d3d4bce9bc96`
- 授权配置：[thesis_neutral_postmerge_reentry_recovery_v3_authorized.yaml](../config/experiment/thesis_neutral_postmerge_reentry_recovery_v3_authorized.yaml)
- 输出根：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_v3_runtime_feasibility`
- 最终判定：`implementation_failure_not_experimental_evidence`

V3 预检已通过，确认 48 个冻结场景、三对手、144 条计划 record、严格 JSBSim、完整授权哈希和全新输出根。正式运行后，在 expert / midrange_balanced 的第 5 条场景首次进入 post-handoff 基线路径时停止，未完成 144-record universe。

## 失败事实

运行复用 `NeutralPostMergeEpisode.step()` 时，底层实现读取
`config["proposed_training"]["geometry_reward"]`，而 V3 的冻结 runtime-feasibility 配置只保留了最大步数，未提供该训练期 reward 字段。因此运行抛出：

```text
KeyError('geometry_reward')
```

失败点位于 `src/uav_vpp_guidance/training/thesis_defext_rangeext_pilot.py:477`。这不是 JSBSim backend fallback、物理不稳定、策略性能或样本覆盖结论，而是 V3 runner 与其复用 episode 引擎之间的未覆盖配置契约。

## 已保留的输出证据

| 文件 | SHA-256 | 含义 |
|---|---|---|
| `authorization_preflight.json` | `f20deaa5bab6bb8284f34bac0d122f2afbf22183742c1b13eb7f327feecfbc57` | 已授权、绑定 SHA 与配置哈希 |
| `resolved_config.json` | `72b9059ca8ff01cba27970b7f572d18db69f4832ffae9c621a8cdf3bc496fdf4` | 运行时解析配置快照 |
| `runner_failure_manifest.json` | `424da8551fd6b92d39aade2343901cfba083803f71108067271fc5aa2aaad8a3` | `completed_record_count=4` 与 `KeyError` |

仅生成了 4 条 expert / midrange_balanced partial record；它们都没有到达 handoff，`qualifying=false`，且 `valid_target_steps=0`。`runtime_feasibility_gate.json` 和 `run_manifest.json` 均未生成。它们不能作为 runtime-feasibility、训练、能力或论文证据。

## 冻结边界与后续约束

1. V3 已停止，不得补充 `geometry_reward`、修改 runner、替换场景或重跑同一 Source ID。
2. 不得从 4 条 partial record 推断任何 opponent、VPP、phase 或物理可达性结果。
3. 若仍需研究该问题，必须另起新的 V4 设计与 Source ID，并在授权前证明纯 record 路径不依赖训练 reward，或将完全不影响动作和 telemetry 的运行字段纳入静态 contract/preflight。
4. V4 需保持与 V3 设计资产、候选策略、训练和论文 heldout 结论相互隔离，且重新获得单独授权后方可执行。
