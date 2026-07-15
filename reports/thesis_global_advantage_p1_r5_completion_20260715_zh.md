# P1 R5 可重复性 Gate 完成归档

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R5-JSON-TELEMETRY-REPRO-V1`
**状态：** `runin_protocol_reproducible`
**范围：** 30 个冻结 dev 场景 x 3 对手 x 3 顺序 repeat = 270 条 fresh-process JSBSim episode

## 通过结论

R5 的 90/90 `opponent x scenario_signature` cell 全部通过。每个 cell 含 3 个不同 OS process 的 repeat；完整 reset envelope、首个 reference/opponent action、逐步 trajectory、first-pass（或末步）boundary envelope 与 terminal reason 均严格相等。所有 telemetry 完整，未发生 backend fallback 或 runtime prediction fallback。

## 完整性复核

| 项目 | 结果 |
|---|---:|
| Raw episode artifact | 270/270 |
| expert / end_to_end / independent_ppo_vpp | 90 / 90 / 90 |
| 比较 cell | 90，每 cell 3 repeat |
| Artifact SHA-256 mismatch | 0 |
| 临时 `.tmp` 文件 | 0 |
| Gate failure count | 0 |
| Clean worktree | true |

| 汇总 artifact | SHA-256 |
|---|---|
| `p1_r3_gate_summary.json` | `71d00a4997d277ccc179f3b8252a8078081a39ebe24c3858ee758448581fcc0b` |
| `run_manifest.json` | `9195f0d7c38cf67ec6bd641e7b09f060529143681cec4e3dc475261f104d7631` |

完整输出根为 `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_r5_json_telemetry`，约 513.8 MB。它是 P1 的正向可重复性证据，但不含新技能或高层 PPO 的性能比较，不能支持“全场景优势”或进入正式 held-out 的结论。

## 后续门

P1 通过仅解锁 P2：冻结连续训练分布、dev30/heldout240 的不相交性、物理可达性与三对手 capability card。P2 未通过前，仍不得训练四技能或高层 PPO。
