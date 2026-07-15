# V3 Runtime-Feasibility 独立静态复核

**审计对象：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V3`  
**结论：** `PASS - design and frozen-baseline execution implementation ready for separate execution authorization`  
**执行状态：** 未运行 JSBSim，未训练，未创建 V3 output root。

## 复核项

| 项目 | 结果 | 证据 |
|---|---|---|
| 设计权限 | PASS | 所有 V3 execution/training/four-skill/combat-finetune 权限均为 `false`；`--execute` 返回 exit 2。 |
| 场景再生 | PASS | 从 builder 临时重建 manifest，和受控 manifest 的文件 SHA-256 均为 `0dbc1748e528dadd6112577f095d05a11a774eb33bc8b9f67c896e22ff49aa30`，逐字节一致。 |
| manifest 身份 | PASS | 48 个唯一 evaluation `pair_key`；4 family 各 12 条，均为 2 package x 3 height x 2 mirror。 |
| 历史隔离 | PASS | manifest 记录的 V1/V2、B2--B6、`heldout240`、`dev30`、历史 `heldout60`、phase-v2 均为 physical/seed intersection `0/0`，并在 preflight 复核源文件 hash。 |
| train/evaluation 分离 | PASS | train key 为 `neutral_postmerge_reentry_v3_train::seed=2026071701`，不含 pair metadata，`paired_delta_eligible=false`。 |
| 三对手 gate | PASS | coverage evaluator 要求冻结 manifest 的完整 144 个 `opponent x pair_key` universe；缺失、重复、额外记录或 family mismatch 都 fail-closed。 |
| 执行 overlay | PASS | 未来 overlay 必须哈希绑定 canonical design、clean implementation SHA 与 code whitelist；除 authorization/execution/预声明 output creation 外不得修改冻结方法字段。 |
| 运行时 provenance | PASS | serializer 只写 frozen-baseline feasibility record，拒绝 backend/telemetry failure，记录 V3 source 与被复用物理引擎 provenance。 |
| Reset/phase receipt | PASS（静态） | future runner 在同一 frozen engine 内拦截即时 reset state，调用 scenario-application receipt，并重放每个 ledger raw-SI phase；依赖导入与 serializer failure path 均通过测试，尚未执行 JSBSim。 |
| 资产与容量 | PASS | runtime template、registry、P3、三个 opponent 与 frozen run-in specialist hash 通过；可用空间约 291 GB，高于 120 GB 门槛。 |
| 定向回归 | PASS | V3 + V2 pairing/runtime 共 `11 passed`；另有 `py_compile` 与 `git diff --check` 通过。 |

## 重要边界

这份复核不证明任何 family 在物理上可达，也不证明 `reentry_recovery` 会优于
head-on/crossing specialist。48 场景尚未进入 JSBSim；manifest 的几何标签和 B6 的
历史 reachability 结果都不能替代 V3 future engine 的真实 handoff/66-D/P3/VPP
telemetry。

V3 仅比 V2 前进一步：如果未来实际运行，身份错配和不完整 record 不能再被解释为
性能数字。它仍不具有执行权限。

## 下一步门槛

V3 execution runner 已完成独立静态复核但尚未被授权。唯一下一步是单独授予一次性
V3 runtime-feasibility execution 权限；运行时必须在同一 frozen pilot engine 中逐
episode 保存 scenario receipt、raw-SI phase replay、handoff state hash、66-D
observation、3-D VPP action、prediction、VPP/guidance/PID response、terminal reason
与完整 manifest identity。届时才允许运行这 144 条 non-learning record。

即使该运行通过，也只允许起草一个新的单技能 pilot 预注册；四共享技能、combat
finetune、高层 PPO、P5/P6/P7 和 formal heldout 继续锁定。
