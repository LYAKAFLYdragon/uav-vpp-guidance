# V16 动态参考路径 Probe 结果与 Go/No-Go 决策

**Source ID：** `THESIS-FIVE-STATE-REENTRY-DYNAMIC-REFERENCE-PATH-V16`  
**实现提交：** `de1e0abf72fbe113a061131aed8d39906a073523`  
**授权提交 / 运行 SHA：** `bfccfc84f9eb24ff88fbe5ccf23b5031410b3bc5`  
**输出根：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v16_dynamic_reference_path`  
**最终判定：** `mechanism_preflight_passed_not_a_performance_or_training_result`

## 1. 执行完整性

V16 在一个与 V1--V15 物理不相交的 12 场景 manifest 上完成了预注册的
`3 methods x 3 opponents x 12 scenarios = 108/108` 条 strict-JSBSim 连续记录。
每条 record 均从一次合法 reset 开始，使用 frozen fixed-head-on specialist run-in，
不在 handoff reset，不训练，不记录 reward，不修改 VPP generator、guidance、PID 或 terminal 语义。

每个 path step 只使用当前 simulator ego/target state、当前 range-rate 和冻结 predictor 的单一预测位置。
为避免动作后 state 与动作前 prediction 混用，runner 在动作前同步 predictor history，环境随后复用该 history 计算同一步 VPP anchor；这不是 future-state 注入。

独立 verifier 已重放 108 条 record 的：三 anchor 构造、range-rate progress、所选 anchor/profile、固定 66-D profile target/weight 段、path-to-action receipt 和未合并对手 gate。结果为 `passes=true`。

| Artifact | SHA-256 |
|---|---|
| `dynamic_reference_path_gate.json` | `eef057a189fa89c02555751ea2addda4b04c4d70da0f264480b77126c1794368` |
| `run_manifest.json` | `6adafbb9e90df682acfe1444b1ce37c976f1891b6304d29c4d763cda2ccd3064` |
| `independent_verification.json` | `e06f526dd408a2981093e5109d9a4e4a68270e720153097c3f41c8461307280f` |

## 2. 预注册机制门

通过条件不是胜率，而是每个 opponent 单独满足：至少 4 个 handoff-hash 一致的 qualifying pair、有效且非零的 anchor/VPP/guidance receipt、ego crash/OOB delta 不超过 `+0.05`，且 escape range-rate error 不恶化超过 `+0.02`。至少两个 opponent 通过才允许起草新的训练 pilot；禁止 pooled gate。

| Opponent | Qualifying pairs | Anchor signal steps | Mean paired VPP delta (m) | Mean guidance delta | Escape rate-error delta | Ego crash/OOB delta | Mechanism |
|---|---:|---:|---:|---:|---:|---:|---|
| expert | 5 | 79 / 100 | 1340.04 | 0.2540 | -0.00865 | 0.0000 | Pass |
| end-to-end | 3 | 45 / 60 | 1210.12 | 0.1867 | -0.00039 | 0.0000 | Insufficient pairs |
| independent PPO/VPP | 6 | 97 / 120 | 1408.52 | 0.2494 | -0.00144 | 0.0000 | Pass |

因此 `expert` 与 `independent PPO/VPP` 独立通过，`end-to-end` 仅有 `3 < 4` 个 qualifying pair。V16 的机制 gate 为 `passes=true`，但该结论只意味着动态参考路径没有被现有 3-D VPP / guidance / PID 执行链吞没，并非对三对手的普适结论。

## 3. 只读结果审计与边界

`progress_triggered_reference_path` 的有效窗口中，anchor 选择分别为：expert `79/0/21`、end-to-end `44/1/15`、independent PPO/VPP `96/1/23`（escape / connector / predicted-reentry）。因此路径进度并非恒定在第一锚点，确实进入了 predicted-reentry 选择。

但完整路径不能被声称为“精确跟踪”。路径 action 在有效窗口被现有 3-D VPP 范围裁剪的次数为 expert `74/100`、end-to-end `32/60`、independent PPO/VPP `81/120`。该裁剪是冻结 action head 的物理/接口边界，不能通过事后扩大动作范围或同一 Source ID 补跑来掩盖。

在与 frozen fixed-head-on 具有相同 handoff hash 的有效配对中，terminal reason 完全相同：expert `5/5`、end-to-end `3/3`、independent PPO/VPP `6/6`。因此 V16 没有提供 win-rate、kill-rate、terminal superiority、长期 re-entry 成功率或训练收益的证据；它也不改变任何 canonical 或论文主结果。

## 4. 正式后续决策

1. 允许起草一个全新 Source ID 的 **dev-only path-conditioned policy feasibility pilot**，因为 V16 已满足其预注册的接口机制前置条件。
2. 不允许从 V16 直接启动四共享技能、combat finetune、P5/P6/P7 或论文级 held-out 评估；V16 没有达到这些阶段所需的性能证据。
3. V17 设计必须把 V16 的 action clipping 显式作为预注册机制指标，冻结新的不相交 dev manifest，并以三个 opponent 分开报告 paired outcome、terminal mix、path selection、clipping、VPP/guidance/actual-response telemetry。
4. V17 的首轮 gate 应先回答“在不增加 ego crash/OOB 的条件下，path-conditioned policy 是否相对 frozen fixed-head-on 改善预注册的脱离/再进入机制指标”，而不是宣称全部空战场景优越。只有该 dev gate 成立，才可以另行预注册 formal held-out。
