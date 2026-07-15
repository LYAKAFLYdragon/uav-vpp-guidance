# V5 五态势 Raw-SI 相位可观测性结果

> **2026-07-15 phase-contract correction.** This report remains valid only as
> evidence for continuous JSBSim recording and post-first-pass 66-D
> observability. Its legacy `PhaseTracker` received normalized observations
> although its thresholds are expressed in metres and metres per second; it
> therefore cannot support a claim of zero physical `re_entry`. See
> `reports/thesis_five_state_v5_v7_phase_contract_audit_20260715_zh.md`.

## 结论

`THESIS-FIVE-STATE-RAW-SI-PHASE-OBSERVABILITY-V5` 的 R1 record run 完成并通过独立复核。五种初始态势、三类对手和固定 head-on reference controller 形成完整 `60 x 3 = 180` 条 strict-JSBSim record。所有 `initial state x opponent` 单元均通过预注册数据契约门槛，故结论为：**五态势下的连续 66-D/3-D VPP 观测与执行数据契约已经建立。**

这不是性能优越性、技能有效性、飞控安全认证或论文主结果。所有控制动作均来自冻结的 legacy fixed head-on specialist；`front_intercept` profile 仅用于构造并记录 66-D 参考向量，不改变 action，也不进行 profile-validity 筛选。

## 运行与复核

| 项目 | 事实 |
|---|---|
| Source ID | `THESIS-FIVE-STATE-RAW-SI-PHASE-OBSERVABILITY-V5` |
| 正式 R1 输出 | `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_raw_si_v5_observability_r1` |
| Manifest | `V5-MANIFEST60`，5 initial states x 2 packages x 3 heights x 2 mirrors |
| 对手 | expert、end-to-end、independent PPO/VPP，严格分开报告 |
| 控制器 | frozen fixed head-on，非学习、无 routing、无 reward 记录 |
| Record 完整性 | `180/180`；每格 `12/12`，无 backend/reset/history/checkpoint contract failure |
| 独立验证 | `independent_verification.json` 为 `pass`；重放 raw-SI event 并重算 15 格 gate |
| R1 前身 | 初版在首条 record 的 telemetry hash 写入前失败，completed records=`0`；仅 serializer implementation failure，详见 `thesis_five_state_raw_si_v5_initial_implementation_failure_20260715_zh.md` |

初版失败目录被保留，R1 使用全新 output root。R1 的唯一实现变化是 ndarray/numpy telemetry 的确定性哈希化；环境、场景、三对手、P3 encoder、VPP、guidance、PID、门槛与动作控制器均未改变。

## 预注册 Gate

门槛按每个 `initial state x opponent` 独立判断：至少 2 条 raw-SI first-pass episode、20 个 post-event 有效 66-D step、2 个镜像和 2 个高度条件；禁止对手池化。

| Opponent | Advantage | Head-on | Disadvantage | Neutral | Crossing-entry |
|---|---:|---:|---:|---:|---:|
| expert：event episodes / post-66D steps | 12 / 1671 | 12 / 2446 | 12 / 2038 | 12 / 824 | 12 / 2518 |
| end-to-end：event episodes / post-66D steps | 12 / 1143 | 12 / 2632 | 12 / 2107 | 10 / 525 | 12 / 2654 |
| independent PPO/VPP：event episodes / post-66D steps | 12 / 1642 | 12 / 2536 | 12 / 2198 | 11 / 809 | 12 / 2433 |

15 个单元均满足门槛。所有 post-event 66-D step 同时具有有效 prediction telemetry；因此本轮没有由 prediction fallback、history padding、非有限 action 或 JSBSim fallback 造成的缺口。

## 动态覆盖与限制

post-event 动态 taxonomy 在每个对手下都出现五种状态：

| Opponent | Advantage | Head-on | Disadvantage | Neutral | Crossing-entry |
|---|---:|---:|---:|---:|---:|
| expert post-event steps | 1020 | 1037 | 3083 | 2408 | 1949 |
| end-to-end post-event steps | 532 | 972 | 3466 | 2535 | 1556 |
| independent PPO/VPP post-event steps | 886 | 1159 | 2787 | 2864 | 1922 |

但这些 post-event steps 全部由冻结的 raw-SI PhaseTracker 标为 `post_merge`，没有 `re_entry` step。因此，V5 的正向发现只建立了**五态势 post-merge 可观测性**，没有建立可供 `reentry_recovery` 学习的跨对手 re-entry 数据支持。

## 决策

1. P4 v1 的正式 NO-GO 与既有 200k 训练负证据保持不变；不得重跑、加步数、改 reward 或放宽 gate。
2. combat finetune、P5 高层 PPO、P6 消融和 P7 heldout60 继续锁定。V5 不是它们的解锁证据。
3. 唯一合理的后续研发线是一个新的、独立且非学习的 **V6 re-entry reachability control matrix**：在全新、物理不相交场景中，仅比较冻结 fixed head-on 与 fixed crossing reference controller 是否能在 raw-SI first pass 后形成连续 re-entry，同时保存相同的 16-D/history/66-D/action/VPP/guidance/PID telemetry，并按三对手分别设 gate。
4. 若 V6 仍不能在三个对手下建立 re-entry 数据契约，应正式将五态势系统范围降为 post-merge interface，而不是启动四共享技能训练或声称 re-entry capability。
