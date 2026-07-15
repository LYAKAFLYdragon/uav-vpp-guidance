# V5--V7 Re-entry 标签合同审计与奖励试验决策

## 结论

当前不能把 V5--V7 中的 `re_entry = 0` 解释为“控制器不会掉头”“训练不够”或“奖励不足”。
只读重算发现，旧 `PhaseTracker` 的阈值以米和米/秒定义，却接收了归一化的
16-D 观测值：`range_m` 实际被除以 `5000`，`range_rate_mps` 实际被除以 `200`。
因此 `range <= 1000 m` 在重置后几乎立即成立，而 `range_rate <= -25 m/s` 在归一化
空间中不可能成立。旧 ledger 的零 `re_entry` 是标签/单位合同错误。

此外，旧 tracker 即使改为物理单位，也没有要求“首次交会已经发生”；它会把首轮进入
1000 m 后的正常接近误标为 re-entry。新的定义必须同时满足：物理量、第一次交会已锁存、
之后再次以不高于 -25 m/s 的速度接近。

## 已完成的只读复核

来源：

- V5: `five_state_raw_si_v5_observability_r1`
- V6: `five_state_reentry_v6_control_matrix`
- V7: `five_state_reentry_v7_teacher`
- 可重复审计输出：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_terminal_horizon_audit_r3`

| 范围 | 首次交会 | 首次交会后中位时长 | ledger re-entry | 物理 re-entry 候选 | 有效候选 66-D step |
|---|---:|---:|---:|---:|---:|
| V5，三对手 | 177 / 180 | 34.8--37.8 s | 0 | 139 | 8,241 |
| V6，二冻结控制器 x 三对手 | 72 / 72 | 40.5--42.8 s | 0 | 72 | 5,190 |
| V7，两种掉头增益 x 三对手 | 72 / 72 | 40.1--42.0 s | 0 | 72 | 5,462 |

V6/V7 合计 `144/144` 条记录在首次交会后存在满足物理再接近条件的有效 66-D 样本，
共 `10,652` step；同时，全部记录仍有 ledger `re_entry=0`。V6/V7 的首次交会后
中位时间为 `41.6 s`，记录到的 post-pass 饱和 step 与 ego crash/OOB 均为 `0`。
因此不能把问题归于“时间不够”“刚交会就被规则中止”或“飞控保护阻止掉头”。

这里的“物理 re-entry 候选”仅指原始遥测满足新定义的输入条件，不代表新技能已经训练、
性能更高或可以写入论文主结论。

## 已落地的保护

- 新增 `src/uav_vpp_guidance/evaluation/raw_si_reentry_phase.py`。
- 新 tracker 只接受米和米/秒，且在 raw-SI first-pass 锁存前绝不输出 `re_entry`。
- 旧 `PhaseTracker` 和 V5--V7 文件不改写，保留为历史研发证据。
- 定向测试覆盖“未交会不得 re-entry”“首次交会后再次接近才可 re-entry”“非有限输入拒绝”。

## 当前决策

**不立即增加胜利奖励，也不立即训练。** 原因不是奖励一定无效，而是训练将使用错误的
phase 标签；这会让任何正负奖励的结果都无法解释。此时加大奖励、增加训练步数或扩展
action grid 都是在掩盖数据合同问题。

下一步只允许一个新的非学习 `V8 phase-contract probe`：使用全新、不与 V5--V7 重合的
物理场景，接入 `RawSIReentryPhaseTracker`，显式记录 raw range/range-rate、归一化观测、
first-pass 事件、phase、66-D、3-D action、VPP、PID、终止原因和 provenance。

## V8 验收门槛

每个对手分别通过，禁止合并：

1. 至少 2 条连续 strict-JSBSim episode。
2. 至少 20 个新的 `re_entry` 66-D -> 3-D VPP step。
3. 至少 2 个镜像方向和 2 个独立场景 signature。
4. 每步满足 `raw_range = normalized_range x 5000`、`raw_rate = normalized_rate x 200`；无 history padding、reset、backend fallback 或 checkpoint fallback。
5. 独立 replay 必须逐步复现 first-pass 与 re-entry 标签。
6. 出现标签不一致则冻结为合同失败；不以修改阈值、奖励或训练步数补救。

只有 V8 在三个对手下都通过，才可以起草一个独立的奖励 pilot 预注册。

## 后续奖励 Pilot 的边界

如果 V8 通过，奖励改动应仅作为一个新的训练变量，固定其余任务、VPP、guidance、PID、
对手、场景和 checkpoint 初始化。最小终端项可写为：

`r_terminal = +b_win * I(win) - b_loss * I(loss) - b_draw * I(draw)`

其中 `ego_crash_or_out_of_bounds` 只能计入 `loss`；`target_crash_or_out_of_bounds` 是合法
胜利，不能误加惩罚。`timeout_hp_advantage`、`timeout_hp_disadvantage` 和 `timeout_draw`
必须在预注册中明确其是否分别计入 win、loss、draw，不能事后按结果选择。奖励 pilot 的
核心机制指标仍是 re-entry 连续性、range opening、能量恢复和攻击区暴露；win rate 只作
次要结果，并受 crash/OOB 不得高于 baseline +5 pp 的安全 stop rule 约束。
