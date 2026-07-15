# V18 同包线相位遥测审计预注册

## 研究问题

V17 在 expert 与 independent PPO/VPP 对手下形成了充分的 74-D path-context window，却在
end-to-end 对手下只形成 4 条 handoff。V17 的 record 保存了 phase、range 和 range-rate，但没有
逐步保存双方速度、高度、specific energy、动态 taxonomy 或 VPP/guidance/PID telemetry。因此，
目前只能确认“post-first-pass phase occupancy 不足”，不能判断是相位标签口径、对手行为，还是共同
执行链导致。

V18 只补足这一**可观测性缺口**。它不是性能实验、训练数据收集、奖励实验或场景扩展；不会授权新的
skill、profile、PPO、reward、VPP、guidance 或 PID 改动。

## 固定范围

- Source ID：`THESIS-FIVE-STATE-V18-PHASE-TELEMETRY-AUDIT`。
- 运行方式：strict-JSBSim，连续 episode，冻结 fixed-head-on controller，3-D VPP action 不变。
- 对手：expert、end-to-end、independent PPO/VPP，分别记录，绝不 pooled。
- 场景：故意复用 V17 的完整 24-scene manifest，三对手共 72 records。理由是诊断 V17 缺失的
  telemetry，必须让相位占用与 V17 逐格可比；它不是 V17 的失败后补跑，也不重开 V17 gate。
- V17 基准：V17 gate 和 run manifest 的 SHA-256 已冻结在 design config。每对手 handoff 数和
  valid path-context step 数必须精确重现，否则 V18 冻结为不可比较证据。

## 每步记录

每个 high-level step 必须写入原始 16-D 观测及可复算 SI 量：range/range-rate、ATA、AA、LOS、
双方速度和高度、双方 specific energy、动态 taxonomy、raw-SI first-pass phase、VPP world offset、
guidance command、实际 `n_z` 与 saturation flags。不得写入 future state、padding、训练 reward 或
候选策略动作。

## 预注册门与停止规则

1. 三对手各 24 records，且无 backend/prediction/checkpoint fallback。
2. V17 的 phase/handoff/path-context 覆盖逐对手精确重放；任何不一致即说明 V18 不可比。
3. 所有必需遥测字段每步完整、有限、单位可复算；否则是 observability-contract failure。
4. V18 的 ego crash/OOB 只记录，不作为通过、选模型或调整控制器的依据。
5. 任一门失败，冻结为负证据；不得在同一 Source ID 下替换场景、加步数、调 reward 或重跑。

## 可回答与不可回答的问题

若 V18 通过，它可以比较三对手在 first-pass 后的速度、能量、几何与执行响应，并将 V17 的
end-to-end 覆盖不足严格缩小为“对手条件下的可观察 phase occupancy 模式”。它仍不能单独证明任何
新路径、奖励或技能会提升 outcome。

只有 V18 通过并显示存在可重复、完整遥测的目标 phase motif，才可以设计全新 Source ID 的低层
机制 pilot；届时还必须另设 train/dev/heldout 划分和安全门。V18 现在保持 `design_only_not_authorized`，
不得执行。
