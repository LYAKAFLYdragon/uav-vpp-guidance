# V14 Temporal-Corridor 实现失败归档

**Source ID：** `THESIS-FIVE-STATE-REENTRY-TEMPORAL-CORRIDOR-V14`
**授权提交：** `17f30469dbf51805281d67097d496b7ecce4c336`
**实现提交：** `91683a0bfde3968bcd71e95acdff23bdd2346fad`
**最终判定：** `implementation_failure_zero_records_not_scientific_evidence`

## 事实

V14 的 design preflight、逐文件 hash 授权和 144 条 strict-JSBSim record 宇宙均已通过。正式运行在第一条 `expert / frozen_fixed_head_on` record 的连续 run-in 后、首次进入 V14 handoff 66-D 组装时 fail-closed：

```text
KeyError('profile')
```

输出根仅包含以下两份文件：

| 文件 | SHA-256 | 含义 |
|---|---|---|
| `authorized_config.json` | `abfff47b4326f4e47dc83195cdd336d38825cb78510bd568edec63b4bb5ea6ef` | 已解析的 V14 冻结配置和授权白名单快照。 |
| `failure.json` | `5bbb1841530dc4b5dd05bf0f1dc69ab876721dba3a6958cb46c650403357523c` | 原始异常及 `completed_records=0`。 |

没有 episode ledger、handoff、66-D window、VPP/guidance/PID telemetry、gate、候选结果或对手比较。因此 V14 不是关于物理可达性、时间走廊有效性、安全性、奖励或训练价值的正/负证据。

## 根因

V14 为了允许 step 0--4 使用 `range_extension`、step 5--9 使用 `reentry_preparation`，从 `semantic_compiler` 中移除了固定 `profile` 字段。可是继承的连续 66-D 提取器 [`_observation66d`](../scripts/run_thesis_five_state_reentry_v11_relative_frame_semantic.py) 仍把该字段作为 observation contract 的一部分：它调用 `compile_profile(config["semantic_compiler"]["profile"])`，并把固定 profile 的 target vector 与 weight vector 写入 66-D 向量。

因此，错误发生在动作注入、VPP/guidance/PID、终端判定或对手行为之前。它不是 JSBSim backend failure，不是物理相位覆盖失败，也不能被解释为“短走廊没有控制权威”。

## 冻结决定

1. 保留上述零记录输出，不删除、不覆盖、不将其纳入论文或汇总统计。
2. V14 不得补加固定 profile、修改 runner、替换场景、重建输出根或重跑。
3. V14 的预注册门槛、方法矩阵和 V13 的单点接口负证据均保持不变；训练、奖励 rerun、单/四技能训练、combat finetune、P5--P7 继续锁定。

## 若继续研发的唯一兼容方向

任何后续方案必须使用全新的 Source ID、全新 manifest 和新输出根，并在 JSBSim 运行前验证两件事：

1. **编码条件固定且可解释：** 66-D contract 使用明确冻结的 encoder profile，例如 `range_extension`，仅用于构造目标/权重输入，不随时间走廊切换。
2. **评价与动作条件可独立变化：** 每一步仍显式记录 time-indexed objective profile 与 action profile，证明动态走廊的比较不靠改变 observation 维度、history、P3 embedding 或未来信息。

这一修复属于新接口契约，不能伪装为 V14 补跑。只有它通过新的 design/runtime preflight 后，才可另行决定是否授权一个新的非学习 probe。
