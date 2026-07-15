# V12 Intent-Controllability 实现失败归档

**Source ID：** `THESIS-FIVE-STATE-REENTRY-INTENT-CONTROLLABILITY-V12`
**运行 SHA：** `c8feffb252be0ba3c7b10eb4d097d73c989614aa`
**最终状态：** `implementation_failure_zero_records_not_scientific_evidence`

## 事实

V12 授权运行在创建第一个 `CloseRangeTrackingEnv` 之前停止。输出根只包含 `authorized_config.json` 与 `failure.json`；`completed_records=0`，`episodes=0`，没有 JSBSim step、handoff、policy action、奖励、终端事件或结果 gate。

原始错误为：`KeyError('execution')`。`build_record_only_env` 需要 `config.execution.runtime_template` 解析冻结 runtime，而 V12 design config 错误地只保留了 `inputs.runtime_template`。原 preflight 只验证了输入文件 SHA，未验证运行时键存在，故没有在执行前阻止该错误。

## 结论与保留

这不是 V12 的正证据或负证据，不能用于判断六维 intent 可控性、对手差异、技能库、奖励或安全。原输出保持只读，不删除、不覆盖，不得在相同 Source ID、相同输出目录或相同授权 overlay 下补跑。

任何后继实现必须使用新的 Source ID、全新物理 manifest、全新输出根，并将以下条件放入 preflight 与单元测试：

1. `execution.runtime_template` 存在、可解析，且其路径和 SHA 与 `inputs.runtime_template` 一致。
2. 预flight 必须在创建输出根前验证这一 runtime contract。
3. 保持 V12 的 non-learning、三对手分开报告、66-D、raw-SI continuity 和 full-factorial telemetry 定义；不得把配置修复伪装成对 V12 的结果补跑。
