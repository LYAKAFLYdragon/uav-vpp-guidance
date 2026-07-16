# Temporal-Corridor Transfer V1 执行无效归档（2026-07-16）

## 结论

`THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V1` 已在授权配置下尝试执行一次，
但在创建第一条 `frozen_fixed_head_on / expert` 记录的 JSBSim 环境时即中止。
本次运行未进入 handoff、未调用 temporal-corridor action compiler、未执行策略控制，
也没有产生 episode telemetry。因此它是**工程执行无效证据**，不是正面或负面的
物理传递、可控性或性能证据。

预注册的单次执行规则仍然有效：不得在本 Source ID 下修改配置或实现后重跑，
不得用本次结果声称 corridor 的有效或无效，也不得据此启动训练。

## 已验证事实

| 项目 | 结果 |
|---|---|
| 实现 SHA | `c6ccb013b28ace4cb05acc99c7bd628f7554e946` |
| 预注册 Source ID | `THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V1` |
| 计划矩阵 | 3 methods x 3 opponents x 12 scenarios = 108 records |
| 预飞 | 通过：授权、输入哈希、清洁 worktree 和空输出根均通过 |
| 实际完成记录 | 0 |
| 失败文件 | `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/temporal_corridor_transfer_v1/failure.json` |
| 失败异常 | `KeyError('execution')` |
| 独立 verifier | 未形成 gate：首个预期 episode JSON 不存在 |

`failure.json` 同时保留已解析的授权配置，因而可确认失败发生在环境初始化，
而非结果写入、统计 gate 或后处理阶段。

## 根因

runner 将已解析的 temporal-corridor 设计配置传入
`build_record_only_env(...)`。后者调用
`thesis_neutral_postmerge_reentry_recovery_pilot._runtime(config)`，其旧接口要求：

```python
config["execution"]["runtime_template"]
```

但本 Source 的冻结设计把该输入表示为：

```yaml
inputs:
  runtime_template: config/experiment/train_thesis_five_state_shared_skills_geometry_v1.yaml
```

并将授权 overlay 的 `execution` 内容有意保存为 `_execution`，所以没有可供旧函数读取的
`execution` 键。预飞仅验证文件哈希、授权和输出根，未构造一次真实 JSBSim record-only
环境，未能拦截这一接口不匹配。

## 后续约束

若继续研究 temporal-corridor physical transfer，必须使用**新的** Source ID、全新的不重合
manifest 与空输出根；不得覆盖本目录或重用本次 108-record 授权。

新 Source 设计前必须先完成下列工程修复与 smoke 验收：

1. 显式把已冻结的 `inputs.runtime_template` 解析为 runtime mapping，或调整
   `build_record_only_env` 接口以接收该 mapping；不允许依赖隐式 `config.execution`。
2. 新增一个不占正式 Source ID 的 build-only smoke test，对三个对手各构造一次 strict-JSBSim
   record-only environment，并验证 predictor、VPP、guidance 和 PID telemetry 可读。
3. 将上述 env-construction smoke 纳入正式 preflight；只有 3/3 通过，才允许新的物理传递
   Source 进入单次执行授权。

本次 V1 不改变 V16 已验证的 run-in/phase/telemetry 结论，也不改变 temporal-corridor
read-only feasibility 结论；它只说明两者在该 runner 中尚未完成可执行接口闭合。
