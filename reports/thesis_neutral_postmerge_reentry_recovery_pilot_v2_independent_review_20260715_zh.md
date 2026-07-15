# Neutral Post-Merge Pilot V2 独立设计与 Adapter 复核

**审计对象：** `51f605fff89e6d6c5739b65c7cd99c0e6a05a98d` 加本复核测试
**结论：** `PASS - ready for separate execution authorization only`

## 通过项

- V2 `dev12`/`heldout24` 各自有 12/24 个唯一 `pair_key`，且真实 YAML payload hash 可再生。
- 每个 V2 pair key 来自 `geometry_cell_id + scenario_seed`；篡改已序列化 key 即使重算 manifest payload SHA，仍会 fail-closed。
- V2 wrapper 不直接读取 V1 的 `metadata.scenario_signature`；V1 的旧 metadata 假设不会成为 V2 runtime 的配对来源。
- scoped adapter 仅在 V2 run/preflight 生命周期内替换 V1 engine 的 source、episode ledger、result serializer 与 authorization validator，退出后完整恢复原对象。
- V2 design preflight 再次验证 P3、runtime registry、输出根、磁盘容量和所有 authorization permission 为 false。

## 不构成的结论

没有启动 JSBSim、训练、baseline 或 heldout。因此这项复核不证明 V2 的物理可行性、性能、安全或新技能库缺口；它只说明 V2 的输入、配对和软件隔离已达到可单独授权的状态。

V2 必须以新的 authorized config、新的 output root 和新的 clean SHA 单独执行；V1 授权已消费，不能继承。
