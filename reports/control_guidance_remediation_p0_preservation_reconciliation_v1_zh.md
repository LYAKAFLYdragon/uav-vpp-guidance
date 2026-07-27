# P0 preservation manifest 哈希证据链收口

- **阶段状态：`pass`（仅限证据接口收口）。** 此结果不恢复 checkpoint，不产生控制器或 strict-JSBSim 物理结论。
- **完成项：** 保留历史 `baseline_freeze.json` 不变；新增 superseding freeze 与原始字节 forensic record。
- **核验方法：** 以 Python 捕获 `git show <commit>:<path>` 的原始 stdout bytes 并计算 SHA-256；没有使用会改写换行或编码的文本管道。
- **事实：** V1 当前 `HEAD`、index 和 worktree 均为 `7deac2330bc400574ed52a0d1b6809b2900688a03c8b11ffc3bc97c8785032ac`（36,036 bytes，LF-only）；旧 freeze 声明的 `4f427f…9368` 与 source revision 的 `25970c…74d2`、V1 canonical blob 及所检查的全部可达历史版本均不匹配。
- **生成物：** `.kiro/specs/control-guidance-remediation/baseline_freeze_superseding_20260727.json`；`.kiro/specs/control-guidance-remediation/preservation_manifest_forensic_record_20260727.json`。
- **门结论：** 旧 freeze 继续作为不可变历史记录；后续 preservation 测试以 superseding freeze 的 canonical V1 SHA 验证。P0 仍为 `physical_unavailable`，因为资产门与三对手 strict-JSBSim smoke 尚未满足。
