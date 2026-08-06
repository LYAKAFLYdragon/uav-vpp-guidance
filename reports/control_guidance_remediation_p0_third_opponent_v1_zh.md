# P0 第三独立对手与 dry-run 阶段总结

- **阶段状态：`pass`（仅限接口与可复现性）。** 这不是 strict-JSBSim episode、物理闭环或性能证据。
- **实现：** 新增 `ScriptedEvasiveVPPOpponent` 和 `scripted_evasive_vpp` registry/CLI stage。规则只使用当前角色反转后的 own/target state 与当前相对位置，输出规范 3D VPP `[dx, dy, dz]`；每轴 clip 到 `[-1, 1]`。它实行“远离当前 LOS + 固定左向 break”，不使用 future、oracle 或 terminal 信息。
- **保持不变：** legacy 默认路径、F01/F02、奖励、PID、F04 限幅、F15 sin/cos 和 observation schema；未知对手 stage 继续按既有两维编码映射为 `[0, 0]`，没有破坏旧 checkpoint 输入维度。
- **验证：** `PYTHONPATH=src python -m pytest tests/test_opponent_controller.py -v`（7 passed）；runner dynamic factory + P0 preservation（5 passed）；`TestArtifactBundle`（3 passed）；`py_compile` 和 `git diff --check` 通过。
- **工件：** dry-run 源输出在 `outputs/control_guidance_remediation_v1/p0_third_independent_opponent_dry_run/p0_third_independent_opponent_dry_run_20260727/`；portable bundle 在 `.kiro/specs/control-guidance-remediation/evidence/p0_third_independent_opponent_dry_run_bundle_portable/`。其中 `backend=jsbsim`、`strict_backend=true` 是 resolved configuration，而 `dry_run=true`，零 episode、无 telemetry；bundle hash 与 artifact contract 已验证。
- **附带修复：** `ArtifactBundle` 现能递归复制并验证契约声明的目录，因此保留 runner 所需的 `manifests/`。
- **阻断：** E 盘 32 个相关根目录的 3,726 个 checkpoint 盘点未找到 VPP policy 登记路径或 predictor SHA `a7c…4491`；故不复制未登记候选，也不执行 three-opponent strict-JSBSim smoke。P0 总状态仍为 `physical_unavailable`。
