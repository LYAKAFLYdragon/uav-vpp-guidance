# 控制与制导整改 P0 阶段总结

- **阶段状态：`physical_unavailable`。** 未执行任何训练、正式实验或 strict-JSBSim 闭环；该状态不能归因于控制器失败。
- **实际完成：** 确认指定 worktree、分支和 clean SHA；原样同步执行清单；核对审计、ledger、preservation baseline、配置哈希、场景矩阵、现有 portable bundle 与登记 checkpoint。
- **命令与验证：** `git rev-parse/status/worktree list`；登记 checkpoint 的跨 worktree 只读搜索与 SHA-256；`python -m pytest tests/test_control_guidance_remediation_phase0.py -v`；10 个既有 portable bundle 完整性验证。
- **生成物：** `outputs/control_guidance_remediation_v1/p0_evidence_freeze/` 及其 portable bundle；本总结；更新后的执行清单。
- **通过门：** 目标 worktree/分支/SHA 和初始 clean 状态；4 个冻结配置哈希；10 个既有 portable bundle 验证。
- **发现：** 当前 worktree 缺少全部四个登记 checkpoint；外部仅发现未登记的 end-to-end 候选和 SHA 不匹配的预测器候选；矩阵只有 expert、end_to_end 两个独立对手，curriculum 解析为 end_to_end；冻结审计报告和 preservation-manifest 自哈希也与 `baseline_freeze.json` 不一致。
- **问题分类：** checkpoint/第三对手缺失为 `physical_unavailable`；冻结哈希不一致为证据接口错误；没有数学错误、控制器安全失败或对手性能结论。
- **是否允许进入下一阶段：** 不允许。P0 验收要求关键输入可独立读取，当前不满足；不得以未登记 checkpoint、simple backend 或旧 telemetry 替代。
- **下一命令与验收：** 仅在恢复四个已登记且 SHA 一致的资产、或经审核发布新版 asset manifest，并补齐第三独立对手后，运行 `python scripts/run_jsbsim_hrl_comparison.py --config config/experiment/jsbsim_hrl_comparison.yaml --run-id p0_smoke_<opponent> --preset main --opponent-stage <opponent> --run-status smoke --seeds 0 --n-episodes 1 --backend jsbsim --device cpu`。每个 opponent 必须完成一条 episode，telemetry 非空、hand-off 连续、无 reset/fallback，且 `backend == "jsbsim"`、`backend_fallback_occurred == false`。
