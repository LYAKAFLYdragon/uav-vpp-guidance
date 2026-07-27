# 控制与制导整改 P0 阶段总结

- **阶段状态：`physical_unavailable`。** 未执行任何训练、正式实验或 strict-JSBSim 闭环；该状态不能归因于控制器失败。
- **实际完成：** 确认指定 worktree、分支和 clean SHA；原样同步执行清单；核对审计、ledger、preservation baseline、配置哈希、场景矩阵、现有 portable bundle 与登记 checkpoint。
- **命令与验证：** `git rev-parse/status/worktree list`；登记 checkpoint 的跨 worktree 只读搜索与 SHA-256；`python -m pytest tests/test_control_guidance_remediation_phase0.py -v`；10 个既有 portable bundle 完整性验证。
- **生成物：** `outputs/control_guidance_remediation_v1/p0_evidence_freeze/` 及其 portable bundle；本总结；更新后的执行清单。
- **通过门：** 目标 worktree/分支/SHA 和初始 clean 状态；4 个冻结配置哈希；10 个既有 portable bundle 验证。
- **发现：** 当前 worktree 缺少全部四个登记 checkpoint；对 E 盘 32 个相关根目录的 3,726 个 loose checkpoint、25 个 ZIP/TAR/TGZ/TAR.GZ 容器和 11 个本地 Git/LFS 元数据的只读 SHA-256 盘点均未找到 `prediction_vpp` 登记路径或 predictor 冻结 SHA。no-prediction/end-to-end 仅有原 freeze 未给 SHA 的外部候选；archive predictor member 也为不匹配的 SHA。已新增 checkpoint-free `scripted_evasive_vpp` 作为第三独立对手；curriculum 仍只是 end_to_end 调度。旧 preservation freeze 的自哈希不一致已用保留历史记录的 superseding freeze 收口。
- **问题分类：** checkpoint 缺失/不匹配为 `physical_unavailable`；第三对手 registry 已就绪；freeze 哈希不一致已作为证据接口错误收口；没有数学错误、控制器安全失败或对手性能结论。
- **是否允许进入下一阶段：** 不允许。P0 的资产门仍不满足；不得以未登记 checkpoint、simple backend、dry-run 或旧 telemetry 替代。
- **下一命令与验收：** 仅在恢复四个已登记且 SHA 一致的资产（对前三个 policy checkpoint 还必须建立经审核、显式 superseding asset manifest）后，运行 `PYTHONPATH=src python scripts/run_jsbsim_hrl_comparison.py --config config/experiment/jsbsim_hrl_comparison.yaml --run-id p0_smoke_<opponent> --preset main --opponent-stage <opponent> --run-status smoke --seeds 0 --n-episodes 1 --backend jsbsim --device cpu`。每个 opponent 必须完成一条 episode，telemetry 非空、hand-off 连续、除初始 reset 外无异常 reset/restart、无 fallback，且 `backend == "jsbsim"`、`backend_fallback_occurred == false`。
