# 控制与制导整改 P0 阶段总结

- **阶段状态：`physical_unavailable`。** 未执行任何训练、正式实验或 strict-JSBSim 闭环；该状态不能归因于控制器失败。
- **实际完成：** 确认指定 worktree、分支和 clean SHA；原样同步执行清单；核对审计、ledger、preservation baseline、配置哈希、场景矩阵、现有 portable bundle 与登记 checkpoint。
- **命令与验证：** `git rev-parse/status/worktree list`；登记 checkpoint 的跨 worktree 只读搜索与 SHA-256；`python -m pytest tests/test_control_guidance_remediation_phase0.py -v`；10 个既有 portable bundle 完整性验证。
- **生成物：** `outputs/control_guidance_remediation_v1/p0_evidence_freeze/` 及其 portable bundle；本总结；更新后的执行清单。
- **通过门：** 目标 worktree/分支/SHA 和初始 clean 状态；4 个冻结配置哈希；10 个既有 portable bundle 验证。
- **发现：** 当前 worktree 缺少全部四个登记 checkpoint；对 E 盘 32 个相关根目录的 3,726 个 loose checkpoint、25 个 ZIP/TAR/TGZ/TAR.GZ 容器和 11 个本地 Git/LFS 元数据的只读 SHA-256 盘点均未找到 `prediction_vpp` 登记路径或 predictor 冻结 SHA。authority-collection receipt 进一步确认其 `outputs` 是 CAN assets Junction、无源 Git/训练 run manifest；predictor `96bc…5931` 的严格结构加载通过但 SHA 不匹配，no-prediction/end-to-end 仍只是原 freeze 未给 SHA 的来源不完整候选。已新增 checkpoint-free `scripted_evasive_vpp` 作为第三独立对手；curriculum 仍只是 end_to_end 调度。旧 preservation freeze 的自哈希不一致已用保留历史记录的 superseding freeze 收口。
- **问题分类：** checkpoint 缺失/不匹配为 `physical_unavailable`；第三对手 registry 已就绪；freeze 哈希不一致已作为证据接口错误收口；没有数学错误、控制器安全失败或对手性能结论。
- **文献证据边界：** 已对第六章、实验计划、对手能力卡与 CAN 权威索引进行原始字节哈希核验。它们仅证明方法/配置记录或候选 identity，不能替代缺失 checkpoint bytes、要求的 predictor 精确 SHA 或独立训练 provenance；因此 predictor 仍为 `rejected_hash_mismatch`，prediction-VPP 仍为 `physical_unavailable`，no-prediction/end-to-end 仍为 `unregistered_candidate_not_adopted`，禁止创建 superseding asset manifest 或执行 strict-JSBSim smoke。
- **是否允许进入下一阶段：** 不允许。历史 P0 的资产门仍不满足；不得以未登记 checkpoint、simple backend、dry-run、旧 telemetry 或文献记录替代。已建立独立的 `UAV-VPP-GUIDANCE-RETRAIN-NON-EQUIVALENT-20260727-R01` 新 Source 路线，但它不能改变历史 P0 状态、替代历史资产或形成等同性主张。
- **下一命令与验收：** 历史路线仅在恢复四个已登记且 SHA 一致的资产（对前三个 policy checkpoint 还必须建立经审核、显式 superseding asset manifest）后，才可运行历史 strict-JSBSim smoke。独立新 Source 路线须先提交新 trajectory CSV corpus 的 SHA-256 数据 manifest、采集配置/命令、种子、Git 信息和审核批准；然后执行 `PYTHONPATH=src python scripts/prepare_non_equivalent_retraining.py --data-dir <new-corpus>`，以生成隔离的 resolved config 和 source preflight，再按新 Source registry 依次训练和登记四个新模型。两条路线的资产、结果和结论不得混用。
