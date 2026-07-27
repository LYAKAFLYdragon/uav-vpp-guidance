# P0 登记资产恢复检索总结

- **阶段状态：`physical_unavailable`。** 未复制、解包、修改或采用 checkpoint；未运行 strict-JSBSim。
- **只读检索：** 哈希 32 个相关根目录中的 3,726 个 loose checkpoint；检查 25 个 ZIP/TAR/TGZ/TAR.GZ 容器；检查 11 个本地 Git 历史与 Git-LFS 元数据；没有剩余 7z/RAR 类容器。
- **结果：** 未找到登记的 `prediction_vpp` 路径，也未找到 predictor 冻结 SHA `a7c00fd2…d74491`。loose predictor 候选为 `96bc8e5e…95931`，两份 archive member 均为 `c4bf3d8d…f8629`，都不匹配。
- **候选政策：** no-prediction `82cc54df…ef2cb` 和 end-to-end `d3882833…5b8bc` 的 checkpoint payload 都声明 `obs_dim=16`、`action_dim=3`，但原 freeze 对二者没有 SHA，且没有经审核的新资产契约；因此未采用。
- **结论与下一门：** 不创建 superseding asset manifest，不复制外部候选。必须先获取登记 prediction-VPP bytes 与精确 predictor bytes，再为两份 policy 候选建立具来源、SHA、大小与兼容性证明的审核 manifest；然后才可开始三对手 strict-JSBSim smoke。
