# Tactical-Basis Continuity Collector V2 工程无效记录

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V2`
**运行提交：** `2ec12407dc505f26106a2c8f4fb72ee3efa4b6b8`
**输出根目录：** `E:/uav-vpp-guidance-disadvantage-capability-v1-results/tactical_basis_continuity_collector_v2`
**最终归类：** `engineering_invalid_openmp_runtime_abort_before_first_step`

V2 仅修复了 V1 的 raw-SI dataclass 字段，并通过预检和相关测试。执行时，JSBSim-ML 在首次启动阶段报告 `libiomp5md.dll already initialized`，进程中止于第一条环境记录之前。没有 episode JSON、VPP action、basis、phase、prediction 或飞控 telemetry，因此 V2 不构成相位覆盖或控制能力负证据。

V2 不得重跑。V3 先执行不占 Source ID 的 runtime smoke：每个对手各完成一次 strict-JSBSim reset、冻结 P3 inference、冻结 head-on action 和 tactical-basis telemetry 检查。smoke 失败时停止该线；通过时才允许新的 V3 单次 collector 授权。
