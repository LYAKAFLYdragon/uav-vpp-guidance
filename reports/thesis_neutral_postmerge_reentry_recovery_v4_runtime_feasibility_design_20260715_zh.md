# V4 Reward-Unobserved Runtime-Feasibility 设计记录（2026-07-15）

## 目的与边界

- Source ID：`THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V4`
- 状态：`preregistered_design_execution_not_authorised`
- 当前阶段：静态设计、合同测试与 preflight 已完成；**尚未实例化 JSBSim，未创建 V4 输出根，未执行任何 episode**。

V4 的唯一目标是回答一个比训练更前置的问题：在连续、严格 JSBSim run-in 下，冻结的 head-on baseline 是否能在独立的 neutral/pre-merge 场景中产生具有完整 66-D、3-D VPP、phase 与执行链 telemetry 的 post-merge 支持。它不训练低层技能，不加载 candidate policy，不训练高层 PPO，也不生成论文 performance claim。

V1、V2 与 V3 都是 `implementation_failure_not_experimental_evidence`。尤其 V3 在首次 post-handoff 时因训练 episode 对 `geometry_reward` 的隐式依赖停止；V4 不通过填充该 reward 字段掩盖失败，而是使用独立的“训练 reward 不使用、environment reward 不记录”的 post-handoff record path。

## 冻结设计

| 项目 | V4 固定值 |
|---|---|
| 场景 | 新的、与 V1/V2/V3 及既有 manifests 不重叠的 48 个 neutral/pre-merge 场景 |
| 结构 | 4 family x 2 distance-speed package x 3 height x 2 mirror |
| 对手 | `expert`、`end_to_end`、`independent_ppo_vpp`，独立报告，禁止 pooled gate |
| 控制器 | `frozen_fixed_head_on` run-in 与 post-handoff baseline；routing 关闭 |
| 观测与动作 | 真实连续 10-frame history、冻结 P3 encoder、66-D observation、归一化 3-D VPP action |
| 物理链 | strict JSBSim、prediction-VPP、guidance、PID、attack-zone telemetry 不变 |
| reward | 禁止训练 reward 使用与 environment reward 记录；V4 config 不含 `proposed_training`，record-only metric/result 禁止 `reward` 字段 |
| 输出 | `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_v4_runtime_feasibility`，当前必须不存在 |

关键输入 SHA-256：

| 输入 | SHA-256 |
|---|---|
| V4 manifest | `8b83069c9e1949b1d5b59711936691dec413349471946c56d3c474c37f5410e4` |
| runtime template | `2a9fb82721d82cd67e13e08157d5cc9d9eac3deafa04b12e253c8b6098c38806` |
| runtime registry | `e3124120b33b55c1018da6d8d133ef15b5cf5e6c5c04361d6b90498b0b279ca8` |
| frozen P3 encoder | `385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59` |
| frozen head-on specialist | `0aeadd11cd8c7723c8963bcc5da21dd97c365e64f234d07b86c2b8f0d7ec5fb6` |

## Gate 与停止规则

每个 family 在每个 opponent 下都必须有完整 12 条 record，并达到：

1. 至少 8 条 qualifying episode。
2. 至少 160 个 valid target step。
3. 同时覆盖 2 个 mirror 和 3 个高度条件。
4. 零 backend/telemetry contract failure。

144 条 `opponent x pair_key` 是不可缺少的 universe；缺失、重复或额外 record 是合同失败，不是零分或性能结果。任何 backend fallback、receipt/phase replay 失败、异常 reset、padding、future-state injection、非有限/越界 action、serializer failure 或 reward 字段泄漏，都必须停止并冻结 V4，禁止修改或重跑同一 Source ID。

即使有 family 通过，结论也只允许起草新的单技能 pilot 预注册；不会自动解锁四共享技能、combat finetune、高层 PPO、P5/P6/P7 或 formal heldout。

## 当前验证

以下检查在未执行 JSBSim 的情况下通过：

- `python scripts/build_thesis_neutral_postmerge_reentry_recovery_v4_feasibility_manifest.py --output <temp>` 与受控 manifest SHA-256 完全一致。
- `python scripts/preflight_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py`：48 场景、144 计划 record、输出根不存在、约 291 GB 可用空间，满足 120 GB 门槛。
- `pytest tests/test_thesis_neutral_postmerge_v4_runtime_feasibility.py tests/test_thesis_neutral_postmerge_v3_runtime_feasibility.py tests/test_common_provenance.py -q`：`23 passed`。

正式运行若被单独授权，runner 会把 authorized overlay、frozen base config、frozen feasibility manifest 和 artifact provenance 复制到输出根。后续独立 verifier 将只读取这些快照与 raw episode records，而不会依赖运行后可能变化的工作树文件。

独立 verifier 为 `scripts/verify_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py`。它必须重新计算 144 条 `opponent x pair_key` universe 与 per-opponent family gate，并验证 raw-SI phase replay、66-D/3-D telemetry、reward 字段隔离、serialized overlay/base/manifest 哈希，以及 runner gate 与独立 gate 一致。它不写入 bundle；可选 `--report` 才会写入用户指定位置。

## 下一步授权条件

当前配置的 `execution_permitted=false`。如需运行，必须在**新的干净提交**上单独创建 authorized overlay；它只能改变 `status`、authorization、execution hash whitelist 与 output-root creation permission。overlay 必须以创建前 clean HEAD 作为 `required_implementation_git_sha`，并重新绑定 base config、runner、preflight、manifest builder、V4 contract、V4 record-only path、run-in engine、环境与对手适配器的 SHA-256。runner 会拒绝缺失或额外的白名单路径，不能以部分哈希列表执行。

在获得该独立授权前，唯一允许动作是只读审计和静态测试；不得执行 V4 JSBSim、训练或调参。
