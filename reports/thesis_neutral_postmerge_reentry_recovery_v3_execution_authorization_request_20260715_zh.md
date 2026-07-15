# V3 Runtime-Feasibility 一次性执行授权请求

**拟执行 Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V3`  
**当前状态：** `request_only_not_authorised`  
**当前 canonical SHA：** `a843f2cdc195d01136547660311516b63e62bbca`  
**基准配置 SHA-256：** `363c8ab6c3443836f6e3d127feb0df0ea5cf471ce5ba6d23f6069c8558e98fc1`

## 请求的唯一操作

在新的一次性 authorized overlay 下，运行 48 个冻结 neutral/pre-merge 场景与三个冻结
opponent，共 `48 x 3 = 144` 条 strict-JSBSim **非学习** record。每条 episode 使用
frozen fixed head-on specialist 进行 run-in 和 post-handoff reference；不加载、训练或
选择 candidate policy。

输出根固定为：

```text
E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/
neutral_postmerge_reentry_recovery_v3_runtime_feasibility
```

它必须在执行前不存在。该输出一律 `paper_safe=false`，只能回答 runtime phase support
是否存在，不能写入论文结果。

**授权路径预检：** 已在仓库外临时构造 overlay，以本节 SHA、base-config hash 和完整
代码白名单运行 `--preflight`。结果为
`authorized_preflight_no_jsbsim_no_output_creation`，确认 48 个场景、144 条计划 record、
clean SHA 与 fresh output root 均可验证；临时 overlay 已删除，未生成正式授权配置、
未创建 output root，亦未运行 JSBSim。

## 必须保持冻结的内容

| 项目 | 固定值 |
|---|---|
| 场景 | manifest48 SHA-256 `0dbc1748e528dadd6112577f095d05a11a774eb33bc8b9f67c896e22ff49aa30` |
| train identity contract | SHA-256 `805e5689f3e0e8ecc72317a929679e6fa115979e1ed77f14fe4237ec1665b52d`，仅审计、不得采样或训练 |
| P3 encoder | SHA-256 `385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`，冻结 |
| run-in specialist | SHA-256 `0aeadd11cd8c7723c8963bcc5da21dd97c365e64f234d07b86c2b8f0d7ec5fb6` |
| runtime template | SHA-256 `2a9fb82721d82cd67e13e08157d5cc9d9eac3deafa04b12e253c8b6098c38806` |
| runtime registry | SHA-256 `e3124120b33b55c1018da6d8d133ef15b5cf5e6c5c04361d6b90498b0b279ca8` |
| 对手 | expert、end-to-end、independent PPO/VPP，分开报告，禁止 pooled gate |
| 控制链 | prediction-VPP、guidance、PID、attack-zone、66-D -> 3-D VPP、AoA60 语义均不变 |

授权 config 必须将所有 training、tuning、candidate loading、高层 PPO、四技能、combat
finetune、reward/VPP/guidance/PID change、reset、future-state injection、history padding
和 heldout claim 权限固定为 `false`。

## 执行实现白名单

authorized overlay 至少应 hash 绑定下列文件，且 `required_implementation_git_sha` 必须是
当前 SHA 或包含当前 SHA 的无关代码变更前 clean descendant：

| 文件 | SHA-256 |
|---|---|
| `scripts/run_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py` | `4fc8e180e49cffab61d1996353f2318c017e4449f94447f14f39d0fa0bb4dee3` |
| `scripts/preflight_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py` | `694e0dd5f7a96607c0d41d19694dcd965a263f50a0c19e1a1fa32556c36539d4` |
| `scripts/build_thesis_neutral_postmerge_reentry_recovery_v3_feasibility_manifest.py` | `35dc5fea3c8d18e49cf6b0b946f53f9b94d374d103d6b21cec7f41b5318ac861` |
| `src/uav_vpp_guidance/evaluation/thesis_neutral_postmerge_v3_contract.py` | `286fdf9f39865a31773144a21ce621aa3287e4e8348d269ef3329c59477950de` |
| `src/uav_vpp_guidance/evaluation/thesis_neutral_postmerge_v3_runtime.py` | `c550a26c359af217a56d693980c0a70b881e4f0d2b125c3754ee18416d799e07` |
| `src/uav_vpp_guidance/training/thesis_neutral_postmerge_reentry_recovery_pilot.py` | `dd55d0cca4f223f0d89deda6e4a45907f6bc8c6282e96e7b6f955f7221fcc49c` |
| `scripts/run_thesis_global_advantage_p2_b3_phase_observability.py` | `808096729748c015f2d348dc09f2acacc3063afa427ed0e5090e49535c03d5ee` |
| `src/uav_vpp_guidance/envs/tracking_env.py` | `7596f6b7250c52f76db5c7612496a1f5afcbb2bdd2dba934ce066b9c4f3805a3` |

同时必须将 runtime registry 的所有 checkpoint、opponent adapter、specialist、prediction
与 encoder 依赖纳入 `authorized_code_files` 或单独的 asset hash verifier；不得仅锁定
V3 wrapper 而遗漏实际 JSBSim 控制链。

## 通过、停止与解释边界

每个 family 在每个 opponent 下都必须有完整 12 条 record，并满足 `>=8` qualifying
episode、`>=160` valid target step、2 mirror、3 height、零 backend/telemetry contract
failure。统计前还必须精确覆盖 144 个 `opponent x pair_key`；缺失、重复或额外记录是
contract failure，不是零分或性能结果。

任何 episode 出现以下项，立即保留已写 output 为 failure evidence，并停止；不调参、
不加步数、不换场景、不重跑同一 Source ID：

- JSBSim backend fallback、非有限/越界 3-D action、checkpoint fallback；
- scenario-application receipt、raw-SI phase replay 或初始 pre-merge 语义失败；
- history padding、future-state injection、异常 reset；
- 144-record universe 不完整、身份不匹配、handoff/telemetry serializer 失败。

若 144 条均完成但没有 family 通过，V3 结论仅为
`v3_no_cross_opponent_runtime_feasible_family`。若存在通过 family，也只允许另起新的
single-skill pilot 预注册；它不自动证明新技能有效，更不解锁四共享技能、combat
finetune、高层 PPO、P5/P6/P7 或 formal heldout。

## 授权后唯一命令

授权 overlay 必须新建，不得修改下列 design config：

```powershell
D:\Anaconda3\envs\jsbenv\python.exe `
  scripts\run_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py `
  --config config\experiment\thesis_neutral_postmerge_reentry_recovery_v3_authorized.yaml `
  --execute
```

在执行前必须先使用同一 overlay 运行 `--preflight`，并确认 clean worktree、输入 hash、
资产 hash、输出根不存在以及至少 120 GB 可用空间。
