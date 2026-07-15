# V4 一次性执行授权请求（未授权）

## 拟执行范围

该请求仅适用于 `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V4`：使用 [V4 设计配置](../config/experiment/thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.yaml) 中的 48 个新场景与三类冻结对手，运行 `48 x 3 = 144` 条 strict-JSBSim reward-unobserved runtime record。

每条 episode 使用 frozen fixed-head-on specialist 完成连续 run-in 与 post-handoff baseline；不加载候选 skill/PPO checkpoint，不调用训练 reward，不训练、不调参、不改变 VPP、guidance、PID 或 attack-zone 语义。

## 授权前必须满足

1. V4 output root 必须仍不存在，且 E 盘可用空间不少于 120 GB。
2. 当前 worktree 必须 clean。
3. 新 overlay 只能含 `base_config`、`status`、`authorization`、`execution`、`outputs` 五个顶级字段。
4. overlay 必须将除 `execution_permitted=true` 外的所有权限保持 `false`。
5. `required_implementation_git_sha` 必须是创建 overlay 前的 clean HEAD；`authorization_delta_paths` 只能是该 overlay 文件。
6. overlay 必须重新哈希绑定 base config、V4 runner/preflight/builder、V4 contract/record-only module、run-in engine、tracking environment、三种 opponent adapter、feature projection、specialist policy 与 intent/profile validity 组件。
7. 同一 overlay 必须先通过 `--preflight`；该命令不得创建输出根或启动 JSBSim。

## 执行与停止

唯一正式命令为：

```powershell
D:\Anaconda3\envs\jsbenv\python.exe `
  scripts\run_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py `
  --config config\experiment\thesis_neutral_postmerge_reentry_recovery_v4_authorized.yaml `
  --execute
```

任何 backend fallback、receipt/phase replay 失败、异常 reset、history padding、future-state injection、非有限/越界 action、reward term 泄漏、serializer failure 或不完整 144-record universe，都必须停止并将 V4 冻结为 `implementation_failure_not_experimental_evidence`。禁止在同一 Source ID 下补字段、换场景、加步数、替换 checkpoint 或重跑。

若完整运行但没有 family 同时通过三对手 gate，结论仅为 `v4_no_cross_opponent_runtime_feasible_family`；不得把它解释为训练不足或修改结果后重跑。若存在通过 family，也仅能起草后续 single-skill pilot 预注册，不能自动进入四共享技能系统训练。
