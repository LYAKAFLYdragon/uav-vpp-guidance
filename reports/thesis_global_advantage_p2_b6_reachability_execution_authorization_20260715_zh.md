# P2-B6 Opponent-Conditional Reachability 一次性执行授权

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-R1`
**授权状态：** `completed_candidate_identified_execution_closed`
**冻结实现 commit：** `1477b75b6152900c1848b53c8c360e4f743a6c4e`

## 唯一允许命令

```powershell
python scripts/run_thesis_global_advantage_p2_b6_reachability.py --execute
```

该命令运行 60 个预注册的五态势场景与三个冻结 opponent，共 180 条 strict-JSBSim profile-free record。控制器固定为 B5 相同的 frozen head-on specialist；不训练、不选 profile、不替换动作，也不评估候选共享技能性能。

## 不可变条件

- 当前工作树必须 clean，且 `1477b75` 是当前提交祖先；config 所列 14 个运行相关源文件必须逐一匹配 SHA-256。
- manifest、runtime config、runtime registry、P3 encoder 和 reference specialist 必须匹配 config 中的冻结哈希。
- 输出根 `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p2_b6_opponent_conditional_reachability_r1` 必须不存在，E 盘可用空间至少 120 GB。
- phase tracker 仅使用 raw-SI relative state；动态 taxonomy 参数顺序固定为 `(ATA, AA)`；每条 episode 只允许一次 reset。

## Readout 与 Stop Rule

每个 `dynamic state x {post_merge,re_entry}` 在每个 opponent 分别统计有效步数、qualifying episode、signature 和 mirror。候选必须在三个 opponent 各自达到 2 episode、20 step、2 signature、2 mirror；候选选择规则已冻结在 B6 config 中。不得 pool opponent，也不得以候选出现作为训练授权。

无论结果如何，执行后立即关闭 `execution_permitted`。无 candidate 时，B6 固定为负证据，禁止改场景、调 reward、增加训练步数或重跑同一 Source ID；有 candidate 时，最多允许起草新的 pilot 输入预注册，四共享技能、P5/P6 与 formal held-out 仍不自动解锁。

## 已执行结果

本授权已于 2026-07-15 执行一次并关闭。180/180 strict-JSBSim record 完成，scenario application、raw-SI phase replay 与 step-0 `pre_merge` 全部通过。预注册排序选出 `neutral -> post_merge`：三个 opponent 的有效 step 为 1,461/1,247/1,430，qualifying episode 为 19/16/16，均覆盖 6 个 signature 和双镜像。因此仅允许**起草**该 cell 的独立 pilot 输入预注册；`training_unlocked=false`，不得直接训练。详见 `reports/thesis_global_advantage_p2_b6_reachability_completion_20260715_zh.md`。
