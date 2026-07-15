# P2-B2 Strict JSBSim Physical Preflight 一次性执行授权

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT-V1`  
**授权状态：** `one_shot_execution_authorized`  
**冻结实现 commit：** `7e9092caf3e109728ae7c863cfdc031c484715c8`

## 授权范围

仅允许使用 frozen `run_in_head_on` specialist，在 60 个预检 geometry cell、三个独立报告 opponent（`expert`、`end_to_end`、`independent_ppo_vpp`）上运行 180 条 fresh-environment strict-JSBSim 记录。该运行是非学习的 physical-reachability preflight：不加载候选五态势策略，不训练，不调参，不产生 held-out 性能或优越性声明。

唯一执行命令：

```powershell
python scripts/run_thesis_global_advantage_p2_physical_preflight.py --execute
```

## 冻结输入与实现

| 项目 | SHA-256 / 标识 |
|---|---|
| 预检 manifest | `78d90d6ea32490cc777d13e0a809061ec300f94ff6d7d012f93a9e9895ba834f` |
| runtime config | `3a050de1eac5258fb68ae52dea71fc9cc8c9ad1bea69a73c0cd403a56e40b1f2` |
| runtime registry | `c87edfed4be69a8293a889c9caa55c520fdab7dd93c44c9c6e057be869f435c5` |
| runner | `27cbbe61a10d871788a1e18a9606e79957c617e032b0312199f6e9cabb9674c4` |
| run-in head-on specialist | `0aeadd11cd8c7723c8963bcc5da21dd97c365e64f234d07b86c2b8f0d7ec5fb6` |
| end-to-end opponent | `d38828338d5e7e22167129b2e9972d52ecf5fc3fda5d52caff95a8374c35b8bc` |
| independent PPO/VPP opponent | `43cd37fc86c0eb01d96a693f68d30894de7e768de01aed6dba41fa4761edfc60` |

执行器会 fail-closed 地检查上述输入、六个授权实现文件、clean worktree、冻结 commit ancestor、仅授权材料差异、空输出根和 `>=120 GB` 空闲磁盘。当前输出根必须不存在：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p2_physical_preflight`。

## 判定与停止规则

- 物理预检通过仅指 `180/180` 条记录完成、strict JSBSim、无 backend/prediction fallback、动作与记录状态有限且每条至少 20 个 step。
- `pre_merge/post_merge/re_entry` 覆盖矩阵独立输出；phase gap 是后续相关技能训练的阻塞证据，不能篡改或否定物理可达性记录。
- 任何异常、输入哈希不一致、fallback、输出根已存在或磁盘不足均立即停止。不得删除失败输出、替换场景、增加 episode、重跑同一 Source ID，或以调参绕过结果。
- 无论 gate 结果如何，运行结束后都将 `execution_permitted` 复位为 `false`，并将输出作为正面或负面证据归档；它不会自动解锁 P3--P7。
