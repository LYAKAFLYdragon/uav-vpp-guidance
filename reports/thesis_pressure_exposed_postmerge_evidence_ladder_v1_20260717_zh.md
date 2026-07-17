# Pressure-Exposed Post-Merge 证据阶梯 V1

**正式 Source（尚未执行）：** `THESIS-DISADVANTAGE-PRESSURE-EXPOSED-POSTMERGE-COLLECTOR-V1`  
**目标：** 先建立真正可判定的 `disadvantage -> post_merge/re_entry -> target pressure` 共同 cohort，再讨论对手压力、模块归因或新技能。

## 已完成的前置事实

1. 对冻结 `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` 进行只读审计：36 条 `initial disadvantage × fixed_head_on` 原始回合中，三名对手各有 0 条满足 40 步连续 post-merge/re-entry 且至少 10 步目标攻击区暴露的窗口。
2. 因而，Taxonomy60 中此前的攻击区暴露发生在 pre-merge；它不能被重新解释为“控制器未能降低 post-merge 压力”。详细索引见 [只读审计](thesis_pressure_exposed_postmerge_taxonomy60_readonly_audit_20260717/pressure_coverage_audit.md)。
3. 新的 16 场景、8 个镜像 signature manifest 已冻结，与 Taxonomy60/旧 temporal-reference manifest 不重合。它只改变初始几何，不改变对手、预测器、VPP、制导、PID、奖励或 checkpoint。
4. 工程 smoke 已在 `expert`、`end_to_end`、`independent_ppo_vpp` 各完成一个完整严格 JSBSim 回合；三者均有完整 66-D 轨迹与预测器 warm-up 后有效遥测。见 [smoke receipt](thesis_pressure_exposed_postmerge_collector_v1_engineering_smoke_20260717.json)。这只证明运行链可用，不构成正式结果。

## Step 1：唯一正式 gate

正式 collector 固定使用 `frozen_fixed_head_on`，关闭高层路由；每个对手运行同一份 16 场景 manifest，共 48 条完整连续回合。每个对手必须独立满足：

- 至少 4 条合格回合、至少 4 个 distinct scenario signature、正负镜像均出现。
- 每条合格回合有 40 步连续 `post_merge/re_entry` 窗口；窗口第一步为 `disadvantage`。
- 窗口内至少 10 步 `target_in_attack_zone=true`，即有真实、持续的待降低压力端点。
- 每步均有有限的 16-D base、66-D history/encoder observation 与 3-D VPP action；不允许 padding、reset、backend/prediction/checkpoint fallback。
- 三名对手分开判定；禁止 pool。

### 预注册决策

| Gate 结果 | 允许的下一步 | 明确禁止 |
| --- | --- | --- |
| 三名对手均通过 | 在这个共同采集包上生成 capability card，并执行紧凑 M0–M4 消融 | 训练、调 reward、换场景、挑选 episode |
| 任一对手未通过 | 冻结为 opponent-specific pressure-availability 负证据；仅报告卡片中的压力缺失 | 以未受压样本评判 defensive-extension；重跑或替换场景 |

## Step 2：Capability Card（仅在 Step 1 通过）

每名对手基于同一个 fixed-head-on cohort 分开报告：目标攻击区暴露比例/连续长度、range/range-rate、AA/ATA、能量和高度变化、VPP 三轴、AoA、`n_z` command/actual、饱和、终端类型。它描述“对共同参考控制器施加什么压力”，不生成 Elo、绝对强度排名，也不把胜率解释为击杀率或飞行安全认证。

## Step 3：紧凑消融（仅在 Step 1 通过）

所有方法使用同一场景、seed、对手与压力窗口定义；不跨对手合并统计。矩阵固定为：

- M0：fixed head-on specialist。
- M1：fixed crossing specialist。
- M2：static Oracle task gate（特权参考）。
- M3：canonical PPO routing（部署接口）。
- M4：PPO 去除 crossing bootstrap/pre-merge lock；若 deterministic recovery override 仍存在，必须单列 requested/effective mode，不能伪装成纯 PPO。

主结论是压力端点、几何变化、VPP/飞控响应与配对 terminal 差异；win rate 仅为次要结果。消融只回答“路由、接口还是已有技能库”在哪个明确压力条件下起作用，不能自动证明缺少新技能。

## Step 4：训练边界

只有 Step 1 通过且 Step 3 显示 M0/M1/M2/M3 均无法在相同压力窗口形成可解释的脱离/能量恢复几何时，才可另起 Source 预注册单一 `defensive_extension` 或状态条件化接口 pilot。它不得继承本 Source 的输出，也不得因 gate 结果修改本 Source 的场景或阈值。
