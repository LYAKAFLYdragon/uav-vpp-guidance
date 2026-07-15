# P2-B6 Opponent-Conditional Target-Geometry Reachability Atlas 设计预注册

**拟议 Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-R1`
**状态：** `design_only_not_authorised`
**前置证据：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-R1` 已完成且为跨对手数据契约负证据。

## 1. 设计动机

B5 的问题不是 raw-SI phase、JSBSim、66-D 构造或 reference action 异常，而是预先指定的
`disadvantage -> post_merge/re_entry` motif 在三个 opponent 下不共同可达。B5 atlas 的最大跨对手
下界是 `neutral:post_merge=17` step、`crossing_entry:post_merge=6` step，且都只有一个 expert episode；
没有任何 state x phase 满足 2 episode、20 step、2 signature、2 mirror 的最小门槛。

因此 B6 不把“劣势再进入”或任何 profile 当成预设训练目标。它先回答更基础的问题：在冻结参考控制器与
不同对手条件下，哪些动态几何状态和交战阶段真的能构成共同、连续、可追溯的数据支撑。

## 2. 冻结方法边界

- 参考控制器固定为 B5 相同的 frozen head-on specialist；无高层 PPO、无 Oracle、无候选新技能、无 profile 选择、无 reward/VPP/guidance/PID 变更。
- 三 opponent 仍为 expert、end-to-end 与 independent PPO/VPP，分开报告，禁止建立 pooled 强度排名。
- 每 episode 仅一次合法 reset；连续运行，禁止 snapshot restore、future-state injection、handoff reset 与 history padding。
- phase 仍只使用 raw-SI `observation.relative_state.range_m/range_rate_mps`；policy vector 的归一化值只能用于诊断。
- 每一步保存 dynamic `(ATA, AA)` taxonomy、phase、first-pass、raw 16-D、10-frame real history、冻结 P3 embedding、原始 3-D action、prediction、VPP/guidance/PID response 与 terminal。
- 这是一项 reachability/observability 实验，不报告候选方法胜率、不产生论文主结果。

## 3. 独立包线

拟定 manifest 为 60 场景：`5 initial states x 3 height conditions x 2 mirrors x 2 new range/speed packages`，每个
opponent 一次，共 180 strict-JSBSim records。距离/速度包、seed、场景名和完整物理 state signature 必须同时与
P2-B2/B3/B4/B5、heldout240、dev30、heldout60、phase-v2 不相交；最终数值在 builder 生成前冻结，不得参考 B5
逐场景结果挑选。

## 4. 预注册 Readout 与候选规则

对每个 `dynamic state x {post_merge, re_entry} x opponent`，报告：

- valid physical step、qualifying episode、distinct signature、distinct mirror。
- first-pass 是否发生，phase replay、66-D/action finite、prediction fallback、terminal 与 PID response。
- profile-free physical reachability 与每个冻结 profile validity mask 分开保存，避免把 mask 当作物理不可达。

一个 cell 只有在**每个 opponent 分别**达到至少 2 qualifying episode、20 valid step、2 signature 与 2 mirror，且无
strict-JSBSim/continuity/prediction/66-D/action 合同失败时，才成为 `pilot-input candidate`。该状态不等价于训练授权。

若存在多个 candidate，选择规则在运行前固定：

1. 先按三 opponent 中最小 valid-step 数从高到低排序。
2. 并列时按最小 qualifying-episode 数、最小 signature 数、最小 mirror 数依次排序。
3. 仍并列时按预定义顺序 `post_merge` 优先于 `re_entry`，再按 `neutral, crossing_entry, advantage, head_on, disadvantage`。

若没有 candidate，B6 也是明确负证据：当前固定 reference 与五态势包线尚不能给共享技能训练建立跨对手输入输出契约。不得改 reward、增加训练步数、重跑或以单一 opponent 结果开始训练。

## 5. 允许的下一步

B6 当前只允许实现 manifest builder、只读 collector、analysis、tests 与一次性授权前 preflight。必须先单独复核
manifest 物理不相交性、ATA/AA 约定、raw-SI phase input、输出容量与 selection rule，才可能申请一轮执行授权。
四共享技能、defensive-extension pilot、P5/P6 和 formal held-out 在 B6 readout 前仍全部锁定。
