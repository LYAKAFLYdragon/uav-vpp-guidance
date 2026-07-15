# V9 Re-entry 3-D VPP 动作可控性 Probe 预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V9`  
**状态：** `preregistered_design_execution_not_authorised`  
**性质：** 非学习、非论文主结果的机制可行性 probe。

## 问题

V8 已在三个冻结对手下建立了物理单位、连续 re-entry 的 66-D 输入契约。随后 V1 终端奖励
pilot 在相同接口上安全完成 20,000 步，但没有达到实用改善门。该结果不能推出“应继续调大奖励”
或“训练步数不足”。在训练新策略前，必须先回答更基础的问题：**冻结 3-D VPP 动作在 raw-SI
re-entry 窗口内，能否安全且可重复地改变 `range_extension` 意图几何？**

## 冻结方法

- 使用与 V1--V8、Taxonomy30 和论文 heldout 物理不相交的 12 个新场景，三对手分别运行。
- 每个 episode 仅合法 reset 一次；先由 frozen fixed-head-on 完成连续 run-in，再在 raw-SI
  re-entry 触发后执行固定 20 个高层步的局部动作窗口。
- reference 为原始 frozen fixed-head-on 动作。六个比较方法只在该动作上加预注册的
  `+/-0.25` forward、lateral 或 vertical normalized VPP delta；所有裁剪都持久化记录。
- 训练、奖励读取、PPO、高层路由、VPP/guidance/PID 参数和对手均冻结；不使用 snapshot、
  future state、history padding 或 candidate checkpoint。

## 预注册判定

每个非 reference 方法按 opponent 分开计算与 reference 的 paired `intent_loss_auc20` delta。
一个方向仅在至少两个对手中同时满足以下条件时，才可标为“可用于下一条新机制线的 action-authority
candidate”：至少 4 个完整配对窗口、双镜像、至少两种高度、handoff hash 完全一致、平均实际
动作差范数 `>=0.15`、ego crash/OOB 差 `<=+0.05`，且 mean paired delta `<=-0.02`。

该 probe 不选择或训练最终策略。通过只允许为一个全新 Source ID 起草 action-conditioned
teacher 或受控 warm-start pilot；失败则说明当前 range-extension objective 在此 VPP 接口中
没有观察到安全、跨对手的局部可控改善，不得以“继续调 reward/多训几步”回应。
