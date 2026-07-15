# V7 Turn-back Teacher Feasibility 结果

> **2026-07-15 phase-contract correction.** The turn-back teacher run is
> preserved, but its legacy `re_entry=0` label is not physical negative
> evidence: the tracker received normalized observations against physical-unit
> thresholds and had no first-pass prerequisite. This report must not be used
> to claim that re-entry was absent or infeasible. See
> `reports/thesis_five_state_v5_v7_phase_contract_audit_20260715_zh.md`.

`THESIS-FIVE-STATE-REENTRY-TEACHER-FEASIBILITY-V7` 在与 V1--V6 物理不相交的 12 场景中完成 72 条 strict-JSBSim record：两种预注册镜像 lateral turn-back gain（`-0.8`、`+0.8`）各对三个 opponent 运行 12 条。

每个 `gain x opponent` 单元均为 0 条 re-entry episode、0 个连续 re-entry 66-D step，未满足最小 2 episode/20 step gate。teacher 在 raw-SI first-pass 后才激活，不读取 future state、不 reset、不训练；VPP、guidance、PID 与现有保护器未改变。

这与 V5（冻结 head-on post-merge）、V6（冻结 head-on/crossing control matrix）构成三轮一致的 re-entry 数据契约负证据。当前项目不能诚实地启动或声称覆盖 reentry_recovery 的完整多阶段共享技能系统。继续在同一任务定义中增加训练步数、修改 reward 或事后扩展 action grid 都不会是可解释的验证。

后续若要恢复全阶段目标，必须由用户/研究设计明确引入新的、独立的任务动力学或可验证的 re-entry reset/trajectory protocol；在此之前，本目标在“多阶段 re-entry”要求上没有可继续的内部实验路径。
