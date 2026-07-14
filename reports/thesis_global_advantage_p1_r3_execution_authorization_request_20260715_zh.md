# P1 R3 一次性执行授权请求清单

**状态：** `not_authorized`
**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R3-FRESH-ENV-REPRO-V1`
**范围：** 仅 30 个 dev 场景 x 3 对手 x 3 repeat 的非学习 run-in 可重复性复核，共 270 episode。它不是训练、技能比较、消融或 formal held-out。

## 已具备的实现证据

| 项目 | 证据 |
|---|---|
| 独立环境/进程 | R3 parent 每个 launch 只启动一个 Python child；child 只创建一个 environment，并在回合后退出 |
| 完整 reset 证据 | `capture_runtime_envelope()` 保存完整 observation、history、predictor、guidance/PID、opponent、FDM property whitelist 与 reference metadata |
| 逐步证据 | 每步记录 reference/opponent action、3-D VPP、runtime envelope hash、VPP/guidance/PID telemetry 与 terminal 语义 |
| 失败关闭 | 任一 hash、PID、fallback、telemetry、checkpoint、backend 或来源不一致都会使 cell 和总 gate 失败 |
| 默认锁定 | config 中 `execution_permitted=false`，`--execute` 已确认返回拒绝；训练与 held-out 权限均为 false |
| 定向测试 | `python -m pytest tests/test_global_advantage_p1_r3_contract.py tests/test_global_advantage_runin_contract.py -q`，当前 11 passed |

## 授权前的两次提交

1. **实现冻结提交**：提交所有 P1 R1/R2/R3 代码、配置、报告和测试；记录该 clean commit 的 SHA 为 `IMPLEMENTATION_SHA`。
2. **独立复核后授权提交**：仅修改 R3 config 的授权字段；将 `execution_permitted` 改为 true，填写 `required_implementation_git_sha=IMPLEMENTATION_SHA` 和下表中的 SHA-256。该提交后 worktree 必须重新 clean。

配置检查的是“`IMPLEMENTATION_SHA` 是当前 HEAD 的祖先 + 已授权代码文件哈希完全一致 + worktree clean”。这样授权 config 自己的提交不会造成 SHA 自指，同时不能靠后续代码修改绕过冻结。

## 必须填写的授权代码哈希

以下列表应在第一步的 clean implementation commit 上计算；所有路径相对仓库根目录。

| 路径 | SHA-256 | 复核人 |
|---|---|---|
| `scripts/run_thesis_global_advantage_p1_r3_fresh_environment.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |
| `src/uav_vpp_guidance/evaluation/global_advantage_p1_r3_contract.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |
| `src/uav_vpp_guidance/evaluation/global_advantage_runin_contract.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |
| `src/uav_vpp_guidance/training/thesis_defext_rangeext_pilot.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |
| `src/uav_vpp_guidance/training/thesis_shared_skill_geometry.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |
| `src/uav_vpp_guidance/envs/tracking_env.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |
| `src/uav_vpp_guidance/envs/jsbsim_env.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |
| `src/uav_vpp_guidance/hierarchy/specialist_policy.py` | `TO_BE_FILLED_AFTER_IMPLEMENTATION_COMMIT` | `TO_BE_FILLED` |

## 独立复核项

- [ ] 复核 child 的唯一输入是 frozen launch record，且 parent 不构造或缓存 `CloseRangeTrackingEnv`。
- [ ] 复核 snapshot 覆盖 observation、history、predictor buffer、guidance、两侧 PID、command filter、opponent、FDM property whitelist 与 reference action。
- [ ] 复核任何未支持对象、非有限值、缺少 FDM property、重复 PID 或 timeout 都 fail-closed。
- [ ] 复核三对手 checkpoint SHA、head-on specialist SHA、P3 checkpoint SHA、runtime registry SHA 和 scenario manifest SHA。
- [ ] 复核 `R3` 的 30 个 dev 场景不等于 heldout240，且没有候选五态势 policy、reward 或 profile 被加载。
- [ ] 复核输出目录不存在，目标磁盘空闲空间不少于 120 GB。
- [ ] 复核 11 个定向测试、`py_compile` 和 `python scripts/run_thesis_global_advantage_p1_r3_fresh_environment.py` 的未执行验证均通过。

## 一次性运行与判定

授权提交后，唯一许可命令为：

```powershell
python scripts/run_thesis_global_advantage_p1_r3_fresh_environment.py --execute
```

允许的通过标准是 **90/90** `opponent x scenario` cell 同时满足：完整 reset envelope、首个双方动作、逐步 trajectory、boundary envelope、terminal reason 一致；三个 PID 不同；270/270 artifact 完整；无 backend/prediction/checkpoint fallback。

任一失败均写为 `runin_protocol_not_reproducible_do_not_train`。禁止在失败后调整 seed、量化精度、场景顺序、超时、控制参数或重新运行失败 cell；允许的下一步只是在已记录的完整状态哈希基础上定位模拟/评估器状态边界。
