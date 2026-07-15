# V17 Path-Context 74-D 非学习预检结果

## 正式结论

`THESIS-FIVE-STATE-PATH-CONTEXT-PREFLIGHT-V17` **未通过**预注册的三对手
path-context coverage gate。72 条严格 JSBSim 记录已完整生成，独立 verifier 已从 episode record
重放 74-D path context，并确认报告矩阵一致；但是 end-to-end 对手只形成 4 条 qualifying
episode 和 80 个有效 path-context step，低于每个对手分别要求的 `8` 条和 `160` 步。

因此，V17 冻结为一条**对手条件下的连续覆盖负证据**。它不授权 path-conditioned PPO、
新的低层 skill、reward 调节、combat finetune 或 heldout 性能评估。

## 预注册门与结果

每个 opponent 必须独立满足：24 records、至少 8 条 qualifying continuous episode、至少
160 个有效 74-D step、至少两种 mirror 与两种 scenario signature、观测与动作全有限、
无 prediction/backend/checkpoint fallback、并且 ego crash/OOB 为零。anchor 0 和 2 也必须
出现。

| 对手 | Records | Qualifying episodes | Valid 74-D steps | Mirrors | Signatures | Anchors | Crash/OOB | Gate |
|---|---:|---:|---:|---|---:|---|---:|---|
| expert | 24 | 9 | 180 | +/- | 9 | 0, 2 | 0 | PASS |
| end-to-end | 24 | 4 | 80 | +/- | 4 | 0, 1, 2 | 0 | **FAIL** |
| independent PPO/VPP | 24 | 11 | 220 | +/- | 11 | 0, 1, 2 | 0 | PASS |

end-to-end 的 failure 不是单一边界条件：qualifying episode 与有效 step 都恰为门槛的一半。
它同时满足镜像、场景 signature、anchor、有限性、预测有效性和零 ego crash/OOB 条件，故可确认的
归因仅是**该冻结 24-scene envelope 下的 post-first-pass phase-occupancy coverage 不足**，而非
74-D 编码损坏、动作接口错误或共同飞控执行链失效。V17 record 没有完整保存 target speed、altitude、
specific energy 或 attack-zone telemetry，不能进一步写成 target dynamics 的因果结论。

## 已证明与未证明的内容

### 已证明

- 固定 head-on 控制下，V1 66-D 前缀加 8-D path context 可连续构造并保存为有限的 74-D 向量。
- expert 与 independent PPO/VPP 两类对手在新、冻结、严格 JSBSim 场景中分别满足完整 coverage
  gate。
- 所有 72 条记录均使用 3-D raw VPP action，未发生 prediction fallback 或 ego crash/OOB。
- 独立 verifier 重放 path-context contract，并与 `path_context_preflight_gate.json` 的结果一致。

### 未证明

- 这不证明 74-D path context 对任意对手均有足够训练覆盖，更不证明其会改善胜率、几何、
  安全或五态势系统性能。
- 这不证明 end-to-end 对手下 path context 不可用；它只表明本预注册 probe 的有效样本量不足。
- 零 crash/OOB 不是安全认证，也不能替代三对手 coverage gate。

## 复现与保留性

- 实现 SHA：`5a4e6ed7a14db9e31d875d3ecfda42ea4c8a417d`
- 授权配置 SHA-256：`bfee408b398df0100a152d5524675eb43ae47e8547fff4f09bf55f0a055cf405`
- 场景 manifest SHA-256：`87bb61c5837541d45e0213a6af5c7b5a4a55a0c172a7c19e6315de50d9d3ba16`
- 输出根：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_v17_path_context_preflight`
- Gate：`path_context_preflight_gate.json`
- Independent replay：`independent_verification.json`
- End-to-end phase audit：[phase occupancy audit](thesis_five_state_v17_path_context_phase_occupancy_audit_20260716_zh.md)
- 代码层回归：`13 passed`（observation contract、manifest、preflight tests）。

本轮未训练策略、未加载 candidate checkpoint、未修改 reward、VPP、guidance 或 PID；这些限制均由
授权配置与 run manifest 记录。

## Go/No-Go 与下一步

**No-go：** 不得放宽 V17 门槛、补跑同一 manifest、添加训练步数、调 reward、替换对手，或将 V17
作为 path-conditioned policy 的训练授权。V17 的 expert 与 independent PPO/VPP 通过不能覆盖
end-to-end 的独立失败。

只读相位审计已经确认 end-to-end 的 24 条 episode 均锁存 first-pass，但其中 20 条没有出现
post-merge phase；详见上述 audit。由于 V17 不含完整的 target kinematics，下一步若要解释这个覆盖缺口，
必须以新 Source ID 预注册一个带 target/own kinematics、phase 和 terminal telemetry 的独立非学习
collector。它需使用全新且不重合的 manifest 和重新预注册的每对手覆盖门，不能被表述为 V17 的 rerun、
阈值放宽或训练授权。
