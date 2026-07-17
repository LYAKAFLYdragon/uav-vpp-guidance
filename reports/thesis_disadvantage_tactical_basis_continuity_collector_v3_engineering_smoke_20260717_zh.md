# Tactical-Basis Continuity Collector V3 Engineering Smoke

**Smoke ID：** `ENGINEERING-TACTICAL-BASIS-CONTINUITY-COLLECTOR-NOT-A-SOURCE-ID`
**Source under test：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V3`
**性质：** 工程运行链检查，不是实验，不产生性能或论文结论。

## 已验证

- 三个对手均完成一次 strict-JSBSim reset、冻结 P3 encoder inference、冻结 fixed-head-on action 和 tactical-basis VPP step。
- 每个对手的 active/configured action semantics 均为 `tactical_basis_v1`，basis rank 均为 3。
- 预测器在 11 个 smoke step 的前 9 步处于历史 warm-up fallback；第 10-frame history 形成后，最终检查步 prediction valid 且无 fallback。
- 进程采用项目既有 Windows 兼容设置 `KMP_DUPLICATE_LIB_OK=TRUE`，避免 JSBSim-ML/PyTorch OpenMP 重复加载中止。

## 复现命令

```powershell
D:\Anaconda3\envs\jsbenv\python.exe scripts\run_thesis_disadvantage_tactical_basis_continuity_collector_v1.py `
  --config config\experiment\thesis_disadvantage_tactical_basis_continuity_collector_v3.yaml --smoke
```

## 边界

smoke 只允许证明运行链可启动，不能证明 `disadvantage -> post_merge/re_entry` 覆盖、40-step trajectory contract 或任何控制性能。V3 正式 collector 仍需单独冻结、授权、运行并由独立 verifier 复核。
