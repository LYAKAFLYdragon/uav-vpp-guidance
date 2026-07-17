# Tactical-Basis Continuity Collector V3 结果与决策

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V3`
**运行提交：** `cad656244f698d5cb5c5ab83db49c9f32e7f3c9f`
**性质：** non-learning continuous baseline data-contract evidence，`paper_safe=false`

## 结果

36 条 strict-JSBSim 连续 baseline 记录均已生成，并由独立 verifier 逐文件哈希和逐窗口重算。预注册 phase + contract gate 通过：

| Opponent | Qualifying handoff | Distinct signatures | 双镜像 | Gate |
|---|---:|---:|---:|---|
| expert | 5 | 3 | 是 | 不通过 |
| end-to-end | 5 | 4 | 是 | 通过 |
| independent PPO/VPP | 7 | 4 | 是 | 通过 |

17 条 qualifying handoff 共提供 680 个完整 window step。它们全部处于 `disadvantage -> post_merge`，均具有完整 10-frame history、rank-3 tactical basis、窗口内零 prediction fallback、零 history padding；basis condition number 范围为 2.67--3.10。

## 可得与不可得结论

- 可以确认：当前冻结 `tactical_basis_v1` fixed-head-on baseline 能在两名独立对手条件下提供足够的连续、同语义、40-step 后合并数据，以设计新的 P2 offline envelope screen。
- 不能确认：re-entry 覆盖、temporal-reference 的几何改善、跨三对手普适性、训练收益或胜率优势。
- expert 的 5 条样本仅覆盖 3 个 signature，不能被补跑或与其它对手合并来满足其门槛。

## 可复核产物

- `continuity_gate.json` SHA-256: `5b5e71455074f19a0d179ab86dc5e19a50c9c6f438b968166e60341994d147ba`
- `continuity_verification.json` SHA-256: `bb968a90c1e067a7d455fe651046eaca3b8195dd569dd10c33212e2b2de14396`
- `records_index.json` SHA-256: `01ae3c96eeb39ac5818984c8d2b4f595c0879e656c6f3acee7b8a1293e2169ea`
- `run_manifest.json` SHA-256: `2d1651970bd60d1cdf851a7508c49e54c1d4689774b9d65a730fbcbdc01e8d9f`

## 正式决策

允许起草新的独立 Source 的 **post-merge P2 offline envelope screen**。该资格不自动授权执行 P2、训练新技能、修改 reward、扩大动作包线或重跑 V3。
