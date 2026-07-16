# V22 Observer-Only Predictor 预注册

V22 在 V21 的接口死锁后引入 observer-only predictor update。warm-up 中，真实 target state 只用于
更新 predictor history；预测位置不得传入 VPP generator、guidance、PID、reward 或 policy observation。
每步必须记录 `observer_updated`、`observer_prediction_valid`、`observer_prediction_consumed=false`。

仅在 history 满 10 帧、observer receipt valid/no-fallback 后，允许一次性 release 到
`predicted_target`；release 后 observer output 才可进入 VPP anchor。三对手分别报告，72 条 strict-JSBSim
连续记录；不训练、不调参、不作性能结论。任一 warm-up prediction-consumed、post-release fallback、
第二次切换或 backend fallback 均立即冻结 Source ID。
