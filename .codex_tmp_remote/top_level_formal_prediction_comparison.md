# Prediction Comparison Summary

Source: `.codex_tmp_remote\top_level_formal_summary.rebuilt.json`

## Scope

- baseline_method: no_prediction_vpp
- methods: no_prediction_vpp,prediction_vpp
- tasks: head_on,crossing_feasible
- opponent_stages: expert,end_to_end
- missing_rows: -

## Rows

| task | opponent_stage | method | win_rate | effective_engagement | damaging_win | damage_dealt | damage_taken | damage_margin | ego_crashes | target_crash_or_oob | delta_win_vs_no_pred | delta_damaging_vs_no_pred | delta_margin_vs_no_pred |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| head_on | expert | no_prediction_vpp | 1 | 1 | 1 | 56.3 | 29.5 | 26.8 | 0 | 1 | 0 | 0 | 0 |
| head_on | expert | prediction_vpp | 0.155 | 1 | 0.152 | 29.3 | 58.1 | -28.7 | 3 | 0 | -0.845 | -0.848 | -55.6 |
| head_on | end_to_end | no_prediction_vpp | 0.969 | 1 | 0.967 | 26.2 | 18.1 | 8.11 | 2 | 5 | 0 | 0 | 0 |
| head_on | end_to_end | prediction_vpp | 0 | 1 | 0 | 25 | 31.5 | -6.52 | 0 | 0 | -0.969 | -0.967 | -14.6 |
| crossing_feasible | expert | no_prediction_vpp | 0.947 | 1 | 0.923 | 45 | 28.5 | 16.5 | 1 | 24 | 0 | 0 | 0 |
| crossing_feasible | expert | prediction_vpp | 1 | 1 | 1 | 80.4 | 24.5 | 55.9 | 0 | 0 | 0.0534 | 0.0771 | 39.4 |
| crossing_feasible | end_to_end | no_prediction_vpp | 0.821 | 1 | 0.821 | 18.6 | 17.1 | 1.47 | 0 | 262 | 0 | 0 | 0 |
| crossing_feasible | end_to_end | prediction_vpp | 0.906 | 1 | 0.906 | 22.1 | 15 | 7.09 | 0 | 312 | 0.0854 | 0.0854 | 5.62 |
