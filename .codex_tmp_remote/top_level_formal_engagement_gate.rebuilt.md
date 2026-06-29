# Formal Engagement Gate

Source: `.codex_tmp_remote\top_level_formal_summary.rebuilt.json`

## Thresholds

- min_win_rate: 0.5
- min_effective_engagement_rate: 0.5
- min_damaging_win_rate: 0.25

## Groups

| group | eps | win_rate | effective_engagement | damaging_win | damage_dealt | damage_taken | ego/target/timeout | ready | action | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| crossing_feasible::end_to_end_rl::end_to_end::d0.5::r3.0::a60.0 | 480 | 0.00208 | 1 | 0.00208 | 41.9 | 49.9 | 0/1/479 | False | hold | win_rate_below_gate,damaging_win_rate_below_gate |
| crossing_feasible::end_to_end_rl::expert::d0.5::r3.0::a60.0 | 480 | 1 | 1 | 1 | 35 | 23 | 0/0/480 | True | formal_expand | - |
| crossing_feasible::no_prediction_vpp::end_to_end::d0.5::r3.0::a60.0 | 480 | 0.821 | 1 | 0.821 | 18.6 | 17.1 | 0/262/218 | True | formal_expand | - |
| crossing_feasible::no_prediction_vpp::expert::d0.5::r3.0::a60.0 | 480 | 0.947 | 1 | 0.923 | 45 | 28.5 | 1/24/455 | True | formal_expand | - |
| crossing_feasible::prediction_vpp::end_to_end::d0.5::r3.0::a60.0 | 480 | 0.906 | 1 | 0.906 | 22.1 | 15 | 0/312/168 | True | formal_expand | - |
| crossing_feasible::prediction_vpp::expert::d0.5::r3.0::a60.0 | 480 | 1 | 1 | 1 | 80.4 | 24.5 | 0/0/480 | True | formal_expand | - |
| head_on::end_to_end_rl::end_to_end::d0.5::r3.0::a40.0 | 480 | 0.998 | 1 | 0.998 | 48.5 | 43 | 0/0/480 | True | formal_expand | - |
| head_on::end_to_end_rl::expert::d0.5::r3.0::a40.0 | 480 | 0.00417 | 1 | 0.00417 | 35.2 | 41.4 | 0/0/480 | False | hold_fix_combat_outcome | win_rate_below_gate,damaging_win_rate_below_gate |
| head_on::no_prediction_vpp::end_to_end::d0.5::r3.0::a40.0 | 480 | 0.969 | 1 | 0.967 | 26.2 | 18.1 | 2/5/473 | True | formal_expand | - |
| head_on::no_prediction_vpp::expert::d0.5::r3.0::a40.0 | 480 | 1 | 1 | 1 | 56.3 | 29.5 | 0/1/479 | True | formal_expand | - |
| head_on::prediction_vpp::end_to_end::d0.5::r3.0::a40.0 | 480 | 0 | 1 | 0 | 25 | 31.5 | 0/0/480 | False | hold_fix_combat_outcome | win_rate_below_gate,damaging_win_rate_below_gate |
| head_on::prediction_vpp::expert::d0.5::r3.0::a40.0 | 480 | 0.155 | 1 | 0.152 | 29.3 | 58.1 | 3/0/402 | False | hold_fix_combat_outcome | win_rate_below_gate,damaging_win_rate_below_gate |
