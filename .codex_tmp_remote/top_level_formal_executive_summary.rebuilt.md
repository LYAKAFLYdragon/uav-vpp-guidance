# Formal Executive Summary

Source: `.codex_tmp_remote\top_level_formal_summary.rebuilt.json`

## Overall

- can_run_large_scale_air_combat_experiments: True
- scope: combat_only_head_on_crossing
- recommended_action: proceed_large_scale_combat_experiment
- missing_required_tasks: -
- not_ready_required_tasks: -
- filters: -

## Tasks

| task | ready | best_config | win | effective | damaging | damage_dealt | damage_taken | ego/target/timeout | next_action |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| crossing_feasible | True | prediction_vpp::end_to_end::d0.5::r3.0::a60.0 | 0.906 | 1 | 0.906 | 22.1 | 15 | 0/312/168 | formal_expand_with_explicit_aoa60 |
| head_on | True | end_to_end_rl::end_to_end::d0.5::r3.0::a40.0 | 0.998 | 1 | 0.998 | 48.5 | 43 | 0/0/480 | formal_expand |
